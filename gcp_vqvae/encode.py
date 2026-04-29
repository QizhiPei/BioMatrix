#!/usr/bin/env python
"""
Protein 3D Structure Tokenization: PDB/CIF → VQ indices CSV

Merges two steps into one:
  1) PDB/CIF → H5 (backbone extraction via BioPython)
  2) H5 → VQ indices (GCP-VQVAE encoding)

Usage:
    python -m gcp_vqvae.encode \
        --input     /path/to/structures/          \
        --output    /path/to/output/               \
        --model_dir /path/to/vqvae3d_ckpt/large    \
        --vq_code_dir /path/to/vq_encoder_decoder

All config values are built-in defaults; no YAML file is needed.
"""

import os
import sys
import argparse
import tempfile
import shutil
import datetime
import csv
import functools
import math
import glob
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import yaml
import torch
import h5py
from box import Box
from tqdm import tqdm
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import broadcast_object_list
from torch.utils.data import DataLoader
from Bio.PDB import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.Polypeptide import PPBuilder

AMINO_ACID_3TO1 = {
    'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
    'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
    'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
    'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M',
    'ASX': 'B', 'GLX': 'Z', 'PYL': 'O', 'SEC': 'U',
}


# ---------------------------------------------------------------------------
# Step 1: PDB/CIF → H5   (adapted from pdb_to_h5_keep_all_chain_v2.py)
# ---------------------------------------------------------------------------

def _estimate_missing_from_distance(prev_ca, next_ca, ideal_ca_ca=3.8):
    try:
        x1, y1, z1 = prev_ca
        x2, y2, z2 = next_ca
        if any(math.isnan(v) for v in (x1, y1, z1, x2, y2, z2)):
            return None
    except Exception:
        return None
    dist = math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
    return max(0, int(math.floor((dist / ideal_ca_ca) * 1.2) - 1))


def _propagate_nan_residues(pos):
    updated = 0
    for i, residue_coords in enumerate(pos):
        is_fully_nan = True
        has_any_missing = False
        for atom_coords in residue_coords:
            if len(atom_coords) != 3:
                has_any_missing = True
                continue
            if any(math.isnan(v) for v in atom_coords):
                has_any_missing = True
            else:
                is_fully_nan = False
        if has_any_missing and not is_fully_nan:
            pos[i] = [[math.nan]*3 for _ in range(4)]
            updated += 1
    return updated


def _evaluate_missing_content(pos, max_missing_ratio=0.2, max_consecutive_missing=15):
    total = len(pos)
    if total == 0:
        return False, 'missing_ratio_exceeded'
    missing_flags = []
    for residue in pos:
        ca = residue[1] if len(residue) > 1 else []
        if len(ca) != 3:
            missing_flags.append(True)
            continue
        missing_flags.append(any(math.isnan(v) for v in ca))
    if sum(missing_flags) / total > max_missing_ratio:
        return False, 'missing_ratio_exceeded'
    longest, current = 0, 0
    for m in missing_flags:
        if m:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    if longest > max_consecutive_missing:
        return False, 'missing_block_exceeded'
    return True, ''


def convert_structure_to_h5(
    file_path, save_dir, max_len=1280, min_len=0, gap_threshold=5,
):
    """Convert a single PDB/CIF file to H5 files (one per chain)."""
    ext = os.path.splitext(file_path)[1].lower()
    use_cif = ext in ('.cif', '.mmcif')
    parser = MMCIFParser(QUIET=True, auth_chains=True) if use_cif else PDBParser(QUIET=True)
    structure = parser.get_structure('protein', file_path)

    ppb = PPBuilder()
    chains = [c for c in structure[0]]
    h5_paths = []

    for chain in chains:
        sequence = ''.join(str(pp.get_sequence()) for pp in ppb.build_peptides(chain))
        if len(sequence) < min_len:
            continue

        chain_id = chain.id
        model = structure[0]

        residues = [res for res in model[chain_id] if res.id[0] == ' ']
        if not residues:
            continue

        protein_seq = ''
        pos = []
        plddt_scores = []
        for residue in residues:
            protein_seq += AMINO_ACID_3TO1.get(residue.resname, 'X')
            try:
                plddt_scores.append(residue['CA'].get_bfactor())
            except KeyError:
                plddt_scores.append(math.nan)
            coords = []
            for key in ['N', 'CA', 'C', 'O']:
                if key in residue:
                    coords.append(list(residue[key].coord))
                else:
                    coords.append([math.nan]*3)
            pos.append(coords)

        # Gap handling
        for i in range(len(residues) - 1, 0, -1):
            cur_id, prev_id = residues[i].id, residues[i-1].id
            if cur_id[1] > prev_id[1] + 1:
                gap = cur_id[1] - prev_id[1] - 1
                insert_count = gap
                if gap > gap_threshold:
                    est = _estimate_missing_from_distance(pos[i-1][1], pos[i][1])
                    insert_count = min(gap, est) if est is not None else gap_threshold
                if insert_count <= 0:
                    continue
                nan_coord = [[math.nan]*3 for _ in range(4)]
                protein_seq = protein_seq[:i] + 'X'*insert_count + protein_seq[i:]
                pos[i:i] = [nan_coord]*insert_count
                plddt_scores[i:i] = [math.nan]*insert_count

        _propagate_nan_residues(pos)

        final_len = len(protein_seq)
        if final_len < min_len or final_len > max_len:
            continue

        is_valid, _ = _evaluate_missing_content(pos)
        if not is_valid:
            continue

        basename = os.path.splitext(os.path.basename(file_path))[0]
        h5_name = f"{basename}_chain_id_{chain_id}.h5"
        h5_path = os.path.join(save_dir, h5_name)
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('seq', data=protein_seq)
            f.create_dataset('N_CA_C_O_coord', data=pos)
            f.create_dataset('plddt_scores', data=plddt_scores)
        h5_paths.append(h5_path)

    return h5_paths


# ---------------------------------------------------------------------------
# Step 2: H5 → VQ indices   (adapted from inference_encode_v1.py)
# ---------------------------------------------------------------------------

DEFAULT_ENCODE_CFG = {
    'checkpoint_path': 'checkpoints/best_valid.pth',
    'config_vqvae': 'config_vqvae.yaml',
    'config_encoder': 'config_gcpnet_encoder.yaml',
    'config_decoder': 'config_geometric_decoder.yaml',
    'batch_size': 128,
    'shuffle': False,
    'num_workers': 0,
    'max_task_samples': 5000000,
    'vq_indices_csv_filename': 'vq_indices.csv',
    'mixed_precision': 'bf16',
    'tqdm_progress_bar': True,
    'compile_model': {'enabled': False},
}


def _record_indices(pids, indices_tensor, sequences, records):
    cpu_inds = indices_tensor.detach().cpu().tolist()
    if not isinstance(cpu_inds, list):
        cpu_inds = [cpu_inds]
    for pid, idx, seq in zip(pids, cpu_inds, sequences):
        if not isinstance(idx, list):
            idx = [idx]
        cleaned_idx, cleaned_seq = [], []
        for v, aa in zip(idx, seq):
            if v != -1:
                cleaned_idx.append(int(v))
                cleaned_seq.append(aa)
        records.append({
            'pid': pid,
            'indices': cleaned_idx,
            'protein_sequence': ''.join(cleaned_seq),
        })


def run_encode(h5_dir, output_dir, model_dir, vq_code_dir, gpu_id=0, batch_size=None):
    """Run VQ encoding on H5 files and produce vq_indices.csv."""
    sys.path.insert(0, vq_code_dir)
    from utils.utils import load_configs, load_checkpoints_simple, get_logging
    from data.dataset import GCPNetDataset, custom_collate_pretrained_gcp
    from models.super_model import (
        prepare_model,
        compile_non_gcp_and_exclude_vq,
        compile_gcp_encoder,
    )

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cfg = dict(DEFAULT_ENCODE_CFG)
    cfg['trained_model_dir'] = model_dir
    cfg['data_path'] = h5_dir
    cfg['output_base_dir'] = output_dir
    if batch_size is not None:
        cfg['batch_size'] = batch_size
    infer_cfg = Box(cfg)

    dataloader_config = DataLoaderConfiguration(non_blocking=True, even_batches=False)
    accelerator = Accelerator(
        mixed_precision=infer_cfg.mixed_precision,
        dataloader_config=dataloader_config,
    )

    now = datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S')
    if accelerator.is_main_process:
        result_dir = os.path.join(infer_cfg.output_base_dir, now)
        os.makedirs(result_dir, exist_ok=True)
        paths = [result_dir]
    else:
        paths = [None]
    broadcast_object_list(paths, from_process=0)
    result_dir = paths[0]

    vqvae_cfg_path = os.path.join(model_dir, infer_cfg.config_vqvae)
    encoder_cfg_path = os.path.join(model_dir, infer_cfg.config_encoder)
    decoder_cfg_path = os.path.join(model_dir, infer_cfg.config_decoder)

    with open(vqvae_cfg_path) as f:
        vqvae_cfg = yaml.full_load(f)
    configs = load_configs(vqvae_cfg)
    configs.train_settings.max_task_samples = infer_cfg.max_task_samples
    configs.model.max_length = infer_cfg.get('max_length', configs.model.max_length)

    with open(encoder_cfg_path) as f:
        enc_cfg = Box(yaml.full_load(f))
    with open(decoder_cfg_path) as f:
        dec_cfg = Box(yaml.full_load(f))

    old_cwd = os.getcwd()
    os.chdir(vq_code_dir)

    dataset = GCPNetDataset(
        h5_dir,
        top_k=enc_cfg.top_k,
        num_positional_embeddings=enc_cfg.num_positional_embeddings,
        configs=configs,
        mode='evaluation',
    )
    collate_fn = functools.partial(
        custom_collate_pretrained_gcp,
        featuriser=dataset.pretrained_featuriser,
        task_transform=dataset.pretrained_task_transform,
    )
    loader = DataLoader(
        dataset, shuffle=infer_cfg.shuffle,
        batch_size=infer_cfg.batch_size,
        num_workers=infer_cfg.num_workers,
        collate_fn=collate_fn,
    )

    logger = get_logging(result_dir, configs)
    model = prepare_model(configs, logger, encoder_configs=enc_cfg, decoder_configs=dec_cfg)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    ckpt_path = os.path.join(model_dir, infer_cfg.checkpoint_path)
    model = load_checkpoints_simple(ckpt_path, model, logger)

    model, loader = accelerator.prepare(model, loader)

    records = []
    pbar = tqdm(range(len(loader)), leave=True,
                disable=not (infer_cfg.tqdm_progress_bar and accelerator.is_main_process))
    pbar.set_description("Encoding")

    for batch in loader:
        with torch.inference_mode():
            batch['graph'] = batch['graph'].to(accelerator.device)
            batch['masks'] = batch['masks'].to(accelerator.device)
            batch['nan_masks'] = batch['nan_masks'].to(accelerator.device)
            output = model(batch, return_vq_layer=True)
            _record_indices(batch['pid'], output['indices'], batch['seq'], records)
            pbar.update(1)

    accelerator.wait_for_everyone()
    records = accelerator.gather_for_metrics(records, use_gather_object=True)

    csv_path = None
    if accelerator.is_main_process:
        csv_path = os.path.join(result_dir, infer_cfg.vq_indices_csv_filename)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['pid', 'indices', 'protein_sequence'])
            for rec in records:
                inds = rec['indices']
                if not isinstance(inds, (list, tuple)):
                    inds = [inds]
                writer.writerow([rec['pid'], ' '.join(map(str, inds)), rec['protein_sequence']])
        logger.info(f"Saved VQ indices to {csv_path}")

    accelerator.wait_for_everyone()
    accelerator.free_memory()
    accelerator.end_training()

    os.chdir(old_cwd)
    return csv_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Protein 3D Structure Tokenization: PDB/CIF → VQ indices')
    parser.add_argument('--input', required=True,
                        help='Path to a single PDB/CIF file or a directory of structure files')
    parser.add_argument('--output', required=True,
                        help='Output directory for vq_indices.csv')
    parser.add_argument('--model_dir', required=True,
                        help='Path to pre-trained GCP-VQVAE model directory '
                             '(contains config_vqvae.yaml, checkpoints/, etc.)')
    parser.add_argument('--vq_code_dir', required=True,
                        help='Path to vq_encoder_decoder source directory')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_len', type=int, default=1280)
    parser.add_argument('--min_len', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Collect input files
    input_path = args.input
    if os.path.isfile(input_path):
        files = [input_path]
    else:
        files = []
        for ext in ('*.pdb', '*.cif', '*.mmcif'):
            files.extend(glob.glob(os.path.join(input_path, '**', ext), recursive=True))

    if not files:
        print(f"[ERROR] No PDB/CIF files found in {input_path}")
        sys.exit(1)

    print(f"Found {len(files)} structure file(s). Converting to H5 ...")

    h5_tmp = tempfile.mkdtemp(prefix='gcp_vqvae_h5_')
    try:
        for fpath in tqdm(files, desc="PDB/CIF → H5"):
            convert_structure_to_h5(fpath, h5_tmp, max_len=args.max_len, min_len=args.min_len)

        h5_count = len(glob.glob(os.path.join(h5_tmp, '*.h5')))
        if h5_count == 0:
            print("[ERROR] No valid H5 files produced. Check your input structures.")
            sys.exit(1)
        print(f"Produced {h5_count} H5 files. Running VQ encoding ...")

        csv_path = run_encode(h5_tmp, args.output, args.model_dir, args.vq_code_dir, args.gpu_id, args.batch_size)
        if csv_path:
            print(f"\nDone! VQ indices saved to: {csv_path}")
    finally:
        shutil.rmtree(h5_tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
