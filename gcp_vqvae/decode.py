#!/usr/bin/env python
"""
Protein 3D Structure De-tokenization: VQ indices CSV → PDB files

Reads a vq_indices.csv (columns: pid, indices, protein_sequence) and
reconstructs backbone PDB files using the GCP-VQVAE decoder.

Usage:
    python -m gcp_vqvae.decode \
        --input     /path/to/vq_indices.csv         \
        --output    /path/to/output/                  \
        --model_dir /path/to/vqvae3d_ckpt/large       \
        --vq_code_dir /path/to/vq_encoder_decoder
"""

import os
import sys
import argparse
import datetime
import shutil

import yaml
import torch
import pandas as pd
from box import Box
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import broadcast_object_list


class VQIndicesDataset(Dataset):
    """Dataset that loads VQ indices from a CSV file."""

    def __init__(self, csv_path, max_length):
        self.data = pd.read_csv(csv_path)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        pid = row['pid']
        indices = [int(i) for i in row['indices'].split()]
        seq = row['protein_sequence']

        current_length = len(indices)
        pad_length = self.max_length - current_length

        padded_indices = indices + [-1] * pad_length
        mask = [True] * current_length + [False] * pad_length
        nan_mask = torch.tensor(mask, dtype=torch.bool)
        for i, v in enumerate(padded_indices):
            if v == -1:
                nan_mask[i] = False

        return {
            'pid': pid,
            'indices': torch.tensor(padded_indices, dtype=torch.long),
            'seq': seq,
            'masks': torch.tensor(mask, dtype=torch.bool),
            'nan_masks': nan_mask,
        }


DEFAULT_DECODE_CFG = {
    'checkpoint_path': 'checkpoints/best_valid.pth',
    'config_vqvae': 'config_vqvae.yaml',
    'config_decoder': 'config_geometric_decoder.yaml',
    'batch_size': 16,
    'shuffle': False,
    'num_workers': 0,
    'mixed_precision': 'bf16',
    'tqdm_progress_bar': True,
    'compile_model': {'enabled': False},
}


def _save_backbone_pdb(coords, masks, save_prefix, atom_names=("N", "CA", "C"), chain_id="A"):
    """Write a backbone-only PDB file."""
    if coords.dim() == 3:
        coords = coords.unsqueeze(0)
        masks = masks.unsqueeze(0)

    B, L = coords.shape[:2]
    for b in range(B):
        out_path = save_prefix if save_prefix.lower().endswith('.pdb') else f"{save_prefix}.pdb"
        with open(out_path, "w") as fh:
            serial = 1
            for r in range(L):
                if masks[b, r].item() != 1:
                    continue
                for a_idx, atom_name in enumerate(atom_names):
                    if not torch.isfinite(coords[b, r, a_idx]).all():
                        continue
                    x, y, z = coords[b, r, a_idx].tolist()
                    element = atom_name[0].upper()
                    fh.write(
                        f"ATOM  {serial:5d} {atom_name:>4s} "
                        f"UNK {chain_id}{r + 1:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}"
                        f"{1.00:6.2f}{0.00:6.2f}          "
                        f"{element:>2s}\n"
                    )
                    serial += 1
            fh.write("TER\nEND\n")


def run_decode(csv_path, output_dir, model_dir, vq_code_dir, gpu_id=0, batch_size=None):
    """Run VQ decoding on a CSV of indices and produce PDB files."""
    sys.path.insert(0, vq_code_dir)
    from utils.utils import load_configs, load_checkpoints_simple, get_logging
    from models.super_model import prepare_model

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cfg = dict(DEFAULT_DECODE_CFG)
    cfg['trained_model_dir'] = model_dir
    cfg['indices_csv_path'] = csv_path
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
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        result_dir = os.path.join(infer_cfg.output_base_dir, now)
        os.makedirs(result_dir, exist_ok=True)
        pdb_dir = os.path.join(result_dir, 'pdb_files')
        os.makedirs(pdb_dir, exist_ok=True)
        paths = [result_dir, pdb_dir]
    else:
        paths = [None, None]
    broadcast_object_list(paths, from_process=0)
    result_dir, pdb_dir = paths

    vqvae_cfg_path = os.path.join(model_dir, infer_cfg.config_vqvae)
    decoder_cfg_path = os.path.join(model_dir, infer_cfg.config_decoder)

    with open(vqvae_cfg_path) as f:
        vqvae_cfg = yaml.full_load(f)
    configs = load_configs(vqvae_cfg)
    configs.model.max_length = infer_cfg.get('max_length', configs.model.max_length)

    with open(decoder_cfg_path) as f:
        dec_cfg = Box(yaml.full_load(f))

    old_cwd = os.getcwd()
    os.chdir(vq_code_dir)

    dataset = VQIndicesDataset(csv_path, max_length=configs.model.max_length)
    loader = DataLoader(
        dataset, shuffle=infer_cfg.shuffle,
        batch_size=infer_cfg.batch_size,
        num_workers=infer_cfg.num_workers,
    )

    logger = get_logging(result_dir, configs)
    model = prepare_model(configs, logger, decoder_configs=dec_cfg, decoder_only=True)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    ckpt_path = os.path.join(model_dir, infer_cfg.checkpoint_path)
    model = load_checkpoints_simple(ckpt_path, model, logger, decoder_only=True)

    model, loader = accelerator.prepare(model, loader)

    pbar = tqdm(range(len(loader)), leave=True,
                disable=not (infer_cfg.tqdm_progress_bar and accelerator.is_main_process))
    pbar.set_description("Decoding")

    for batch in loader:
        with torch.inference_mode():
            output = model(batch, decoder_only=True)
            bb_pred = output["outputs"]
            preds = bb_pred.view(bb_pred.shape[0], bb_pred.shape[1], 3, 3)
            for pid, coord, mask in zip(batch['pid'], preds.detach().cpu(), batch['masks'].cpu()):
                prefix = os.path.join(pdb_dir, pid)
                _save_backbone_pdb(coord, mask, prefix)
            pbar.update(1)

    logger.info(f"Decoding completed. PDB files saved in {pdb_dir}")

    accelerator.wait_for_everyone()
    accelerator.free_memory()
    accelerator.end_training()

    os.chdir(old_cwd)
    return pdb_dir


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Protein 3D Structure De-tokenization: VQ indices → PDB')
    parser.add_argument('--input', required=True,
                        help='Path to vq_indices.csv')
    parser.add_argument('--output', required=True,
                        help='Output directory for reconstructed PDB files')
    parser.add_argument('--model_dir', required=True,
                        help='Path to pre-trained GCP-VQVAE model directory')
    parser.add_argument('--vq_code_dir', required=True,
                        help='Path to vq_encoder_decoder source directory')
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    pdb_dir = run_decode(
        args.input, args.output, args.model_dir, args.vq_code_dir, args.gpu_id
    )
    print(f"\nDone! Reconstructed PDB files saved in: {pdb_dir}")


if __name__ == '__main__':
    main()
