"""
MolStructTok: self-contained pipeline for molecular 3D structure tokenization
and de-tokenization using MolStrucTok VQ-VAE.
"""

import re

import numpy as np
import torch
import selfies as sf
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from rdkit import RDLogger

from .vqvae import VQVAE
from .descriptors import process_one_mol
from .reconstruct import generate_coords

RDLogger.DisableLog('rdApp.*')


class MolStructTok:
    """Molecular 3D structure tokenizer / de-tokenizer.

    Args:
        model_path: Path to a VQ-VAE checkpoint (.bin / .pt).
        device: 'cuda' or 'cpu'.
        enable_mmff: Whether to apply MMFF force-field refinement after
            de-tokenization (improves bond-length fidelity).
    """

    def __init__(self, model_path, device='cuda', enable_mmff=True):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
        self.args = self._get_default_args(model_path, state_dict)
        self.enable_mmff = enable_mmff
        if 'both' in model_path:
            self.args.descriptors = 'both'
        self.model = VQVAE(self.args)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _parse_dim_from_path(model_path, prefix, default):
        m = re.search(rf'(?<![a-zA-Z]){prefix}(\d+)', model_path)
        if m:
            return int(m.group(1))
        return default

    def _get_default_args(self, model_path, state_dict):
        hidden_dim = self._parse_dim_from_path(model_path, 'hid', 64)
        latent_dim = self._parse_dim_from_path(model_path, 'lat', 64)
        vocab_size = self._parse_dim_from_path(model_path, 'voc', 512)

        resmlp_match = re.search(r'resmlp(\d+)', model_path)
        use_resmlp = resmlp_match is not None
        resmlp_num_blocks = int(resmlp_match.group(1)) if resmlp_match else 3

        has_multi_head_encoder = any(k.startswith('encoder.fusion.') for k in state_dict)
        has_multi_head = any(k.startswith('multi_head_decoder.') for k in state_dict)
        has_diffusion = any(k.startswith('diffusion_decoder.') for k in state_dict)

        encoder_hidden_dim = hidden_dim
        encoder_num_layers = 3
        decoder_hidden_dim = hidden_dim
        decoder_num_layers = 3
        diffusion_timesteps = 1000
        diffusion_num_blocks = 6
        diffusion_inference_steps = 50

        if has_multi_head_encoder:
            encoder_hidden_dim = state_dict['encoder.gen_len_branch.0.weight'].shape[0]
            max_idx = 0
            for k in state_dict:
                m_enc = re.match(r'encoder\.gen_len_branch\.(\d+)\.', k)
                if m_enc:
                    max_idx = max(max_idx, int(m_enc.group(1)))
            encoder_num_layers = max_idx // 3 + 1

        if has_multi_head:
            decoder_hidden_dim = state_dict['multi_head_decoder.gen_len_branch.0.weight'].shape[0]
            max_idx = 0
            for k in state_dict:
                m_dec = re.match(r'multi_head_decoder\.gen_len_branch\.(\d+)\.', k)
                if m_dec:
                    max_idx = max(max_idx, int(m_dec.group(1)))
            decoder_num_layers = max_idx // 3 + 1

        if has_diffusion:
            decoder_hidden_dim = state_dict['diffusion_decoder.denoiser.input_proj.weight'].shape[0]
            diffusion_timesteps = state_dict['diffusion_decoder.alphas_cumprod'].shape[0]
            block_ids = set()
            for k in state_dict:
                m_blk = re.match(r'diffusion_decoder\.denoiser\.blocks\.(\d+)\.', k)
                if m_blk:
                    block_ids.add(int(m_blk.group(1)))
            if block_ids:
                diffusion_num_blocks = len(block_ids)

        encoder_type = 'multi_head' if has_multi_head_encoder else 'default'
        decoder_type = 'diffusion' if has_diffusion else ('multi_head' if has_multi_head else 'default')

        class Args:
            pass

        args = Args()
        args.load_model_path = model_path
        args.hidden_dim = hidden_dim
        args.latent_dim = latent_dim
        args.vocab_size = vocab_size
        args.descriptors = 'generation'
        args.additional_sign_feature = 'True'
        args.gen_normalize_angle = 'True'
        args.gen_normalize_length = 'log'
        args.und_normalize_angle = 'True'
        args.last_act = 'True'
        args.output_mlp = 'True'
        args.conv1d = 'False'
        args.relubn = 'False'
        args.ema_vocab = 'True'
        args.ema_decay = 0.99
        args.ema_epsilon = 1e-5
        args.commitment_cost = 0.25
        args.ring_pred = 'False'
        args.torsion_rm_sign = 'False'
        args.pct_arch = 'False'
        args.use_bindgpt = False
        args.use_kl = False
        args.resmlp = 'True' if use_resmlp else 'False'
        args.resmlp_num_blocks = resmlp_num_blocks
        args.resmlp_expand_ratio = 4
        args.resmlp_dropout = 0.0
        args.multi_head_encoder = 'True' if has_multi_head_encoder else 'False'
        args.encoder_hidden_dim = encoder_hidden_dim
        args.encoder_num_layers = encoder_num_layers
        args.multi_head_decoder = 'True' if has_multi_head else 'False'
        args.diffusion_decoder = 'True' if has_diffusion else 'False'
        args.decoder_hidden_dim = decoder_hidden_dim
        args.decoder_num_layers = decoder_num_layers
        args.freeze_encoder = 'False'
        args.diffusion_timesteps = diffusion_timesteps
        args.diffusion_num_blocks = diffusion_num_blocks
        args.diffusion_inference_steps = diffusion_inference_steps
        return args

    def encode(self, mol):
        """Encode an RDKit Mol (with 3D conformer) into a struct_sfi string.

        Args:
            mol: An RDKit Mol object with at least one 3D conformer.

        Returns:
            A tuple of (mol_renumbered, struct_sfi, struct_token_indices):
            - mol_renumbered: The RDKit Mol re-ordered to canonical SMILES atom order.
            - struct_sfi: A string like ``[C 42][Branch1 -1][H 305]...`` where each
              SELFIES token is paired with its VQ code index (-1 for non-atom tokens).
            - struct_token_indices: A list of integer VQ code indices (atom tokens only).
        """
        cano_smi = Chem.MolToSmiles(mol, canonical=True, kekuleSmiles=True, isomericSmiles=False)
        m_order = list(
            mol.GetPropsAsDict(includePrivate=True, includeComputed=True)['_smilesAtomOutputOrder']
        )
        mol_renumbered = Chem.RenumberAtoms(mol, m_order)

        sfi, attr = sf.encoder(cano_smi, attribute=True)

        understanding_descriptors, generation_descriptors = process_one_mol(
            mol_renumbered, choose_c2='recurrent-index',
        )
        if self.args.descriptors == 'both':
            data_array = np.concatenate([understanding_descriptors, generation_descriptors], axis=1)
            data_tensor = torch.from_numpy(data_array).to(self.device).float()
        elif self.args.descriptors == 'generation':
            data_tensor = torch.from_numpy(generation_descriptors).to(self.device).float()
        else:
            raise ValueError(f'Invalid descriptors: {self.args.descriptors}')

        if self.args.descriptors == 'both':
            if self.args.gen_normalize_length == 'log':
                data_tensor[:, :4] = torch.log(data_tensor[:, :4] + 1.0)
                data_tensor[:, 10:11] = torch.log(data_tensor[:, 10:11] + 1.0)
            data_tensor[:, 4:10] = data_tensor[:, 4:10] / torch.pi
            data_tensor[:, 11:13] = data_tensor[:, 11:13] / torch.pi
        elif self.args.descriptors == 'generation':
            if self.args.gen_normalize_length == 'log':
                data_tensor[:, :1] = torch.log(data_tensor[:, :1] + 1.0)
            data_tensor[:, 1:3] = data_tensor[:, 1:3] / torch.pi

        with torch.no_grad():
            _, _, _, encodings, _, _, _ = self.model(data_tensor)
            indices = encodings.argmax(dim=1)

        matches = re.findall(r'\[([^]]+)]', sfi)
        _, attr_rev = sf.decoder(sfi, attribute=True)
        all_indexes = [i for i in range(len(matches))]
        all_rev_indexes = [item[-1].index for item in [item.attribution for item in attr_rev]]
        meaningless_indexes = set(all_indexes) - set(all_rev_indexes)

        atom_ptr = 0
        new_token_list = []
        struct_tokens_only = []

        for t_idx, match in enumerate(matches):
            if t_idx in meaningless_indexes:
                new_token_list.append(f'[{match} -1]')
            else:
                idx_val = indices[atom_ptr].item()
                new_token_list.append(f'[{match} {idx_val}]')
                struct_tokens_only.append(idx_val)
                atom_ptr += 1

        struct_sfi = ''.join(new_token_list)
        return mol_renumbered, struct_sfi, struct_tokens_only

    def decode(self, mol_3d_seq):
        """Decode a struct_sfi string back into an RDKit Mol with 3D coordinates.

        Args:
            mol_3d_seq: A struct_sfi string, e.g.
                ``[C 42][Branch1 -1][H 305][C 70]...``

        Returns:
            An RDKit Mol object with a 3D conformer. If ``enable_mmff=True``,
            MMFF force-field optimization is applied to refine the geometry.
        """
        pattern = r'\[[^\[\]]+\]'
        result = re.findall(pattern, mol_3d_seq)
        sfi_tokens = []
        struct_tokens = []
        for token in result:
            parts = token[1:-1].split()
            sfi_tokens.append('[' + parts[0] + ']')
            struct_token = int(parts[1])
            if struct_token != -1:
                struct_tokens.append(struct_token)

        smi = sf.decoder(''.join(sfi_tokens))
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass

        vocab_num = torch.tensor(struct_tokens).to(self.device)
        mu, sign, _ = self.model.decode(vocab_num)

        if self.args.descriptors == 'both':
            gen_feat = mu[:, 10:].clone()
        elif self.args.descriptors == 'generation':
            gen_feat = mu.clone()
        else:
            raise ValueError(f'Invalid descriptors: {self.args.descriptors}')

        gen_feat[:, :1] = torch.exp(gen_feat[:, :1]) - 1.0
        gen_feat[:, 1:3] *= torch.pi
        sign_feat = torch.argmax(sign, dim=-1) - 1
        gen_feat = torch.cat([gen_feat, sign_feat.unsqueeze(-1).float()], dim=-1)
        gen_feat = gen_feat.cpu().detach().numpy()

        num_atoms = mol.GetNumAtoms()
        assert num_atoms == len(gen_feat), (
            f'num_atoms: {num_atoms}, len(feats): {len(gen_feat)}'
        )

        positions = torch.tensor(generate_coords(gen_feat, mol, choose_c2='recurrent-index'))
        mol.Compute2DCoords()
        conf = mol.GetConformer()
        assert mol.GetNumAtoms() == positions.shape[0]
        for jdx in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(
                jdx, Point3D(positions[jdx, 0].item(), positions[jdx, 1].item(), positions[jdx, 2].item()),
            )

        if self.enable_mmff:
            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except Exception:
                pass

        return mol


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='MolStructTok tokenization demo')
    parser.add_argument('--model_path', type=str, required=True, help='Path to VQ-VAE checkpoint (.bin)')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    tok = MolStructTok(args.model_path, device=args.device)

    sfi_3d = (
        '[H 3][C 0][Branch1 -1][C -1][H 304][Branch1 -1][C -1][H 140]'
        '[C 315][Branch1 -1][C -1][H 158][C 70][=C 166][Branch1 -1]'
        '[=Branch1 -1][O 129][N 38][=N 17][Ring1 -1][Branch1 -1]'
        '[C 107][Branch1 -1][C -1][H 380][Branch1 -1][C -1][H 182]'
        '[C 54][Ring1 -1][O -1][Branch1 -1][C -1][H 108][H 148]'
    )

    print('=== Decode (struct_sfi -> 3D Mol) ===')
    mol = tok.decode(sfi_3d)
    print(f'  Decoded mol: {Chem.MolToSmiles(mol)}')
    print(f'  Num atoms: {mol.GetNumAtoms()}')

    print('\n=== Encode (3D Mol -> struct_sfi) ===')
    _, struct_sfi, token_indices = tok.encode(mol)
    print(f'  struct_sfi: {struct_sfi}')
    print(f'  token indices: {token_indices}')
