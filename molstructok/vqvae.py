"""
VQ-VAE model for molecular 3D structure tokenization.
Reference: Foldseek (https://github.com/steineggerlab/foldseek-analysis/blob/main/training/train_vqvae.py)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import MixerLayer


class VectorQuantizer(nn.Module):
    def __init__(self, args, N_STATES, Z_DIM):
        super().__init__()
        self.args = args
        self.COMMITMENT_COST = args.commitment_cost
        self.N_STATES = N_STATES

        self.embedding = nn.Embedding(N_STATES, Z_DIM)
        self.embedding.weight.data.uniform_(-1 / N_STATES, 1 / N_STATES)

        if args.ema_vocab == 'True':
            self.decay = args.ema_decay
            self.epsilon = args.ema_epsilon
            self.register_buffer('ema_cluster_size', torch.zeros(N_STATES))
            self.register_buffer('ema_cluster_sum', torch.zeros(N_STATES, Z_DIM))

    def forward(self, inputs):
        distances = (
            torch.sum(inputs ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(inputs, self.embedding.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.N_STATES, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embedding.weight)

        if self.args.ema_vocab == 'True':
            e_latent_loss = F.mse_loss(quantized.detach(), inputs)
            q_latent_loss = torch.zeros_like(e_latent_loss)
            loss = self.COMMITMENT_COST * e_latent_loss
            if self.training and not getattr(self, 'frozen', False):
                with torch.no_grad():
                    updated_cluster_size = encodings.sum(0)
                    updated_cluster_sum = torch.matmul(encodings.t(), inputs)
                    self.ema_cluster_size.mul_(self.decay).add_(updated_cluster_size, alpha=1 - self.decay)
                    self.ema_cluster_sum.mul_(self.decay).add_(updated_cluster_sum, alpha=1 - self.decay)
                    n = self.ema_cluster_size.sum()
                    cluster_size = (
                        (self.ema_cluster_size + self.epsilon) / (n + self.N_STATES * self.epsilon) * n
                    )
                    self.embedding.weight.data.copy_(self.ema_cluster_sum / cluster_size.unsqueeze(1))
        else:
            e_latent_loss = F.mse_loss(quantized.detach(), inputs)
            q_latent_loss = F.mse_loss(quantized, inputs.detach())
            loss = q_latent_loss + self.COMMITMENT_COST * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized, perplexity, encodings, (q_latent_loss, e_latent_loss)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class ResMLPBlock(nn.Module):
    def __init__(self, dim, expand_ratio=4, dropout=0.0):
        super().__init__()
        inner_dim = dim * expand_ratio
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x):
        return x + self.net(x)


class ResMLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_blocks=3, expand_ratio=4, dropout=0.0):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList([
            ResMLPBlock(hidden_dim, expand_ratio, dropout) for _ in range(num_blocks)
        ])
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = self.input_norm(F.gelu(self.input_proj(x)))
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)


class ResMLPDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_blocks=3, expand_ratio=4, dropout=0.0):
        super().__init__()
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList([
            ResMLPBlock(hidden_dim, expand_ratio, dropout) for _ in range(num_blocks)
        ])

    def forward(self, x):
        x = self.input_norm(F.gelu(self.input_proj(x)))
        for block in self.blocks:
            x = block(x)
        return x


class MultiHeadEncoder(nn.Module):
    def __init__(self, args, latent_dim, hidden_dim, num_layers=3):
        super().__init__()
        self.args = args

        if args.descriptors == 'both':
            self.und_len_branch = self._make_branch(4, hidden_dim, num_layers)
            self.und_ang_branch = self._make_branch(6, hidden_dim, num_layers)
            self.gen_len_branch = self._make_branch(1, hidden_dim, num_layers)
            self.gen_ang_branch = self._make_branch(1, hidden_dim, num_layers)
            self.gen_tor_branch = self._make_branch(1, hidden_dim, num_layers)
            n_branches = 5
            if args.torsion_rm_sign != 'True':
                self.sign_branch = self._make_branch(1, hidden_dim, num_layers)
                n_branches += 1
            if args.ring_pred == 'True':
                self.ring_branch = self._make_branch(1, hidden_dim, num_layers)
                n_branches += 1
        elif args.descriptors == 'generation':
            self.gen_len_branch = self._make_branch(1, hidden_dim, num_layers)
            self.gen_ang_branch = self._make_branch(1, hidden_dim, num_layers)
            self.gen_tor_branch = self._make_branch(1, hidden_dim, num_layers)
            self.sign_branch = self._make_branch(1, hidden_dim, num_layers)
            n_branches = 4
        elif args.descriptors == 'understanding':
            self.und_len_branch = self._make_branch(4, hidden_dim, num_layers)
            self.und_ang_branch = self._make_branch(6, hidden_dim, num_layers)
            n_branches = 2

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * n_branches, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    @staticmethod
    def _make_branch(input_dim, hidden_dim, num_layers=3):
        assert num_layers >= 2, "num_layers must be >= 2"
        layers = [nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.args.descriptors == 'generation':
            feats = [
                self.gen_len_branch(x[:, 0:1]),
                self.gen_ang_branch(x[:, 1:2]),
                self.gen_tor_branch(x[:, 2:3]),
                self.sign_branch(x[:, 3:4]),
            ]
        elif self.args.descriptors == 'both':
            feats = [
                self.und_len_branch(x[:, :4]),
                self.und_ang_branch(x[:, 4:10]),
                self.gen_len_branch(x[:, 10:11]),
                self.gen_ang_branch(x[:, 11:12]),
                self.gen_tor_branch(x[:, 12:13]),
            ]
            if self.args.torsion_rm_sign != 'True':
                feats.append(self.sign_branch(x[:, 13:14]))
            if self.args.ring_pred == 'True':
                idx = 14 if self.args.torsion_rm_sign != 'True' else 13
                feats.append(self.ring_branch(x[:, idx:idx + 1]))
        elif self.args.descriptors == 'understanding':
            feats = [
                self.und_len_branch(x[:, :4]),
                self.und_ang_branch(x[:, 4:10]),
            ]
        return self.fusion(torch.cat(feats, dim=1))


class MultiHeadDecoder(nn.Module):
    def __init__(self, args, latent_dim, hidden_dim, num_layers=3):
        super().__init__()
        self.args = args

        if args.descriptors == 'both':
            self.und_len_branch = self._make_branch(latent_dim, hidden_dim, 4, num_layers)
            self.und_ang_branch = self._make_branch(latent_dim, hidden_dim, 6, num_layers)
            self.gen_len_branch = self._make_branch(latent_dim, hidden_dim, 1, num_layers)
            self.gen_ang_branch = self._make_branch(latent_dim, hidden_dim, 1, num_layers)
            self.gen_tor_branch = self._make_branch(latent_dim, hidden_dim, 1, num_layers)
            if args.torsion_rm_sign != 'True':
                self.sign_branch = self._make_branch(latent_dim, hidden_dim, 3, num_layers)
            if args.ring_pred == 'True':
                self.ring_branch = self._make_branch(latent_dim, hidden_dim, 1, num_layers)
        elif args.descriptors == 'generation':
            self.gen_len_branch = self._make_branch(latent_dim, hidden_dim, 1, num_layers)
            self.gen_ang_branch = self._make_branch(latent_dim, hidden_dim, 1, num_layers)
            self.gen_tor_branch = self._make_branch(latent_dim, hidden_dim, 1, num_layers)
            self.sign_branch = self._make_branch(latent_dim, hidden_dim, 3, num_layers)
        elif args.descriptors == 'understanding':
            self.und_len_branch = self._make_branch(latent_dim, hidden_dim, 4, num_layers)
            self.und_ang_branch = self._make_branch(latent_dim, hidden_dim, 6, num_layers)

    @staticmethod
    def _make_branch(input_dim, hidden_dim, output_dim, num_layers=3):
        assert num_layers >= 2, "num_layers must be >= 2"
        layers = [nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*layers)

    def forward(self, quantized_z):
        sign = None
        ring = None

        if self.args.descriptors == 'both':
            und_len = self.und_len_branch(quantized_z)
            und_ang = self.und_ang_branch(quantized_z)
            gen_len = self.gen_len_branch(quantized_z)
            gen_ang = self.gen_ang_branch(quantized_z)
            gen_tor = self.gen_tor_branch(quantized_z)
            mu = torch.cat([und_len, und_ang, gen_len, gen_ang, gen_tor], dim=1)
            if self.args.torsion_rm_sign != 'True':
                sign = self.sign_branch(quantized_z)
            if self.args.ring_pred == 'True':
                ring = self.ring_branch(quantized_z)
        elif self.args.descriptors == 'generation':
            gen_len = self.gen_len_branch(quantized_z)
            gen_ang = self.gen_ang_branch(quantized_z)
            gen_tor = self.gen_tor_branch(quantized_z)
            mu = torch.cat([gen_len, gen_ang, gen_tor], dim=1)
            sign = self.sign_branch(quantized_z)
        elif self.args.descriptors == 'understanding':
            und_len = self.und_len_branch(quantized_z)
            und_ang = self.und_ang_branch(quantized_z)
            mu = torch.cat([und_len, und_ang], dim=1)

        return mu, sign, ring


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ConditionalDenoiser(nn.Module):
    def __init__(self, data_dim, cond_dim, hidden_dim, num_blocks=6):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_proj = nn.Linear(data_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim),
            ) for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, data_dim)

    def forward(self, x_t, t, cond):
        h = self.input_proj(x_t) + self.time_embed(t) + self.cond_proj(cond)
        for block in self.blocks:
            h = h + block(h)
        return self.output_proj(self.final_norm(h))


class DiffusionDecoder(nn.Module):
    def __init__(self, args, cond_dim, hidden_dim,
                 num_timesteps=1000, num_blocks=6, num_inference_steps=50):
        super().__init__()
        self.args = args
        self.num_timesteps = num_timesteps
        self.num_inference_steps = num_inference_steps

        if args.use_bindgpt:
            self.data_dim = 3
        elif args.descriptors == 'generation':
            self.data_dim = 3
        elif args.descriptors == 'understanding':
            self.data_dim = 10
        elif args.descriptors == 'both':
            self.data_dim = 13

        self.denoiser = ConditionalDenoiser(self.data_dim, cond_dim, hidden_dim, num_blocks)

        if not args.use_bindgpt:
            if args.descriptors in ('both', 'generation'):
                if args.descriptors == 'generation' or args.torsion_rm_sign != 'True':
                    self.sign_branch = self._make_clf(cond_dim, hidden_dim, 3)
            if args.descriptors == 'both' and args.ring_pred == 'True':
                self.ring_branch = self._make_clf(cond_dim, hidden_dim, 1)

        betas = self._cosine_beta_schedule(num_timesteps)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

    @staticmethod
    def _make_clf(in_dim, hid_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, hid_dim), nn.BatchNorm1d(hid_dim), nn.ReLU(),
            nn.Linear(hid_dim, hid_dim), nn.BatchNorm1d(hid_dim), nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
        )

    @staticmethod
    def _cosine_beta_schedule(T, s=0.008):
        t = torch.linspace(0, T, T + 1)
        ac = torch.cos(((t / T) + s) / (1 + s) * math.pi * 0.5) ** 2
        ac = ac / ac[0]
        betas = 1 - (ac[1:] / ac[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        a = self.sqrt_alphas_cumprod[t][:, None]
        b = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        return a * x_0 + b * noise

    def forward(self, quantized_z, target):
        B = target.shape[0]
        t = torch.randint(0, self.num_timesteps, (B,), device=target.device)
        noise = torch.randn_like(target)
        x_t = self.q_sample(target, t, noise)
        noise_pred = self.denoiser(x_t, t, quantized_z)
        diff_loss = F.mse_loss(noise_pred, noise)

        with torch.no_grad():
            x0_pred = (
                (x_t - self.sqrt_one_minus_alphas_cumprod[t][:, None] * noise_pred)
                / self.sqrt_alphas_cumprod[t][:, None]
            )

        sign = self.sign_branch(quantized_z) if hasattr(self, 'sign_branch') else None
        ring = self.ring_branch(quantized_z) if hasattr(self, 'ring_branch') else None
        return diff_loss, x0_pred, sign, ring

    @torch.no_grad()
    def sample(self, quantized_z, num_inference_steps=None):
        """DDIM deterministic sampling."""
        B = quantized_z.shape[0]
        device = quantized_z.device
        steps = num_inference_steps or self.num_inference_steps
        x = torch.randn(B, self.data_dim, device=device)

        step_size = max(self.num_timesteps // steps, 1)
        ts = list(range(self.num_timesteps - 1, -1, -step_size))
        if ts[-1] != 0:
            ts.append(0)

        for i, t in enumerate(ts):
            t_b = torch.full((B,), t, device=device, dtype=torch.long)
            eps = self.denoiser(x, t_b, quantized_z)
            ab = self.alphas_cumprod[t]
            x0 = (x - torch.sqrt(1.0 - ab) * eps) / torch.sqrt(ab)
            x0 = torch.clamp(x0, -10, 10)
            if i == len(ts) - 1:
                x = x0
            else:
                ab_prev = (
                    self.alphas_cumprod[ts[i + 1]] if ts[i + 1] > 0
                    else torch.ones(1, device=device)
                )
                d = (x - torch.sqrt(ab) * x0) / torch.sqrt(1.0 - ab)
                x = torch.sqrt(ab_prev) * x0 + torch.sqrt(1.0 - ab_prev) * d

        sign = self.sign_branch(quantized_z) if hasattr(self, 'sign_branch') else None
        ring = self.ring_branch(quantized_z) if hasattr(self, 'ring_branch') else None
        return x, sign, ring


class Conv1dEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.conv1 = nn.Conv1d(1, hidden_dim // 2, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim // 2)
        self.conv2 = nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=4, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=4, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim * (input_dim // 4), latent_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.input_fc(x))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Conv1dDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc = nn.Linear(latent_dim, hidden_dim * (output_dim // 4))
        self.conv_trans1 = nn.ConvTranspose1d(hidden_dim, hidden_dim // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim // 2)
        self.conv_trans2 = nn.ConvTranspose1d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 4)
        self.conv_trans3 = nn.ConvTranspose1d(hidden_dim // 4, 1, kernel_size=3, stride=1, padding=1)
        self.output_fc = nn.Linear(hidden_dim // 4, output_dim)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), self.hidden_dim, self.output_dim // 4)
        x = F.relu(self.bn1(self.conv_trans1(x)))
        x = F.relu(self.bn2(self.conv_trans2(x)))
        x = self.conv_trans3(x)
        x = x.squeeze(1)
        return x


class PCTEncoder(nn.Module):
    def __init__(self, args, input_dim):
        super().__init__()
        self.args = args
        self.und_len_embed = nn.Linear(4, args.pct_encoder_hidden_dim)
        self.und_ang_embed = nn.Linear(6, args.pct_encoder_hidden_dim)
        self.gen_len_embed = nn.Linear(1, args.pct_encoder_hidden_dim)
        self.gen_ang_embed = nn.Linear(1, args.pct_encoder_hidden_dim)
        self.gen_tor_embed = nn.Linear(1, args.pct_encoder_hidden_dim)
        self.gen_sign_embed = nn.Linear(1, args.pct_encoder_hidden_dim)

        self.encoder = nn.ModuleList([
            MixerLayer(
                args.pct_encoder_hidden_dim, args.pct_encoder_hidden_dim,
                6, args.pct_encoder_token_dim, args.pct_encoder_dropout,
            ) for _ in range(args.pct_encoder_num_blocks)
        ])
        self.encoder_layer_norm = nn.LayerNorm(args.pct_encoder_hidden_dim)
        self.feature_readout = MLP(
            input_dim=6 * args.pct_encoder_hidden_dim,
            hidden_dim=args.pct_encoder_hidden_dim,
            output_dim=args.latent_dim,
        )

    def forward(self, input):
        und_len_embedded = self.und_len_embed(input[:, :4]).unsqueeze(1)
        und_ang_embedded = self.und_ang_embed(input[:, 4:10]).unsqueeze(1)
        gen_len_embedded = self.gen_len_embed(input[:, 10:11]).unsqueeze(1)
        gen_ang_embedded = self.gen_ang_embed(input[:, 11:12]).unsqueeze(1)
        gen_tor_embedded = self.gen_tor_embed(input[:, 12:13]).unsqueeze(1)
        gen_sign_embedded = self.gen_sign_embed(input[:, 13:14]).unsqueeze(1)
        encode_feat = torch.cat(
            (und_len_embedded, und_ang_embedded, gen_len_embedded,
             gen_ang_embedded, gen_tor_embedded, gen_sign_embedded), dim=1,
        )

        for num_layer in self.encoder:
            encode_feat = num_layer(encode_feat)
        encode_feat = self.encoder_layer_norm(encode_feat)
        encode_feat = self.feature_readout(encode_feat.view(encode_feat.size(0), -1).contiguous())
        return encode_feat


class PCTDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.und_len_start = nn.Linear(args.latent_dim, args.pct_decoder_hidden_dim)
        self.und_ang_start = nn.Linear(args.latent_dim, args.pct_decoder_hidden_dim)
        self.gen_len_start = nn.Linear(args.latent_dim, args.pct_decoder_hidden_dim)
        self.gen_ang_start = nn.Linear(args.latent_dim, args.pct_decoder_hidden_dim)
        self.gen_tor_start = nn.Linear(args.latent_dim, args.pct_decoder_hidden_dim)
        self.gen_sign_start = nn.Linear(args.latent_dim, args.pct_decoder_hidden_dim)

        self.decoder = nn.ModuleList([
            MixerLayer(
                args.pct_decoder_hidden_dim, args.pct_decoder_hidden_dim,
                6, args.pct_decoder_token_dim, args.pct_decoder_dropout,
            ) for _ in range(args.pct_decoder_num_blocks)
        ])
        self.decoder_layer_norm = nn.LayerNorm(args.pct_decoder_hidden_dim)
        self.und_len_head = nn.Linear(args.pct_decoder_hidden_dim, 4)
        self.und_ang_head = nn.Linear(args.pct_decoder_hidden_dim, 6)
        self.gen_len_head = nn.Linear(args.pct_decoder_hidden_dim, 1)
        self.gen_ang_head = nn.Linear(args.pct_decoder_hidden_dim, 1)
        self.gen_tor_head = nn.Linear(args.pct_decoder_hidden_dim, 1)

    def forward(self, input):
        und_len_start = self.und_len_start(input).unsqueeze(1)
        und_ang_start = self.und_ang_start(input).unsqueeze(1)
        gen_len_start = self.gen_len_start(input).unsqueeze(1)
        gen_ang_start = self.gen_ang_start(input).unsqueeze(1)
        gen_tor_start = self.gen_tor_start(input).unsqueeze(1)
        gen_sign_start = self.gen_sign_start(input).unsqueeze(1)
        decode_feat = torch.cat(
            (und_len_start, und_ang_start, gen_len_start,
             gen_ang_start, gen_tor_start, gen_sign_start), dim=1,
        )

        for num_layer in self.decoder:
            decode_feat = num_layer(decode_feat)
        decode_feat = self.decoder_layer_norm(decode_feat)

        und_len = self.und_len_head(decode_feat[:, 0])
        und_ang = self.und_ang_head(decode_feat[:, 1])
        gen_len = self.gen_len_head(decode_feat[:, 2])
        gen_ang = self.gen_ang_head(decode_feat[:, 3])
        gen_tor = self.gen_tor_head(decode_feat[:, 4])
        mu = torch.cat((und_len, und_ang, gen_len, gen_ang, gen_tor), dim=1)
        return decode_feat[:, 5], mu


class VQVAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        if args.use_bindgpt:
            input_dim = 3
        else:
            if args.descriptors == 'both':
                input_dim = 14
                if args.ring_pred == 'True':
                    input_dim += 1
                if args.torsion_rm_sign == 'True':
                    input_dim -= 1
            elif args.descriptors == 'generation':
                input_dim = 4
            elif args.descriptors == 'understanding':
                input_dim = 10
        hidden_dim = args.hidden_dim
        latent_dim = args.latent_dim

        self.use_multi_head_encoder = getattr(args, 'multi_head_encoder', 'False') == 'True'

        if self.use_multi_head_encoder:
            enc_hidden = getattr(args, 'encoder_hidden_dim', None) or hidden_dim
            enc_num_layers = getattr(args, 'encoder_num_layers', 3)
            self.encoder = MultiHeadEncoder(args, latent_dim, enc_hidden, num_layers=enc_num_layers)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            )
        elif args.resmlp == 'True':
            self.encoder = ResMLPEncoder(
                input_dim, hidden_dim, latent_dim,
                num_blocks=args.resmlp_num_blocks,
                expand_ratio=args.resmlp_expand_ratio,
                dropout=args.resmlp_dropout,
            )
            self.decoder = ResMLPDecoder(
                latent_dim, hidden_dim,
                num_blocks=args.resmlp_num_blocks,
                expand_ratio=args.resmlp_expand_ratio,
                dropout=args.resmlp_dropout,
            )
        elif args.conv1d == 'True':
            self.encoder = Conv1dEncoder(input_dim, hidden_dim, latent_dim)
            self.decoder = Conv1dDecoder(latent_dim, hidden_dim, hidden_dim)
        elif args.pct_arch == 'True':
            self.encoder = PCTEncoder(args, input_dim)
            self.decoder = PCTDecoder(args)
        else:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            )

        self.quantizer = VectorQuantizer(args, self.vocab_size, latent_dim)

        if args.use_bindgpt:
            self.mu = nn.Linear(hidden_dim, input_dim)
        else:
            if args.descriptors == 'both':
                if args.torsion_rm_sign == 'True':
                    self.mu = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=input_dim)
                else:
                    self.sign_head = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=3)
                    if args.ring_pred == 'True':
                        self.ring_head = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=1)
                        self.mu = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=input_dim - 2)
                    else:
                        self.mu = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=input_dim - 1)
            elif args.descriptors == 'understanding':
                self.mu = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=input_dim)
            elif args.descriptors == 'generation':
                self.sign_head = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=3)
                self.mu = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=input_dim - 1)

        self.use_multi_head_decoder = getattr(args, 'multi_head_decoder', 'False') == 'True'
        if self.use_multi_head_decoder:
            dec_hidden = getattr(args, 'decoder_hidden_dim', None) or hidden_dim
            dec_num_layers = getattr(args, 'decoder_num_layers', 3)
            self.multi_head_decoder = MultiHeadDecoder(args, latent_dim, dec_hidden, num_layers=dec_num_layers)

        self.use_diffusion_decoder = getattr(args, 'diffusion_decoder', 'False') == 'True'
        if self.use_diffusion_decoder:
            dec_hidden = getattr(args, 'decoder_hidden_dim', None) or hidden_dim
            self.diffusion_decoder = DiffusionDecoder(
                args, cond_dim=latent_dim, hidden_dim=dec_hidden,
                num_timesteps=getattr(args, 'diffusion_timesteps', 1000),
                num_blocks=getattr(args, 'diffusion_num_blocks', 6),
                num_inference_steps=getattr(args, 'diffusion_inference_steps', 50),
            )

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.001)

    def forward(self, x):
        z = self.encoder(x)
        if self.args.use_kl:
            quantized_z = z
            loss = perplexity = encodings = q_latent_loss = e_latent_loss = torch.tensor(0, device=x.device)
            encodings = quantized_z
            import joblib
            loaded_kmeans = joblib.load('kmeans_model.pkl')
            labels = loaded_kmeans.predict(quantized_z.detach().cpu().numpy())
            quantized_z = torch.tensor(loaded_kmeans.cluster_centers_[labels], device=x.device)
        else:
            loss, quantized_z, perplexity, encodings, (q_latent_loss, e_latent_loss) = self.quantizer(z)

        if self.use_diffusion_decoder:
            if self.args.use_bindgpt:
                target = x
            elif self.args.descriptors == 'generation':
                target = x[:, :-1]
            elif self.args.descriptors == 'both':
                target = x[:, :13] if self.args.torsion_rm_sign != 'True' else x
            elif self.args.descriptors == 'understanding':
                target = x
            if self.training:
                diff_loss, mu, sign, ring = self.diffusion_decoder(quantized_z, target)
            else:
                mu, sign, ring = self.diffusion_decoder.sample(quantized_z)
                diff_loss = torch.tensor(0.0, device=x.device)
            return diff_loss, mu, perplexity, encodings, (q_latent_loss, e_latent_loss), sign, ring
        elif self.use_multi_head_decoder:
            mu, sign, ring = self.multi_head_decoder(quantized_z)
        elif self.args.pct_arch == 'True':
            hidden, mu = self.decoder(quantized_z)
            sign = self.sign_head(hidden)
            ring = None
        else:
            hidden = self.decoder(quantized_z)
            mu = self.mu(hidden)
            if self.args.use_bindgpt:
                sign = None
                ring = None
            else:
                if self.args.descriptors == 'both':
                    sign = None if self.args.torsion_rm_sign == 'True' else self.sign_head(hidden)
                    ring = self.ring_head(hidden) if self.args.ring_pred == 'True' else None
                elif self.args.descriptors == 'understanding':
                    sign, ring = None, None
                elif self.args.descriptors == 'generation':
                    sign = self.sign_head(hidden)
                    ring = None

        return loss, mu, perplexity, encodings, (q_latent_loss, e_latent_loss), sign, ring

    def decode(self, vocab_index):
        quantized_z = self.quantizer.embedding(vocab_index)
        if self.use_diffusion_decoder:
            mu, sign, ring = self.diffusion_decoder.sample(quantized_z)
            return mu, sign, ring
        elif self.use_multi_head_decoder:
            mu, sign, ring = self.multi_head_decoder(quantized_z)
        else:
            hidden = self.decoder(quantized_z)
            mu = self.mu(hidden)
            if self.args.use_bindgpt:
                sign = None
                ring = None
            else:
                sign = None if self.args.torsion_rm_sign == 'True' else self.sign_head(hidden)
                ring = self.ring_head(hidden) if self.args.ring_pred == 'True' else None

        return mu, sign, ring
