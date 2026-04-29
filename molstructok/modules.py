"""
MLP-Mixer modules for the VQ-VAE structure tokenizer.
Borrowed from unofficial MLPMixer (https://github.com/920232796/MlpMixer-pytorch).
"""

import torch
import torch.nn as nn


class FCBlock(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.ff(x)


class MLPBlock(nn.Module):
    def __init__(self, dim, inter_dim, dropout_ratio):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(dim, inter_dim),
            nn.GELU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(inter_dim, dim),
            nn.Dropout(dropout_ratio),
        )

    def forward(self, x):
        return self.ff(x)


class MixerLayer(nn.Module):
    def __init__(self, hidden_dim, hidden_inter_dim, token_dim, token_inter_dim, dropout_ratio):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(hidden_dim)
        self.MLP_token = MLPBlock(token_dim, token_inter_dim, dropout_ratio)
        self.layernorm2 = nn.LayerNorm(hidden_dim)
        self.MLP_channel = MLPBlock(hidden_dim, hidden_inter_dim, dropout_ratio)

    def forward(self, x):
        y = self.layernorm1(x)
        y = y.transpose(2, 1)
        y = self.MLP_token(y)
        y = y.transpose(2, 1)
        z = self.layernorm2(x + y)
        z = self.MLP_channel(z)
        out = x + y + z
        return out
