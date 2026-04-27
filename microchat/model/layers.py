"""Small neural network layers used by the GPT model."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


def norm(x):
    """Use RMS norm for better stability with low-precision training."""
    return F.rms_norm(x, (x.size(-1),))


class Linear(nn.Linear):
    """Match weight dtype to activation dtype during forward."""

    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype), self.bias)


def has_ve(layer_idx, n_layer):
    """Only add value embeddings to alternating layers to save compute."""
    return layer_idx % 2 == (n_layer - 1) % 2


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        from microchat.model.attention import CausalSelfAttention

        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x
