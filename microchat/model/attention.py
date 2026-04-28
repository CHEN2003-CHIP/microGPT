"""Attention helpers and causal self-attention for the GPT model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from microchat.model.layers import Linear, has_ve, norm


def apply_rotary_emb(x, cos, sin):
    """Apply rotary position embeddings to the last tensor dimension."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=-1)


def repeat_kv_heads(x, n_head):
    #x.size(1) is n_kv_head
    """Repeat the key-value heads to match the number of attention heads."""
    if x.size(1) == n_head:
        return x
    repeat = n_head // x.size(1)
    return x.repeat_interleave(repeat, dim=1)


def build_sliding_mask(query_positions, key_length, left_window, device):
    """Build a causal sliding attention mask."""
    keys = torch.arange(key_length, device=device)
    # Create a mask for allowed positions
    allow = keys.unsqueeze(0) <= query_positions.unsqueeze(1)
    # 窗口
    if left_window >= 0:
        allow &= keys.unsqueeze(0) >= (query_positions.unsqueeze(1) - left_window + 1)
    return allow


class CausalSelfAttention(nn.Module):
    #掩码因子注意力
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 12
        self.ve_gate = (
            Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )

    def _scaled_attention(self, q, k, v, window_size, kv_cache):
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if kv_cache is None:
            k = repeat_kv_heads(k, self.n_head)
            v = repeat_kv_heads(v, self.n_head)
            if window_size < 0 or window_size >= q.size(-2):
                return F.scaled_dot_product_attention(q, k, v, is_causal=True)
            query_positions = torch.arange(q.size(-2), device=q.device)
            mask = build_sliding_mask(query_positions, k.size(-2), window_size, q.device)
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        cache_position = kv_cache.get_pos()
        key_cache, value_cache = kv_cache.get_layer_cache(self.layer_idx)
        current_length = q.size(-2)
        key_cache[:, cache_position:cache_position + current_length] = k.transpose(1, 2)
        value_cache[:, cache_position:cache_position + current_length] = v.transpose(1, 2)
        key_length = cache_position + current_length

        full_k = repeat_kv_heads(key_cache[:, :key_length].transpose(1, 2), self.n_head)
        full_v = repeat_kv_heads(value_cache[:, :key_length].transpose(1, 2), self.n_head)
        query_positions = torch.arange(cache_position, cache_position + current_length, device=q.device)
        left_window = window_size if window_size >= 0 else -1
        mask = build_sliding_mask(query_positions, key_length, left_window, q.device)
        out = F.scaled_dot_product_attention(q, full_k, full_v, attn_mask=mask)
        if self.layer_idx == kv_cache.n_layers - 1:
            kv_cache.advance(current_length)
        return out

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        batch_size, seq_len, _ = x.shape
        q = self.c_q(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        k = self.c_k(x).view(batch_size, seq_len, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(batch_size, seq_len, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(batch_size, seq_len, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q = norm(apply_rotary_emb(q, cos, sin)) * 1.2
        k = norm(apply_rotary_emb(k, cos, sin)) * 1.2
        y = self._scaled_attention(q, k, v, window_size, kv_cache)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.c_proj(y)
