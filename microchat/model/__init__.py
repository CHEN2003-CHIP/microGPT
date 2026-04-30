"""Structured model package for microGPT."""

from microchat.model.attention import (
    CausalSelfAttention,
    apply_rotary_emb,
    build_sliding_mask,
    repeat_kv_heads,
)
from microchat.model.config import GPTConfig
from microchat.model.gpt import GPT
from microchat.model.layers import Block, Linear, MiniMoE, MLP, has_ve, norm
from microchat.model.optim import CombinedOptimizer, Muon

__all__ = [
    "GPT",
    "GPTConfig",
    "Block",
    "CausalSelfAttention",
    "MLP",
    "MiniMoE",
    "Linear",
    "Muon",
    "CombinedOptimizer",
    "norm",
    "has_ve",
    "apply_rotary_emb",
    "repeat_kv_heads",
    "build_sliding_mask",
]
