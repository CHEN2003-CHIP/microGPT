"""Backward-compatible GPT imports.

The implementation lives in :mod:`microchat.model`; this module preserves the
original public import path used by scripts and checkpoints.
"""

from microchat.model import (
    Block,
    CausalSelfAttention,
    CombinedOptimizer,
    GPT,
    GPTConfig,
    Linear,
    MiniMoE,
    MLP,
    Muon,
    apply_rotary_emb,
    build_sliding_mask,
    has_ve,
    norm,
    repeat_kv_heads,
)

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
