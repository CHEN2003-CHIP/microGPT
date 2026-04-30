"""Configuration for the teaching GPT model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GPTConfig:
    sequence_len: int = 256
    vocab_size: int = 8192
    n_layer: int = 4
    n_head: int = 4
    n_kv_head: int = 4
    n_embd: int = 256
    window_pattern: str = "L"
    standard_gpt_block: bool = False
    ffn_type: str = "dense"
    num_experts: int = 4
    moe_top_k: int = 2
    moe_aux_loss_weight: float = 0.01
    use_shared_expert: bool = True
