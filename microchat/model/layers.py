"""Small neural network layers used by the GPT model."""

from __future__ import annotations

import torch
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


class MiniMoE(nn.Module):
    """Tiny top-k MoE FFN for teaching routing mechanics."""

    def __init__(self, config):
        super().__init__()
        if config.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if config.moe_top_k <= 0:
            raise ValueError("moe_top_k must be positive")
        if config.moe_top_k > config.num_experts:
            raise ValueError("moe_top_k must be <= num_experts")
        #专家路由
        self.num_experts = config.num_experts
        self.top_k = config.moe_top_k
        self.router = nn.Linear(config.n_embd, config.num_experts, bias=False)
        self.experts = nn.ModuleList([MLP(config) for _ in range(config.num_experts)])
        self.shared_expert = MLP(config) if config.use_shared_expert else None
        self.latest_aux_loss = None #保存均衡损失
        self.latest_stats = None    #保存路由统计信息

    def reset_stats(self):
        self.latest_aux_loss = None
        self.latest_stats = None

    def forward(self, x):
        batch_size, seq_len, n_embd = x.shape
        #把x维度拍扁成(batch_size * seq_len, n_embd)，方便后续计算
        flat_x = x.reshape(batch_size * seq_len, n_embd)
        router_logits = F.linear(flat_x, self.router.weight.to(dtype=flat_x.dtype))
        router_probs = F.softmax(router_logits.float(), dim=-1)
        topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        topk_weights = (topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)).to(flat_x.dtype)

        flat_out = torch.zeros_like(flat_x)
        #选中的token走专家计算，结果加权后累加到flat_out中
        for expert_idx, expert in enumerate(self.experts):
            expert_positions, route_positions = torch.where(topk_indices == expert_idx)
            if expert_positions.numel() == 0:
                continue
            expert_input = flat_x.index_select(0, expert_positions)
            expert_output = expert(expert_input)
            route_weight = topk_weights[expert_positions, route_positions].unsqueeze(-1)
            flat_out.index_add_(0, expert_positions, expert_output * route_weight)
        #共享专家计算
        shared_fraction = None
        if self.shared_expert is not None:
            flat_out = flat_out + self.shared_expert(flat_x)
            shared_fraction = 1.0

        assignment_counts = torch.bincount(topk_indices.reshape(-1), minlength=self.num_experts).to(router_probs.dtype)
        total_assignments = max(int(topk_indices.numel()), 1)
        assignment_fraction = assignment_counts / total_assignments
        router_prob_mean = router_probs.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(router_prob_mean * assignment_fraction)

        self.latest_aux_loss = aux_loss
        self.latest_stats = {
            "num_tokens": int(flat_x.size(0)),
            "top_k": int(self.top_k),
            "total_assignments": int(topk_indices.numel()),
            "expert_counts": [int(value) for value in assignment_counts.detach().cpu().tolist()],
            "expert_fractions": [float(value) for value in assignment_fraction.detach().cpu().tolist()],
            "router_prob_mean": [float(value) for value in router_prob_mean.detach().cpu().tolist()],
            "aux_loss": float(aux_loss.detach().cpu().item()),
            "shared_expert": self.shared_expert is not None,
            "shared_fraction": shared_fraction,
        }
        return flat_out.reshape(batch_size, seq_len, n_embd)


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        from microchat.model.attention import CausalSelfAttention

        self.attn = CausalSelfAttention(config, layer_idx)
        if config.ffn_type == "dense":
            self.mlp = MLP(config)
        elif config.ffn_type == "moe":
            self.moe = MiniMoE(config)
        else:
            raise ValueError(f"Invalid ffn_type: {config.ffn_type!r}")

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        if hasattr(self, "mlp"):
            x = x + self.mlp(norm(x))
        else:
            x = x + self.moe(norm(x))
        return x
