"""GPT model used by the microGPT teaching project."""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from microchat.common import COMPUTE_DTYPE
from microchat.model.config import GPTConfig
from microchat.model.layers import Block, Linear, has_ve, norm
from microchat.model.optim import CombinedOptimizer, Muon


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(padded_vocab_size, config.n_embd),
                "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
            }
        )
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer), requires_grad=not config.standard_gpt_block)
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer), requires_grad=not config.standard_gpt_block)
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1), requires_grad=not config.standard_gpt_block)
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1), requires_grad=not config.standard_gpt_block)

        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim

        self.value_embeds = nn.ModuleDict(
            {
                str(i): nn.Embedding(padded_vocab_size, kv_dim)
                for i in range(config.n_layer)
                if has_ve(i, config.n_layer) and not config.standard_gpt_block
            }
        )
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """Weight initialization as per GPT-2, with some adjustments for stability."""
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        n_embd = self.config.n_embd
        scale = (3 ** 0.5) * n_embd ** -0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -scale, scale)
            torch.nn.init.uniform_(block.attn.c_k.weight, -scale, scale)
            torch.nn.init.uniform_(block.attn.c_v.weight, -scale, scale)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -scale * 0.4, scale * 0.4)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)
        for layer_idx in range(self.config.n_layer):
            self.resid_lambdas.data[layer_idx] = 1.15 - (0.10 * layer_idx / max(self.config.n_layer - 1, 1))
            self.x0_lambdas.data[layer_idx] = 0.20 - (0.15 * layer_idx / max(self.config.n_layer - 1, 1))
        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.constant_(self.backout_lambda, 0.2)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)
        for value_embedding in self.value_embeds.values():
            torch.nn.init.uniform_(value_embedding.weight, -scale, scale)

        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for value_embedding in self.value_embeds.values():
                value_embedding.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        """Precompute cosine and sine values for rotary embeddings."""
        if device is None:
            device = self.transformer.wte.weight.device
        channels = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channels / head_dim))
        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(positions, inv_freq)
        cos = freqs.cos().to(COMPUTE_DTYPE)[None, :, None, :]
        sin = freqs.sin().to(COMPUTE_DTYPE)[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(char in "SL" for char in pattern), f"Invalid window pattern: {pattern}"
        long_window = config.sequence_len
        short_window = max(128, long_window // 4)
        char_to_window = {"L": -1, "S": short_window}
        window_sizes = [char_to_window[pattern[layer_idx % len(pattern)]] for layer_idx in range(config.n_layer)]
        window_sizes[-1] = -1
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    @staticmethod
    def _is_behavior_param(name):
        behavior_markers = (
            "smear_gate",
            "smear_lambda",
            "resid_lambdas",
            "x0_lambdas",
            "backout_lambda",
            "value_embeds",
            "ve_gate",
        )
        return any(marker in name for marker in behavior_markers)

    @staticmethod
    def _is_no_decay_param(name, param):
        return (
            param.ndim < 2
            or "transformer.wte" in name
            or "value_embeds" in name
            or "lm_head" in name
        )

    def setup_optimizer(self, lr=3e-4, weight_decay=0.01):
        muon_params = []
        adamw_decay_params = []
        adamw_no_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if (
                param.ndim >= 2
                and "transformer.wte" not in name
                and "value_embeds" not in name
                and "lm_head" not in name
            ):
                muon_params.append(param)
            elif param.ndim < 2 or "transformer.wte" in name or "value_embeds" in name:
                adamw_no_decay_params.append(param)
            else:
                adamw_decay_params.append(param)

        adamw_kwargs = {"lr": lr, "betas": (0.9, 0.95)}
        if "fused" in inspect.signature(torch.optim.AdamW).parameters:
            adamw_kwargs["fused"] = self.get_device().type == "cuda"

        muon = Muon(muon_params, lr=lr * 20, momentum=0.95) if muon_params else None
        adamw = torch.optim.AdamW(
            [
                {"params": adamw_decay_params, "weight_decay": weight_decay},
                {"params": adamw_no_decay_params, "weight_decay": 0.0},
            ],
            **adamw_kwargs,
        )
        return CombinedOptimizer(muon, adamw)

    def setup_sft_optimizer(
        self,
        lr=2e-5,
        weight_decay=0.01,
        optimizer_type="adamw_full",
        behavior_lr_scale=0.2,
    ):
        decay_params = []
        no_decay_params = []
        behavior_decay_params = []
        behavior_no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            is_behavior = self._is_behavior_param(name)
            is_no_decay = self._is_no_decay_param(name, param)

            if is_behavior and optimizer_type == "adamw_behavior_low_lr":
                if is_no_decay:
                    behavior_no_decay_params.append(param)
                else:
                    behavior_decay_params.append(param)
                continue

            if is_no_decay:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay, "lr_scale": 1.0},
            {"params": no_decay_params, "weight_decay": 0.0, "lr_scale": 1.0},
        ]
        if optimizer_type == "adamw_behavior_low_lr":
            param_groups.extend(
                [
                    {
                        "params": behavior_decay_params,
                        "weight_decay": weight_decay,
                        "lr_scale": behavior_lr_scale,
                    },
                    {
                        "params": behavior_no_decay_params,
                        "weight_decay": 0.0,
                        "lr_scale": behavior_lr_scale,
                    },
                ]
            )

        param_groups = [group for group in param_groups if group["params"]]
        adamw_kwargs = {"lr": lr, "betas": (0.9, 0.95)}
        if "fused" in inspect.signature(torch.optim.AdamW).parameters:
            adamw_kwargs["fused"] = self.get_device().type == "cuda"
        return torch.optim.AdamW(param_groups, **adamw_kwargs)

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        _, seq_len = idx.size()
        start = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, start:start + seq_len], self.sin[:, start:start + seq_len]
        x = norm(self.transformer.wte(idx).to(COMPUTE_DTYPE))

        if kv_cache is None:
            if seq_len > 1:
                if not self.config.standard_gpt_block:
                    gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                    x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            previous = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if seq_len > 1 and not self.config.standard_gpt_block:
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif previous is not None and not self.config.standard_gpt_block:
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                x = x + gate * previous

        x0 = x
        backout = None
        backout_layer = self.config.n_layer // 2
        for layer_idx, block in enumerate(self.transformer.h):
            if not self.config.standard_gpt_block:
                x = self.resid_lambdas[layer_idx] * x + self.x0_lambdas[layer_idx] * x0
            value_embedding = self.value_embeds[str(layer_idx)](idx).to(x.dtype) if str(layer_idx) in self.value_embeds else None
            x = block(x, value_embedding, cos_sin, self.window_sizes[layer_idx], kv_cache)
            if layer_idx == backout_layer and not self.config.standard_gpt_block:
                backout = x

        if backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * backout
        logits = self.lm_head(norm(x))[..., :self.config.vocab_size].float()
        logits = 15 * torch.tanh(logits / 15)

        if targets is None:
            return logits
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction=loss_reduction,
        )
        return loss

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """Generate text using the GPT-2 model."""
        device = self.get_device()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            logits = self.forward(ids)[:, -1, :]
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            if temperature == 0.0:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            ids = torch.cat([ids, next_ids], dim=1)
            yield next_ids.item()
