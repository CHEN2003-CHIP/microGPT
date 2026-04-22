"""GPT model used by the microGPT teaching project."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from microchat.common import COMPUTE_DTYPE



def norm(x):
    # Use RMS norm for better stability with low-precision training.
    return F.rms_norm(x, (x.size(-1),))


class Linear(nn.Linear):
    """Match weight dtype to activation dtype during forward."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype), self.bias)


def has_ve(layer_idx, n_layer):
    """ve = value embedding, an additional input to the attention mechanism. Only add it to some layers to save compute."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    """旋转位置编码，将输入张量x的前半部分和后半部分分别乘以cos和sin，并进行旋转变换。"""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=-1)


def repeat_kv_heads(x, n_head):
    """Repeat the key-value heads to match the number of attention heads."""
    if x.size(1) == n_head:
        return x
    repeat = n_head // x.size(1)
    return x.repeat_interleave(repeat, dim=1)


def build_sliding_mask(query_positions, key_length, left_window, device):
    """Build a sliding attention mask for the given query positions and key length.""" 
    
    keys = torch.arange(key_length, device=device)
    allow = keys.unsqueeze(0) <= query_positions.unsqueeze(1)
    if left_window >= 0:
        allow &= keys.unsqueeze(0) >= (query_positions.unsqueeze(1) - left_window + 1)
    return allow


@dataclass
class GPTConfig:
    sequence_len: int = 256
    vocab_size: int = 8192
    n_layer: int = 4
    n_head: int = 4
    n_kv_head: int = 4
    n_embd: int = 256
    window_pattern: str = "L"


class CausalSelfAttention(nn.Module):
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
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def _scaled_attention(self, q, k, v, window_size, kv_cache):
        q = q.transpose(1, 2)  # (B, H, Tq, D)
        k = repeat_kv_heads(k.transpose(1, 2), self.n_head)  # (B, H, Tk, D)
        v = repeat_kv_heads(v.transpose(1, 2), self.n_head)

        if kv_cache is None:
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
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


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
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict(
            {str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)}
        )
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
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

    def setup_optimizer(self, lr=3e-4, weight_decay=0.01):
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        _, seq_len = idx.size()
        start = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, start:start + seq_len], self.sin[:, start:start + seq_len]

        x = norm(self.transformer.wte(idx).to(COMPUTE_DTYPE))
        if kv_cache is None:
            if seq_len > 1:
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            previous = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if seq_len > 1:
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif previous is not None:
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                x = x + gate * previous

        x0 = x
        backout = None
        backout_layer = self.config.n_layer // 2
        for layer_idx, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[layer_idx] * x + self.x0_lambdas[layer_idx] * x0
            value_embedding = self.value_embeds[str(layer_idx)](idx).to(x.dtype) if str(layer_idx) in self.value_embeds else None
            x = block(x, value_embedding, cos_sin, self.window_sizes[layer_idx], kv_cache)
            if layer_idx == backout_layer:
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
