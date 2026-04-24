"""Simple autoregressive inference helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from microchat.common import COMPUTE_DTYPE


class KVCache:
    """Per-layer KV cache stored as (layer, batch, time, heads, head_dim)."""

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers, device, dtype):
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim
        self.k_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        self.v_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        self.prev_embedding = None

    def reset(self):
        self.cache_seqlens.zero_()
        self.prev_embedding = None

    def get_pos(self):
        return self.cache_seqlens[0].item()

    def get_layer_cache(self, layer_idx):
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, num_tokens):
        self.cache_seqlens += num_tokens

    def prefill(self, other):
        position = other.get_pos()
        self.k_cache[:, :, :position] = other.k_cache[:, :, :position]
        self.v_cache[:, :, :position] = other.v_cache[:, :, :position]
        self.cache_seqlens.fill_(position)
        if other.prev_embedding is not None:
            self.prev_embedding = other.prev_embedding.expand(self.batch_size, -1, -1).clone()


@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None, top_p=None):
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, indices = torch.topk(logits, top_k, dim=-1)
        logits = torch.full_like(logits, -float("inf"))
        logits.scatter_(1, indices, values)
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits / temperature, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[:, 0] = False
        sorted_logits = sorted_logits.masked_fill(sorted_mask, -float("inf"))
        logits = torch.full_like(logits, -float("inf"))
        logits.scatter_(1, sorted_indices, sorted_logits)
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=rng)


class Engine:
    """Streaming token generation around a cached GPT forward pass."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer


    #prefill先把提示词运算kvcache进行预热，然后在generate里用这个预热好的kvcache进行解码，直到生成结束或者达到最大生成长度
    @torch.inference_mode()
    def generate(
        self,
        tokens,
        num_samples=1,
        max_tokens=None,
        temperature=1.0,
        top_k=None,
        top_p=None,
        repetition_penalty=1.0,
        stop_token_ids=None,
        seed=42,
    ):
        """Generate tokens autoregressively, yielding one token at a time. Generation stops when max_tokens is reached or when all samples generate an end token."""
        assert isinstance(tokens, list) and tokens and isinstance(tokens[0], int)
        device = self.model.get_device()
        dtype = COMPUTE_DTYPE if device.type == "cuda" else torch.float32
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        stop_ids = set(stop_token_ids or [])
        stop_ids.update({assistant_end, bos})
        config = self.model.config
        # 1. Prefill the KV cache with the prompt tokens, getting the initial logits for the next token
        prefill_cache = KVCache(
            batch_size=1,
            num_heads=config.n_kv_head,
            seq_len=len(tokens),
            head_dim=config.n_embd // config.n_head,
            num_layers=config.n_layer,
            device=device,
            dtype=dtype,
        )
        prompt_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(prompt_ids, kv_cache=prefill_cache)[:, -1, :].expand(num_samples, -1)

        cache_length = len(tokens) + (max_tokens or config.sequence_len)
        decode_cache = KVCache(
            batch_size=num_samples,
            num_heads=config.n_kv_head,
            seq_len=cache_length,
            head_dim=config.n_embd // config.n_head,
            num_layers=config.n_layer,
            device=device,
            dtype=dtype,
        )
        decode_cache.prefill(prefill_cache)

        completed = [False] * num_samples
        seen_tokens = [set(tokens) for _ in range(num_samples)]
        generated = 0
        while True:
            if max_tokens is not None and generated >= max_tokens:
                break
            if all(completed):
                break

            adjusted_logits = logits.clone()
            if repetition_penalty > 1.0:
                for row_index, seen in enumerate(seen_tokens):
                    if seen:
                        adjusted_logits[row_index, list(seen)] /= repetition_penalty
            next_ids = sample_next_token(adjusted_logits, rng, temperature, top_k, top_p)
            token_column = next_ids[:, 0].tolist()
            token_masks = [1] * num_samples

            for row, token in enumerate(token_column):
                seen_tokens[row].add(token)
                if token in stop_ids:
                    completed[row] = True

            yield token_column, token_masks
            generated += 1
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
            logits = self.model.forward(ids, kv_cache=decode_cache)[:, -1, :]

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        stop_ids = set(kwargs.get("stop_token_ids") or [])
        stop_ids.update({assistant_end, bos})

        for token_column, token_masks in self.generate(tokens, num_samples=num_samples, **kwargs):
            for row_index, (token, mask) in enumerate(zip(token_column, token_masks)):
                if completed[row_index]:
                    continue
                if token in stop_ids:
                    completed[row_index] = True
                    continue
                results[row_index].append(token)
                masks[row_index].append(mask)
            if all(completed):
                break

        return results, masks
