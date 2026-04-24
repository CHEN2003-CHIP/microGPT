"""GPT model used by the microGPT teaching project."""

from __future__ import annotations

from dataclasses import dataclass
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from microchat.common import COMPUTE_DTYPE


#归一化层-rmsNorm
def norm(x):
    # Use RMS norm for better stability with low-precision training.
    return F.rms_norm(x, (x.size(-1),))

#线形层y=xW+b
class Linear(nn.Linear):
    """Match weight dtype to activation dtype during forward."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype), self.bias)

#判断是否需要ve,隔层一个
def has_ve(layer_idx, n_layer):
    """ve = value embedding, an additional input to the attention mechanism. Only add it to some layers to save compute."""
    return layer_idx % 2 == (n_layer - 1) % 2

#旋转位置编码,ROPE优于传统transformer的原因之一
def apply_rotary_emb(x, cos, sin):
    """旋转位置编码，将输入张量x的前半部分和后半部分分别乘以cos和sin，并进行旋转变换。"""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=-1)

#GQA-分组注意力需要把少的KV进行复制几倍，以匹配查询头的数量
def repeat_kv_heads(x, n_head):
    """Repeat the key-value heads to match the number of attention heads."""
    if x.size(1) == n_head:
        return x
    repeat = n_head // x.size(1)
    return x.repeat_interleave(repeat, dim=1)

#做mask掩码矩阵+窗口限制，窗口限制是为了减少计算量，掩码矩阵是为了保证自回归的性质，即每个位置只能看到之前的位置
def build_sliding_mask(query_positions, key_length, left_window, device):
    """Build a sliding attention mask for the given query positions and key length.""" 
    keys = torch.arange(key_length, device=device)
    """
    [                   
    [T, . , . , . ],
    [T, T , . , . ],          
    [T, T , T , . ],
    [T, T , T , T ]
    ]
    """
    allow = keys.unsqueeze(0) <= query_positions.unsqueeze(1)
    if left_window >= 0:
        allow &= keys.unsqueeze(0) >= (query_positions.unsqueeze(1) - left_window + 1)
    """
    [[T . . .]
    [T T . .]
    [. T T .]
    [. . T T]]
    """
    return allow


def _zeropower_via_newtonschulz5(x, steps=5, eps=1e-7):
    """Approximate the nearest orthogonal matrix using a low-cost Newton-Schulz iteration."""
    orig_dtype = x.dtype
    y = x.float()
    if y.size(-2) > y.size(-1):
        y = y.mT
        transposed = True
    else:
        transposed = False
    y = y / (y.norm() + eps)
    for _ in range(steps):
        a = y @ y.mT
        b = 0.5 * (3.0 * torch.eye(a.size(-1), device=a.device, dtype=a.dtype) - a)
        y = b @ y
    if transposed:
        y = y.mT
    return y.to(orig_dtype)


class Muon(torch.optim.Optimizer):
    """Muon optimizer for matrix parameters."""
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if grad.ndim < 2:
                    raise ValueError("Muon only supports matrix-like parameters")
                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)
                update = grad.add(buf, alpha=momentum) if nesterov else buf
                update = _zeropower_via_newtonschulz5(update, steps=ns_steps)
                param.add_(update, alpha=-lr)
        return loss


class CombinedOptimizer:
    """Small wrapper that presents Muon + AdamW as one optimizer."""
    def __init__(self, *optimizers):
        self.optimizers = [opt for opt in optimizers if opt is not None]
        self.param_groups = [group for opt in self.optimizers for group in opt.param_groups]

    def zero_grad(self, set_to_none=True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        for i, opt in enumerate(self.optimizers):
            opt_loss = opt.step(closure if i == 0 else None)
            if opt_loss is not None:
                loss = opt_loss
        return loss

    def state_dict(self):
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict):
        for opt, opt_state in zip(self.optimizers, state_dict["optimizers"]):
            opt.load_state_dict(opt_state)

    def train(self):
        for opt in self.optimizers:
            if hasattr(opt, "train"):
                opt.train()

    def eval(self):
        for opt in self.optimizers:
            if hasattr(opt, "eval"):
                opt.eval()


@dataclass
class GPTConfig:
    sequence_len: int = 256         #序列长度
    vocab_size: int = 8192          #词表大小
    n_layer: int = 4                #层数
    n_head: int = 4                 #查询头数量
    n_kv_head: int = 4              #KV头数量
    n_embd: int = 256               #嵌入维度
    window_pattern: str = "L"       #窗口模式，S短，L全
    standard_gpt_block: bool = False


#自注意力机制
class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx                                                       #层号
        self.n_head = config.n_head                                                      #查询头数量
        self.n_kv_head = config.n_kv_head                                                #KV头数量
        self.n_embd = config.n_embd                                                      #嵌入维度
        self.head_dim = self.n_embd // self.n_head                                       #每个头维度
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)          #Q变换层
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)       #K变换层
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)       #V变换层
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)                       #映射组装层，把多头的输出映射回嵌入维度
        self.ve_gate_channels = 12                                                       #VE门控通道数，经验值，越大越能利用VE信息，但也增加计算量
        #y=wt
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    #缩放注意力
    def _scaled_attention(self, q, k, v, window_size, kv_cache):
        q = q.transpose(1, 2)  # (B, H, Tq, D)
        k = repeat_kv_heads(k.transpose(1, 2), self.n_head)  # (B, H, Tk, D)
        v = repeat_kv_heads(v.transpose(1, 2), self.n_head)

        #没有KVcache，说明是训练或者生成的第一步，直接计算注意力
        if kv_cache is None:
            if window_size < 0 or window_size >= q.size(-2):
                #缩放点积注意力+因果掩码，保证自回归性质
                return F.scaled_dot_product_attention(q, k, v, is_causal=True)
            #有窗口就构造窗口掩码
            query_positions = torch.arange(q.size(-2), device=q.device)
            mask = build_sliding_mask(query_positions, k.size(-2), window_size, q.device)
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        #有KVcache，说明是训练或者生成的中间步骤，需要更新KVcache
        cache_position = kv_cache.get_pos()
        key_cache, value_cache = kv_cache.get_layer_cache(self.layer_idx)
        current_length = q.size(-2)
        #写入缓存
        key_cache[:, cache_position:cache_position + current_length] = k.transpose(1, 2)
        value_cache[:, cache_position:cache_position + current_length] = v.transpose(1, 2)
        key_length = cache_position + current_length

        full_k = repeat_kv_heads(key_cache[:, :key_length].transpose(1, 2), self.n_head)
        full_v = repeat_kv_heads(value_cache[:, :key_length].transpose(1, 2), self.n_head)
        #构造mask+QKV计算注意力
        query_positions = torch.arange(cache_position, cache_position + current_length, device=q.device)
        left_window = window_size if window_size >= 0 else -1
        mask = build_sliding_mask(query_positions, key_length, left_window, q.device)
        out = F.scaled_dot_product_attention(q, full_k, full_v, attn_mask=mask)
        if self.layer_idx == kv_cache.n_layers - 1:
            kv_cache.advance(current_length)
        return out

    #前向处理
    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        #计算Q,K,V矩阵，并调整形状以适应多头注意力机制
        batch_size, seq_len, _ = x.shape
        q = self.c_q(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        k = self.c_k(x).view(batch_size, seq_len, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(batch_size, seq_len, self.n_kv_head, self.head_dim)
        #VE门控
        if ve is not None:
            ve = ve.view(batch_size, seq_len, self.n_kv_head, self.head_dim) #（B, T, Hk, D）
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            #给输入增加ve信息，ve_gate的作用是根据输入动态调整ve的影响力，gate的值越大，ve对v的影响越大
            v = v + gate.unsqueeze(-1) * ve

        #Q,K,V矩阵进行缩放和旋转位置编码，然后计算注意力输出
        cos, sin = cos_sin
        q = norm(apply_rotary_emb(q, cos, sin)) * 1.2
        k = norm(apply_rotary_emb(k, cos, sin)) * 1.2
        y = self._scaled_attention(q, k, v, window_size, kv_cache)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.c_proj(y)

#MLP层，作为前馈神经网络
class MLP(nn.Module):
    #放大特征维度，增加非线性，最后映射回原始维度
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


#Transformer块，包含一个自注意力层和一个MLP层，并且有残差连接和层归一化
class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x

#GPT模型
class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        #把词表大小填充到pad_vocab_size_to的倍数，以优化嵌入矩阵的内存布局和计算效率
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        #构造tranformer层，包含词嵌入层和多个Block层
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(padded_vocab_size, config.n_embd),
                "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
            }
        )
        #输出层，把Transformer的输出映射到词表大小的维度，用于预测下一个词的概率分布
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        #公式：x_next = x_cur + λ_resid * block_out + λ_x0 * x0
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer), requires_grad=not config.standard_gpt_block)
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer), requires_grad=not config.standard_gpt_block)
        # 门控投影：将24维输入特征映射为1维门控分数
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1), requires_grad=not config.standard_gpt_block)
        #out = λ_backout * x
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
        #缓存cos,sin计算结果，避免每次前向计算时都要重新计算，节省计算资源和时间
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)


    @torch.no_grad()
    def init_weights(self):
        """weight initialization as per GPT-2, with some adjustments for stability."""
        #词嵌入层权重初始化为均值为0，标准差为0.8的正态分布，以提供更大的初始激活范围，促进训练初期的梯度流动
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        #输出层权重初始化为较小的值，以避免初始阶段过大的梯度，导致训练不稳定
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)


        n_embd = self.config.n_embd
        #这里用的均匀分布，限制大小范围
        scale = (3 ** 0.5) * n_embd ** -0.5
        #各个模块初始权值
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -scale, scale)
            torch.nn.init.uniform_(block.attn.c_k.weight, -scale, scale)
            torch.nn.init.uniform_(block.attn.c_v.weight, -scale, scale)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -scale * 0.4, scale * 0.4)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)
        #初始输出权值（x_next = x_cur + λ_resid * block_out + λ_x0 * x0）
        for layer_idx in range(self.config.n_layer):
            self.resid_lambdas.data[layer_idx] = 1.15 - (0.10 * layer_idx / max(self.config.n_layer - 1, 1))
            self.x0_lambdas.data[layer_idx] = 0.20 - (0.15 * layer_idx / max(self.config.n_layer - 1, 1))
        #平滑门控权值小一点，避免初始阶段过大的影响，导致训练不稳定
        torch.nn.init.zeros_(self.smear_lambda)
        
        torch.nn.init.constant_(self.backout_lambda, 0.2)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)
        #ve嵌入层权重初始化为均匀分布，范围根据经验值调整，以提供适当的初始激活范围，促进训练初期的梯度流动
        for value_embedding in self.value_embeds.values():
            torch.nn.init.uniform_(value_embedding.weight, -scale, scale)

        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        #统一计算单位
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for value_embedding in self.value_embeds.values():
                value_embedding.to(dtype=COMPUTE_DTYPE)

    #提前计算cos,sin，节省计算资源和时间
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        """Precompute the cosine and sine values for rotary embeddings to speed up training and inference."""
        if device is None:
            device = self.transformer.wte.weight.device
        #生成[0, 2, 4, ..., head_dim-2]的序列，作为频率计算的基础
        channels = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        #计算频率，使用指数衰减的方式，较高维度的频率更快衰减，以提供更丰富的位置编码信息
        inv_freq = 1.0 / (base ** (channels / head_dim))
        #生成cos,sin序列
        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(positions, inv_freq)
        cos = freqs.cos().to(COMPUTE_DTYPE)[None, :, None, :]
        sin = freqs.sin().to(COMPUTE_DTYPE)[None, :, None, :]
        return cos, sin

    #计算每个层的窗口大小
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

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        _, seq_len = idx.size()
        start = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, start:start + seq_len], self.sin[:, start:start + seq_len]
        #归一化
        x = norm(self.transformer.wte(idx).to(COMPUTE_DTYPE))
        #加上平滑
        if kv_cache is None:
            if seq_len > 1:
                #平滑输入，增强模型对连续输入的敏感性，促进更连贯的生成。gate的值根据输入动态调整，越大越能利用前一个位置的信息，但也增加计算量和可能的过拟合风险
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
            #公式：x_next = x_cur + λ_resid * block_out + λ_x0 * x0
            if not self.config.standard_gpt_block:
                x = self.resid_lambdas[layer_idx] * x + self.x0_lambdas[layer_idx] * x0
            #拿到嵌入向量，隔层一个，节省计算资源和时间
            value_embedding = self.value_embeds[str(layer_idx)](idx).to(x.dtype) if str(layer_idx) in self.value_embeds else None
            #前向计算，得到下一层的输入
            x = block(x, value_embedding, cos_sin, self.window_sizes[layer_idx], kv_cache)
            #只在某些层存储backout信息，节省内存，同时提供足够的梯度流动路径，促进训练稳定性和性能提升
            if layer_idx == backout_layer and not self.config.standard_gpt_block:
                backout = x
        #减去backout信息，提供一个直接的梯度流动路径，帮助训练更深的模型，同时也可以看作是一种正则化，防止模型过拟合和梯度爆炸
        #让模型更加专注于学习新的残差
        if backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * backout
        #输出层，把Transformer的输出映射到词表大小的维度，用于预测下一个词的概率分布
        logits = self.lm_head(norm(x))[..., :self.config.vocab_size].float()
        logits = 15 * torch.tanh(logits / 15)

        if targets is None:
            return logits
        #计算损失，使用交叉熵损失函数，忽略掉标签为-1的位置，这些位置通常是填充位置，不参与损失计算，同时支持不同的损失缩放方式，如平均或求和，以适应不同的训练需求和资源限制
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
        #宇宙数字42，随机数种子，用于生成随机数
        rng.manual_seed(seed)
        # 输入初始 tokens，一步步生成下一个 token，直到 max_tokens
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            # 1. 前向推理，取【最后一个token】的输出 logits
            logits = self.forward(ids)[:, -1, :]
            # 2. 取Top-K 采样：只保留概率最高的 K 个 token，其余屏蔽
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            # 3. 生成下一个 token
            if temperature == 0.0:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # 随机采样：按概率随机选词（更有创意）
                probs = F.softmax(logits / temperature, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            ids = torch.cat([ids, next_ids], dim=1)
            yield next_ids.item()
