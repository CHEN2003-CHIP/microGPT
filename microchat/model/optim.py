"""Optimizer helpers for the GPT model."""

from __future__ import annotations

import torch


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
