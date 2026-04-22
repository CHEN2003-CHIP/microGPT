"""Train the base GPT model for the microGPT teaching project."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import os
import time

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


parser = argparse.ArgumentParser(description="Train the base GPT model")
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--depth", type=int, default=4, help="Number of transformer blocks")
parser.add_argument("--aspect-ratio", type=int, default=64, help="Model width is depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=64, help="Attention head dimension")
parser.add_argument("--max-seq-len", type=int, default=256, help="Context length")
parser.add_argument("--window-pattern", type=str, default="L", help="Attention window pattern")
parser.add_argument("--num-iterations", type=int, default=20, help="Number of optimizer steps")
parser.add_argument("--device-batch-size", type=int, default=1, help="Per-device batch size")
parser.add_argument("--total-batch-size", type=int, default=512, help="Total tokens per optimizer step")
parser.add_argument("--learning-rate", type=float, default=3e-4, help="AdamW learning rate")
parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
parser.add_argument("--eval-every", type=int, default=10, help="Run validation every N steps")
parser.add_argument("--eval-tokens", type=int, default=1024, help="Validation token budget")
parser.add_argument("--model-tag", type=str, default=None, help="Checkpoint directory name")
args = parser.parse_args()

import torch

from microchat.checkpoint_manager import save_checkpoint
from microchat.common import autodetect_device_type, compute_cleanup, compute_init, get_base_dir, print0
from microchat.dataloader import token_batch_loader
from microchat.gpt import GPT, GPTConfig
from microchat.loss_eval import evaluate_bpb
from microchat.tokenizer import get_token_bytes, get_tokenizer


def build_model_config(tokenizer):
    base_dim = args.depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    return GPTConfig(
        sequence_len=args.max_seq_len,
        vocab_size=tokenizer.get_vocab_size(),
        n_layer=args.depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=args.window_pattern,
    )


def save_final_checkpoint(model, step, validation_bpb, config):
    base_dir = get_base_dir()
    model_tag = args.model_tag or f"d{args.depth}"
    checkpoint_dir = os.path.join(base_dir, "base_checkpoints", model_tag)
    save_checkpoint(
        checkpoint_dir,
        step,
        model.state_dict(),
        {
            "step": step,
            "val_bpb": validation_bpb,
            "model_config": asdict(config),
            "training_config": vars(args),
        },
    )
    print0(f"Saved checkpoint to {checkpoint_dir}")


def train():
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _, _, _, _, device = compute_init(device_type)
    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device=device)

    config = build_model_config(tokenizer)
    model = GPT(config).to(device)
    model.init_weights()
    optimizer = model.setup_optimizer(lr=args.learning_rate, weight_decay=args.weight_decay)

    tokens_per_micro_step = args.device_batch_size * args.max_seq_len
    assert args.total_batch_size % tokens_per_micro_step == 0, "total_batch_size must divide into micro-steps"
    grad_accum_steps = args.total_batch_size // tokens_per_micro_step
    print0(f"Model config: {asdict(config)}")
    print0(f"Gradient accumulation steps: {grad_accum_steps}")

    train_loader = token_batch_loader(tokenizer, args.device_batch_size, args.max_seq_len, "train", device)
    val_loader = lambda: token_batch_loader(tokenizer, args.device_batch_size, args.max_seq_len, "val", device)

    current_inputs, current_targets = next(train_loader)
    best_val_bpb = float("inf")
    ema_loss = 0.0
    ema_beta = 0.9
    started = time.time()

    for step in range(1, args.num_iterations + 1):
        model.train()
        step_loss = 0.0
        for _ in range(grad_accum_steps):
            loss = model(current_inputs, current_targets)
            step_loss += loss.item()
            (loss / grad_accum_steps).backward()
            current_inputs, current_targets = next(train_loader)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        ema_loss = ema_beta * ema_loss + (1 - ema_beta) * step_loss
        smooth_loss = ema_loss / (1 - ema_beta ** step)

        if step == 1 or step % args.eval_every == 0 or step == args.num_iterations:
            model.eval()
            eval_steps = max(1, args.eval_tokens // tokens_per_micro_step)
            val_bpb = evaluate_bpb(model, val_loader(), eval_steps, token_bytes)
            best_val_bpb = min(best_val_bpb, val_bpb)
            print0(f"step {step:04d} | loss {smooth_loss:.4f} | val_bpb {val_bpb:.4f}")
        else:
            print0(f"step {step:04d} | loss {smooth_loss:.4f}")

    elapsed = time.time() - started
    final_val_bpb = best_val_bpb if best_val_bpb != float("inf") else None
    save_final_checkpoint(model, args.num_iterations, final_val_bpb, config)
    print0(f"Training finished in {elapsed/60:.2f} minutes")
    compute_cleanup()


if __name__ == "__main__":
    train()
