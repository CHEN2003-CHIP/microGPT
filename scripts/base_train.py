"""Train the base GPT model for the microGPT teaching project."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
import sys
import time

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


parser = argparse.ArgumentParser(description="Train the base GPT model")
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--depth", type=int, default=4, help="Number of transformer blocks")
parser.add_argument("--aspect-ratio", type=int, default=64, help="Model width is depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=64, help="Attention head dimension")
parser.add_argument("--max-seq-len", type=int, default=256, help="Context length")
parser.add_argument("--window-pattern", type=str, default="L", help="Attention window pattern")
parser.add_argument("--num-iterations", type=int, default=20, help="Number of optimizer steps")
parser.add_argument("--max-train-tokens", type=int, default=0, help="Optional token budget cap; 0 disables")
parser.add_argument("--device-batch-size", type=int, default=1, help="Per-device batch size")
parser.add_argument("--total-batch-size", type=int, default=512, help="Total tokens per optimizer step")
parser.add_argument("--learning-rate", type=float, default=3e-4, help="Peak AdamW learning rate")
parser.add_argument("--min-lr", type=float, default=3e-5, help="Minimum learning rate after cosine decay")
parser.add_argument("--warmup-iters", type=int, default=100, help="Linear warmup iterations")
parser.add_argument("--lr-decay-iters", type=int, default=0, help="Cosine decay length in optimizer steps; 0 uses total steps")
parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping max norm; <=0 disables")
parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw"], help="Optimizer choice")
parser.add_argument("--eval-every", type=int, default=10, help="Run validation every N steps")
parser.add_argument("--eval-tokens", type=int, default=1024, help="Validation token budget")
parser.add_argument("--save-every", type=int, default=0, help="Save a checkpoint every N steps; 0 disables periodic save")
parser.add_argument("--resume", action="store_true", help="Resume the latest checkpoint for the selected model tag")
parser.add_argument("--model-tag", type=str, default=None, help="Checkpoint directory name")
parser.add_argument("--standard-gpt-block", action="store_true", help="Disable experimental residual/value-embedding paths")
parser.add_argument("--ffn-type", type=str, default="dense", choices=["dense", "moe"], help="Feed-forward layer type")
parser.add_argument("--num-experts", type=int, default=4, help="Number of MoE experts when --ffn-type moe")
parser.add_argument("--moe-top-k", type=int, default=2, help="Number of routed experts per token in MoE mode")
parser.add_argument("--moe-aux-loss-weight", type=float, default=0.01, help="MoE load-balance loss weight")
parser.add_argument("--no-shared-expert", action="store_true", help="Disable the shared expert in MoE mode")
args = parser.parse_args()

import torch

from microchat.checkpoint_manager import find_last_step, load_checkpoint, load_optimizer_checkpoint, save_checkpoint
from microchat.common import autodetect_device_type, compute_cleanup, compute_init, get_base_dir, print0
from microchat.dataloader import token_batch_loader
from microchat.gpt import GPT, GPTConfig
from microchat.loss_eval import evaluate_bpb
from microchat.tokenizer import get_token_bytes, get_tokenizer


def resolve_model_tag():
    return args.model_tag or f"d{args.depth}"


def get_checkpoint_dir():
    return os.path.join(get_base_dir(), "base_checkpoints", resolve_model_tag())


def get_report_paths():
    reports_dir = os.path.join(get_base_dir(), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    model_tag = resolve_model_tag()
    return (
        os.path.join(reports_dir, f"{model_tag}_base_train_metrics.jsonl"),
        os.path.join(reports_dir, f"{model_tag}_base_train_report.md"),
    )


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
        standard_gpt_block=args.standard_gpt_block,
        ffn_type=args.ffn_type,
        num_experts=args.num_experts,
        moe_top_k=args.moe_top_k,
        moe_aux_loss_weight=args.moe_aux_loss_weight,
        use_shared_expert=not args.no_shared_expert,
    )


def estimate_total_steps(tokens_per_step: int):
    if args.max_train_tokens > 0:
        return max(1, math.ceil(args.max_train_tokens / tokens_per_step))
    return max(1, args.num_iterations)


def compute_learning_rate(step: int, total_steps: int):
    warmup_iters = max(0, args.warmup_iters)
    decay_iters = args.lr_decay_iters if args.lr_decay_iters > 0 else total_steps
    decay_iters = max(decay_iters, 1)
    if warmup_iters > 0 and step <= warmup_iters:
        return args.learning_rate * step / warmup_iters
    if step >= decay_iters:
        return args.min_lr
    if decay_iters <= warmup_iters:
        return args.min_lr
    progress = (step - warmup_iters) / (decay_iters - warmup_iters)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.min_lr + cosine * (args.learning_rate - args.min_lr)


def set_optimizer_lr(optimizer, lr: float):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def append_metric(path: str, payload):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def render_markdown_report(path: str, *, model_tag: str, config, summary: dict, history: list[dict]):
    last_metric = history[-1] if history else {}
    lines = [
        "# Base Train Report",
        "",
        f"- Model tag: `{model_tag}`",
        f"- Qualified to continue scaling: `{'YES' if summary['qualified_to_scale'] else 'NO'}`",
        f"- Final step: `{summary['final_step']}`",
        f"- Tokens seen: `{summary['tokens_seen']}`",
        f"- Best val_bpb: `{summary['best_val_bpb']}`",
        f"- Final val_bpb: `{summary['final_val_bpb']}`",
        f"- Final train_loss_ema: `{last_metric.get('train_loss_ema')}`",
        f"- Standard GPT block: `{config.standard_gpt_block}`",
        "",
        "## Recommendation",
        "",
        summary["recommendation"],
        "",
    ]
    if history:
        lines.extend(["## Latest Metrics", ""])
        for key in ("step", "lr", "train_loss_ema", "val_bpb", "tokens_seen", "tok_per_sec"):
            if key in last_metric:
                lines.append(f"- {key}: `{last_metric[key]}`")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")


def summarize_training(history: list[dict], best_val_bpb: float):
    val_points = [item for item in history if item.get("val_bpb") is not None]
    final_val = val_points[-1]["val_bpb"] if val_points else None
    qualified_to_scale = bool(final_val is not None and final_val <= best_val_bpb + 0.03)
    if final_val is None:
        recommendation = "Validation did not run; increase --eval-every or --eval-tokens before making scaling decisions."
    elif not qualified_to_scale:
        recommendation = "Validation has not stabilized enough yet. Keep the same model size and train on more tokens first."
    else:
        recommendation = "Validation looks stable enough to continue this recipe. Increase token budget before increasing model size."
    return qualified_to_scale, final_val, recommendation


def maybe_resume(model, optimizer, checkpoint_dir, device):
    if not args.resume:
        return 0, float("inf"), 0, []
    step = find_last_step(checkpoint_dir)
    model_state, metadata = load_checkpoint(checkpoint_dir, step, device)
    clean_state = {key.removeprefix("_orig_mod."): value for key, value in model_state.items()}
    if device.type in {"cpu", "mps"}:
        clean_state = {
            key: value.float() if getattr(value, "dtype", None) == torch.bfloat16 else value
            for key, value in clean_state.items()
        }
    model.load_state_dict(clean_state, strict=True)
    optimizer.load_state_dict(load_optimizer_checkpoint(checkpoint_dir, step, device))
    best_val_bpb = metadata.get("best_val_bpb", metadata.get("val_bpb", float("inf")))
    tokens_seen = metadata.get("tokens_seen", 0)
    history = metadata.get("recent_metrics", [])
    print0(f"Resumed {checkpoint_dir} at step {step}")
    return step, best_val_bpb, tokens_seen, history


def save_training_checkpoint(model, optimizer, step, validation_bpb, best_val_bpb, config, tokens_seen, recent_metrics):
    checkpoint_dir = get_checkpoint_dir()
    save_checkpoint(
        checkpoint_dir,
        step,
        model.state_dict(),
        {
            "step": step,
            "val_bpb": validation_bpb,
            "best_val_bpb": best_val_bpb,
            "tokens_seen": tokens_seen,
            "model_config": asdict(config),
            "training_config": vars(args),
            "recent_metrics": recent_metrics[-10:],
        },
        optimizer_state=optimizer.state_dict(),
    )
    print0(f"Saved checkpoint to {checkpoint_dir} at step {step}")


def train():
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _, _, _, _, device = compute_init(device_type)
    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device=device)

    config = build_model_config(tokenizer)
    model = GPT(config).to(device)
    model.init_weights()
    optimizer = model.setup_optimizer(lr=args.learning_rate, weight_decay=args.weight_decay)

    checkpoint_dir = get_checkpoint_dir()
    start_step, best_val_bpb, tokens_seen, history = maybe_resume(model, optimizer, checkpoint_dir, device)

    tokens_per_micro_step = args.device_batch_size * args.max_seq_len
    assert args.total_batch_size % tokens_per_micro_step == 0, "total_batch_size must divide into micro-steps"
    grad_accum_steps = args.total_batch_size // tokens_per_micro_step
    tokens_per_step = grad_accum_steps * tokens_per_micro_step
    target_steps = estimate_total_steps(tokens_per_step)
    final_step = max(args.num_iterations, start_step)
    if args.max_train_tokens > 0:
        final_step = max(final_step, target_steps)

    print0(f"Model config: {asdict(config)}")
    print0(f"Gradient accumulation steps: {grad_accum_steps}")
    print0(f"Tokens per optimizer step: {tokens_per_step}")
    print0(f"Target training steps: {final_step}")

    train_loader = token_batch_loader(tokenizer, args.device_batch_size, args.max_seq_len, "train", device)
    val_loader = lambda: token_batch_loader(tokenizer, args.device_batch_size, args.max_seq_len, "val", device)
    current_inputs, current_targets = next(train_loader)

    ema_loss = 0.0
    ema_beta = 0.9
    started = time.time()
    metrics_path, report_path = get_report_paths()

    if start_step == 0 and os.path.exists(metrics_path):
        os.remove(metrics_path)

    step = start_step
    while step < final_step:
        if args.max_train_tokens > 0 and tokens_seen >= args.max_train_tokens:
            break
        step += 1
        model.train()
        step_loss = 0.0
        step_started = time.time()
        lr = compute_learning_rate(step, final_step)
        set_optimizer_lr(optimizer, lr)

        for _ in range(grad_accum_steps):
            loss = model(current_inputs, current_targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Encountered a non-finite base training loss.")
            step_loss += loss.item()
            (loss / grad_accum_steps).backward()
            current_inputs, current_targets = next(train_loader)

        grad_norm = None
        if args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        ema_loss = ema_beta * ema_loss + (1 - ema_beta) * step_loss
        smooth_loss = ema_loss / (1 - ema_beta ** step)
        tokens_seen += tokens_per_step
        step_elapsed = max(time.time() - step_started, 1e-6)
        tok_per_sec = tokens_per_step / step_elapsed

        metric = {
            "step": step,
            "lr": round(lr, 10),
            "train_loss_ema": round(float(smooth_loss), 6),
            "tokens_seen": int(tokens_seen),
            "tok_per_sec": round(tok_per_sec, 2),
        }
        if grad_norm is not None:
            metric["grad_norm"] = round(float(grad_norm), 6)

        if step == 1 or step % args.eval_every == 0 or step == final_step:
            model.eval()
            eval_steps = max(1, args.eval_tokens // tokens_per_micro_step)
            val_bpb = evaluate_bpb(model, val_loader(), eval_steps, token_bytes)
            best_val_bpb = min(best_val_bpb, val_bpb)
            metric["val_bpb"] = round(float(val_bpb), 6)
            print0(
                f"step {step:05d} | loss {smooth_loss:.4f} | val_bpb {val_bpb:.4f} | "
                f"lr {lr:.6f} | tokens {tokens_seen} | tok/s {tok_per_sec:.1f}"
            )
        else:
            metric["val_bpb"] = None
            print0(
                f"step {step:05d} | loss {smooth_loss:.4f} | lr {lr:.6f} | "
                f"tokens {tokens_seen} | tok/s {tok_per_sec:.1f}"
            )

        history.append(metric)
        append_metric(metrics_path, metric)

        if args.save_every > 0 and step % args.save_every == 0:
            save_training_checkpoint(
                model,
                optimizer,
                step,
                metric["val_bpb"],
                best_val_bpb,
                config,
                tokens_seen,
                history,
            )

    elapsed = time.time() - started
    qualified_to_scale, final_val_bpb, recommendation = summarize_training(history, best_val_bpb)
    save_training_checkpoint(
        model,
        optimizer,
        step,
        final_val_bpb,
        best_val_bpb,
        config,
        tokens_seen,
        history,
    )
    render_markdown_report(
        report_path,
        model_tag=resolve_model_tag(),
        config=config,
        summary={
            "qualified_to_scale": qualified_to_scale,
            "final_step": step,
            "tokens_seen": tokens_seen,
            "best_val_bpb": best_val_bpb,
            "final_val_bpb": final_val_bpb,
            "recommendation": recommendation,
        },
        history=history,
    )
    print0(f"Training finished in {elapsed/60:.2f} minutes")
    print0(f"Metrics written to {metrics_path}")
    print0(f"Report written to {report_path}")
    compute_cleanup()


if __name__ == "__main__":
    train()
