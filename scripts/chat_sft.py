"""Supervised fine-tuning for the microGPT teaching project."""

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


parser = argparse.ArgumentParser(description="Run supervised fine-tuning")
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--model-tag", type=str, default=None, help="Base model tag to load and SFT tag to write")
parser.add_argument("--model-step", type=int, default=None, help="Base checkpoint step to load")
parser.add_argument("--num-iterations", type=int, default=1000, help="Number of optimizer steps")
parser.add_argument("--max-train-tokens", type=int, default=0, help="Optional token budget cap; 0 disables")
parser.add_argument("--max-seq-len", type=int, default=256, help="Context length")
parser.add_argument("--device-batch-size", type=int, default=1, help="Per-device batch size")
parser.add_argument("--total-batch-size", type=int, default=512, help="Total tokens per optimizer step")
parser.add_argument("--learning-rate", type=float, default=2e-5, help="Peak AdamW learning rate")
parser.add_argument("--min-lr", type=float, default=2e-6, help="Minimum learning rate after cosine decay")
parser.add_argument("--warmup-iters", type=int, default=100, help="Linear warmup iterations")
parser.add_argument("--lr-decay-iters", type=int, default=0, help="Cosine decay length in optimizer steps; 0 uses total steps")
parser.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay")
parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping max norm; <=0 disables")
parser.add_argument("--eval-every", type=int, default=100, help="Run validation every N steps")
parser.add_argument("--eval-tokens", type=int, default=2048, help="Validation token budget")
parser.add_argument("--save-every", type=int, default=250, help="Save a checkpoint every N steps; 0 disables periodic save")
parser.add_argument("--resume", action="store_true", help="Resume the latest SFT checkpoint for the selected model tag")
parser.add_argument("--smoltalk-limit", type=int, default=0, help="Limit SmolTalk rows; 0 disables SmolTalk")
parser.add_argument("--ultrachat-limit", type=int, default=20_000, help="Limit UltraChat 200k rows; 0 disables UltraChat")
parser.add_argument("--local-repeat", type=int, default=40, help="Oversample local curated identity/instruction examples")
args = parser.parse_args()

import torch

from microchat.checkpoint_manager import (
    find_last_step,
    load_checkpoint,
    load_model,
    load_optimizer_checkpoint,
    save_checkpoint,
)
from microchat.common import autodetect_device_type, compute_cleanup, compute_init, get_base_dir, print0
from microchat.loss_eval import evaluate_bpb
from microchat.tokenizer import get_token_bytes
from tasks.common import TaskMixture
from tasks.customjson import CustomJSON
from tasks.smoltalk import SmolTalk
from tasks.ultrachat import UltraChat


def get_sft_checkpoint_dir():
    return os.path.join(get_base_dir(), "chatsft_checkpoints", args.model_tag or "d4")


def get_report_paths():
    reports_dir = os.path.join(get_base_dir(), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    model_tag = args.model_tag or "d4"
    return (
        os.path.join(reports_dir, f"{model_tag}_chat_sft_metrics.jsonl"),
        os.path.join(reports_dir, f"{model_tag}_chat_sft_report.md"),
    )


def resolve_identity_dataset(base_dir):
    preferred = os.path.join(base_dir, "identity_conversations.jsonl")
    anchor_en = os.path.join(os.path.dirname(__file__), "identity_conversations.anchor_en.jsonl")
    curated = os.path.join(os.path.dirname(__file__), "identity_conversations.curated.jsonl")
    bundled = os.path.join(os.path.dirname(__file__), "identity_conversations.sample.jsonl")
    if os.path.exists(preferred):
        return preferred
    if os.path.exists(anchor_en):
        return anchor_en
    if os.path.exists(curated):
        return curated
    return bundled


def build_sft_datasets(base_dir):
    """
    Prefer UltraChat/SmolTalk mixed with a curated local identity/instruction set.
    If remote datasets are unavailable, fall back to the local curated set only.
    """
    identity_path = resolve_identity_dataset(base_dir)
    local_task = CustomJSON(filepath=identity_path)
    local_tasks = [CustomJSON(filepath=identity_path) for _ in range(max(args.local_repeat, 1))]

    if args.smoltalk_limit == 0 and args.ultrachat_limit == 0:
        train_dataset = TaskMixture(local_tasks)
        val_dataset = TaskMixture([local_task])
        dataset_label = f"CustomJSON only (curated local mode, x{max(args.local_repeat, 1)})"
        return train_dataset, val_dataset, dataset_label

    try:
        train_tasks = []
        val_tasks = [local_task]
        labels = []

        if args.smoltalk_limit > 0:
            train_tasks.append(SmolTalk(split="train", stop=args.smoltalk_limit))
            val_tasks.append(SmolTalk(split="test", stop=min(args.smoltalk_limit, 2_000)))
            labels.append(f"SmolTalk(limit={args.smoltalk_limit})")

        if args.ultrachat_limit > 0:
            train_tasks.append(UltraChat(split="train", stop=args.ultrachat_limit))
            val_tasks.append(UltraChat(split="test", stop=min(args.ultrachat_limit, 2_000)))
            labels.append(f"UltraChat(limit={args.ultrachat_limit})")

        if not train_tasks:
            train_dataset = TaskMixture(local_tasks)
            val_dataset = TaskMixture([local_task])
            dataset_label = f"CustomJSON only (curated local mode, x{max(args.local_repeat, 1)})"
            return train_dataset, val_dataset, dataset_label

        train_dataset = TaskMixture([*train_tasks, *local_tasks])
        val_dataset = TaskMixture(val_tasks)
        dataset_label = " + ".join([*labels, f"CustomJSON(x{max(args.local_repeat, 1)})"])
    except Exception as exc:
        print0(f"Remote SFT datasets unavailable on this machine, falling back to bundled local conversations: {exc!r}")
        train_dataset = TaskMixture([CustomJSON(filepath=identity_path) for _ in range(max(args.local_repeat, 3))])
        val_dataset = TaskMixture([CustomJSON(filepath=identity_path)])
        dataset_label = f"CustomJSON only (offline fallback, x{max(args.local_repeat, 3)})"

    return train_dataset, val_dataset, dataset_label


def build_sft_loader(tokenizer, dataset, batch_size, seq_len, device, buffer_size=128):
    bos_token = tokenizer.get_bos_token_id()
    row_capacity = seq_len + 1
    use_cuda = device.type == "cuda"
    buffer = []
    cursor = 0

    def refill_buffer():
        nonlocal cursor
        while len(buffer) < buffer_size:
            conversation = dataset[cursor % len(dataset)]
            cursor += 1
            token_ids, loss_mask = tokenizer.render_conversation(conversation, max_tokens=row_capacity)
            has_supervised_targets = len(loss_mask) > 1 and any(loss_mask[1:])
            if has_supervised_targets:
                buffer.append((token_ids, loss_mask))

    while True:
        refill_buffer()
        rows, masks, lengths = [], [], []
        for _ in range(batch_size):
            row, mask = [], []
            while len(row) < row_capacity:
                remaining = row_capacity - len(row)
                candidate_index = -1
                candidate_len = 0
                for index, (token_ids, _) in enumerate(buffer):
                    if len(token_ids) <= remaining and len(token_ids) > candidate_len:
                        candidate_index = index
                        candidate_len = len(token_ids)
                if candidate_index >= 0:
                    token_ids, loss_mask = buffer.pop(candidate_index)
                    row.extend(token_ids)
                    mask.extend(loss_mask)
                    continue
                content_length = len(row)
                row.extend([bos_token] * remaining)
                mask.extend([0] * remaining)
                lengths.append(content_length)
                break
            if len(lengths) < len(rows) + 1:
                lengths.append(row_capacity)
            rows.append(row[:row_capacity])
            masks.append(mask[:row_capacity])

        batch = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
        inputs = batch[:, :-1].to(device=device, dtype=torch.int64, non_blocking=use_cuda).contiguous()
        targets = batch[:, 1:].to(device=device, dtype=torch.int64, non_blocking=use_cuda).contiguous()
        mask_tensor = torch.tensor(masks, dtype=torch.int8)
        target_mask = mask_tensor[:, 1:].to(device=device)
        targets[target_mask == 0] = -1
        for row_index, content_length in enumerate(lengths):
            if content_length < row_capacity:
                targets[row_index, content_length - 1 :] = -1
        yield inputs, targets


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


def render_markdown_report(path: str, *, model_tag: str, summary: dict, history: list[dict], dataset_label: str):
    last_metric = history[-1] if history else {}
    lines = [
        "# Chat SFT Report",
        "",
        f"- Model tag: `{model_tag}`",
        f"- Dataset mix: `{dataset_label}`",
        f"- Qualified to evaluate: `{'YES' if summary['qualified_to_evaluate'] else 'NO'}`",
        f"- Final step: `{summary['final_step']}`",
        f"- Tokens seen: `{summary['tokens_seen']}`",
        f"- Best val_bpb: `{summary['best_val_bpb']}`",
        f"- Final val_bpb: `{summary['final_val_bpb']}`",
        f"- Final train_loss_ema: `{last_metric.get('train_loss_ema')}`",
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
    qualified_to_evaluate = bool(final_val is not None and math.isfinite(final_val))
    if final_val is None:
        recommendation = "Validation did not run; increase --eval-every or --eval-tokens before choosing a checkpoint."
    elif final_val > best_val_bpb + 0.05:
        recommendation = "Validation drifted after the best checkpoint. Prefer evaluating an earlier saved checkpoint."
    else:
        recommendation = "Validation stayed stable. Run chat_eval across saved checkpoints and pick the best step by score."
    return qualified_to_evaluate, final_val, recommendation


def save_training_checkpoint(model, optimizer, step, validation_bpb, best_val_bpb, tokens_seen, recent_metrics, dataset_label):
    checkpoint_dir = get_sft_checkpoint_dir()
    save_checkpoint(
        checkpoint_dir,
        step,
        model.state_dict(),
        {
            "step": step,
            "val_bpb": validation_bpb,
            "best_val_bpb": best_val_bpb,
            "tokens_seen": tokens_seen,
            "model_config": asdict(model.config),
            "training_config": vars(args),
            "dataset_label": dataset_label,
            "recent_metrics": recent_metrics[-10:],
        },
        optimizer_state=optimizer.state_dict(),
    )
    print0(f"Saved checkpoint to {checkpoint_dir} at step {step}")


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


def train():
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _, _, _, _, device = compute_init(device_type)
    model, tokenizer, metadata = load_model("base", device, phase="train", model_tag=args.model_tag, step=args.model_step)
    token_bytes = get_token_bytes(device=device)
    optimizer = model.setup_optimizer(lr=args.learning_rate, weight_decay=args.weight_decay)

    checkpoint_dir = get_sft_checkpoint_dir()
    start_step, best_val_bpb, tokens_seen, history = maybe_resume(model, optimizer, checkpoint_dir, device)

    base_dir = get_base_dir()
    train_dataset, val_dataset, dataset_label = build_sft_datasets(base_dir)
    train_loader = build_sft_loader(tokenizer, train_dataset, args.device_batch_size, args.max_seq_len, device)
    val_loader = lambda: build_sft_loader(tokenizer, val_dataset, args.device_batch_size, args.max_seq_len, device)

    tokens_per_micro_step = args.device_batch_size * args.max_seq_len
    assert args.total_batch_size % tokens_per_micro_step == 0, "total_batch_size must divide into micro-steps"
    grad_accum_steps = args.total_batch_size // tokens_per_micro_step
    tokens_per_step = grad_accum_steps * tokens_per_micro_step
    target_steps = estimate_total_steps(tokens_per_step)
    final_step = max(args.num_iterations, start_step)
    if args.max_train_tokens > 0:
        final_step = max(final_step, target_steps)

    print0(f"Loaded base checkpoint step {metadata['step']}")
    print0(f"Training mixture: {len(train_dataset):,} conversations ({dataset_label})")
    print0(f"Gradient accumulation steps: {grad_accum_steps}")
    print0(f"Tokens per optimizer step: {tokens_per_step}")
    print0(f"Target training steps: {final_step}")

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
                raise RuntimeError(
                    "Encountered a non-finite SFT loss. "
                    "This usually means the batch had no supervised assistant targets "
                    "or the optimization became numerically unstable."
                )
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
                f"lr {lr:.7f} | tokens {tokens_seen} | tok/s {tok_per_sec:.1f}"
            )
        else:
            metric["val_bpb"] = None
            print0(
                f"step {step:05d} | loss {smooth_loss:.4f} | lr {lr:.7f} | "
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
                tokens_seen,
                history,
                dataset_label,
            )

    elapsed = time.time() - started
    qualified_to_evaluate, final_val_bpb, recommendation = summarize_training(history, best_val_bpb)
    save_training_checkpoint(
        model,
        optimizer,
        step,
        final_val_bpb,
        best_val_bpb,
        tokens_seen,
        history,
        dataset_label,
    )
    render_markdown_report(
        report_path,
        model_tag=args.model_tag or "d4",
        summary={
            "qualified_to_evaluate": qualified_to_evaluate,
            "final_step": step,
            "tokens_seen": tokens_seen,
            "best_val_bpb": best_val_bpb,
            "final_val_bpb": final_val_bpb,
            "recommendation": recommendation,
        },
        history=history,
        dataset_label=dataset_label,
    )
    print0(f"SFT finished in {elapsed/60:.2f} minutes")
    print0(f"Metrics written to {metrics_path}")
    print0(f"Report written to {report_path}")
    compute_cleanup()


if __name__ == "__main__":
    train()
