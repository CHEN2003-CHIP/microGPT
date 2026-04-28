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
parser.add_argument("--num-iterations", type=int, default=800, help="Number of optimizer steps")
parser.add_argument("--max-train-tokens", type=int, default=0, help="Optional token budget cap; 0 disables")
parser.add_argument("--max-seq-len", type=int, default=512, help="Context length")
parser.add_argument("--device-batch-size", type=int, default=1, help="Per-device batch size")
parser.add_argument("--total-batch-size", type=int, default=512, help="Total tokens per optimizer step")
parser.add_argument("--learning-rate", type=float, default=3e-6, help="Peak AdamW learning rate")
parser.add_argument("--min-lr", type=float, default=3e-7, help="Minimum learning rate after cosine decay")
parser.add_argument("--warmup-iters", type=int, default=80, help="Linear warmup iterations")
parser.add_argument("--lr-decay-iters", type=int, default=0, help="Cosine decay length in optimizer steps; 0 uses total steps")
parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping max norm; <=0 disables")
parser.add_argument("--eval-every", type=int, default=100, help="Run validation every N steps")
parser.add_argument("--eval-tokens", type=int, default=2048, help="Validation token budget")
parser.add_argument("--save-every", type=int, default=100, help="Save a checkpoint every N steps; 0 disables periodic save")
parser.add_argument("--resume", action="store_true", help="Resume the latest SFT checkpoint for the selected model tag")
parser.add_argument("--pack-conversations", action="store_true", help="Pack multiple conversations into one training row")
parser.add_argument(
    "--sft-optimizer",
    type=str,
    default="adamw_full",
    choices=["adamw_full", "adamw_behavior_low_lr"],
    help="SFT optimizer grouping strategy",
)
parser.add_argument(
    "--behavior-lr-scale",
    type=float,
    default=0.2,
    help="LR multiplier for experimental behavior parameters in adamw_behavior_low_lr mode",
)
parser.add_argument("--smoltalk-limit", type=int, default=0, help="Limit SmolTalk rows; 0 disables SmolTalk")
parser.add_argument("--ultrachat-limit", type=int, default=3_000, help="Limit UltraChat 200k rows; 0 disables UltraChat")
parser.add_argument("--local-repeat", type=int, default=24, help="Oversample the bundled local phase-2 SFT mix")
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


def resolve_local_sft_specs(base_dir):
    scripts_dir = os.path.dirname(__file__)
    preferred = os.path.join(base_dir, "identity_conversations.jsonl")
    anchor_en = os.path.join(scripts_dir, "identity_conversations.anchor_en.jsonl")
    phase2_mix = os.path.join(scripts_dir, "chat_phase2_mix_en.jsonl")
    sample = preferred if os.path.exists(preferred) else os.path.join(scripts_dir, "identity_conversations.sample.jsonl")

    specs = []
    if os.path.exists(anchor_en):
        specs.append({"name": "anchor_en", "path": anchor_en, "weight": 1})
    if os.path.exists(phase2_mix):
        specs.append({"name": "phase2_mix_en", "path": phase2_mix, "weight": 3})
    if os.path.exists(sample):
        sample_name = "sample" if sample.endswith("identity_conversations.sample.jsonl") else "local_override"
        specs.append({"name": sample_name, "path": sample, "weight": 1})
    if not specs:
        fallback = resolve_identity_dataset(base_dir)
        specs.append({"name": "fallback", "path": fallback, "weight": 1})
    return specs


def build_local_sft_tasks(base_dir, repeat_count):
    repeat_count = max(repeat_count, 1)
    train_tasks = []
    val_tasks = []
    labels = []
    for spec in resolve_local_sft_specs(base_dir):
        val_tasks.append(CustomJSON(filepath=spec["path"]))
        copies = repeat_count * spec["weight"]
        train_tasks.extend(CustomJSON(filepath=spec["path"]) for _ in range(copies))
        labels.append(f"{spec['name']} x{copies}")
    return train_tasks, val_tasks, ", ".join(labels)


def build_sft_datasets(base_dir):
    """
    Prefer a phase-2 local mix of anchor/style/general-answer datasets, optionally
    blended with a light amount of remote chat data.
    """
    local_train_tasks, local_val_tasks, local_label = build_local_sft_tasks(base_dir, args.local_repeat)

    if args.smoltalk_limit == 0 and args.ultrachat_limit == 0:
        train_dataset = TaskMixture(local_train_tasks)
        val_dataset = TaskMixture(local_val_tasks)
        dataset_label = f"CustomJSON local mix ({local_label})"
        return train_dataset, val_dataset, dataset_label

    try:
        train_tasks = []
        val_tasks = list(local_val_tasks)
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
            train_dataset = TaskMixture(local_train_tasks)
            val_dataset = TaskMixture(local_val_tasks)
            dataset_label = f"CustomJSON local mix ({local_label})"
            return train_dataset, val_dataset, dataset_label

        train_dataset = TaskMixture([*train_tasks, *local_train_tasks])
        val_dataset = TaskMixture(val_tasks)
        dataset_label = " + ".join([*labels, f"LocalMix({local_label})"])
    except Exception as exc:
        print0(f"Remote SFT datasets unavailable on this machine, falling back to bundled local conversations: {exc!r}")
        offline_repeat = max(args.local_repeat, 3)
        local_train_tasks, local_val_tasks, local_label = build_local_sft_tasks(base_dir, offline_repeat)
        train_dataset = TaskMixture(local_train_tasks)
        val_dataset = TaskMixture(local_val_tasks)
        dataset_label = f"CustomJSON local mix (offline fallback: {local_label})"

    return train_dataset, val_dataset, dataset_label


def build_sft_loader(tokenizer, dataset, batch_size, seq_len, device, buffer_size=128, pack_conversations=False):
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
            if pack_conversations:
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
                    break
            else:
                token_ids, loss_mask = buffer.pop(0)
                row.extend(token_ids)
                mask.extend(loss_mask)

            content_length = len(row)
            if content_length < row_capacity:
                row.extend([bos_token] * (row_capacity - content_length))
                mask.extend([0] * (row_capacity - content_length))
            lengths.append(content_length)
            rows.append(row[:row_capacity])
            masks.append(mask[:row_capacity])

        batch = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
        inputs = batch[:, :-1].to(device=device, dtype=torch.int64, non_blocking=use_cuda).contiguous()
        targets = batch[:, 1:].to(device=device, dtype=torch.int64, non_blocking=use_cuda).contiguous()
        mask_tensor = torch.tensor(masks, dtype=torch.int8)
        # The renderer decides which tokens are supervised. After shifting we
        # simply ignore every target position that does not belong to the
        # assistant side of the conversation.
        target_mask = mask_tensor[:, 1:].to(device=device)
        targets[target_mask == 0] = -1
        for row_index, content_length in enumerate(lengths):
            if content_length <= 0:
                targets[row_index, :] = -1
            elif content_length < row_capacity:
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
        param_group["lr"] = lr * param_group.get("lr_scale", 1.0)


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
        for key in ("step", "lr", "train_loss_ema", "val_bpb", "tokens_seen", "supervised_tokens", "tok_per_sec"):
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
    #选择运行设备
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _, _, _, _, device = compute_init(device_type)
    #load model and tokenizer
    model, tokenizer, metadata = load_model("base", device, phase="train", model_tag=args.model_tag, step=args.model_step)
    #获取token字节数以便正确计算bpb
    token_bytes = get_token_bytes(device=device)
    #设置优化器
    optimizer = model.setup_sft_optimizer(
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer_type=args.sft_optimizer,
        behavior_lr_scale=args.behavior_lr_scale,
    )
    #获取sft目录
    checkpoint_dir = get_sft_checkpoint_dir()
    #maybe resume from checkpoint
    start_step, best_val_bpb, tokens_seen, history = maybe_resume(model, optimizer, checkpoint_dir, device)

    #获取base模型的训练数据集和验证数据集
    base_dir = get_base_dir()
    train_dataset, val_dataset, dataset_label = build_sft_datasets(base_dir)
    train_loader = build_sft_loader(
        tokenizer,
        train_dataset,
        args.device_batch_size,
        args.max_seq_len,
        device,
        pack_conversations=args.pack_conversations,
    )
    val_loader = lambda: build_sft_loader(
        tokenizer,
        val_dataset,
        args.device_batch_size,
        args.max_seq_len,
        device,
        pack_conversations=args.pack_conversations,
    )
    #一次forward和backward的token数量
    tokens_per_micro_step = args.device_batch_size * args.max_seq_len

    assert args.total_batch_size % tokens_per_micro_step == 0, "total_batch_size must divide into micro-steps"
    #计算梯度累积步数以达到目标总批量大小
    grad_accum_steps = args.total_batch_size // tokens_per_micro_step
    #每grad_accum_steps个step更新一次模型参数,更新了多少token
    tokens_per_step = grad_accum_steps * tokens_per_micro_step
    #计算训练总步数
    target_steps = estimate_total_steps(tokens_per_step)
    #最后的训练步数取用户指定的迭代次数和token预算计算出的步数中的较大者，确保至少训练用户指定的迭代次数
    final_step = max(args.num_iterations, start_step)
    if args.max_train_tokens > 0:
        final_step = max(final_step, target_steps)

    print0(f"Loaded base checkpoint step {metadata['step']}")
    print0(f"Training mixture: {len(train_dataset):,} conversations ({dataset_label})")
    print0(f"SFT optimizer: {args.sft_optimizer} (behavior_lr_scale={args.behavior_lr_scale})")
    print0(f"Conversation packing: {'ON' if args.pack_conversations else 'OFF'}")
    print0(f"Gradient accumulation steps: {grad_accum_steps}")
    print0(f"Tokens per optimizer step: {tokens_per_step}")
    print0(f"Target training steps: {final_step}")


    current_inputs, current_targets = next(train_loader)
    ema_loss = 0.0
    ema_beta = 0.9
    ema_count = 0
    started = time.time()
    metrics_path, report_path = get_report_paths()

    if start_step == 0 and os.path.exists(metrics_path):
        os.remove(metrics_path)

    #训练主循环
    step = start_step
    while step < final_step:
        if args.max_train_tokens > 0 and tokens_seen >= args.max_train_tokens:
            break
        step += 1
        model.train()
        step_loss = 0.0
        supervised_tokens_this_step = 0
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
            step_loss += loss.item() / grad_accum_steps
            (loss / grad_accum_steps).backward()
            supervised_tokens_this_step += int((current_targets != -1).sum().item())
            current_inputs, current_targets = next(train_loader)

        grad_norm = None
        if args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        ema_count += 1
        ema_loss = ema_beta * ema_loss + (1 - ema_beta) * step_loss
        smooth_loss = ema_loss / (1 - ema_beta ** ema_count)
        tokens_seen += tokens_per_step
        step_elapsed = max(time.time() - step_started, 1e-6)
        tok_per_sec = tokens_per_step / step_elapsed

        metric = {
            "step": step,
            "lr": round(lr, 10),
            "train_loss_ema": round(float(smooth_loss), 6),
            "tokens_seen": int(tokens_seen),
            "supervised_tokens": supervised_tokens_this_step,
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
            checkpoint_val_bpb = metric["val_bpb"] if metric["val_bpb"] is not None else best_val_bpb
            save_training_checkpoint(
                model,
                optimizer,
                step,
                checkpoint_val_bpb,
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
