"""Supervised fine-tuning for the microGPT teaching project."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import os
import time

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


parser = argparse.ArgumentParser(description="Run supervised fine-tuning")
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--model-tag", type=str, default=None, help="Base model tag to load")
parser.add_argument("--model-step", type=int, default=None, help="Checkpoint step to load")
parser.add_argument("--num-iterations", type=int, default=20, help="Number of optimizer steps")
parser.add_argument("--max-seq-len", type=int, default=256, help="Context length")
parser.add_argument("--device-batch-size", type=int, default=1, help="Per-device batch size")
parser.add_argument("--total-batch-size", type=int, default=512, help="Total tokens per optimizer step")
parser.add_argument("--learning-rate", type=float, default=1e-4, help="AdamW learning rate")
parser.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay")
parser.add_argument("--eval-every", type=int, default=10, help="Run validation every N steps")
parser.add_argument("--eval-tokens", type=int, default=1024, help="Validation token budget")
parser.add_argument("--smoltalk-limit", type=int, default=20_000, help="Limit SmolTalk rows for faster local SFT; set 0 for local-only SFT")
parser.add_argument("--local-repeat", type=int, default=50, help="Oversample local identity/instruction examples")
args = parser.parse_args()

import torch

from microchat.checkpoint_manager import load_model, save_checkpoint
from microchat.common import autodetect_device_type, compute_cleanup, compute_init, get_base_dir, print0
from microchat.loss_eval import evaluate_bpb
from microchat.tokenizer import get_token_bytes
from tasks.common import TaskMixture
from tasks.customjson import CustomJSON
from tasks.smoltalk import SmolTalk


def resolve_identity_dataset(base_dir):
    preferred = os.path.join(base_dir, "identity_conversations.jsonl")
    bundled = os.path.join(os.path.dirname(__file__), "identity_conversations.sample.jsonl")
    return preferred if os.path.exists(preferred) else bundled


def build_sft_datasets(base_dir):
    """
    Prefer SmolTalk + CustomJSON for the teaching flow.
    If the remote SmolTalk dataset cannot be downloaded on the local machine,
    fall back to the bundled JSONL conversations so SFT still runs end-to-end.
    """
    identity_path = resolve_identity_dataset(base_dir)
    local_task = CustomJSON(filepath=identity_path)
    local_tasks = [CustomJSON(filepath=identity_path) for _ in range(args.local_repeat)]

    if args.smoltalk_limit == 0:
        train_dataset = TaskMixture(local_tasks)
        val_dataset = TaskMixture([local_task])
        dataset_label = f"CustomJSON only (local accuracy mode, x{args.local_repeat})"
        return train_dataset, val_dataset, dataset_label

    smoltalk_kwargs = {}
    smoltalk_kwargs["stop"] = args.smoltalk_limit

    try:
        train_dataset = TaskMixture(
            [
                SmolTalk(split="train", **smoltalk_kwargs),
                *local_tasks,
            ]
        )
        val_dataset = TaskMixture(
            [
                SmolTalk(split="test", stop=min(args.smoltalk_limit, 2_000) if args.smoltalk_limit > 0 else None),
                CustomJSON(filepath=identity_path),
            ]
        )
        dataset_label = f"SmolTalk(limit={args.smoltalk_limit}) + CustomJSON(x{args.local_repeat})"
    except Exception as exc:
        print0(f"SmolTalk unavailable on this machine, falling back to bundled local conversations: {exc}")
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
            # Skip rows that contain no assistant tokens after shifting to next-token targets.
            # Otherwise cross-entropy would see only ignore_index values and return NaN.
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


def save_final_checkpoint(model, step, validation_bpb):
    base_dir = get_base_dir()
    model_tag = args.model_tag or "d4"
    checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", model_tag)
    save_checkpoint(
        checkpoint_dir,
        step,
        model.state_dict(),
        {
            "step": step,
            "val_bpb": validation_bpb,
            "model_config": asdict(model.config),
            "training_config": vars(args),
        },
    )
    print0(f"Saved checkpoint to {checkpoint_dir}")


def train():
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _, _, _, _, device = compute_init(device_type)
    model, tokenizer, metadata = load_model("base", device, phase="train", model_tag=args.model_tag, step=args.model_step)
    token_bytes = get_token_bytes(device=device)
    optimizer = model.setup_optimizer(lr=args.learning_rate, weight_decay=args.weight_decay)

    base_dir = get_base_dir()
    train_dataset, val_dataset, dataset_label = build_sft_datasets(base_dir)

    train_loader = build_sft_loader(tokenizer, train_dataset, args.device_batch_size, args.max_seq_len, device)
    val_loader = lambda: build_sft_loader(tokenizer, val_dataset, args.device_batch_size, args.max_seq_len, device)

    tokens_per_micro_step = args.device_batch_size * args.max_seq_len
    assert args.total_batch_size % tokens_per_micro_step == 0, "total_batch_size must divide into micro-steps"
    grad_accum_steps = args.total_batch_size // tokens_per_micro_step
    print0(f"Loaded base checkpoint step {metadata['step']}")
    print0(f"Training mixture: {len(train_dataset):,} conversations ({dataset_label})")
    print0(f"Gradient accumulation steps: {grad_accum_steps}")

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
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Encountered a non-finite SFT loss. "
                    "This usually means the batch had no supervised assistant targets "
                    "or the optimization became numerically unstable."
                )
            step_loss += loss.item()
            (loss / grad_accum_steps).backward()
            current_inputs, current_targets = next(train_loader)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    save_final_checkpoint(model, args.num_iterations, final_val_bpb)
    print0(f"SFT finished in {elapsed/60:.2f} minutes")
    compute_cleanup()


if __name__ == "__main__":
    train()
