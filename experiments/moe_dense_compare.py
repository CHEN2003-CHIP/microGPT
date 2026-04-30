"""Compare dense FFN and Mini-MoE FFN on synthetic random batches."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from microchat.common import autodetect_device_type, get_base_dir
from microchat.model import GPT, GPTConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Compare dense and Mini-MoE FFNs on synthetic data")
    parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
    parser.add_argument("--batch-size", type=int, default=2, help="Synthetic batch size")
    parser.add_argument("--seq-len", type=int, default=16, help="Synthetic sequence length")
    parser.add_argument("--steps", type=int, default=3, help="Measured forward passes")
    parser.add_argument("--n-layer", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--n-head", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n-kv-head", type=int, default=2, help="Number of KV heads")
    parser.add_argument("--n-embd", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--vocab-size", type=int, default=512, help="Synthetic vocabulary size")
    parser.add_argument("--num-experts", type=int, default=4, help="Number of MoE experts")
    parser.add_argument("--moe-top-k", type=int, default=2, help="Routed MoE experts per token")
    parser.add_argument("--moe-aux-loss-weight", type=float, default=0.01, help="MoE aux loss weight")
    parser.add_argument("--no-shared-expert", action="store_true", help="Disable the shared MoE expert")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--report-dir", type=str, default=None, help="Directory for JSON, Markdown, and CSV reports")
    return parser.parse_args()


def validate_args(args):
    for field in (
        "batch_size",
        "seq_len",
        "steps",
        "n_layer",
        "n_head",
        "n_kv_head",
        "n_embd",
        "vocab_size",
        "num_experts",
        "moe_top_k",
    ):
        if getattr(args, field) <= 0:
            raise ValueError(f"{field} must be positive")
    if args.n_embd % args.n_head != 0:
        raise ValueError(f"n_embd ({args.n_embd}) must be divisible by n_head ({args.n_head})")
    if args.n_head % args.n_kv_head != 0:
        raise ValueError(f"n_head ({args.n_head}) must be divisible by n_kv_head ({args.n_kv_head})")
    if args.moe_top_k > args.num_experts:
        raise ValueError("moe_top_k must be <= num_experts")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def count_params(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def make_config(args, ffn_type):
    return GPTConfig(
        sequence_len=args.seq_len,
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        n_embd=args.n_embd,
        ffn_type=ffn_type,
        num_experts=args.num_experts,
        moe_top_k=args.moe_top_k,
        moe_aux_loss_weight=args.moe_aux_loss_weight,
        use_shared_expert=not args.no_shared_expert,
    )


def make_batches(args, device):
    return [
        (
            torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device),
            torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device),
        )
        for _ in range(args.steps)
    ]


def average_layer_fractions(stats, num_experts):
    if not stats["layers"]:
        return []
    totals = [0.0 for _ in range(num_experts)]
    for layer in stats["layers"]:
        for expert_idx, fraction in enumerate(layer["expert_fractions"]):
            totals[expert_idx] += fraction
    return [value / len(stats["layers"]) for value in totals]


def measure_model(label, model, batches, args, device):
    total_params, trainable_params = count_params(model)
    tokens = args.batch_size * args.seq_len * args.steps
    model.train()

    losses = []
    ce_losses = []
    moe_aux_losses = []
    total_losses = []
    synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for idx, targets in batches:
            loss = model(idx, targets)
            losses.append(float(loss.detach().cpu().item()))
            latest_ce = model.latest_ce_loss if model.latest_ce_loss is not None else loss
            latest_aux = model.latest_moe_aux_loss
            latest_total = model.latest_total_loss if model.latest_total_loss is not None else loss
            ce_losses.append(float(latest_ce.detach().cpu().item()))
            moe_aux_losses.append(float(latest_aux.detach().cpu().item()) if latest_aux is not None else 0.0)
            total_losses.append(float(latest_total.detach().cpu().item()))
    synchronize(device)
    elapsed = max(time.perf_counter() - started, 1e-9)

    stats = model.get_moe_stats() if model.config.ffn_type == "moe" else {}
    expert_usage_ratios = average_layer_fractions(stats, args.num_experts) if stats else []
    active_ratio = 1.0
    if model.config.ffn_type == "moe":
        active_experts = args.moe_top_k + (1 if model.config.use_shared_expert else 0)
        total_experts = args.num_experts + (1 if model.config.use_shared_expert else 0)
        active_ratio = active_experts / total_experts

    return {
        "mode": label,
        "ffn_type": model.config.ffn_type,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "estimated_active_expert_ratio": float(active_ratio),
        "forward_time_sec": float(elapsed),
        "tokens_per_sec": float(tokens / elapsed),
        "ce_loss": float(sum(ce_losses) / len(ce_losses)),
        "moe_aux_loss": float(sum(moe_aux_losses) / len(moe_aux_losses)),
        "total_loss": float(sum(total_losses) / len(total_losses)),
        "expert_usage_ratios": expert_usage_ratios,
        "moe_stats": stats,
    }


def get_metadata(device, args):
    cuda_available = torch.cuda.is_available()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "device_type": device.type,
        "cuda_available": cuda_available,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "run_config": vars(args),
    }


def run_experiment(args):
    validate_args(args)
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    device = torch.device(device_type)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.set_float32_matmul_precision("high")

    batches = make_batches(args, device)
    dense = GPT(make_config(args, "dense")).to(device)
    moe = GPT(make_config(args, "moe")).to(device)
    dense.init_weights()
    moe.init_weights()

    rows = [
        measure_model("dense", dense, batches, args, device),
        measure_model("moe", moe, batches, args, device),
    ]
    return {"metadata": get_metadata(device, args), "rows": rows}


def render_markdown(payload):
    lines = [
        "# Dense vs Mini-MoE Compare",
        "",
        "This synthetic report compares dense FFN and Mini-MoE FFN on random input and target batches.",
        "",
        "| mode | params | trainable | active expert ratio | sec | tokens/sec | CE loss | MoE aux | total loss |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['mode']} | {row['total_params']} | {row['trainable_params']} | "
            f"{row['estimated_active_expert_ratio']:.3f} | {row['forward_time_sec']:.6f} | "
            f"{row['tokens_per_sec']:.2f} | {row['ce_loss']:.6f} | "
            f"{row['moe_aux_loss']:.6f} | {row['total_loss']:.6f} |"
        )
    lines.extend(["", "## Expert Usage", ""])
    for row in payload["rows"]:
        usage = ", ".join(f"{value:.3f}" for value in row["expert_usage_ratios"]) or "n/a"
        lines.append(f"- {row['mode']}: `{usage}`")
    return "\n".join(lines) + "\n"


def write_reports(payload, report_dir):
    os.makedirs(report_dir, exist_ok=True)
    json_path = os.path.join(report_dir, "moe_dense_compare.json")
    csv_path = os.path.join(report_dir, "moe_dense_compare.csv")
    md_path = os.path.join(report_dir, "moe_dense_compare.md")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "mode",
            "total_params",
            "trainable_params",
            "estimated_active_expert_ratio",
            "forward_time_sec",
            "tokens_per_sec",
            "ce_loss",
            "moe_aux_loss",
            "total_loss",
            "expert_usage_ratios",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload["rows"]:
            csv_row = {key: row[key] for key in fieldnames[:-1]}
            csv_row["expert_usage_ratios"] = json.dumps(row["expert_usage_ratios"])
            writer.writerow(csv_row)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(render_markdown(payload))
    return json_path, csv_path, md_path


def main():
    args = parse_args()
    payload = run_experiment(args)
    report_dir = args.report_dir or os.path.join(get_base_dir(), "reports", "experiments")
    json_path, csv_path, md_path = write_reports(payload, report_dir)
    for row in payload["rows"]:
        print(
            f"{row['mode']}: {row['tokens_per_sec']:.2f} tok/s | "
            f"params {row['total_params']} | loss {row['total_loss']:.4f}"
        )
    print(f"JSON report: {json_path}")
    print(f"CSV report: {csv_path}")
    print(f"Markdown report: {md_path}")


if __name__ == "__main__":
    main()
