"""Inspect Mini-MoE router usage on a synthetic batch."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from microchat.common import autodetect_device_type, get_base_dir
from microchat.model import GPT, GPTConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect Mini-MoE router usage")
    parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
    parser.add_argument("--n-layer", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--n-head", type=int, default=2, help="Number of attention heads")
    parser.add_argument("--n-kv-head", type=int, default=2, help="Number of KV heads")
    parser.add_argument("--n-embd", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--vocab-size", type=int, default=256, help="Synthetic vocabulary size")
    parser.add_argument("--seq-len", type=int, default=16, help="Synthetic sequence length")
    parser.add_argument("--batch-size", type=int, default=2, help="Synthetic batch size")
    parser.add_argument("--num-experts", type=int, default=4, help="Number of MoE experts")
    parser.add_argument("--moe-top-k", type=int, default=2, help="Routed experts per token")
    parser.add_argument("--moe-aux-loss-weight", type=float, default=0.01, help="MoE aux loss weight")
    parser.add_argument("--no-shared-expert", action="store_true", help="Disable the shared expert")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--report-dir", type=str, default=None, help="Directory for JSON, Markdown, and CSV reports")
    return parser.parse_args()


def validate_args(args):
    for field in ("n_layer", "n_head", "n_kv_head", "n_embd", "vocab_size", "seq_len", "batch_size"):
        if getattr(args, field) <= 0:
            raise ValueError(f"{field} must be positive")
    if args.n_embd % args.n_head != 0:
        raise ValueError(f"n_embd ({args.n_embd}) must be divisible by n_head ({args.n_head})")
    if args.n_head % args.n_kv_head != 0:
        raise ValueError(f"n_head ({args.n_head}) must be divisible by n_kv_head ({args.n_kv_head})")
    if args.moe_top_k > args.num_experts:
        raise ValueError("moe_top_k must be <= num_experts")


def get_experiment_metadata(device, model_config):
    cuda_available = torch.cuda.is_available()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "device_type": device.type,
        "cuda_available": cuda_available,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "model_config": model_config,
    }


def run_experiment(args):
    validate_args(args)
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    device = torch.device(device_type)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.set_float32_matmul_precision("high")

    config = GPTConfig(
        sequence_len=args.seq_len,
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        n_embd=args.n_embd,
        ffn_type="moe",
        num_experts=args.num_experts,
        moe_top_k=args.moe_top_k,
        moe_aux_loss_weight=args.moe_aux_loss_weight,
        use_shared_expert=not args.no_shared_expert,
    )
    model = GPT(config).to(device)
    model.init_weights()
    model.train()

    idx = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)
    targets = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)
    with torch.no_grad():
        loss = model(idx, targets)

    stats = model.get_moe_stats()
    return {
        "metadata": get_experiment_metadata(device, vars(config)),
        "run_config": {"seed": args.seed, "batch_size": args.batch_size, "seq_len": args.seq_len},
        "results": {
            "loss": float(loss.detach().cpu().item()),
            "ce_loss": float(model.latest_ce_loss.detach().cpu().item()),
            "moe_aux_loss": float(model.latest_moe_aux_loss.detach().cpu().item()),
            "total_assignments": stats["total_assignments"],
            "expected_assignments_per_layer": args.batch_size * args.seq_len * args.moe_top_k,
            "expected_total_assignments": args.n_layer * args.batch_size * args.seq_len * args.moe_top_k,
            "stats": stats,
        },
    }


def render_markdown(payload):
    results = payload["results"]
    stats = results["stats"]
    lines = [
        "# Mini-MoE Router Stats",
        "",
        "This report shows top-k routed expert usage for one synthetic batch.",
        "",
        "## Results",
        "",
        f"- total_assignments: `{results['total_assignments']}`",
        f"- expected_total_assignments: `{results['expected_total_assignments']}`",
        f"- moe_aux_loss: `{results['moe_aux_loss']:.6f}`",
        f"- ce_loss: `{results['ce_loss']:.6f}`",
        "",
        "## Layer Usage",
        "",
        "| layer | assignments | expert counts | expert fractions | aux loss |",
        "| ---: | ---: | --- | --- | ---: |",
    ]
    for layer in stats["layers"]:
        counts = ", ".join(str(value) for value in layer["expert_counts"])
        fractions = ", ".join(f"{value:.3f}" for value in layer["expert_fractions"])
        lines.append(
            f"| {layer['layer_idx']} | {layer['total_assignments']} | {counts} | "
            f"{fractions} | {layer['aux_loss']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def write_reports(payload, report_dir):
    os.makedirs(report_dir, exist_ok=True)
    json_path = os.path.join(report_dir, "moe_router_stats.json")
    md_path = os.path.join(report_dir, "moe_router_stats.md")
    csv_path = os.path.join(report_dir, "moe_router_stats.csv")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(render_markdown(payload))
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["layer_idx", "total_assignments", "expert_counts", "expert_fractions", "aux_loss"],
        )
        writer.writeheader()
        for layer in payload["results"]["stats"]["layers"]:
            writer.writerow(
                {
                    "layer_idx": layer["layer_idx"],
                    "total_assignments": layer["total_assignments"],
                    "expert_counts": json.dumps(layer["expert_counts"]),
                    "expert_fractions": json.dumps(layer["expert_fractions"]),
                    "aux_loss": layer["aux_loss"],
                }
            )
    return json_path, md_path, csv_path


def main():
    args = parse_args()
    payload = run_experiment(args)
    report_dir = args.report_dir or os.path.join(get_base_dir(), "reports", "experiments")
    json_path, md_path, csv_path = write_reports(payload, report_dir)
    results = payload["results"]
    print(f"assignments: {results['total_assignments']} / expected {results['expected_total_assignments']}")
    print(f"moe_aux_loss: {results['moe_aux_loss']:.6f}")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    print(f"CSV report: {csv_path}")


if __name__ == "__main__":
    main()
