"""Estimate KV cache memory savings from grouped-query attention."""

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

from microchat.common import get_base_dir


DTYPE_BYTES = {
    "float32": 4,
    "bfloat16": 2,
    "float16": 2,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate GQA KV cache memory usage")
    parser.add_argument("--n-layer", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--n-head", type=int, default=4, help="Number of query heads")
    parser.add_argument("--n-embd", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--n-kv-heads", type=int, nargs="*", default=None, help="KV head counts to compare")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[256, 512, 1024], help="Sequence lengths")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1], help="Batch sizes")
    parser.add_argument("--dtype", choices=sorted(DTYPE_BYTES), default="bfloat16", help="KV cache dtype")
    parser.add_argument("--report-dir", type=str, default=None, help="Directory for JSON and Markdown reports")
    return parser.parse_args()


def get_experiment_metadata(dtype, model_config):
    cuda_available = torch.cuda.is_available()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "device_type": "cuda" if cuda_available else "cpu",
        "cuda_available": cuda_available,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "dtype": dtype,
        "model_config": model_config,
    }


def default_kv_heads(n_head):
    candidates = [n_head, max(1, n_head // 2), max(1, n_head // 4), 1]
    return sorted(set(candidates), reverse=True)


def validate_config(n_layer, n_head, n_embd, n_kv_heads, seq_lens, batch_sizes):
    if n_layer <= 0 or n_head <= 0 or n_embd <= 0:
        raise ValueError("n_layer, n_head, and n_embd must be positive")
    if n_embd % n_head != 0:
        raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")
    for n_kv_head in n_kv_heads:
        if n_kv_head <= 0:
            raise ValueError("n_kv_head values must be positive")
        if n_head % n_kv_head != 0:
            raise ValueError(f"n_head ({n_head}) must be divisible by n_kv_head ({n_kv_head})")
    if any(seq_len <= 0 for seq_len in seq_lens):
        raise ValueError("seq_lens must be positive")
    if any(batch_size <= 0 for batch_size in batch_sizes):
        raise ValueError("batch_sizes must be positive")


def estimate_cache_bytes(n_layer, batch_size, seq_len, n_kv_head, head_dim, dtype_bytes):
    return n_layer * batch_size * seq_len * n_kv_head * head_dim * 2 * dtype_bytes


def bytes_to_mib(num_bytes):
    return num_bytes / (1024 ** 2)


def build_rows(args):
    n_kv_heads = args.n_kv_heads or default_kv_heads(args.n_head)
    validate_config(args.n_layer, args.n_head, args.n_embd, n_kv_heads, args.seq_lens, args.batch_sizes)

    head_dim = args.n_embd // args.n_head
    dtype_bytes = DTYPE_BYTES[args.dtype]
    rows = []
    for batch_size in args.batch_sizes:
        for seq_len in args.seq_lens:
            baseline_bytes = estimate_cache_bytes(
                args.n_layer,
                batch_size,
                seq_len,
                args.n_head,
                head_dim,
                dtype_bytes,
            )
            for n_kv_head in n_kv_heads:
                cache_bytes = estimate_cache_bytes(
                    args.n_layer,
                    batch_size,
                    seq_len,
                    n_kv_head,
                    head_dim,
                    dtype_bytes,
                )
                rows.append(
                    {
                        "batch_size": batch_size,
                        "seq_len": seq_len,
                        "n_head": args.n_head,
                        "n_kv_head": n_kv_head,
                        "head_dim": head_dim,
                        "cache_bytes": cache_bytes,
                        "cache_mib": bytes_to_mib(cache_bytes),
                        "memory_ratio_vs_mha": cache_bytes / baseline_bytes,
                        "memory_saving_percent": 100 * (1 - cache_bytes / baseline_bytes),
                    }
                )
    return rows


def render_console_table(rows):
    headers = ["batch", "seq", "q_heads", "kv_heads", "MiB", "vs_mha", "saved"]
    lines = [" | ".join(headers), " | ".join("-" * len(header) for header in headers)]
    for row in rows:
        lines.append(
            " | ".join(
                [
                    str(row["batch_size"]),
                    str(row["seq_len"]),
                    str(row["n_head"]),
                    str(row["n_kv_head"]),
                    f"{row['cache_mib']:.3f}",
                    f"{row['memory_ratio_vs_mha']:.3f}x",
                    f"{row['memory_saving_percent']:.1f}%",
                ]
            )
        )
    return "\n".join(lines)


def render_markdown(payload):
    lines = [
        "# GQA KV Cache Memory Estimate",
        "",
        "This report is a theoretical memory estimate, not a measured runtime benchmark.",
        "",
        "It estimates KV cache memory from the cache shape used by `microchat.engine.KVCache`:",
        "",
        "`layers * batch * seq_len * n_kv_head * head_dim * 2(K,V) * dtype_bytes`",
        "",
        "## Metadata",
        "",
    ]
    for key, value in payload["metadata"].items():
        if key == "model_config":
            continue
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Model Config", ""])
    for key, value in payload["metadata"]["model_config"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Estimated Results",
            "",
            "| batch | seq | q_heads | kv_heads | cache MiB | vs MHA | saved |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["rows"]:
        lines.append(
            f"| {row['batch_size']} | {row['seq_len']} | {row['n_head']} | "
            f"{row['n_kv_head']} | {row['cache_mib']:.3f} | "
            f"{row['memory_ratio_vs_mha']:.3f}x | {row['memory_saving_percent']:.1f}% |"
        )
    return "\n".join(lines) + "\n"


def write_reports(payload, report_dir):
    os.makedirs(report_dir, exist_ok=True)
    json_path = os.path.join(report_dir, "gqa_cache_memory.json")
    md_path = os.path.join(report_dir, "gqa_cache_memory.md")
    csv_path = os.path.join(report_dir, "gqa_cache_memory.csv")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(render_markdown(payload))
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "batch_size",
            "seq_len",
            "n_head",
            "n_kv_head",
            "head_dim",
            "cache_bytes",
            "cache_mib",
            "memory_ratio_vs_mha",
            "memory_saving_percent",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(payload["rows"])
    return json_path, md_path, csv_path


def main():
    args = parse_args()
    rows = build_rows(args)
    model_config = {
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "n_embd": args.n_embd,
        "n_kv_heads": args.n_kv_heads or default_kv_heads(args.n_head),
        "seq_lens": args.seq_lens,
        "batch_sizes": args.batch_sizes,
        "dtype_bytes": DTYPE_BYTES[args.dtype],
    }
    payload = {
        "metadata": get_experiment_metadata(args.dtype, model_config),
        "rows": rows,
    }
    report_dir = args.report_dir or os.path.join(get_base_dir(), "reports", "experiments")
    json_path, md_path, csv_path = write_reports(payload, report_dir)
    print(render_console_table(rows))
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    print(f"CSV report: {csv_path}")


if __name__ == "__main__":
    main()
