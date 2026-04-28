"""Benchmark autoregressive generation with and without KV cache."""

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

from microchat.common import COMPUTE_DTYPE, autodetect_device_type, get_base_dir
from microchat.engine import KVCache
from microchat.model import GPT, GPTConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark GPT generation with and without KV cache")
    parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
    parser.add_argument("--n-layer", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--n-head", type=int, default=4, help="Number of query heads")
    parser.add_argument("--n-kv-head", type=int, default=2, help="Number of KV heads")
    parser.add_argument("--n-embd", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--vocab-size", type=int, default=1024, help="Synthetic vocabulary size")
    parser.add_argument("--max-seq-len", type=int, default=64, help="Maximum model/cache sequence length")
    parser.add_argument("--prompt-len", type=int, default=32, help="Synthetic prompt length")
    parser.add_argument("--new-tokens", type=int, default=32, help="Number of generated tokens to benchmark")
    parser.add_argument("--batch-size", type=int, default=1, help="Synthetic batch size")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Warmup runs before measurement")
    parser.add_argument("--runs", type=int, default=3, help="Measured runs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--report-dir", type=str, default=None, help="Directory for JSON and Markdown reports")
    return parser.parse_args()


def validate_args(args):
    positive_fields = (
        "n_layer",
        "n_head",
        "n_kv_head",
        "n_embd",
        "vocab_size",
        "max_seq_len",
        "prompt_len",
        "new_tokens",
        "batch_size",
        "runs",
    )
    for field in positive_fields:
        if getattr(args, field) <= 0:
            raise ValueError(f"{field} must be positive")
    if args.warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative")
    if args.n_embd % args.n_head != 0:
        raise ValueError(f"n_embd ({args.n_embd}) must be divisible by n_head ({args.n_head})")
    if args.n_head % args.n_kv_head != 0:
        raise ValueError(f"n_head ({args.n_head}) must be divisible by n_kv_head ({args.n_kv_head})")
    if args.prompt_len + args.new_tokens > args.max_seq_len:
        raise ValueError(
            f"prompt_len + new_tokens ({args.prompt_len + args.new_tokens}) "
            f"must be <= max_seq_len ({args.max_seq_len})"
        )


def get_experiment_metadata(device, dtype, model_config):
    cuda_available = torch.cuda.is_available()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "device_type": device.type,
        "cuda_available": cuda_available,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "dtype": str(dtype).replace("torch.", ""),
        "model_config": model_config,
    }


def build_model(args, device):
    config = GPTConfig(
        sequence_len=args.max_seq_len,
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        n_embd=args.n_embd,
    )
    model = GPT(config).to(device)
    model.init_weights()
    model.eval()
    return model


def make_prompt(args, device):
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    return torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.prompt_len),
        dtype=torch.long,
        device=device,
        generator=generator,
    )


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def generate_without_cache(model, prompt, new_tokens):
    ids = prompt
    for _ in range(new_tokens):
        logits = model(ids)
        next_ids = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        ids = torch.cat([ids, next_ids], dim=1)
    return ids


@torch.inference_mode()
def generate_with_cache(model, prompt, new_tokens):
    config = model.config
    device = model.get_device()
    dtype = COMPUTE_DTYPE if device.type == "cuda" else torch.float32
    cache_len = prompt.size(1) + new_tokens
    if cache_len > config.sequence_len:
        raise ValueError(f"cache length {cache_len} exceeds config.sequence_len {config.sequence_len}")

    kv_cache = KVCache(
        batch_size=prompt.size(0),
        num_heads=config.n_kv_head,
        seq_len=cache_len,
        head_dim=config.n_embd // config.n_head,
        num_layers=config.n_layer,
        device=device,
        dtype=dtype,
    )
    logits = model(prompt, kv_cache=kv_cache)
    generated = []
    for token_index in range(new_tokens):
        next_ids = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(next_ids)
        if token_index < new_tokens - 1:
            logits = model(next_ids, kv_cache=kv_cache)
    return torch.cat([prompt, *generated], dim=1)


def time_generation(fn, model, prompt, new_tokens, device):
    synchronize(device)
    started = time.perf_counter()
    output = fn(model, prompt, new_tokens)
    synchronize(device)
    return time.perf_counter() - started, output


def run_benchmark(args):
    validate_args(args)
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    device = torch.device(device_type)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.set_float32_matmul_precision("high")

    model = build_model(args, device)
    prompt = make_prompt(args, device)

    for _ in range(args.warmup_runs):
        time_generation(generate_without_cache, model, prompt, args.new_tokens, device)
        time_generation(generate_with_cache, model, prompt, args.new_tokens, device)

    no_cache_times = []
    kv_cache_times = []
    for _ in range(args.runs):
        elapsed, _ = time_generation(generate_without_cache, model, prompt, args.new_tokens, device)
        no_cache_times.append(elapsed)
        elapsed, _ = time_generation(generate_with_cache, model, prompt, args.new_tokens, device)
        kv_cache_times.append(elapsed)

    total_tokens = args.batch_size * args.new_tokens
    no_cache_avg = sum(no_cache_times) / len(no_cache_times)
    kv_cache_avg = sum(kv_cache_times) / len(kv_cache_times)
    model_config = {
        "sequence_len": args.max_seq_len,
        "vocab_size": args.vocab_size,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "n_kv_head": args.n_kv_head,
        "n_embd": args.n_embd,
        "prompt_len": args.prompt_len,
        "new_tokens": args.new_tokens,
        "batch_size": args.batch_size,
    }
    return {
        "metadata": get_experiment_metadata(device, COMPUTE_DTYPE if device.type == "cuda" else torch.float32, model_config),
        "run_config": {
            "warmup_runs": args.warmup_runs,
            "runs": args.runs,
            "seed": args.seed,
        },
        "results": {
            "generated_tokens_per_run": total_tokens,
            "no_cache_times_sec": no_cache_times,
            "kv_cache_times_sec": kv_cache_times,
            "no_cache_avg_sec": no_cache_avg,
            "kv_cache_avg_sec": kv_cache_avg,
            "no_cache_tokens_per_sec": total_tokens / no_cache_avg,
            "kv_cache_tokens_per_sec": total_tokens / kv_cache_avg,
            "no_cache_ms_per_token": 1000 * no_cache_avg / total_tokens,
            "kv_cache_ms_per_token": 1000 * kv_cache_avg / total_tokens,
            "speedup": no_cache_avg / kv_cache_avg,
        },
    }


def render_markdown(payload):
    metadata = payload["metadata"]
    results = payload["results"]
    lines = [
        "# KV Cache Generation Benchmark",
        "",
        "This report contains measured runtime results for this exact benchmark run.",
        "",
        "It compares full-prefix autoregressive generation against one-token decoding with the existing project `KVCache`.",
        "",
        "KV cache is not guaranteed to be faster for every configuration. On tiny CPU runs, cache setup and one-token overhead may dominate.",
        "",
        "## Metadata",
        "",
    ]
    for key, value in metadata.items():
        if key == "model_config":
            continue
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Model Config", ""])
    for key, value in metadata["model_config"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Run Config", ""])
    for key, value in payload["run_config"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Measured Results",
            "",
            "| mode | avg sec | tokens/sec | ms/token |",
            "| --- | ---: | ---: | ---: |",
            f"| no cache | {results['no_cache_avg_sec']:.6f} | "
            f"{results['no_cache_tokens_per_sec']:.2f} | {results['no_cache_ms_per_token']:.3f} |",
            f"| kv cache | {results['kv_cache_avg_sec']:.6f} | "
            f"{results['kv_cache_tokens_per_sec']:.2f} | {results['kv_cache_ms_per_token']:.3f} |",
            "",
            f"Speedup: `{results['speedup']:.3f}x`",
        ]
    )
    return "\n".join(lines) + "\n"


def write_reports(payload, report_dir):
    os.makedirs(report_dir, exist_ok=True)
    json_path = os.path.join(report_dir, "kv_cache_generation.json")
    md_path = os.path.join(report_dir, "kv_cache_generation.md")
    csv_path = os.path.join(report_dir, "kv_cache_generation.csv")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(render_markdown(payload))
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        fieldnames = ["mode", "avg_sec", "tokens_per_sec", "ms_per_token", "speedup_vs_kv_cache"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        results = payload["results"]
        writer.writerow(
            {
                "mode": "no_cache",
                "avg_sec": results["no_cache_avg_sec"],
                "tokens_per_sec": results["no_cache_tokens_per_sec"],
                "ms_per_token": results["no_cache_ms_per_token"],
                "speedup_vs_kv_cache": results["speedup"],
            }
        )
        writer.writerow(
            {
                "mode": "kv_cache",
                "avg_sec": results["kv_cache_avg_sec"],
                "tokens_per_sec": results["kv_cache_tokens_per_sec"],
                "ms_per_token": results["kv_cache_ms_per_token"],
                "speedup_vs_kv_cache": 1.0,
            }
        )
    return json_path, md_path, csv_path


def main():
    args = parse_args()
    payload = run_benchmark(args)
    report_dir = args.report_dir or os.path.join(get_base_dir(), "reports", "experiments")
    json_path, md_path, csv_path = write_reports(payload, report_dir)
    results = payload["results"]
    print(f"no cache: {results['no_cache_tokens_per_sec']:.2f} tok/s ({results['no_cache_ms_per_token']:.3f} ms/token)")
    print(f"kv cache: {results['kv_cache_tokens_per_sec']:.2f} tok/s ({results['kv_cache_ms_per_token']:.3f} ms/token)")
    print(f"speedup: {results['speedup']:.3f}x")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    print(f"CSV report: {csv_path}")


if __name__ == "__main__":
    main()
