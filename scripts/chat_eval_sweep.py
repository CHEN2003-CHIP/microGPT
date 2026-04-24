"""Evaluate multiple chat checkpoints and pick the best step."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

parser = argparse.ArgumentParser(description="Run chat_eval across saved checkpoints")
parser.add_argument("-g", "--model-tag", type=str, required=True, help="SFT model tag to evaluate")
parser.add_argument("--device-type", type=str, default="", choices=["cuda", "cpu", "mps"], help="Device override")
parser.add_argument("--eval-file", type=str, default=None, help="Path to benchmark JSONL")
parser.add_argument("--pass-threshold", type=float, default=0.8, help="Overall pass line from 0.0 to 1.0")
parser.add_argument("--max-tokens", type=int, default=128, help="Maximum generated tokens per answer")
parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling (0 disables)")
args = parser.parse_args()

from microchat.chat_eval import default_eval_path, evaluate_response, load_eval_cases, summarize_results
from microchat.checkpoint_manager import load_model
from microchat.common import autodetect_device_type, compute_init, get_base_dir, print0
from microchat.engine import Engine


def build_prompt_tokens(tokenizer, prompt: str) -> list[int]:
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    return [bos, user_start, *tokenizer.encode(prompt), user_end, assistant_start]


def generate_response(engine, tokenizer, prompt: str) -> str:
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    tokens = build_prompt_tokens(tokenizer, prompt)
    generated = []
    for token_column, _ in engine.generate(
        tokens,
        num_samples=1,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=(args.top_k or None),
    ):
        token = token_column[0]
        if token == assistant_end:
            break
        generated.append(token)
    return tokenizer.decode(generated).strip()


def list_steps(checkpoint_dir: str):
    model_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "model_*.pt")))
    if not model_paths:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return [int(os.path.basename(path).split("_")[-1].split(".")[0]) for path in model_paths]


def main():
    base_dir = get_base_dir()
    checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", args.model_tag)
    steps = list_steps(checkpoint_dir)
    eval_path = args.eval_file or default_eval_path(base_dir)
    cases = load_eval_cases(eval_path)

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _, _, _, _, device = compute_init(device_type)

    best = None
    rows = []
    for step in steps:
        model, tokenizer, metadata = load_model("sft", device, phase="eval", model_tag=args.model_tag, step=step)
        engine = Engine(model, tokenizer)
        results = []
        for case in cases:
            results.append(evaluate_response(case, generate_response(engine, tokenizer, case.prompt)))
        summary = summarize_results(results, pass_threshold=args.pass_threshold)
        row = {
            "step": step,
            "score": summary["overall_score"],
            "qualified": summary["qualified"],
            "passed_cases": summary["passed_cases"],
            "total_cases": summary["total_cases"],
            "required_failures": [item.case.case_id for item in summary["required_failures"]],
            "val_bpb": metadata.get("val_bpb"),
        }
        rows.append(row)
        if best is None or row["score"] > best["score"] or (row["score"] == best["score"] and step < best["step"]):
            best = row
        print0(
            f"step {step:05d} | score {row['score'] * 100:.1f}% | "
            f"cases {row['passed_cases']}/{row['total_cases']} | val_bpb {row['val_bpb']}"
        )

    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    out_path = os.path.join(reports_dir, f"{args.model_tag}_chat_eval_sweep.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump({"model_tag": args.model_tag, "eval_file": eval_path, "best": best, "rows": rows}, handle, ensure_ascii=False, indent=2)

    print0("")
    print0(f"Best step: {best['step']}")
    print0(f"Best score: {best['score'] * 100:.1f}%")
    print0(f"Sweep report: {out_path}")


if __name__ == "__main__":
    main()
