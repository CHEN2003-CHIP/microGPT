"""Evaluate base checkpoints with short completion prompts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


parser = argparse.ArgumentParser(description="Evaluate a base model with short prompts")
parser.add_argument("-g", "--model-tag", type=str, default=None, help="Model tag to load")
parser.add_argument("-s", "--step", type=int, default=None, help="Checkpoint step to load")
parser.add_argument("--device-type", type=str, default="", choices=["cuda", "cpu", "mps"], help="Device override")
parser.add_argument("--eval-file", type=str, default=None, help="Path to base eval JSONL")
parser.add_argument("--report-dir", type=str, default=None, help="Directory to write reports")
parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling")
parser.add_argument("--max-tokens", type=int, default=48, help="Maximum generated tokens per prompt")
args = parser.parse_args()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from microchat.checkpoint_manager import load_model
from microchat.common import autodetect_device_type, compute_init, get_base_dir, print0
from microchat.engine import Engine
import torch


def resolve_eval_path(base_dir: str) -> str:
    preferred = os.path.join(base_dir, "base_eval.jsonl")
    bundled = os.path.join(os.path.dirname(__file__), "base_eval.sample.jsonl")
    return args.eval_file or (preferred if os.path.exists(preferred) else bundled)


def build_report_dir(base_dir: str) -> str:
    report_dir = args.report_dir or os.path.join(base_dir, "reports", "base_eval")
    os.makedirs(report_dir, exist_ok=True)
    return report_dir


def load_rows(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No eval rows found in {path}")
    return rows


def generate_completion(engine, tokenizer, prompt: str) -> str:
    tokens = [tokenizer.get_bos_token_id(), *tokenizer.encode(prompt)]
    completion = []
    for token_column, _ in engine.generate(
        tokens,
        num_samples=1,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=(args.top_k or None),
    ):
        completion.append(token_column[0])
    return tokenizer.decode(completion).strip()


@torch.inference_mode()
def continuation_nll(model, tokenizer, prompt: str, continuation: str):
    device = model.get_device()
    prompt_tokens = [tokenizer.get_bos_token_id(), *tokenizer.encode(prompt)]
    continuation_tokens = tokenizer.encode(continuation)
    if not continuation_tokens:
        return float("inf")

    full_tokens = prompt_tokens + continuation_tokens
    inputs = torch.tensor([full_tokens[:-1]], dtype=torch.long, device=device)
    targets = torch.tensor([full_tokens[1:]], dtype=torch.long, device=device)

    # Target position i predicts full_tokens[i + 1]. Keep only continuation targets.
    first_continuation_target = len(prompt_tokens) - 1
    targets[:, :first_continuation_target] = -1
    loss_sum = model(inputs, targets, loss_reduction="sum")
    return loss_sum.item() / len(continuation_tokens)


def evaluate_row(row, completion: str):
    lower = completion.lower()
    contains_any = row.get("contains_any", [])
    forbidden = row.get("forbidden_contains", [])
    pass_contains = (not contains_any) or any(item.lower() in lower for item in contains_any)
    pass_forbidden = not any(item.lower() in lower for item in forbidden)
    return {
        "passed": bool(pass_contains and pass_forbidden),
        "contains_ok": pass_contains,
        "forbidden_ok": pass_forbidden,
    }


def evaluate_continuation_row(model, tokenizer, row):
    expected = row["expected"]
    expected_nll = continuation_nll(model, tokenizer, row["prompt"], expected)
    alternatives = row.get("alternatives", [])
    alternative_scores = [
        {"text": alternative, "nll": continuation_nll(model, tokenizer, row["prompt"], alternative)}
        for alternative in alternatives
    ]
    best_alternative = min((item["nll"] for item in alternative_scores), default=float("inf"))
    passed = expected_nll < best_alternative if alternative_scores else math.isfinite(expected_nll)
    return {
        "id": row["id"],
        "prompt": row["prompt"],
        "expected": expected,
        "expected_nll": expected_nll,
        "alternatives": alternative_scores,
        "passed": passed,
    }


def main():
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _, _, _, _, device = compute_init(device_type)
    model, tokenizer, metadata = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)
    engine = Engine(model, tokenizer)

    base_dir = get_base_dir()
    eval_path = resolve_eval_path(base_dir)
    rows = load_rows(eval_path)

    results = []
    passed = 0
    for index, row in enumerate(rows, start=1):
        if "expected" in row:
            result = evaluate_continuation_row(model, tokenizer, row)
            passed += int(result["passed"])
            print0(
                f"[{index:02d}/{len(rows):02d}] {'PASS' if result['passed'] else 'FAIL'} "
                f"{row['id']} -> expected_nll {result['expected_nll']:.4f}"
            )
            results.append(result)
            continue

        completion = generate_completion(engine, tokenizer, row["prompt"])
        verdict = evaluate_row(row, completion)
        passed += int(verdict["passed"])
        result = {"id": row["id"], "prompt": row["prompt"], "completion": completion, **verdict}
        results.append(result)
        print0(f"[{index:02d}/{len(rows):02d}] {'PASS' if verdict['passed'] else 'FAIL'} {row['id']} -> {completion[:80] or '<empty>'}")

    report_dir = build_report_dir(base_dir)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = args.model_tag or "latest"
    stem = f"{model_tag}_{timestamp}"
    json_path = os.path.join(report_dir, f"{stem}.json")
    md_path = os.path.join(report_dir, f"{stem}.md")

    payload = {
        "model_tag": model_tag,
        "step": metadata.get("step"),
        "val_bpb": metadata.get("val_bpb"),
        "score": passed / len(rows),
        "results": results,
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    lines = [
        "# Base Eval Report",
        "",
        f"- Model tag: `{model_tag}`",
        f"- Step: `{metadata.get('step')}`",
        f"- val_bpb: `{metadata.get('val_bpb')}`",
        f"- Score: `{passed}/{len(rows)} ({(passed / len(rows)) * 100:.1f}%)`",
        "",
    ]
    for result in results:
        if "expected" in result:
            lines.extend(
                [
                    f"## {'PASS' if result['passed'] else 'FAIL'} `{result['id']}`",
                    "",
                    f"Prompt: `{result['prompt']}`",
                    f"Expected continuation: `{result['expected']}`",
                    f"Expected NLL/token: `{result['expected_nll']:.4f}`",
                    "",
                ]
            )
            if result["alternatives"]:
                lines.append("Alternatives:")
                for alternative in result["alternatives"]:
                    lines.append(f"- `{alternative['text']}`: `{alternative['nll']:.4f}`")
                lines.append("")
            continue
        lines.extend(
            [
                f"## {'PASS' if result['passed'] else 'FAIL'} `{result['id']}`",
                "",
                f"Prompt: `{result['prompt']}`",
                "",
                "```text",
                result["completion"] or "<empty>",
                "```",
                "",
            ]
        )
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")

    print0(f"Base eval score: {passed}/{len(rows)} ({(passed / len(rows)) * 100:.1f}%)")
    print0(f"Markdown report: {md_path}")
    print0(f"JSON report: {json_path}")


if __name__ == "__main__":
    main()
