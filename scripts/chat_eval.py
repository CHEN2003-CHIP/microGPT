"""Evaluate chat checkpoints with a simple rule-based benchmark."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


parser = argparse.ArgumentParser(description="Evaluate a trained chat model")
parser.add_argument("-i", "--source", type=str, default="sft", choices=["base", "sft"], help="Checkpoint source")
parser.add_argument("-g", "--model-tag", type=str, default=None, help="Model tag to load")
parser.add_argument("-s", "--step", type=int, default=None, help="Checkpoint step to load")
parser.add_argument("--device-type", type=str, default="", choices=["cuda", "cpu", "mps"], help="Device override")
parser.add_argument("--eval-file", type=str, default=None, help="Path to benchmark JSONL")
parser.add_argument("--report-dir", type=str, default=None, help="Directory to write report files")
parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling (0 disables)")
parser.add_argument("--max-tokens", type=int, default=128, help="Maximum generated tokens per answer")
parser.add_argument("--pass-threshold", type=float, default=0.8, help="Overall pass line from 0.0 to 1.0")
args = parser.parse_args()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from microchat.chat_eval import (
    default_eval_path,
    evaluate_response,
    load_eval_cases,
    render_markdown_report,
    summarize_results,
)
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
    generated: list[int] = []
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


def ensure_report_dir(base_dir: str) -> str:
    report_dir = args.report_dir or os.path.join(base_dir, "reports", "chat_eval")
    os.makedirs(report_dir, exist_ok=True)
    return report_dir


def main():
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _, _, _, _, device = compute_init(device_type)
    model, tokenizer, metadata = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
    engine = Engine(model, tokenizer)

    base_dir = get_base_dir()
    eval_path = args.eval_file or default_eval_path(base_dir)
    cases = load_eval_cases(eval_path)

    print0(f"Loaded {len(cases)} eval cases from {eval_path}")
    results = []
    for index, case in enumerate(cases, start=1):
        response = generate_response(engine, tokenizer, case.prompt)
        result = evaluate_response(case, response)
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print0(f"[{index:02d}/{len(cases):02d}] {status} {case.case_id} -> {response[:80] or '<empty>'}")

    summary = summarize_results(results, pass_threshold=args.pass_threshold)

    report_dir = ensure_report_dir(base_dir)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = args.model_tag or "auto"
    stem = f"{args.source}_{model_tag}_{timestamp}"
    json_path = os.path.join(report_dir, f"{stem}.json")
    md_path = os.path.join(report_dir, f"{stem}.md")

    payload = {
        "summary": {
            "qualified": summary["qualified"],
            "overall_score": summary["overall_score"],
            "pass_threshold": summary["pass_threshold"],
            "passed_cases": summary["passed_cases"],
            "total_cases": summary["total_cases"],
            "categories": summary["categories"],
            "required_failures": [result.case.case_id for result in summary["required_failures"]],
        },
        "model": {
            "source": args.source,
            "model_tag": args.model_tag,
            "requested_step": args.step,
            "loaded_step": metadata.get("step"),
            "val_bpb": metadata.get("val_bpb"),
        },
        "eval_file": eval_path,
        "results": [
            {
                "id": result.case.case_id,
                "category": result.case.category,
                "prompt": result.case.prompt,
                "response": result.response,
                "passed": result.passed,
                "score": result.score,
                "required": result.case.required,
                "reasons": result.reasons,
            }
            for result in results
        ],
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    report = render_markdown_report(
        results=results,
        summary=summary,
        report_title="microGPT Chat Evaluation",
        model_source=args.source,
        model_tag=args.model_tag or "latest",
        model_step=metadata.get("step"),
        metrics={"val_bpb": metadata.get("val_bpb")},
    )
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(report)

    print0("")
    print0("=" * 72)
    print0(f"Qualified: {'YES' if summary['qualified'] else 'NO'}")
    print0(f"Score: {summary['overall_score'] * 100:.1f}%")
    print0(f"Pass line: {summary['pass_threshold'] * 100:.1f}%")
    print0(f"Cases: {summary['passed_cases']}/{summary['total_cases']}")
    print0(f"Loaded step: {metadata.get('step')}")
    print0(f"Validation bpb: {metadata.get('val_bpb')}")
    print0(f"Markdown report: {md_path}")
    print0(f"JSON report: {json_path}")
    if summary["required_failures"]:
        print0("Required failures:")
        for result in summary["required_failures"]:
            print0(f"  - {result.case.case_id}: {'; '.join(result.reasons) or 'failed'}")


if __name__ == "__main__":
    main()
