"""Rule-based chat evaluation helpers for microGPT."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import re
from typing import Any, Optional


@dataclass
class EvalChecks:
    exact: list[str] = field(default_factory=list)
    contains_all: list[str] = field(default_factory=list)
    contains_any: list[str] = field(default_factory=list)
    regex_any: list[str] = field(default_factory=list)
    forbidden_contains: list[str] = field(default_factory=list)
    min_chars: int = 0
    max_chars: int = 0
    max_ngram_repeats: int = 0


@dataclass
class EvalCase:
    case_id: str
    prompt: str
    checks: EvalChecks
    category: str = "general"
    weight: float = 1.0
    required: bool = False
    notes: str = ""


@dataclass
class EvalResult:
    case: EvalCase
    response: str
    passed: bool
    score: float
    reasons: list[str]


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _count_adjacent_ngram_repeats(text: str, n: int) -> int:
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < 2 * n or n <= 0:
        return 0

    max_repeats = 1
    for start in range(0, len(tokens) - (2 * n) + 1):
        phrase = tuple(tokens[start : start + n])
        current_repeats = 1
        cursor = start + n
        while cursor + n <= len(tokens) and tuple(tokens[cursor : cursor + n]) == phrase:
            current_repeats += 1
            cursor += n
        max_repeats = max(max_repeats, current_repeats)
    return max_repeats


def load_eval_cases(path: str) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = line.strip()
            if not row:
                continue
            data = json.loads(row)
            checks_raw = data.get("checks", {})
            checks = EvalChecks(
                exact=_as_list(checks_raw.get("exact")),
                contains_all=_as_list(checks_raw.get("contains_all")),
                contains_any=_as_list(checks_raw.get("contains_any")),
                regex_any=_as_list(checks_raw.get("regex_any")),
                forbidden_contains=_as_list(checks_raw.get("forbidden_contains")),
                min_chars=int(checks_raw.get("min_chars", 0)),
                max_chars=int(checks_raw.get("max_chars", 0)),
                max_ngram_repeats=int(checks_raw.get("max_ngram_repeats", 0)),
            )
            case = EvalCase(
                case_id=str(data.get("id", f"case_{line_number:03d}")),
                prompt=str(data["prompt"]),
                category=str(data.get("category", "general")),
                checks=checks,
                weight=float(data.get("weight", 1.0)),
                required=bool(data.get("required", False)),
                notes=str(data.get("notes", "")),
            )
            cases.append(case)
    if not cases:
        raise ValueError(f"No evaluation cases found in {path}")
    return cases


def evaluate_response(case: EvalCase, response: str) -> EvalResult:
    reasons: list[str] = []
    passed_checks = 0
    total_checks = 0

    normalized_response = _normalize_text(response)
    raw_response = response.strip()

    if case.checks.min_chars > 0:
        total_checks += 1
        if len(raw_response) >= case.checks.min_chars:
            passed_checks += 1
        else:
            reasons.append(f"too short (< {case.checks.min_chars} chars)")

    if case.checks.max_chars > 0:
        total_checks += 1
        if len(raw_response) <= case.checks.max_chars:
            passed_checks += 1
        else:
            reasons.append(f"too long (> {case.checks.max_chars} chars)")

    if case.checks.exact:
        total_checks += 1
        expected = {_normalize_text(item) for item in case.checks.exact}
        if normalized_response in expected:
            passed_checks += 1
        else:
            reasons.append("exact match failed")

    if case.checks.contains_all:
        total_checks += 1
        missing = [item for item in case.checks.contains_all if item.lower() not in response.lower()]
        if not missing:
            passed_checks += 1
        else:
            reasons.append(f"missing required phrases: {', '.join(missing)}")

    if case.checks.contains_any:
        total_checks += 1
        if any(item.lower() in response.lower() for item in case.checks.contains_any):
            passed_checks += 1
        else:
            reasons.append("missing any expected phrase")

    if case.checks.regex_any:
        total_checks += 1
        if any(re.search(pattern, response, flags=re.IGNORECASE) for pattern in case.checks.regex_any):
            passed_checks += 1
        else:
            reasons.append("regex match failed")

    if case.checks.forbidden_contains:
        total_checks += 1
        forbidden_hits = [item for item in case.checks.forbidden_contains if item.lower() in response.lower()]
        if not forbidden_hits:
            passed_checks += 1
        else:
            reasons.append(f"contains forbidden phrases: {', '.join(forbidden_hits)}")

    if case.checks.max_ngram_repeats > 0:
        total_checks += 1
        bigram_repeats = _count_adjacent_ngram_repeats(raw_response, 2)
        trigram_repeats = _count_adjacent_ngram_repeats(raw_response, 3)
        max_repeats = max(bigram_repeats, trigram_repeats)
        if max_repeats <= case.checks.max_ngram_repeats:
            passed_checks += 1
        else:
            reasons.append("repeated phrase loop detected")

    if total_checks == 0:
        total_checks = 1
        passed_checks = 1 if raw_response else 0
        if not raw_response:
            reasons.append("empty response")

    score = passed_checks / total_checks
    passed = score >= 1.0
    return EvalResult(case=case, response=response, passed=passed, score=score, reasons=reasons)


def summarize_results(results: list[EvalResult], pass_threshold: float) -> dict[str, Any]:
    total_weight = sum(result.case.weight for result in results)
    passed_weight = sum(result.case.weight for result in results if result.passed)
    required_failures = [result for result in results if result.case.required and not result.passed]

    category_summary: dict[str, dict[str, float]] = {}
    for result in results:
        bucket = category_summary.setdefault(
            result.case.category,
            {"passed": 0.0, "total": 0.0},
        )
        bucket["total"] += result.case.weight
        if result.passed:
            bucket["passed"] += result.case.weight

    overall_score = (passed_weight / total_weight) if total_weight else 0.0
    qualified = overall_score >= pass_threshold and not required_failures
    return {
        "qualified": qualified,
        "overall_score": overall_score,
        "pass_threshold": pass_threshold,
        "required_failures": required_failures,
        "categories": category_summary,
        "total_cases": len(results),
        "passed_cases": sum(1 for result in results if result.passed),
    }


def render_markdown_report(
    *,
    results: list[EvalResult],
    summary: dict[str, Any],
    report_title: str,
    model_source: str,
    model_tag: str,
    model_step: Optional[int],
    metrics: Optional[dict[str, Any]] = None,
) -> str:
    lines = [
        f"# {report_title}",
        "",
        f"- Model: `{model_source}` / `{model_tag}`",
        f"- Step: `{model_step}`" if model_step is not None else "- Step: latest",
        f"- Qualified: `{'YES' if summary['qualified'] else 'NO'}`",
        f"- Score: `{summary['overall_score'] * 100:.1f}%`",
        f"- Pass line: `{summary['pass_threshold'] * 100:.1f}%`",
        f"- Cases: `{summary['passed_cases']}/{summary['total_cases']}`",
    ]
    if metrics:
        for key, value in metrics.items():
            lines.append(f"- {key}: `{value}`")

    lines.extend(["", "## By Category", ""])
    for category, bucket in sorted(summary["categories"].items()):
        total = bucket["total"] or 1.0
        ratio = bucket["passed"] / total
        lines.append(f"- `{category}`: {bucket['passed']:.1f}/{bucket['total']:.1f} ({ratio * 100:.1f}%)")

    if summary["required_failures"]:
        lines.extend(["", "## Required Failures", ""])
        for result in summary["required_failures"]:
            lines.append(f"- `{result.case.case_id}`: {'; '.join(result.reasons) or 'failed'}")

    lines.extend(["", "## Case Results", ""])
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        lines.append(f"### {status} `{result.case.case_id}` [{result.case.category}]")
        lines.append("")
        lines.append(f"Prompt: `{result.case.prompt}`")
        if result.case.notes:
            lines.append(f"Notes: {result.case.notes}")
        lines.append(f"Score: `{result.score * 100:.1f}%`")
        if result.reasons:
            lines.append(f"Reasons: `{' | '.join(result.reasons)}`")
        lines.append("")
        lines.append("```text")
        lines.append(result.response.strip() or "<empty>")
        lines.append("```")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def default_eval_path(base_dir: str) -> str:
    preferred = os.path.join(base_dir, "chat_eval.jsonl")
    bundled = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "chat_eval.sample.jsonl")
    return preferred if os.path.exists(preferred) else bundled
