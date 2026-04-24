import os

from microchat.chat_eval import EvalCase, EvalChecks, evaluate_response, load_eval_cases, summarize_results


def test_evaluate_response_passes_all_checks():
    case = EvalCase(
        case_id="demo",
        prompt="say hello",
        category="basic",
        required=True,
        checks=EvalChecks(
            contains_all=["hello"],
            forbidden_contains=["goodbye"],
            min_chars=3,
        ),
    )

    result = evaluate_response(case, "hello there")

    assert result.passed is True
    assert result.score == 1.0
    assert result.reasons == []


def test_evaluate_response_collects_failures():
    case = EvalCase(
        case_id="demo",
        prompt="say apple",
        category="basic",
        checks=EvalChecks(
            exact=["apple"],
            forbidden_contains=["banana"],
            min_chars=6,
        ),
    )

    result = evaluate_response(case, "banana")

    assert result.passed is False
    assert result.score < 1.0
    assert any("exact match failed" in reason for reason in result.reasons)
    assert any("contains forbidden phrases" in reason for reason in result.reasons)


def test_evaluate_response_supports_max_chars_and_ngram_repeats():
    case = EvalCase(
        case_id="repeat",
        prompt="say something short",
        category="anti_repeat",
        checks=EvalChecks(max_chars=200, max_ngram_repeats=1),
    )

    result = evaluate_response(case, "They are built. They are built.")

    assert result.passed is False
    assert any("repeated phrase loop detected" in reason for reason in result.reasons)


def test_summarize_results_respects_required_failures():
    passing_case = EvalCase(case_id="pass", prompt="a", checks=EvalChecks(exact=["ok"]))
    required_case = EvalCase(case_id="required", prompt="b", checks=EvalChecks(exact=["ok"]), required=True)

    results = [
        evaluate_response(passing_case, "ok"),
        evaluate_response(required_case, "no"),
    ]

    summary = summarize_results(results, pass_threshold=0.4)

    assert summary["overall_score"] == 0.5
    assert summary["qualified"] is False
    assert [item.case.case_id for item in summary["required_failures"]] == ["required"]


def test_phase2_eval_file_loads():
    path = os.path.join(os.getcwd(), "scripts", "chat_eval.phase2_en.jsonl")
    cases = load_eval_cases(path)

    assert len(cases) >= 6
    assert any(case.case_id == "tree_definition" for case in cases)
    assert any(case.required for case in cases)


def test_repetition_eval_file_loads():
    path = os.path.join(os.getcwd(), "scripts", "chat_eval.repetition_en.jsonl")
    cases = load_eval_cases(path)

    assert len(cases) >= 6
    assert any(case.case_id == "tree_short_stop" for case in cases)
    assert any(case.checks.max_chars > 0 for case in cases)
    assert any(case.checks.max_ngram_repeats > 0 for case in cases)
