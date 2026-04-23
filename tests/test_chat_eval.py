from microchat.chat_eval import EvalCase, EvalChecks, evaluate_response, summarize_results


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
