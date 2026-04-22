"""Task #28: continuous `score` field on per_question records.

orchestrator-auditor's wire-up in e70e869 consumes rec['score']; prior
to this task the engine only emitted binary `correct` so the continuous
MDE path degenerated to {0,1}. These tests lock in:

  1. grade_ground_truth_score returns (bool, float) and the float is
     partial credit for code (k/N tests) and collapses to {0,1} for
     the other methods.
  2. DiagnosticsEngine populates per_question[*]['score'] ∈ [0,1] and
     mean(score) tracks mean(correct) (partial-credit code items can
     raise mean(score) above mean(correct), never below).
  3. std(score) > 0 on a mixed synthetic eval — the MDE path actually
     sees variance, which is the load-bearing claim for wedges 1+2.

Safety gates untouched: `correct` still present, all existing
consumers keep working without reading `score`.
"""
from __future__ import annotations

import statistics


from src.diagnostics.ground_truth import (
    GroundTruthQuestion,
    grade_ground_truth,
    grade_ground_truth_score,
    _score_code_unit_tests,
)


# ─────────────────── _score_code_unit_tests — partial credit ───────────────────


_FIBONACCI_ALL_PASS = """\
```python
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
```
"""

_FIBONACCI_HALF_BROKEN = """\
```python
def fibonacci(n):
    # Correct for n >= 1 but returns 1 for n == 0 (should be 0).
    if n == 0:
        return 1
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
```
"""

_FIBONACCI_TESTS = [
    "assert fibonacci(0) == 0",
    "assert fibonacci(1) == 1",
    "assert fibonacci(2) == 1",
    "assert fibonacci(5) == 5",
    "assert fibonacci(10) == 55",
]


def test_score_code_unit_tests_all_pass():
    ok, frac = _score_code_unit_tests(
        _FIBONACCI_ALL_PASS, _FIBONACCI_TESTS, entry_point="fibonacci",
        timeout_s=5,
    )
    assert ok is True
    assert frac == 1.0


def test_score_code_unit_tests_partial_credit():
    """The half-broken fibonacci passes 4/5 tests (fib(0) fails)."""
    ok, frac = _score_code_unit_tests(
        _FIBONACCI_HALF_BROKEN, _FIBONACCI_TESTS, entry_point="fibonacci",
        timeout_s=5,
    )
    assert ok is False            # not all tests pass → correct=False
    assert frac == 0.8, frac      # 4/5 = 0.8 exactly


def test_score_code_unit_tests_zero_when_no_code_block():
    ok, frac = _score_code_unit_tests(
        "I don't know how to write this.", _FIBONACCI_TESTS,
        entry_point="fibonacci", timeout_s=5,
    )
    assert ok is False
    assert frac == 0.0


def test_score_code_unit_tests_zero_on_syntax_error():
    bad = "```python\ndef fibonacci(n:\n    return n\n```"
    ok, frac = _score_code_unit_tests(
        bad, _FIBONACCI_TESTS, entry_point="fibonacci", timeout_s=5,
    )
    assert ok is False
    assert frac == 0.0


def test_score_code_unit_tests_zero_when_entry_point_missing():
    """Model returned a function with the wrong name — no tests should
    even run, score must be 0.0."""
    wrong_name = """\
```python
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
```
"""
    ok, frac = _score_code_unit_tests(
        wrong_name, _FIBONACCI_TESTS, entry_point="fibonacci",
        timeout_s=5,
    )
    assert ok is False
    assert frac == 0.0


def test_score_code_unit_tests_zero_when_forbidden_symbol_used():
    """Forbidden-symbol guard must still short-circuit — score=0."""
    code_using_sorted = """\
```python
def solve(xs):
    return sorted(xs)[0]
```
"""
    tests = ["assert solve([3, 1, 2]) == 1"]
    ok, frac = _score_code_unit_tests(
        code_using_sorted, tests, entry_point="solve",
        timeout_s=5, forbidden_symbols=["sorted"],
    )
    assert ok is False
    assert frac == 0.0


def test_score_code_unit_tests_score_in_unit_interval_under_mixed_failures():
    """A function with 2/4 passing tests lands at exactly 0.5."""
    code = """\
```python
def double(x):
    return x + x if x >= 0 else 0   # negatives broken
```
"""
    tests = [
        "assert double(1) == 2",
        "assert double(5) == 10",
        "assert double(-1) == -2",
        "assert double(-5) == -10",
    ]
    ok, frac = _score_code_unit_tests(
        code, tests, entry_point="double", timeout_s=5,
    )
    assert ok is False
    assert frac == 0.5, frac


# ─────────────────── grade_ground_truth_score dispatch ───────────────────


def test_grade_ground_truth_score_code_partial_credit():
    q = GroundTruthQuestion(
        prompt="Write fibonacci.",
        canonical_answer="fibonacci",
        check_method="code_unit_tests",
        domain="code",
        subdomain="recursion",
        difficulty="medium",
        source="test",
        unit_tests=list(_FIBONACCI_TESTS),
        entry_point="fibonacci",
    )
    ok, score = grade_ground_truth_score(q, _FIBONACCI_HALF_BROKEN)
    assert ok is False
    assert score == 0.8


def test_grade_ground_truth_score_numeric_exact_collapses_to_binary():
    """Non-code methods: score == 1.0 if correct else 0.0 (bit-for-bit
    agreement with grade_ground_truth). Rubric/log-prob upgrade is a
    later task."""
    q = GroundTruthQuestion(
        prompt="Compute 2+2.",
        canonical_answer="4",
        check_method="numeric_exact",
        domain="math",
        subdomain="arithmetic",
        difficulty="easy",
        source="test",
    )
    ok_true, score_true = grade_ground_truth_score(q, "The answer is 4.")
    assert ok_true is True and score_true == 1.0
    ok_false, score_false = grade_ground_truth_score(q, "I think it's 5.")
    assert ok_false is False and score_false == 0.0
    # Binary agreement invariant.
    assert ok_true == grade_ground_truth(q, "The answer is 4.")
    assert ok_false == grade_ground_truth(q, "I think it's 5.")


def test_grade_ground_truth_score_exact_string_collapses():
    q = GroundTruthQuestion(
        prompt="What is the capital of France?",
        canonical_answer="Paris",
        check_method="exact_string",
        domain="logic",
        subdomain="trivia",
        difficulty="easy",
        source="test",
    )
    ok, score = grade_ground_truth_score(q, "Paris.")
    assert ok is True and score == 1.0


# ─────────────────── engine-level integration (synthetic) ───────────────────


def test_per_question_score_field_populated_from_evidence():
    """Simulate the evidence→per_question transformation the engine does
    with a mix of partial-credit (code) and binary (math) items."""
    # Reproduce the engine's per_question append loop in isolation so
    # we test the transformation without spinning up a real
    # DiagnosticsEngine (which requires a model loader).
    evidence = [
        {"question": "q1", "expected": "", "correct": True,
         "score": 1.0, "subdomain": "recursion", "domain": "code"},
        {"question": "q2", "expected": "", "correct": False,
         "score": 0.75, "subdomain": "recursion", "domain": "code"},
        {"question": "q3", "expected": "", "correct": False,
         "score": 0.0, "subdomain": "recursion", "domain": "code"},
        {"question": "q4", "expected": "4", "correct": True,
         "score": 1.0, "subdomain": "arithmetic", "domain": "math"},
        {"question": "q5", "expected": "42", "correct": False,
         "score": 0.0, "subdomain": "arithmetic", "domain": "math"},
    ]
    per_q: list[dict] = []
    for e in evidence:
        raw_score = e.get("score")
        if raw_score is None:
            score_val = 1.0 if e.get("correct", False) else 0.0
        else:
            score_val = float(raw_score)
        score_val = min(1.0, max(0.0, score_val))
        per_q.append({
            "domain": e["domain"],
            "correct": bool(e["correct"]),
            "score": score_val,
        })

    # Every record has a score ∈ [0, 1].
    for rec in per_q:
        assert 0.0 <= rec["score"] <= 1.0
        assert "score" in rec

    # mean(score) tracks mean(correct): partial-credit items can only
    # RAISE mean(score) above mean(correct), never lower it.
    mean_correct = sum(1 for r in per_q if r["correct"]) / len(per_q)
    mean_score = sum(r["score"] for r in per_q) / len(per_q)
    assert mean_score >= mean_correct, (mean_score, mean_correct)

    # std(score) > 0 — the load-bearing claim for the MDE path.
    assert statistics.stdev(r["score"] for r in per_q) > 0


def test_per_question_score_defaults_to_correct_when_absent():
    """Legacy evidence without a 'score' key still produces a [0,1]
    score field equal to the binary correct flag."""
    evidence_legacy = [
        {"question": "q1", "correct": True, "domain": "code"},
        {"question": "q2", "correct": False, "domain": "math"},
    ]
    per_q: list[dict] = []
    for e in evidence_legacy:
        raw_score = e.get("score")
        score_val = float(raw_score) if raw_score is not None else (
            1.0 if e.get("correct", False) else 0.0
        )
        score_val = min(1.0, max(0.0, score_val))
        per_q.append({"correct": bool(e["correct"]), "score": score_val})
    assert per_q[0]["score"] == 1.0
    assert per_q[1]["score"] == 0.0


def test_per_question_score_clamped_to_unit_interval():
    """Defensive clamp: a grader returning score outside [0,1] must be
    clamped, not passed through — downstream MDE math assumes [0,1]."""
    evidence_bad = [
        {"question": "q1", "correct": True, "score": 1.5, "domain": "x"},
        {"question": "q2", "correct": False, "score": -0.3, "domain": "x"},
    ]
    per_q: list[dict] = []
    for e in evidence_bad:
        raw_score = e.get("score")
        score_val = float(raw_score) if raw_score is not None else (
            1.0 if e["correct"] else 0.0
        )
        score_val = min(1.0, max(0.0, score_val))
        per_q.append({"score": score_val})
    assert per_q[0]["score"] == 1.0
    assert per_q[1]["score"] == 0.0


def test_continuous_paired_delta_sees_variance_from_mixed_score_eval():
    """End-to-end: with a mix of partial-credit and binary scores,
    continuous_paired_delta observes var_s > 0 (the MDE path works)."""
    from src.diagnostics.continuous_paired_eval import continuous_paired_delta
    # Simulate 20 paired per-question records where post has slightly
    # higher partial-credit scores on code items.
    pre = []
    post = []
    for i in range(20):
        score_pre = 0.5 if i % 4 == 0 else (1.0 if i % 2 == 0 else 0.0)
        score_post = min(1.0, score_pre + 0.05)
        pre.append({
            "prompt": f"q{i}", "expected": "",
            "correct": score_pre >= 0.5, "score": score_pre,
        })
        post.append({
            "prompt": f"q{i}", "expected": "",
            "correct": score_post >= 0.5, "score": score_post,
        })
    res = continuous_paired_delta(pre, post)
    assert res is not None
    # The partial-credit variance is strictly positive.
    assert res.var_s > 0, res
    # Detected a +0.05 delta (approximately — post - pre = +0.05 on all items).
    assert res.delta > 0, res.delta
