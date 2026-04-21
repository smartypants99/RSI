"""Tests for the component proposer (meta-meta-meta)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.orchestrator.component_proposer import (
    ALLOWED_KINDS,
    BenchResult,
    ComponentProposal,
    append_verdict_log,
    decide_merge,
    evaluate_batch,
    evaluate_proposal,
    load_verdict_log,
    run_test_bench,
    safety_gate_review,
)


def _good_patch(path: str = "src/verifier/new_prop.py") -> str:
    return (
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        "@@ -0,0 +1,2 @@\n"
        "+def verify(x):\n"
        "+    return True\n"
    )


def _proposal(**over) -> ComponentProposal:
    base = dict(
        name="prop_v1",
        kind="verifier",
        rationale="adds a property check",
        patch=_good_patch(),
        entrypoint="src.verifier.new_prop:verify",
        smoke_cycles=3,
    )
    base.update(over)
    return ComponentProposal(**base)


# ------------------------------ safety gate ------------------------------


def test_safety_gate_accepts_well_formed_proposal():
    r = safety_gate_review(_proposal())
    assert r.ok, r.reasons


def test_safety_gate_rejects_unknown_kind():
    r = safety_gate_review(_proposal(kind="bogus"))
    assert not r.ok
    assert any("unknown component kind" in reason for reason in r.reasons)


def test_safety_gate_rejects_bad_entrypoint():
    r = safety_gate_review(_proposal(entrypoint="no_colon"))
    assert not r.ok


def test_safety_gate_rejects_out_of_range_smoke_cycles():
    r = safety_gate_review(_proposal(smoke_cycles=0))
    assert not r.ok
    r = safety_gate_review(_proposal(smoke_cycles=999))
    assert not r.ok


def test_safety_gate_rejects_patch_touching_safety_module():
    patch = _good_patch("src/safety/pwn.py")
    r = safety_gate_review(_proposal(patch=patch))
    assert not r.ok
    assert any("safety module" in reason for reason in r.reasons)


def test_safety_gate_rejects_forbidden_import():
    patch = (
        "--- a/src/verifier/evil.py\n"
        "+++ b/src/verifier/evil.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+import subprocess\n"
        "+subprocess.run(['ls'])\n"
    )
    r = safety_gate_review(_proposal(patch=patch))
    assert not r.ok


def test_allowed_kinds_matches_expected():
    assert ALLOWED_KINDS == {
        "verifier",
        "curriculum",
        "data_filter",
        "reasoning_strategy",
    }


# ------------------------------ test-bench -------------------------------


def test_run_test_bench_aggregates_mean():
    p = _proposal(smoke_cycles=4)
    vals = iter([
        {"anchor_delta": 0.1, "diversity_delta": 0.2, "gradient_health": 0.9},
        {"anchor_delta": 0.3, "diversity_delta": 0.0, "gradient_health": 0.7},
        {"anchor_delta": 0.2, "diversity_delta": 0.4, "gradient_health": 0.8},
        {"anchor_delta": 0.0, "diversity_delta": 0.0, "gradient_health": 1.0},
    ])
    runner = lambda _p, _i: next(vals)
    res = run_test_bench(p, runner)
    assert res.cycles_run == 4
    assert res.error is None
    assert res.anchor_delta == pytest.approx(0.15)
    assert res.diversity_delta == pytest.approx(0.15)
    assert res.gradient_health == pytest.approx(0.85)


def test_run_test_bench_catches_runner_exception():
    p = _proposal(smoke_cycles=3)
    def runner(_p, i):
        if i == 1:
            raise RuntimeError("sandbox blew up")
        return {"anchor_delta": 0.1, "diversity_delta": 0.1, "gradient_health": 0.5}
    res = run_test_bench(p, runner)
    assert res.error is not None
    assert "sandbox blew up" in res.error
    assert res.cycles_run == 1


# ------------------------------ merge decision ---------------------------


def test_decide_merge_all_positive_merges():
    bench = BenchResult(anchor_delta=0.1, diversity_delta=0.05, gradient_health=0.8, cycles_run=3)
    merge, reasons = decide_merge(bench)
    assert merge
    assert reasons == []


def test_decide_merge_fails_on_regression():
    bench = BenchResult(anchor_delta=-0.1, diversity_delta=0.05, gradient_health=0.8, cycles_run=3)
    merge, reasons = decide_merge(bench)
    assert not merge
    assert any("anchor_delta" in r for r in reasons)


def test_decide_merge_rejects_on_bench_error():
    bench = BenchResult(error="boom", cycles_run=0)
    merge, reasons = decide_merge(bench)
    assert not merge
    assert any("bench error" in r for r in reasons)


def test_decide_merge_rejects_zero_cycles():
    bench = BenchResult(cycles_run=0)
    merge, reasons = decide_merge(bench)
    assert not merge


# ------------------------------ end-to-end -------------------------------


def test_evaluate_proposal_blocked_by_safety_never_runs_bench():
    p = _proposal(kind="bogus")
    calls = []
    def runner(prop, i):
        calls.append(i)
        return {"anchor_delta": 1.0, "diversity_delta": 1.0, "gradient_health": 1.0}
    v = evaluate_proposal(p, runner)
    assert not v.merge
    assert v.bench is None
    assert calls == []  # runner must NOT be invoked


def test_evaluate_proposal_happy_path():
    p = _proposal(smoke_cycles=2)
    runner = lambda _p, _i: {"anchor_delta": 0.1, "diversity_delta": 0.1, "gradient_health": 0.8}
    v = evaluate_proposal(p, runner)
    assert v.review.ok
    assert v.merge, v.reasons
    assert v.bench is not None
    assert v.bench.cycles_run == 2


def test_evaluate_proposal_regresses_does_not_merge():
    p = _proposal(smoke_cycles=2)
    runner = lambda _p, _i: {"anchor_delta": -0.01, "diversity_delta": 0.1, "gradient_health": 0.9}
    v = evaluate_proposal(p, runner)
    assert v.review.ok
    assert not v.merge


def test_evaluate_batch_writes_audit_log(tmp_path: Path):
    log = tmp_path / "verdicts.jsonl"
    ps = [_proposal(name="a"), _proposal(name="b", kind="bogus")]
    runner = lambda _p, _i: {"anchor_delta": 0.1, "diversity_delta": 0.1, "gradient_health": 0.9}
    verdicts = evaluate_batch(ps, runner, log_path=log)
    assert len(verdicts) == 2
    loaded = load_verdict_log(log)
    assert len(loaded) == 2
    assert loaded[0]["proposal"]["name"] == "a"
    assert loaded[1]["merge"] is False


def test_proposal_roundtrip_dict():
    p = _proposal()
    p2 = ComponentProposal.from_dict(p.to_dict())
    assert p == p2


def test_append_verdict_log_is_valid_jsonl(tmp_path: Path):
    log = tmp_path / "v.jsonl"
    p = _proposal()
    runner = lambda _p, _i: {"anchor_delta": 0.1, "diversity_delta": 0.1, "gradient_health": 0.9}
    v = evaluate_proposal(p, runner)
    append_verdict_log(log, v)
    append_verdict_log(log, v)
    lines = log.read_text().splitlines()
    assert len(lines) == 2
    for ln in lines:
        json.loads(ln)  # must parse
