"""Tests for src/utils/external_benchmarks.py — the verifier-capture canary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils.external_benchmarks import (
    ANCHOR_SEED,
    SUPPORTED_BENCHMARKS,
    BenchmarkItem,
    detect_verifier_capture,
    fire_capture_alarm,
    load_benchmark,
    run_anchor_eval,
    sample_anchor_set,
    seed_adversarial_bank_from_external,
)


def test_load_benchmark_offline_all_supported(tmp_path):
    for bench in SUPPORTED_BENCHMARKS:
        items = load_benchmark(bench, cache_dir=tmp_path, force_offline=True)
        assert len(items) > 0
        assert all(isinstance(it, BenchmarkItem) for it in items)
        assert all(it.benchmark == bench for it in items)
        assert all(it.prompt and it.answer for it in items)


def test_load_benchmark_uses_cache(tmp_path):
    items1 = load_benchmark("humaneval", cache_dir=tmp_path, force_offline=True)
    cache_file = tmp_path / "humaneval.jsonl"
    assert cache_file.exists()
    # Corrupt in-memory fixture would still return from disk.
    items2 = load_benchmark("humaneval", cache_dir=tmp_path, force_offline=True)
    assert [i.item_id for i in items1] == [i.item_id for i in items2]


def test_unknown_benchmark_raises(tmp_path):
    with pytest.raises(ValueError):
        load_benchmark("not-a-benchmark", cache_dir=tmp_path, force_offline=True)


def test_sample_anchor_set_is_deterministic(tmp_path):
    items_by_bench = {
        b: load_benchmark(b, cache_dir=tmp_path, force_offline=True)
        for b in SUPPORTED_BENCHMARKS
    }
    a = sample_anchor_set(items_by_bench, per_benchmark=5, seed=ANCHOR_SEED)
    b = sample_anchor_set(items_by_bench, per_benchmark=5, seed=ANCHOR_SEED)
    assert [x.item_id for x in a] == [x.item_id for x in b]
    assert len(a) == 5 * len(SUPPORTED_BENCHMARKS)


def test_sample_anchor_set_seed_varies_partition(tmp_path):
    items_by_bench = {
        b: load_benchmark(b, cache_dir=tmp_path, force_offline=True)
        for b in SUPPORTED_BENCHMARKS
    }
    a = [x.item_id for x in sample_anchor_set(items_by_bench, 4, seed=1)]
    b = [x.item_id for x in sample_anchor_set(items_by_bench, 4, seed=2)]
    # Overwhelmingly likely to differ given 12 items per benchmark.
    assert a != b


def test_run_anchor_eval_perfect_grader(tmp_path):
    def oracle(prompt: str) -> str:
        # Return something that the default grader will accept: echo for code,
        # ensure math items still work by extracting from prompt.
        # The default grader checks substring for code, numeric match for math.
        # We emit the canonical answer by consulting the fixture.
        return oracle._answers.get(prompt, "")

    # Build answer lookup from the same fixture the eval will use.
    items_by_bench = {
        b: load_benchmark(b, cache_dir=tmp_path, force_offline=True)
        for b in SUPPORTED_BENCHMARKS
    }
    oracle._answers = {it.prompt: it.answer for its in items_by_bench.values() for it in its}

    result = run_anchor_eval(
        oracle,
        benchmarks=SUPPORTED_BENCHMARKS,
        per_benchmark=5,
        cache_dir=tmp_path,
    )
    assert result["n"] == 5 * len(SUPPORTED_BENCHMARKS)
    assert result["anchor_score"] == pytest.approx(1.0)
    assert set(result["per_benchmark"].keys()) == set(SUPPORTED_BENCHMARKS)


def test_humaneval_grader_executes_known_correct(tmp_path):
    """Proof the anchor harness isn't stuck at 0.0: a known-correct HumanEval
    completion must score >0 under the execution-based grader."""
    items = load_benchmark("humaneval", cache_dir=tmp_path, force_offline=True)
    # Body-only completion (real HumanEval canonical_solution shape).
    completions = {it.prompt: "    return a + b\n" for it in items}
    result = run_anchor_eval(
        lambda p: completions.get(p, ""),
        benchmarks=["humaneval"],
        per_benchmark=5,
        cache_dir=tmp_path,
    )
    assert result["n"] == 5
    assert result["anchor_score"] == pytest.approx(1.0)
    # Wrong completion must still score 0 — grader actually runs tests.
    wrong = {it.prompt: "    return a - b\n" for it in items}
    bad = run_anchor_eval(
        lambda p: wrong.get(p, ""),
        benchmarks=["humaneval"],
        per_benchmark=5,
        cache_dir=tmp_path,
    )
    assert bad["anchor_score"] == 0.0


def test_mbpp_grader_executes_known_correct(tmp_path):
    items = load_benchmark("mbpp", cache_dir=tmp_path, force_offline=True)
    # Full-def completion wrapped in a ```python fence — grader must strip it.
    def completion(prompt: str) -> str:
        for it in items:
            if it.prompt == prompt:
                entry = it.meta["entry_point"]
                return f"```python\ndef {entry}(a, b):\n    return a * b\n```"
        return ""
    result = run_anchor_eval(
        completion,
        benchmarks=["mbpp"],
        per_benchmark=5,
        cache_dir=tmp_path,
    )
    assert result["n"] == 5
    assert result["anchor_score"] == pytest.approx(1.0)


def test_run_anchor_eval_batch_model_fn_used(tmp_path):
    """batch_model_fn must be called once with all prompts and short-circuit
    the serial model_fn path (perf optimization for vLLM)."""
    items_by_bench = {
        b: load_benchmark(b, cache_dir=tmp_path, force_offline=True)
        for b in SUPPORTED_BENCHMARKS
    }
    answers = {it.prompt: it.answer for its in items_by_bench.values() for it in its}

    serial_calls = {"n": 0}
    batch_calls = {"n": 0, "seen_sizes": []}

    def serial_fn(prompt):
        serial_calls["n"] += 1
        return answers.get(prompt, "")

    def batch_fn(prompts):
        batch_calls["n"] += 1
        batch_calls["seen_sizes"].append(len(prompts))
        return [answers.get(p, "") for p in prompts]

    result = run_anchor_eval(
        serial_fn,
        benchmarks=SUPPORTED_BENCHMARKS,
        per_benchmark=3,
        cache_dir=tmp_path,
        batch_model_fn=batch_fn,
    )
    assert serial_calls["n"] == 0, "serial fn must not be called when batch_model_fn succeeds"
    assert batch_calls["n"] == 1
    assert batch_calls["seen_sizes"] == [3 * len(SUPPORTED_BENCHMARKS)]
    assert result["anchor_score"] == pytest.approx(1.0)


def test_run_anchor_eval_batch_failure_falls_back_to_serial(tmp_path):
    """If batch_model_fn raises, the eval must fall back to serial model_fn
    rather than crashing the cycle."""
    items_by_bench = {
        b: load_benchmark(b, cache_dir=tmp_path, force_offline=True)
        for b in SUPPORTED_BENCHMARKS
    }
    answers = {it.prompt: it.answer for its in items_by_bench.values() for it in its}

    serial_calls = {"n": 0}

    def serial_fn(prompt):
        serial_calls["n"] += 1
        return answers.get(prompt, "")

    def bad_batch(prompts):
        raise RuntimeError("vLLM OOM")

    result = run_anchor_eval(
        serial_fn,
        benchmarks=SUPPORTED_BENCHMARKS,
        per_benchmark=2,
        cache_dir=tmp_path,
        batch_model_fn=bad_batch,
    )
    assert serial_calls["n"] == 2 * len(SUPPORTED_BENCHMARKS)
    assert result["anchor_score"] == pytest.approx(1.0)


def test_run_anchor_eval_null_model_scores_zero(tmp_path):
    result = run_anchor_eval(
        lambda _p: "",
        benchmarks=SUPPORTED_BENCHMARKS,
        per_benchmark=3,
        cache_dir=tmp_path,
    )
    assert result["anchor_score"] == 0.0


def test_detect_verifier_capture_triggers_on_divergence():
    # Internal +2% but anchor -1% → alarm.
    assert detect_verifier_capture(internal_delta=0.02, anchor_delta=-0.01) is True
    # Both improving → no alarm.
    assert detect_verifier_capture(internal_delta=0.02, anchor_delta=0.01) is False
    # Internal flat → no alarm.
    assert detect_verifier_capture(internal_delta=0.0, anchor_delta=-0.02) is False
    # Both regressing → no alarm (it's not verifier capture, it's just regression).
    assert detect_verifier_capture(internal_delta=-0.01, anchor_delta=-0.02) is False


def test_fire_capture_alarm_writes_log(tmp_path):
    log = tmp_path / "update-log.txt"
    fire_capture_alarm(cycle=7, internal_delta=0.03, anchor_delta=-0.02, log_path=log)
    assert log.exists()
    content = log.read_text()
    assert "VERIFIER-CAPTURE-ALARM" in content
    assert "cycle=7" in content


def test_seed_adversarial_bank_from_external(tmp_path):
    # Minimal stub bank with the same .append signature as AdversarialBank.
    class StubBank:
        def __init__(self):
            self.entries = []

        def append(self, *, problem_id, candidate, domain, problem_ctx=None,
                   cycle=-1, reason=""):
            self.entries.append({
                "problem_id": problem_id,
                "candidate": candidate,
                "domain": domain,
                "problem_ctx": dict(problem_ctx or {}),
                "cycle": cycle,
                "reason": reason,
            })

    # Pre-populate cache with the offline fixture.
    for b in ("humaneval", "mbpp"):
        load_benchmark(b, cache_dir=tmp_path, force_offline=True)

    bank = StubBank()
    added = seed_adversarial_bank_from_external(bank, per_benchmark=3, cache_dir=tmp_path, cycle=0)
    assert added > 0
    assert added == len(bank.entries)
    assert all(e["domain"] == "code" for e in bank.entries)
    assert all(e["problem_ctx"]["source"] == "external_benchmark" for e in bank.entries)
    # Candidates must differ from any pristine canonical in the fixture.
    for e in bank.entries:
        assert e["candidate"]
        assert "return a + b" not in e["candidate"] or "return a - b" in e["candidate"] or "+ 1" in e["candidate"]
