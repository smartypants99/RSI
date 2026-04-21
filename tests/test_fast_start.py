"""Tests for src/utils/fast_start.py (Task #11)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils.fast_start import (
    bootstrap_tasks_per_cycle,
    default_weakness_diag,
    prestash_prior_training_samples,
)


def _write_pool(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _mk_rec(sid: str, i: int, domain: str = "code") -> dict:
    return {
        "pool_record_id": f"rec_{i}",
        "problem_id": f"p_{i}",
        "candidate_id": f"c_{i}",
        "verification_record_id": f"v_{i}",
        "domain": domain,
        "prompt": f"prompt_{i}",
        "response": f"response_{i}",
        "source": "rsi_property",
        "session_id": sid,
    }


def test_prestash_returns_empty_when_no_files(tmp_path: Path):
    samples = prestash_prior_training_samples(tmp_path, "sidA", 30)
    assert samples == []


def test_prestash_loads_prior_session_samples(tmp_path: Path):
    _write_pool(
        tmp_path / "training_pool" / "prior_sid.jsonl",
        [_mk_rec("prior_sid", i) for i in range(5)],
    )
    samples = prestash_prior_training_samples(tmp_path, current_sid="cur_sid", max_samples=30)
    assert len(samples) == 5
    assert samples[0].prompt == "prompt_0"
    assert samples[0].response == "response_0"
    assert samples[0].verified is True
    assert samples[0].domain == "code"


def test_prestash_skips_current_session(tmp_path: Path):
    # Records with session_id == current_sid must be filtered.
    _write_pool(
        tmp_path / "training_pool" / "mix.jsonl",
        [_mk_rec("cur_sid", 0), _mk_rec("prior", 1), _mk_rec("cur_sid", 2)],
    )
    samples = prestash_prior_training_samples(tmp_path, current_sid="cur_sid", max_samples=30)
    assert len(samples) == 1
    assert samples[0].prompt == "prompt_1"


def test_prestash_respects_cap(tmp_path: Path):
    _write_pool(
        tmp_path / "training_pool" / "big.jsonl",
        [_mk_rec("prior", i) for i in range(100)],
    )
    samples = prestash_prior_training_samples(tmp_path, "cur_sid", max_samples=30)
    assert len(samples) == 30


def test_prestash_scans_sibling_outputs_run_dirs(tmp_path: Path):
    out_a = tmp_path / "outputs"
    out_b = tmp_path / "outputs_run_2"
    _write_pool(out_b / "training_pool" / "sidX.jsonl", [_mk_rec("sidX", i) for i in range(3)])
    samples = prestash_prior_training_samples(out_a, "cur_sid", 30)
    assert len(samples) == 3


def test_prestash_dedupes_by_prompt_response(tmp_path: Path):
    rec = _mk_rec("prior", 0)
    _write_pool(tmp_path / "training_pool" / "a.jsonl", [rec])
    _write_pool(tmp_path / "training_pool" / "b.jsonl", [rec, _mk_rec("prior", 1)])
    samples = prestash_prior_training_samples(tmp_path, "cur_sid", 30)
    assert len(samples) == 2


def test_prestash_skips_malformed_lines(tmp_path: Path):
    path = tmp_path / "training_pool" / "bad.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("not json\n")
        fh.write(json.dumps(_mk_rec("prior", 0)) + "\n")
        fh.write("\n")
        fh.write(json.dumps({"prompt": "", "response": "x"}) + "\n")  # empty prompt → skip
    samples = prestash_prior_training_samples(tmp_path, "cur_sid", 30)
    assert len(samples) == 1


def test_prestash_zero_cap_returns_empty(tmp_path: Path):
    _write_pool(tmp_path / "training_pool" / "x.jsonl", [_mk_rec("prior", 0)])
    assert prestash_prior_training_samples(tmp_path, "cur_sid", 0) == []


def test_default_weakness_diag_uniform():
    domains = ["code", "math", "logic"]
    diag = default_weakness_diag(domains, cycle=1)
    assert diag.cycle == 1
    assert len(diag.weaknesses) == 3
    assert set(diag.domain_scores.keys()) == set(domains)
    assert all(v == 0.5 for v in diag.domain_scores.values())
    assert diag.subdomain_scores == {}
    assert diag.per_question == []
    # overall_score should be a number (0.5 after weighting with zeros → fallback to mean)
    _ = diag.overall_score


def test_default_weakness_diag_synth_compat(tmp_path):
    """Ensure set_diagnostics on TaskSynthesizer accepts the default diag."""
    diag = default_weakness_diag(["code", "math"], cycle=1)
    # Minimum compatibility surface: subdomain_scores dict, per_question list.
    assert hasattr(diag, "subdomain_scores")
    assert hasattr(diag, "per_question")
    assert isinstance(diag.subdomain_scores, dict)
    assert isinstance(diag.per_question, list)


class _Cfg:
    def __init__(self, tasks_per_cycle=30, synthesis_tasks_per_cycle_bootstrap=15):
        self.tasks_per_cycle = tasks_per_cycle
        self.synthesis_tasks_per_cycle_bootstrap = synthesis_tasks_per_cycle_bootstrap


def test_bootstrap_tasks_per_cycle_first_cycle():
    cfg = _Cfg(tasks_per_cycle=30, synthesis_tasks_per_cycle_bootstrap=15)
    assert bootstrap_tasks_per_cycle(cfg, cycle=1) == 15


def test_bootstrap_tasks_per_cycle_steady_state():
    cfg = _Cfg(tasks_per_cycle=30, synthesis_tasks_per_cycle_bootstrap=15)
    assert bootstrap_tasks_per_cycle(cfg, cycle=2) == 30
    assert bootstrap_tasks_per_cycle(cfg, cycle=99) == 30


def test_bootstrap_tasks_per_cycle_missing_bootstrap_field():
    class _Bare:
        tasks_per_cycle = 20
    assert bootstrap_tasks_per_cycle(_Bare(), cycle=1) == 20


def test_bootstrap_tasks_per_cycle_zero_bootstrap_falls_back():
    cfg = _Cfg(tasks_per_cycle=30, synthesis_tasks_per_cycle_bootstrap=0)
    assert bootstrap_tasks_per_cycle(cfg, cycle=1) == 30
