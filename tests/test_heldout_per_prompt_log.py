"""Test outputs/heldout_per_prompt.jsonl emission.

Calls ImprovementLoop._emit_heldout_per_prompt_log directly with a
synthetic per_question list — avoids the ~20s vLLM / diagnostics init
cost of standing up a full orchestrator.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.orchestrator.loop import ImprovementLoop as _Orch
from src.utils.config import OrchestratorConfig
from src.utils.structured_logs import SINK_FILENAMES


REQUIRED_FIELDS = {
    "cycle", "heldout_kind", "prompt_id", "domain", "subdomain",
    "base_score", "base_correct",
    "trained_score", "trained_correct", "trained_completion_length",
    "score_delta", "eval_time_ms",
}


class _FakeBaseCache:
    """Minimal stand-in for BaseHeldoutCache — just .get()."""
    def __init__(self, entries: dict):
        self._entries = entries

    def get(self, prompt, expected):
        return self._entries.get((prompt, expected or ""))


def _orchestrator_stub(tmp_path: Path, base_cache=None) -> ImprovementLoop:
    """Build an instance-like object with only what the emit helper reads."""
    ocfg = OrchestratorConfig()
    ocfg.output_dir = tmp_path
    ocfg.structured_observability_enabled = True
    # Bind the emit helper to a minimal instance.
    stub = SimpleNamespace()
    stub.config = SimpleNamespace(orchestrator=ocfg)
    stub._heldout_base_cache = base_cache
    stub._emit_heldout_per_prompt_log = (
        _Orch._emit_heldout_per_prompt_log.__get__(stub)
    )
    return stub


def test_heldout_per_prompt_log_writes_joined_records(tmp_path: Path):
    base = _FakeBaseCache({
        ("what is 2+2?", "4"): {"correct": True, "score": 1.0},
        ("what is 3+5?", "8"): {"correct": False, "score": 0.3},
    })
    stub = _orchestrator_stub(tmp_path, base_cache=base)

    per_q = [
        {
            "question_id": "q1", "question": "what is 2+2?", "expected": "4",
            "domain": "math", "subdomain": "arith",
            "correct": True, "score": 1.0,
        },
        {
            "question_id": "q2", "question": "what is 3+5?", "expected": "8",
            "domain": "math", "subdomain": "arith",
            "correct": True, "score": 0.9,  # trained improved over base
        },
        {
            # Not in base cache → base_score/base_correct None, delta None.
            "question_id": "q3", "question": "novel?", "expected": "yes",
            "domain": "code", "subdomain": "implementation",
            "correct": False, "score": 0.0,
        },
    ]
    stub._emit_heldout_per_prompt_log(
        cycle=4, heldout_kind="full", per_question=per_q,
    )
    path = tmp_path / SINK_FILENAMES["heldout_per_prompt"]
    assert path.exists()
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 3
    recs = [json.loads(l) for l in lines]
    for r in recs:
        assert REQUIRED_FIELDS.issubset(r.keys()), (
            f"missing: {REQUIRED_FIELDS - set(r.keys())}"
        )
        assert r["cycle"] == 4
        assert r["heldout_kind"] == "full"
    # Second record: trained improved from 0.3 → 0.9 (delta ≈ 0.6).
    r2 = next(r for r in recs if r["prompt_id"] == "q2")
    assert r2["base_score"] == pytest.approx(0.3)
    assert r2["trained_score"] == pytest.approx(0.9)
    assert r2["score_delta"] == pytest.approx(0.6)
    # Third record has no base entry → deltas None.
    r3 = next(r for r in recs if r["prompt_id"] == "q3")
    assert r3["base_score"] is None
    assert r3["score_delta"] is None


def test_heldout_per_prompt_log_disabled_writes_nothing(tmp_path: Path):
    ocfg = OrchestratorConfig()
    ocfg.output_dir = tmp_path
    ocfg.structured_observability_enabled = False
    stub = SimpleNamespace()
    stub.config = SimpleNamespace(orchestrator=ocfg)
    stub._heldout_base_cache = None
    stub._emit_heldout_per_prompt_log = (
        _Orch._emit_heldout_per_prompt_log.__get__(stub)
    )
    stub._emit_heldout_per_prompt_log(
        cycle=1, heldout_kind="quick",
        per_question=[{"question": "x", "correct": True, "score": 1.0}],
    )
    assert not (tmp_path / SINK_FILENAMES["heldout_per_prompt"]).exists()
