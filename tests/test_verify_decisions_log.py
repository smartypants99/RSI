"""Test outputs/verify_decisions.jsonl emission.

Builds a minimal VerificationRecord in-memory and invokes
property_engine._emit_verify_decision — avoids the cost of spinning up
a full sandbox + Property fan-out.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.verifier import property_engine as pe
from src.verifier.property_engine import (
    PropertyVerdict, VerificationRecord,
)
from src.utils.config import OrchestratorConfig
from src.utils.structured_logs import SINK_FILENAMES


REQUIRED_FIELDS = {
    "cycle", "problem_id", "candidate_idx",
    "n_backends_tried", "per_backend",
    "accepted", "accept_policy", "verdict_warnings",
    "independence_classes_count",
}


@pytest.fixture(autouse=True)
def _reset_pe_obs():
    """Reset module-level observability state between tests."""
    pe.set_observability_config(None)
    pe.set_observability_context(cycle=None, candidate_idx=None)
    yield
    pe.set_observability_config(None)
    pe.set_observability_context(cycle=None, candidate_idx=None)


def _make_record(accepted: bool = True) -> VerificationRecord:
    pv1 = PropertyVerdict(
        property_id="p1", verdict="PASS",
        independence_class="algebraic.substitution",
        author="builtin:z3_backend", name="prop1",
        duration_ms=12, backend="z3",
    )
    pv2 = PropertyVerdict(
        property_id="p2", verdict="PASS",
        independence_class="behavioral.tests",
        author="builtin:simulator_backend", name="prop2",
        duration_ms=5, backend="simulator",
    )
    pv3 = PropertyVerdict(
        property_id="p3", verdict="FAIL",
        independence_class="algebraic.substitution",
        author="model:run1", name="prop3", reason="mismatch",
        duration_ms=30, backend="python",
    )
    return VerificationRecord(
        record_id="ver_abc",
        problem_id="prob_001",
        candidate_hash="deadbeef",
        per_property=[pv1, pv2, pv3],
        accepted=accepted,
        quorum_n=3, pass_count=2, fail_count=1, error_count=0,
        distinct_classes=("algebraic.substitution", "behavioral.tests"),
        reject_reason="" if accepted else "some reason",
        created_at=0.0,
        accept_policy="majority",
        verdict_warnings=("any_fail",) if accepted else (),
    )


def test_verify_decisions_log_writes_valid_record(tmp_path: Path):
    ocfg = OrchestratorConfig()
    ocfg.output_dir = tmp_path
    ocfg.structured_observability_enabled = True
    pe.set_observability_config(ocfg)
    pe.set_observability_context(cycle=7, candidate_idx=2)

    rec = _make_record(accepted=True)
    pe._emit_verify_decision(rec)

    path = tmp_path / SINK_FILENAMES["verify_decisions"]
    assert path.exists()
    row = json.loads(path.read_text().strip())
    assert REQUIRED_FIELDS.issubset(row.keys()), (
        f"missing: {REQUIRED_FIELDS - set(row.keys())}"
    )
    assert row["cycle"] == 7
    assert row["candidate_idx"] == 2
    assert row["problem_id"] == "prob_001"
    assert row["accepted"] is True
    assert row["accept_policy"] == "majority"
    assert row["verdict_warnings"] == ["any_fail"]
    # 3 distinct backends (z3, simulator, python) were tried.
    assert row["n_backends_tried"] == 3
    assert len(row["per_backend"]) == 3
    assert {b["backend"] for b in row["per_backend"]} == {"z3", "simulator", "python"}
    # independence_classes_count reflects DISTINCT PASS classes.
    assert row["independence_classes_count"] == 2


def test_verify_decisions_log_disabled_writes_nothing(tmp_path: Path):
    ocfg = OrchestratorConfig()
    ocfg.output_dir = tmp_path
    ocfg.structured_observability_enabled = False
    pe.set_observability_config(ocfg)

    pe._emit_verify_decision(_make_record())
    assert not (tmp_path / SINK_FILENAMES["verify_decisions"]).exists()


def test_verify_decisions_log_no_cfg_installed_is_noop(tmp_path: Path):
    # Module-level _OBS_CFG never set → emit is a silent no-op.
    pe._emit_verify_decision(_make_record())
    # No file anywhere was written; the function returned cleanly.
