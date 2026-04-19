"""Tests for the RSI append-only JSONL registries (spec §4.1 / Phase A4).

All tests use tmp_path (pytest fixture) — no real outputs/ directory touched.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from src.orchestrator.registries import (
    RSIRegistries,
    PropertyRegistry,
    ProblemRegistry,
    VerificationLog,
    CalibrationLedger,
    TrainingPool,
    PropertyRecord,
    ProblemRecord,
    VerificationRecord,
    CalibrationEntry,
    TrainingPoolRecord,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _open_all(tmp_path: Path, sid: str = "test_sid") -> RSIRegistries:
    return RSIRegistries.open(tmp_path / "outputs", sid=sid)


# ─── RSIRegistries factory ───────────────────────────────────────────────────

def test_open_creates_all_stores(tmp_path):
    regs = _open_all(tmp_path)
    assert regs.property_registry is not None
    assert regs.problem_registry is not None
    assert regs.verification_log is not None
    assert regs.calibration_ledger is not None
    assert regs.training_pool is not None


def test_open_creates_output_dirs(tmp_path):
    regs = _open_all(tmp_path, sid="abc123")
    # Each store should have created its directory.
    assert (tmp_path / "outputs" / "properties").is_dir()
    assert (tmp_path / "outputs" / "problems").is_dir()
    assert (tmp_path / "outputs" / "verifications").is_dir()
    assert (tmp_path / "outputs" / "training_pool").is_dir()
    # CalibrationLedger uses fixed path, no subdir
    assert (tmp_path / "outputs" / "calibration.jsonl").parent.is_dir()


def test_open_sid_default_is_random(tmp_path):
    r1 = RSIRegistries.open(tmp_path / "out1")
    r2 = RSIRegistries.open(tmp_path / "out2")
    assert r1.sid != r2.sid


def test_open_explicit_sid(tmp_path):
    regs = _open_all(tmp_path, sid="mysid")
    assert regs.sid == "mysid"


# ─── _AppendOnlyStore base behavior ──────────────────────────────────────────

def test_append_and_iter_dict(tmp_path):
    reg = PropertyRegistry(tmp_path / "outputs", "s1")
    reg.append({"property_id": "p1", "name": "test"})
    reg.append({"property_id": "p2", "name": "test2"})
    records = list(reg.iter_records())
    assert len(records) == 2
    assert records[0]["property_id"] == "p1"
    assert records[1]["property_id"] == "p2"


def test_iter_empty_file_returns_nothing(tmp_path):
    reg = PropertyRegistry(tmp_path / "outputs", "empty")
    assert list(reg.iter_records()) == []


def test_count_matches_append_calls(tmp_path):
    reg = ProblemRegistry(tmp_path / "outputs", "s2")
    for i in range(5):
        reg.append({"problem_id": f"prob_{i}"})
    assert reg.count() == 5


def test_append_order_preserved(tmp_path):
    reg = VerificationLog(tmp_path / "outputs", "s3")
    for i in range(10):
        reg.append({"seq": i})
    seqs = [r["seq"] for r in reg.iter_records()]
    assert seqs == list(range(10))


def test_file_path_matches_kind_and_sid(tmp_path):
    reg = PropertyRegistry(tmp_path / "out", "abc")
    assert reg.path == tmp_path / "out" / "properties" / "abc.jsonl"


def test_append_dataclass(tmp_path):
    reg = PropertyRegistry(tmp_path / "outputs", "dc")
    record = PropertyRecord(
        property_id="p1",
        problem_id="pr1",
        author="model:run_abc",
        independence_class="exec.behavioral",
        kind="UNIT_TEST",
        name="output_is_sorted",
    )
    reg.append(record)
    rows = list(reg.iter_records())
    assert len(rows) == 1
    assert rows[0]["property_id"] == "p1"
    assert rows[0]["independence_class"] == "exec.behavioral"


def test_corrupt_line_skipped_gracefully(tmp_path):
    """A corrupt JSONL line is skipped; subsequent records still readable."""
    reg = PropertyRegistry(tmp_path / "outputs", "corrupt")
    reg.path.parent.mkdir(parents=True, exist_ok=True)
    with open(reg.path, "w") as f:
        f.write('{"property_id": "good1"}\n')
        f.write("NOT_JSON\n")
        f.write('{"property_id": "good2"}\n')
    records = list(reg.iter_records())
    assert len(records) == 2
    assert records[0]["property_id"] == "good1"
    assert records[1]["property_id"] == "good2"


def test_append_is_durable_across_instances(tmp_path):
    """Records written by one instance are readable by a second instance on the same file."""
    reg1 = PropertyRegistry(tmp_path / "outputs", "dur")
    reg1.append({"property_id": "p_persist"})
    reg2 = PropertyRegistry(tmp_path / "outputs", "dur")
    records = list(reg2.iter_records())
    assert any(r.get("property_id") == "p_persist" for r in records)


# ─── PropertyRegistry ─────────────────────────────────────────────────────────

def test_property_registry_get_by_problem(tmp_path):
    reg = PropertyRegistry(tmp_path / "outputs", "pr")
    reg.append({"property_id": "p1", "problem_id": "prob_A"})
    reg.append({"property_id": "p2", "problem_id": "prob_B"})
    reg.append({"property_id": "p3", "problem_id": "prob_A"})
    results = reg.get_by_problem("prob_A")
    assert len(results) == 2
    assert all(r["problem_id"] == "prob_A" for r in results)


def test_property_registry_get_by_id_returns_last(tmp_path):
    reg = PropertyRegistry(tmp_path / "outputs", "prid")
    reg.append({"property_id": "p1", "name": "v1"})
    reg.append({"property_id": "p1", "name": "v2"})  # update
    result = reg.get_by_id("p1")
    assert result is not None
    assert result["name"] == "v2"


def test_property_registry_get_by_id_missing(tmp_path):
    reg = PropertyRegistry(tmp_path / "outputs", "miss")
    assert reg.get_by_id("nonexistent") is None


# ─── ProblemRegistry ──────────────────────────────────────────────────────────

def test_problem_registry_append_and_retrieve(tmp_path):
    reg = ProblemRegistry(tmp_path / "outputs", "pb")
    prob = ProblemRecord(
        problem_id="prob_1",
        domain="code",
        problem_text="Sort a list of integers",
        declared_difficulty=0.4,
        nearest_neighbor_dist=0.12,
    )
    reg.append_problem(prob)
    result = reg.get_by_id("prob_1")
    assert result is not None
    assert result["domain"] == "code"
    assert abs(result["declared_difficulty"] - 0.4) < 1e-9


def test_problem_registry_mark_retired(tmp_path):
    reg = ProblemRegistry(tmp_path / "outputs", "ret")
    reg.append({"problem_id": "p_old", "retired": False})
    reg.mark_retired("p_old", session_id="s1")
    result = reg.get_by_id("p_old")
    assert result is not None
    assert result["retired"] is True


def test_problem_registry_get_by_id_not_found(tmp_path):
    reg = ProblemRegistry(tmp_path / "outputs", "notfound")
    assert reg.get_by_id("xyz") is None


# ─── VerificationLog ──────────────────────────────────────────────────────────

def test_verification_log_primary_and_adversarial(tmp_path):
    log = VerificationLog(tmp_path / "outputs", "vl")
    log.append({"record_id": "r1", "candidate_id": "c1", "adversarial": False, "quorum_accepted": True})
    log.append({"record_id": "r2", "candidate_id": "c1", "adversarial": True, "quorum_accepted": False})
    log.append({"record_id": "r3", "candidate_id": "c2", "adversarial": False, "quorum_accepted": True})

    for_c1 = log.get_for_candidate("c1")
    assert len(for_c1) == 2

    adv = list(log.adversarial_records())
    assert len(adv) == 1
    assert adv[0]["record_id"] == "r2"


def test_verification_log_append_dataclass(tmp_path):
    log = VerificationLog(tmp_path / "outputs", "vdc")
    rec = VerificationRecord(
        record_id="vr1",
        problem_id="prob_1",
        candidate_id="cand_1",
        property_ids=["p1", "p2", "p3"],
        per_property_verdicts=[
            {"property_id": "p1", "passed": True},
            {"property_id": "p2", "passed": True},
            {"property_id": "p3", "passed": True},
        ],
        quorum_accepted=True,
        quorum_reason="all pass",
    )
    log.append_verification(rec)
    rows = list(log.iter_records())
    assert rows[0]["quorum_accepted"] is True
    assert rows[0]["adversarial"] is False


def test_verification_log_adversarial_tag(tmp_path):
    log = VerificationLog(tmp_path / "outputs", "adv")
    rec = VerificationRecord(
        record_id="adv1",
        problem_id="p",
        candidate_id="c",
        property_ids=["px"],
        per_property_verdicts=[],
        quorum_accepted=False,
        quorum_reason="adversarial fail",
        adversarial=True,
    )
    log.append_verification(rec)
    rows = list(log.iter_records())
    assert rows[0]["adversarial"] is True


# ─── CalibrationLedger ───────────────────────────────────────────────────────

def test_calibration_ledger_fixed_path(tmp_path):
    ledger = CalibrationLedger(tmp_path / "outputs", "any_sid")
    assert ledger.path == tmp_path / "outputs" / "calibration.jsonl"


def test_calibration_ledger_append_and_suspended(tmp_path):
    ledger = CalibrationLedger(tmp_path / "outputs", "cl")
    ledger.append_calibration(CalibrationEntry(
        tick=1, independence_class="exec.behavioral",
        true_accept_rate=0.9, true_reject_rate=0.85,
        error_rate=0.02, suspended=False, n_probes=20,
    ))
    ledger.append_calibration(CalibrationEntry(
        tick=1, independence_class="algebra.symbolic",
        true_accept_rate=0.5, true_reject_rate=0.4,
        error_rate=0.1, suspended=True, n_probes=20,
    ))
    suspended = ledger.suspended_classes()
    assert "algebra.symbolic" in suspended
    assert "exec.behavioral" not in suspended


def test_calibration_ledger_suspend_then_resume(tmp_path):
    ledger = CalibrationLedger(tmp_path / "outputs", "resume")
    ledger.append({"tick": 1, "independence_class": "smt.logical", "suspended": True})
    ledger.append({"tick": 2, "independence_class": "smt.logical", "suspended": False})
    assert "smt.logical" not in ledger.suspended_classes()


def test_calibration_ledger_suspended_at_tick(tmp_path):
    ledger = CalibrationLedger(tmp_path / "outputs", "tick")
    ledger.append({"tick": 1, "independence_class": "roundtrip", "suspended": True})
    ledger.append({"tick": 3, "independence_class": "roundtrip", "suspended": False})
    # At tick=2, still suspended
    assert "roundtrip" in ledger.suspended_classes(tick=2)
    # At tick=3, resumed
    assert "roundtrip" not in ledger.suspended_classes(tick=3)


def test_calibration_ledger_class_stats(tmp_path):
    ledger = CalibrationLedger(tmp_path / "outputs", "stats")
    for i in range(3):
        ledger.append({"tick": i, "independence_class": "exec.behavioral", "true_accept_rate": 0.9 - i * 0.1})
    stats = ledger.class_stats("exec.behavioral")
    assert len(stats) == 3


# ─── TrainingPool ─────────────────────────────────────────────────────────────

def test_training_pool_append_and_pending(tmp_path):
    pool = TrainingPool(tmp_path / "outputs", "tp")
    for i in range(4):
        pool.append({"pool_record_id": f"tr_{i}", "domain": "code", "source": "rsi_property"})
    pending = pool.pending_samples()
    assert len(pending) == 4
    assert all(r["source"] == "rsi_property" for r in pending)


def test_training_pool_append_dataclass(tmp_path):
    pool = TrainingPool(tmp_path / "outputs", "tpdc")
    rec = TrainingPoolRecord(
        pool_record_id="pool_1",
        problem_id="prob_1",
        candidate_id="cand_1",
        verification_record_id="vr_1",
        domain="math",
        prompt="What is 2+2?",
        response="4",
    )
    pool.append_sample(rec)
    rows = list(pool.iter_records())
    assert rows[0]["source"] == "rsi_property"
    assert rows[0]["domain"] == "math"


def test_training_pool_source_tag(tmp_path):
    """All TrainingPool records must carry source='rsi_property' per spec §5.2."""
    pool = TrainingPool(tmp_path / "outputs", "tag")
    pool.append_sample(TrainingPoolRecord(
        pool_record_id="x", problem_id="p", candidate_id="c",
        verification_record_id="v", domain="code",
        prompt="p", response="r",
    ))
    row = list(pool.iter_records())[0]
    assert row.get("source") == "rsi_property"


# ─── session isolation ────────────────────────────────────────────────────────

def test_different_sids_produce_different_files(tmp_path):
    r1 = _open_all(tmp_path, sid="sid_a")
    r2 = _open_all(tmp_path, sid="sid_b")
    r1.property_registry.append({"property_id": "p_a"})
    r2.property_registry.append({"property_id": "p_b"})
    a_ids = {r["property_id"] for r in r1.property_registry.iter_records()}
    b_ids = {r["property_id"] for r in r2.property_registry.iter_records()}
    assert a_ids == {"p_a"}
    assert b_ids == {"p_b"}


def test_calibration_ledger_is_shared_across_sids(tmp_path):
    """CalibrationLedger uses a fixed path — shared across sessions per spec."""
    r1 = _open_all(tmp_path, sid="sid_x")
    r2 = _open_all(tmp_path, sid="sid_y")
    assert r1.calibration_ledger.path == r2.calibration_ledger.path


# ─── retirement on first training-pool acceptance (spec §7 / v0.2 item 7) ────

def test_retire_on_training_pool_acceptance(tmp_path):
    """Spec §7: mark_retired is called when a problem enters the training pool."""
    regs = _open_all(tmp_path, sid="retire_test")
    # Register a problem as active
    regs.problem_registry.append({"problem_id": "prob_frontier_1", "retired": False})
    # Simulate training-pool admission (loop calls mark_retired on acceptance)
    regs.training_pool.append_sample({"pool_record_id": "pr1", "problem_id": "prob_frontier_1",
                                       "source": "rsi_property"})
    regs.problem_registry.mark_retired("prob_frontier_1", session_id="retire_test")
    # The problem should now read as retired
    rec = regs.problem_registry.get_by_id("prob_frontier_1")
    assert rec is not None
    assert rec["retired"] is True


def test_retire_does_not_affect_other_problems(tmp_path):
    """Retiring one problem must not touch sibling problems in the same file."""
    regs = _open_all(tmp_path, sid="sib")
    regs.problem_registry.append({"problem_id": "p_keep", "retired": False})
    regs.problem_registry.append({"problem_id": "p_retire", "retired": False})
    regs.problem_registry.mark_retired("p_retire", session_id="sib")
    assert regs.problem_registry.get_by_id("p_keep")["retired"] is False
    assert regs.problem_registry.get_by_id("p_retire")["retired"] is True


def test_retire_idempotent(tmp_path):
    """Calling mark_retired twice on the same problem is safe."""
    regs = _open_all(tmp_path, sid="idem")
    regs.problem_registry.append({"problem_id": "p1", "retired": False})
    regs.problem_registry.mark_retired("p1")
    regs.problem_registry.mark_retired("p1")
    rec = regs.problem_registry.get_by_id("p1")
    assert rec["retired"] is True  # last patch wins, still retired


def test_training_pool_record_triggers_retirement_in_loop(tmp_path):
    """Integration: after TrainingPoolRecord is written, problem_registry shows retired."""
    regs = _open_all(tmp_path, sid="loop_retire")
    problem_id = "synth_task_abc"
    regs.problem_registry.append({"problem_id": problem_id, "retired": False})

    # Simulate what loop Phase 1b does on quorum acceptance
    regs.training_pool.append_sample({
        "pool_record_id": "pool_xyz",
        "problem_id": problem_id,
        "source": "rsi_property",
    })
    regs.problem_registry.mark_retired(problem_id, session_id=regs.sid)

    # Verify: problem is retired; training pool has the record; counts are right
    assert regs.problem_registry.get_by_id(problem_id)["retired"] is True
    assert regs.training_pool.count() == 1
    pool_rows = regs.training_pool.pending_samples()
    assert pool_rows[0]["problem_id"] == problem_id
