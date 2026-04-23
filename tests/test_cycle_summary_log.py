"""Test outputs/cycle_summary.jsonl emission.

Constructs a CycleResult in-memory, binds _emit_cycle_summary_log to a
lightweight stub, and verifies one denormalized row lands with the
expected fields. Skips the full run() loop which needs a model.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.orchestrator.loop import ImprovementLoop, CycleResult
from src.utils.config import OrchestratorConfig
from src.utils.structured_logs import SINK_FILENAMES


REQUIRED_FIELDS = {
    "cycle", "start_ts", "end_ts", "total_time_s",
    "propose_s", "solve_s", "verify_s", "train_s", "heldout_s", "anchor_s",
    "accepts", "held_out_score",
    "paired_delta", "paired_delta_se", "rho", "mde_80",
    "best_checkpoint_cycle", "pending_best_streak",
    "any_alarm",
}


def _stub(tmp_path: Path, enabled: bool = True) -> SimpleNamespace:
    ocfg = OrchestratorConfig()
    ocfg.output_dir = tmp_path
    ocfg.structured_observability_enabled = enabled
    stub = SimpleNamespace()
    stub.config = SimpleNamespace(orchestrator=ocfg)
    stub._best_checkpoint_cycle = 4
    stub._pending_best_streak = 2
    stub._emit_cycle_summary_log = (
        ImprovementLoop._emit_cycle_summary_log.__get__(stub)
    )
    return stub


def test_cycle_summary_log_writes_denormalized_row(tmp_path: Path):
    stub = _stub(tmp_path, enabled=True)
    cr = CycleResult(cycle=7)
    cr.duration = 312.5
    cr.timestamp = 1_700_000_000.0
    cr.phase_times = {
        "generate": 42.1, "solve": 15.0, "verify": 3.5,
        "train": 90.0, "eval": 160.0, "anchor": 1.9,
    }
    cr.samples_verified = 11
    cr.eval_score = 0.42
    cr.paired_delta = 0.018
    cr.paired_delta_se = 0.006
    cr.paired_delta_rho = 0.91
    cr.paired_delta_mde_80 = 0.012
    cr.mode_collapse_detected = True
    cr.heldout_eval_kind = "full"

    stub._emit_cycle_summary_log(7, cr)
    path = tmp_path / SINK_FILENAMES["cycle_summary"]
    assert path.exists()
    row = json.loads(path.read_text().strip())
    assert REQUIRED_FIELDS.issubset(row.keys())
    assert row["cycle"] == 7
    assert row["total_time_s"] == pytest.approx(312.5)
    assert row["propose_s"] == pytest.approx(42.1)
    assert row["train_s"] == pytest.approx(90.0)
    assert row["heldout_s"] == pytest.approx(160.0)
    assert row["accepts"] == 11
    assert row["held_out_score"] == pytest.approx(0.42)
    assert row["paired_delta"] == pytest.approx(0.018)
    assert row["mde_80"] == pytest.approx(0.012)
    assert row["rho"] == pytest.approx(0.91)
    assert row["best_checkpoint_cycle"] == 4
    assert row["pending_best_streak"] == 2
    # mode_collapse_detected flips any_alarm.
    assert row["any_alarm"] is True
    assert row["mode_collapse_detected"] is True


def test_cycle_summary_log_disabled_writes_nothing(tmp_path: Path):
    stub = _stub(tmp_path, enabled=False)
    cr = CycleResult(cycle=1)
    stub._emit_cycle_summary_log(1, cr)
    assert not (tmp_path / SINK_FILENAMES["cycle_summary"]).exists()


def test_cycle_summary_log_no_alarm_flag_off(tmp_path: Path):
    stub = _stub(tmp_path, enabled=True)
    cr = CycleResult(cycle=2)
    cr.duration = 10.0
    stub._emit_cycle_summary_log(2, cr)
    row = json.loads(
        (tmp_path / SINK_FILENAMES["cycle_summary"]).read_text().strip()
    )
    assert row["any_alarm"] is False
