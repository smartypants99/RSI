"""Test ImprovementLoop._auto_diagnose_cycle wiring.

Drops synthetic jsonl sinks into a temp outputs dir, binds the helper to a
lightweight stub (no model, no full run() loop), invokes it for a cycle,
and asserts a structured row lands in outputs/auto_diagnosis.jsonl with
the expected schema.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from src.orchestrator.loop import ImprovementLoop, CycleResult
from src.utils.config import OrchestratorConfig


def _write_synth_logs(out_dir: Path, cycle: int) -> None:
    """Populate the minimum jsonl rows analyze_cycle.py expects."""
    cycle_dir = out_dir / f"cycle_{cycle}"
    cycle_dir.mkdir(parents=True, exist_ok=True)

    def _dump(name: str, rows: list[dict]) -> None:
        with (cycle_dir / name).open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    # Training — enough to populate loss_initial / loss_final / frac_B_moved.
    training = [
        {"cycle": cycle, "step": i, "loss": 1.0 - 0.01 * i,
         "lr_B": 1e-4, "dB_norm": 1e-3 if i % 2 == 0 else 1e-6}
        for i in range(10)
    ]
    _dump("training_steps.jsonl", training)

    heldout = [
        {"cycle": cycle, "prompt_id": f"p{i}", "domain": "math",
         "score": 0.5, "delta_vs_ref": -0.01}
        for i in range(5)
    ]
    _dump("heldout_per_prompt.jsonl", heldout)

    verify = [
        {"cycle": cycle, "accepted": True, "any_fail": False}
        for _ in range(5)
    ]
    _dump("verify_decisions.jsonl", verify)

    propose = [{"cycle": cycle, "status": "ok"} for _ in range(3)]
    _dump("propose_attempts.jsonl", propose)

    summary = [{"cycle": cycle, "held_out_score": 0.5}]
    _dump("cycle_summary.jsonl", summary)


def _stub(tmp_path: Path) -> SimpleNamespace:
    ocfg = OrchestratorConfig()
    ocfg.output_dir = tmp_path
    stub = SimpleNamespace()
    stub.config = SimpleNamespace(orchestrator=ocfg)
    stub._auto_diagnose_cycle = (
        ImprovementLoop._auto_diagnose_cycle.__get__(stub)
    )
    return stub


def test_auto_diagnose_writes_jsonl_row(tmp_path: Path):
    cycle = 3
    _write_synth_logs(tmp_path, cycle)
    stub = _stub(tmp_path)

    result = CycleResult(cycle=cycle)
    # Clean cycle — no alarms → cycle_eligible=True.
    stub._auto_diagnose_cycle(cycle, result)

    jsonl = tmp_path / "auto_diagnosis.jsonl"
    assert jsonl.exists(), "auto_diagnosis.jsonl must be created"
    lines = [l for l in jsonl.read_text().splitlines() if l.strip()]
    assert len(lines) == 1, f"expected exactly 1 row, got {len(lines)}"
    row = json.loads(lines[0])
    assert row["cycle"] == cycle
    assert isinstance(row["ts"], float)
    assert isinstance(row["reason_bullets"], list)
    assert len(row["reason_bullets"]) == 3
    assert isinstance(row["verbatim_tldr"], str)
    assert row["cycle_eligible"] is True


def test_auto_diagnose_alarmed_cycle_marked_ineligible(tmp_path: Path):
    cycle = 4
    _write_synth_logs(tmp_path, cycle)
    stub = _stub(tmp_path)

    result = CycleResult(cycle=cycle)
    result.regression_reverted = True
    stub._auto_diagnose_cycle(cycle, result)

    row = json.loads((tmp_path / "auto_diagnosis.jsonl").read_text().splitlines()[-1])
    assert row["cycle_eligible"] is False


def test_auto_diagnose_never_raises_on_missing_logs(tmp_path: Path):
    """No jsonl sinks at all — helper must swallow & still write a row."""
    stub = _stub(tmp_path)
    result = CycleResult(cycle=99)
    # Must not raise.
    stub._auto_diagnose_cycle(99, result)
    # Even with missing logs analyze_cycle.py exits 0 and still prints a TL;DR
    # (with "signals missing" bullets), so a row should land.
    jsonl = tmp_path / "auto_diagnosis.jsonl"
    assert jsonl.exists()
    row = json.loads(jsonl.read_text().splitlines()[-1])
    assert row["cycle"] == 99
