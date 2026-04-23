"""Sanity-check scripts/analyze_cycle.py against a synthetic fixture.

Not a deep unit test — this is a tool, not a library. We verify:
  - runs end-to-end on a tiny cycle (3 heldout prompts, 2 training steps)
  - produces outputs/cycle_N_analysis.md
  - handles empty/partial logs gracefully (no crash)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "analyze_cycle.py"


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_analyzer_on_synthetic_fixture(tmp_path: Path):
    logs = tmp_path / "outputs"
    cycle_dir = logs / "cycle_7"
    _write_jsonl(cycle_dir / "training_steps.jsonl", [
        {"step": 0, "loss": 1.23, "grad_norm_B": 0.5,
         "post_step_B_max_abs": 2e-4, "lr_B": 9e-5,
         "sample_id": "s1", "domain": "code"},
        {"step": 1, "loss": 1.10, "grad_norm_B": 0.6,
         "post_step_B_max_abs": 3e-4, "lr_B": 9e-5,
         "sample_id": "s2", "domain": "math"},
    ])
    _write_jsonl(cycle_dir / "heldout_per_prompt.jsonl", [
        {"prompt_id": "p1", "domain": "code", "score_pre": 0.5, "score_post": 0.6},
        {"prompt_id": "p2", "domain": "code", "score_pre": 0.4, "score_post": 0.35},
        {"prompt_id": "p3", "domain": "math", "score_pre": 0.7, "score_post": 0.72},
    ])
    _write_jsonl(cycle_dir / "verify_decisions.jsonl", [
        {"sample_id": "s1", "accepted": True, "verdict_warnings": ["any_fail"]},
        {"sample_id": "s2", "accepted": True, "verdict_warnings": []},
    ])
    _write_jsonl(cycle_dir / "propose_attempts.jsonl", [
        {"attempt_id": 1, "success": True,  "time_s": 2.0},
        {"attempt_id": 2, "success": False, "failure_reason": "compile_error", "time_s": 1.5},
        {"attempt_id": 3, "success": False, "failure_reason": "compile_error", "time_s": 1.1},
        {"attempt_id": 4, "success": False, "failure_reason": "timeout",       "time_s": 8.0},
    ])
    _write_jsonl(cycle_dir / "cycle_summary.jsonl", [
        {"cycle": 7, "promoted": False, "delta_continuous": -0.003},
    ])

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "7", "--logs-dir", str(logs)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr={result.stderr}\nstdout={result.stdout}"
    stdout = result.stdout
    # Spot-check sections rendered
    for marker in [
        "Training health",
        "Training damage probe",
        "Verifier noise",
        "ρ decomposition",
        "Proposer bottleneck",
        "Bottom line",
        "compile_error",
    ]:
        assert marker in stdout, f"missing section marker {marker!r}"
    assert (logs / "cycle_7_analysis.md").exists()


def test_analyzer_handles_missing_logs(tmp_path: Path):
    logs = tmp_path / "outputs"
    logs.mkdir()
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "latest", "--logs-dir", str(logs)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "MISSING" in result.stdout
