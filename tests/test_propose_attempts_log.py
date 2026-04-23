"""Test outputs/propose_attempts.jsonl emission.

Calls TaskSynthesizer._emit_propose_attempt_log directly — the full
propose_batch_code path requires a vLLM-resident model and is out of
scope for a unit test.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.generator.task_synthesizer import TaskSynthesizer
from src.utils.config import OrchestratorConfig, SynthesisConfig
from src.utils.structured_logs import SINK_FILENAMES


REQUIRED_FIELDS = {
    "cycle", "attempt_idx", "domain", "frontier",
    "parsed_successfully", "failure_reason",
    "generation_time_ms", "num_tokens_generated",
    "entry_point", "num_tests",
}


def _make_synth() -> TaskSynthesizer:
    cfg = SynthesisConfig()
    return TaskSynthesizer(cfg, model_loader=MagicMock(), run_vov=False)


def test_propose_attempt_log_writes_success_record(tmp_path: Path):
    ocfg = OrchestratorConfig()
    ocfg.output_dir = tmp_path
    ocfg.structured_observability_enabled = True
    synth = _make_synth()
    synth.set_observability_config(ocfg)
    synth.set_observability_cycle(5)

    synth._emit_propose_attempt_log(
        attempt_idx=0,
        domain="code",
        frontier="code/implementation",
        parsed_successfully=True,
        failure_reason=None,
        generation_time_ms=123.4,
        num_tokens_generated=512,
        entry_point="solve",
        num_tests=4,
    )
    synth._emit_propose_attempt_log(
        attempt_idx=1,
        domain="code",
        frontier="code/implementation",
        parsed_successfully=False,
        failure_reason="missing_problem;too_few_tests",
        generation_time_ms=88.0,
        num_tokens_generated=120,
        entry_point=None,
        num_tests=None,
    )

    path = tmp_path / SINK_FILENAMES["propose_attempts"]
    assert path.exists()
    rows = [json.loads(l) for l in path.read_text().strip().splitlines()]
    assert len(rows) == 2
    for r in rows:
        assert REQUIRED_FIELDS.issubset(r.keys())
        assert r["cycle"] == 5
        assert r["domain"] == "code"
    assert rows[0]["parsed_successfully"] is True
    assert rows[0]["entry_point"] == "solve"
    assert rows[0]["num_tests"] == 4
    assert rows[1]["parsed_successfully"] is False
    assert "missing_problem" in rows[1]["failure_reason"]


def test_propose_attempt_log_disabled_writes_nothing(tmp_path: Path):
    ocfg = OrchestratorConfig()
    ocfg.output_dir = tmp_path
    ocfg.structured_observability_enabled = False
    synth = _make_synth()
    synth.set_observability_config(ocfg)
    synth._emit_propose_attempt_log(
        attempt_idx=0, domain="code", frontier="",
        parsed_successfully=True, failure_reason=None,
        generation_time_ms=1.0, num_tokens_generated=10,
        entry_point="x", num_tests=1,
    )
    assert not (tmp_path / SINK_FILENAMES["propose_attempts"]).exists()


def test_propose_attempt_log_no_cfg_is_noop(tmp_path: Path):
    synth = _make_synth()
    # No set_observability_config → _obs_cfg stays None.
    synth._emit_propose_attempt_log(
        attempt_idx=0, domain="code", frontier="",
        parsed_successfully=True, failure_reason=None,
        generation_time_ms=1.0, num_tokens_generated=10,
        entry_point="x", num_tests=1,
    )
