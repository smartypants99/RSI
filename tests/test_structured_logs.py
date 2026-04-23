"""Tests for src/utils/structured_logs.py — the shared append-only JSONL sink.

Covers:
  - happy path: enabled sink writes one parseable line per emit()
  - gate off: master flag False → no file, no overhead
  - sub-flag off: master True + sub-flag False → no file
  - crash safety: json-unserializable record does not raise
  - unknown sink name is a no-op
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.utils.structured_logs import SINK_FILENAMES, emit, is_enabled


class _Cfg:
    """Minimal config stub with attribute access."""
    def __init__(self, **kw):
        self.output_dir = kw.pop("output_dir", None)
        self.structured_observability_enabled = kw.pop(
            "structured_observability_enabled", False
        )
        for k, v in kw.items():
            setattr(self, k, v)


def test_emit_writes_line_when_enabled(tmp_path: Path):
    cfg = _Cfg(output_dir=tmp_path, structured_observability_enabled=True)
    emit("training_steps", {"cycle": 3, "step_idx": 7, "loss_unweighted": 0.42}, cfg)
    p = tmp_path / SINK_FILENAMES["training_steps"]
    assert p.exists()
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec == {"cycle": 3, "step_idx": 7, "loss_unweighted": 0.42}


def test_emit_appends_multiple_lines(tmp_path: Path):
    cfg = _Cfg(output_dir=tmp_path, structured_observability_enabled=True)
    for i in range(5):
        emit("training_steps", {"cycle": 1, "step_idx": i}, cfg)
    p = tmp_path / SINK_FILENAMES["training_steps"]
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 5
    assert [json.loads(l)["step_idx"] for l in lines] == [0, 1, 2, 3, 4]


def test_master_flag_off_produces_no_file(tmp_path: Path):
    cfg = _Cfg(output_dir=tmp_path, structured_observability_enabled=False)
    emit("training_steps", {"x": 1}, cfg)
    # Neither the jsonl nor the parent dir should be created by emit.
    assert not (tmp_path / SINK_FILENAMES["training_steps"]).exists()
    # Directory was passed as tmp_path (already exists); ensure emit wrote nothing.
    assert os.listdir(tmp_path) == []


def test_sub_flag_off_silences_one_sink(tmp_path: Path):
    cfg = _Cfg(
        output_dir=tmp_path,
        structured_observability_enabled=True,
        structured_log_training_steps=False,
    )
    emit("training_steps", {"x": 1}, cfg)
    emit("verify_decisions", {"y": 2}, cfg)
    assert not (tmp_path / SINK_FILENAMES["training_steps"]).exists()
    assert (tmp_path / SINK_FILENAMES["verify_decisions"]).exists()


def test_none_cfg_is_noop(tmp_path: Path):
    # Must not raise even when cfg is None (defensive path for early call sites).
    emit("training_steps", {"x": 1}, None)


def test_unknown_sink_is_noop(tmp_path: Path):
    cfg = _Cfg(output_dir=tmp_path, structured_observability_enabled=True)
    emit("not_a_real_sink", {"x": 1}, cfg)
    assert os.listdir(tmp_path) == []


def test_non_serializable_record_does_not_raise(tmp_path: Path):
    cfg = _Cfg(output_dir=tmp_path, structured_observability_enabled=True)

    class Weird:
        def __repr__(self):
            return "<weird>"

    # default=str in emit coerces unknown types; must not raise.
    emit("training_steps", {"cycle": 1, "obj": Weird()}, cfg)
    p = tmp_path / SINK_FILENAMES["training_steps"]
    assert p.exists()
    rec = json.loads(p.read_text().strip())
    assert rec["cycle"] == 1
    assert "weird" in rec["obj"]


def test_is_enabled_respects_sub_flag():
    cfg = _Cfg(
        output_dir="/tmp/doesntmatter",
        structured_observability_enabled=True,
        structured_log_verify_decisions=False,
    )
    assert is_enabled(cfg, "training_steps") is True
    assert is_enabled(cfg, "verify_decisions") is False
    cfg.structured_observability_enabled = False
    assert is_enabled(cfg, "training_steps") is False


def test_output_dir_override(tmp_path: Path):
    cfg = _Cfg(output_dir=None, structured_observability_enabled=True)
    emit("cycle_summary", {"cycle": 9}, cfg, output_dir=tmp_path)
    assert (tmp_path / SINK_FILENAMES["cycle_summary"]).exists()


def test_all_five_sinks_have_distinct_filenames():
    expected = {
        "training_steps",
        "heldout_per_prompt",
        "verify_decisions",
        "propose_attempts",
        "cycle_summary",
    }
    assert set(SINK_FILENAMES.keys()) == expected
    # All filenames unique.
    assert len(set(SINK_FILENAMES.values())) == 5
