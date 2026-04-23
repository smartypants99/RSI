"""Test that trainer emits training_steps.jsonl records.

Invokes the minimal path: construct a CustomLoRATrainer (no model load
needed — __init__ is cheap), wire an OrchestratorConfig with
structured_observability_enabled=True, call _emit_training_step_log
directly with a tiny synthetic lora_params list, and assert a valid
JSON line lands with all required fields.

Also asserts the gate-off case: disabling the flag → no file written.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from src.trainer.custom_lora import CustomLoRATrainer
from src.utils.config import OrchestratorConfig, TrainerConfig
from src.utils.structured_logs import SINK_FILENAMES


REQUIRED_FIELDS = {
    "cycle", "step_idx", "sample_idx_in_batch",
    "loss_unweighted", "loss_weighted", "sample_weight", "verdict_warnings",
    "grad_norm_lora_A", "grad_norm_lora_B", "grad_norm_magnitude",
    "grad_norm_total",
    "lr_A", "lr_B",
    "post_step_B_max_abs", "post_step_B_mean_abs",
    "clip_fraction", "time_ms",
}


def _make_trainer() -> CustomLoRATrainer:
    """Build a CustomLoRATrainer without loading any model."""
    # TrainerConfig dataclass with defaults is fine — we won't actually train.
    trainer_cfg = TrainerConfig()
    loader = MagicMock()
    return CustomLoRATrainer(trainer_cfg, loader)


def _fake_lora_params_with_grad():
    """Return two tensors shaped like LoRA A (rank, in) and B (out, rank),
    both with .grad populated."""
    # A: rank=4, in=32  → shape[0] < shape[1] → bucketed as A
    a = torch.randn(4, 32, requires_grad=True)
    a.grad = torch.randn_like(a) * 0.01
    # B: out=32, rank=4 → shape[0] > shape[1] → bucketed as B
    b = torch.randn(32, 4, requires_grad=True)
    b.grad = torch.randn_like(b) * 0.001
    return [a, b]


def test_training_step_log_writes_valid_record(tmp_path: Path):
    obs = OrchestratorConfig()
    obs.output_dir = tmp_path
    obs.structured_observability_enabled = True
    obs.structured_log_training_steps = True

    trainer = _make_trainer()
    trainer.set_observability_config(obs)

    params = _fake_lora_params_with_grad()
    trainer._emit_training_step_log(
        cycle=2,
        step_idx=0,
        loss_unweighted=0.47,
        loss_weighted=0.50,
        sample_weight=1.0,
        verdict_warnings=("any_fail",),
        lora_params=params,
        lr_A=1.8e-3,
        lr_B=9e-5,
        clip_fraction=0.0,
        time_ms=42.3,
        sample_idx_in_batch=None,
    )

    path = tmp_path / SINK_FILENAMES["training_steps"]
    assert path.exists(), "training_steps.jsonl must be written"
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert REQUIRED_FIELDS.issubset(rec.keys()), (
        f"missing fields: {REQUIRED_FIELDS - set(rec.keys())}"
    )
    assert rec["cycle"] == 2
    assert rec["step_idx"] == 0
    assert rec["loss_unweighted"] == pytest.approx(0.47)
    assert rec["verdict_warnings"] == ["any_fail"]
    # Grad norms must be non-negative floats.
    assert rec["grad_norm_lora_A"] >= 0.0
    assert rec["grad_norm_lora_B"] >= 0.0
    assert rec["grad_norm_total"] >= 0.0
    # Post-step B stats computed (B tensor was in params list).
    assert rec["post_step_B_max_abs"] > 0.0


def test_training_step_log_disabled_writes_nothing(tmp_path: Path):
    obs = OrchestratorConfig()
    obs.output_dir = tmp_path
    obs.structured_observability_enabled = False  # master off

    trainer = _make_trainer()
    trainer.set_observability_config(obs)
    trainer._emit_training_step_log(
        cycle=1, step_idx=0,
        loss_unweighted=0.1, loss_weighted=0.1,
        sample_weight=None, verdict_warnings=None,
        lora_params=_fake_lora_params_with_grad(),
        lr_A=1e-3, lr_B=1e-4,
        clip_fraction=0.0, time_ms=1.0,
    )
    assert not (tmp_path / SINK_FILENAMES["training_steps"]).exists()


def test_training_step_log_subflag_off_writes_nothing(tmp_path: Path):
    obs = OrchestratorConfig()
    obs.output_dir = tmp_path
    obs.structured_observability_enabled = True
    obs.structured_log_training_steps = False  # sub-flag off

    trainer = _make_trainer()
    trainer.set_observability_config(obs)
    trainer._emit_training_step_log(
        cycle=1, step_idx=0,
        loss_unweighted=0.1, loss_weighted=0.1,
        sample_weight=None, verdict_warnings=None,
        lora_params=_fake_lora_params_with_grad(),
        lr_A=1e-3, lr_B=1e-4,
        clip_fraction=0.0, time_ms=1.0,
    )
    assert not (tmp_path / SINK_FILENAMES["training_steps"]).exists()


def test_training_step_log_no_obs_cfg_is_noop(tmp_path: Path):
    """When set_observability_config is never called, emit is a no-op."""
    trainer = _make_trainer()
    # Default _obs_cfg is None → emit must silently return.
    trainer._emit_training_step_log(
        cycle=1, step_idx=0,
        loss_unweighted=0.1, loss_weighted=0.1,
        sample_weight=None, verdict_warnings=None,
        lora_params=_fake_lora_params_with_grad(),
        lr_A=1e-3, lr_B=1e-4,
        clip_fraction=0.0, time_ms=1.0,
    )
    # Should not write anywhere; no assertion needed beyond "does not raise".
