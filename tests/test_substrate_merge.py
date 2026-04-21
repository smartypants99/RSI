"""Substrate-update tests: periodic LoRA merge-into-base promotion.

LoRA on a frozen 4-bit base has a fixed ceiling — every N trained cycles
the orchestrator promotes the current merged checkpoint to a new "base"
checkpoint (``outputs/checkpoints/base_epoch_K``) and restarts LoRA fresh
on top. Guardrails:

  - Only TRAINED cycles count toward the N-cycle counter.
  - Skip promotion when cumulative held-out improvement since the last
    promotion is below ``substrate_merge_min_improvement`` (default 0.005).
  - Write a one-line event to ``update-log.txt`` for every promotion /
    skip / defer so the user can see merge epochs.

The trainer is fully mocked — these tests don't load any model.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.orchestrator.loop import CycleResult, ImprovementLoop
from src.utils.config import OrchestratorConfig


# ---------------------------------------------------------------------------
# Config wiring
# ---------------------------------------------------------------------------

def test_orchestrator_config_defaults():
    cfg = OrchestratorConfig()
    assert cfg.merge_into_base_every == 10
    assert cfg.substrate_merge_min_improvement == 0.005


def test_orchestrator_config_rejects_negative_merge_every():
    with pytest.raises(ValueError):
        OrchestratorConfig(merge_into_base_every=-1)


def test_orchestrator_config_rejects_negative_min_improvement():
    with pytest.raises(ValueError):
        OrchestratorConfig(substrate_merge_min_improvement=-0.01)


def test_orchestrator_config_disabled_when_zero():
    cfg = OrchestratorConfig(merge_into_base_every=0)
    assert cfg.merge_into_base_every == 0  # 0 = disabled


# ---------------------------------------------------------------------------
# _maybe_substrate_merge behaviour (trainer mocked)
# ---------------------------------------------------------------------------

def _make_loop(tmp_path: Path, *, every: int = 3, min_improvement: float = 0.005) -> ImprovementLoop:
    """Construct a bare ImprovementLoop without loading any model."""
    loop = ImprovementLoop.__new__(ImprovementLoop)
    orchestrator = OrchestratorConfig(
        merge_into_base_every=every,
        substrate_merge_min_improvement=min_improvement,
        output_dir=tmp_path,
        log_dir=tmp_path / "logs",
    )
    loop.config = SimpleNamespace(orchestrator=orchestrator)
    loop.model_loader = SimpleNamespace(model_path="base-model-repo-id")
    loop._substrate_epoch = 0
    loop._substrate_baseline_eval = None
    loop._substrate_last_merge_cycle = 0
    loop._substrate_trained_cycles_since_merge = 0
    return loop


def _make_trained_result(cycle: int, eval_score: float) -> CycleResult:
    r = CycleResult(cycle)
    r.training_metrics = SimpleNamespace(steps=3)
    r.eval_score = eval_score
    return r


def _make_no_train_result(cycle: int, eval_score: float) -> CycleResult:
    r = CycleResult(cycle)
    r.training_metrics = None
    r.eval_score = eval_score
    return r


def _seed_cycle_checkpoint(tmp_path: Path, cycle: int) -> Path:
    """Create a minimal checkpoint dir resembling an HF save_pretrained output."""
    ckpt = tmp_path / "checkpoints" / f"cycle_{cycle}"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "config.json").write_text(json.dumps({"model_type": "mock"}))
    (ckpt / "model.safetensors").write_bytes(b"fake-weights")
    return ckpt


def test_disabled_when_every_is_zero(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    loop = _make_loop(tmp_path, every=0)
    _seed_cycle_checkpoint(tmp_path, 1)
    loop._maybe_substrate_merge(1, _make_trained_result(1, 0.6))
    assert loop._substrate_epoch == 0
    assert not (tmp_path / "checkpoints" / "base_epoch_1").exists()


def test_promotes_after_n_trained_cycles_with_improvement(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    loop = _make_loop(tmp_path, every=3)

    # Cycle 1: baseline is captured from the first eval; counter=1.
    _seed_cycle_checkpoint(tmp_path, 1)
    loop._maybe_substrate_merge(1, _make_trained_result(1, 0.500))
    assert loop._substrate_baseline_eval == pytest.approx(0.500)
    assert loop._substrate_trained_cycles_since_merge == 1
    assert loop._substrate_epoch == 0

    # Cycle 2: counter=2, no promotion yet.
    _seed_cycle_checkpoint(tmp_path, 2)
    loop._maybe_substrate_merge(2, _make_trained_result(2, 0.520))
    assert loop._substrate_trained_cycles_since_merge == 2
    assert loop._substrate_epoch == 0

    # Cycle 3: counter hits 3, delta 0.550 − 0.500 = 0.050 ≥ 0.005 — PROMOTE.
    _seed_cycle_checkpoint(tmp_path, 3)
    r3 = _make_trained_result(3, 0.550)
    loop._maybe_substrate_merge(3, r3)
    dest = tmp_path / "checkpoints" / "base_epoch_1"
    assert dest.exists()
    assert (dest / "config.json").exists()
    assert (dest / "model.safetensors").exists()
    assert loop._substrate_epoch == 1
    assert loop._substrate_last_merge_cycle == 3
    assert loop._substrate_trained_cycles_since_merge == 0
    assert loop._substrate_baseline_eval == pytest.approx(0.550)
    # Model-loader's fallback base is redirected to the promoted checkpoint.
    assert loop.model_loader.model_path == str(dest)
    # Event recorded on the CycleResult.
    assert any("substrate_merge:base_epoch_1" in e for e in r3.escalation_events)


def test_no_train_cycles_do_not_advance_counter(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    loop = _make_loop(tmp_path, every=2)

    # Two no-train cycles: counter stays 0 but baseline is still captured.
    loop._maybe_substrate_merge(1, _make_no_train_result(1, 0.500))
    loop._maybe_substrate_merge(2, _make_no_train_result(2, 0.600))
    assert loop._substrate_trained_cycles_since_merge == 0
    assert loop._substrate_epoch == 0
    assert loop._substrate_baseline_eval == pytest.approx(0.500)

    # Trained cycles now drive the counter to 2, triggering a promotion.
    _seed_cycle_checkpoint(tmp_path, 3)
    loop._maybe_substrate_merge(3, _make_trained_result(3, 0.610))
    _seed_cycle_checkpoint(tmp_path, 4)
    loop._maybe_substrate_merge(4, _make_trained_result(4, 0.620))
    assert loop._substrate_epoch == 1
    assert (tmp_path / "checkpoints" / "base_epoch_1").exists()


def test_skip_when_improvement_below_threshold(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    loop = _make_loop(tmp_path, every=2, min_improvement=0.01)

    _seed_cycle_checkpoint(tmp_path, 1)
    loop._maybe_substrate_merge(1, _make_trained_result(1, 0.500))
    _seed_cycle_checkpoint(tmp_path, 2)
    # Delta = 0.502 − 0.500 = 0.002 < 0.01 — skip.
    loop._maybe_substrate_merge(2, _make_trained_result(2, 0.502))

    assert loop._substrate_epoch == 0
    assert not (tmp_path / "checkpoints" / "base_epoch_1").exists()
    # Counter is NOT reset on skip — next trained cycle re-checks.
    assert loop._substrate_trained_cycles_since_merge == 2


def test_skip_on_regression(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    loop = _make_loop(tmp_path, every=2)

    _seed_cycle_checkpoint(tmp_path, 1)
    loop._maybe_substrate_merge(1, _make_trained_result(1, 0.600))
    _seed_cycle_checkpoint(tmp_path, 2)
    # Regression: 0.400 < 0.600 — delta negative, skip.
    loop._maybe_substrate_merge(2, _make_trained_result(2, 0.400))

    assert loop._substrate_epoch == 0
    assert not (tmp_path / "checkpoints" / "base_epoch_1").exists()


def test_defer_when_no_cycle_checkpoint_exists(tmp_path, monkeypatch):
    """If training wrote no checkpoint (e.g. quantized-base skip), defer the
    promotion instead of producing an empty base_epoch_K directory."""
    monkeypatch.chdir(tmp_path)
    loop = _make_loop(tmp_path, every=2)

    # Seed cycle 1 but NOT cycle 2 — cycle 2 is the promotion attempt.
    _seed_cycle_checkpoint(tmp_path, 1)
    loop._maybe_substrate_merge(1, _make_trained_result(1, 0.500))
    loop._maybe_substrate_merge(2, _make_trained_result(2, 0.600))  # no ckpt for 2

    assert loop._substrate_epoch == 0
    assert not (tmp_path / "checkpoints" / "base_epoch_1").exists()
    # Counter not reset on defer — retry next trained cycle once a checkpoint exists.
    assert loop._substrate_trained_cycles_since_merge == 2


def test_update_log_records_promotion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    loop = _make_loop(tmp_path, every=1)

    _seed_cycle_checkpoint(tmp_path, 1)
    loop._maybe_substrate_merge(1, _make_trained_result(1, 0.500))

    # First trained cycle captures baseline AND counter=1=every, triggers promote
    # attempt but delta=0 so it skips (below 0.005). Check the skip line appears.
    log_path = tmp_path / "update-log.txt"
    assert log_path.exists()
    contents = log_path.read_text()
    assert "substrate-merge" in contents
    assert "cycle 1" in contents


def test_update_log_records_skip_with_reason(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    loop = _make_loop(tmp_path, every=1, min_improvement=0.01)
    _seed_cycle_checkpoint(tmp_path, 1)
    loop._maybe_substrate_merge(1, _make_trained_result(1, 0.5))
    _seed_cycle_checkpoint(tmp_path, 2)
    loop._maybe_substrate_merge(2, _make_trained_result(2, 0.502))
    contents = (tmp_path / "update-log.txt").read_text()
    assert "SKIPPED" in contents


def test_second_epoch_promotes_from_new_baseline(tmp_path, monkeypatch):
    """After epoch 1 is promoted, the baseline resets to that eval score and
    only further-cumulative improvement triggers epoch 2."""
    monkeypatch.chdir(tmp_path)
    loop = _make_loop(tmp_path, every=1)

    _seed_cycle_checkpoint(tmp_path, 1)
    loop._maybe_substrate_merge(1, _make_trained_result(1, 0.50))  # baseline
    _seed_cycle_checkpoint(tmp_path, 2)
    loop._maybe_substrate_merge(2, _make_trained_result(2, 0.60))  # promote epoch 1
    assert loop._substrate_epoch == 1
    assert loop._substrate_baseline_eval == pytest.approx(0.60)

    # Cycle 3 eval barely above new baseline (delta=0.003 < 0.005) — skip.
    _seed_cycle_checkpoint(tmp_path, 3)
    loop._maybe_substrate_merge(3, _make_trained_result(3, 0.603))
    assert loop._substrate_epoch == 1

    # Cycle 4 clears the threshold — promote epoch 2.
    _seed_cycle_checkpoint(tmp_path, 4)
    loop._maybe_substrate_merge(4, _make_trained_result(4, 0.62))
    assert loop._substrate_epoch == 2
    assert (tmp_path / "checkpoints" / "base_epoch_2").exists()
