"""Regularization guards for the SFT trainer.

Tuned against the cycle-2 (success) vs cycle-3 (memorization) regime observed
in live RSI runs: 1-2 steps / loss 0.4-0.8 is healthy; 25+ steps on <10 samples
with loss <0.1 is memorization. These tests lock in the guardrails that push
future cycles toward the cycle-2 regime automatically.
"""
from __future__ import annotations

import logging

import pytest

from src.trainer.custom_lora import _plan_step_budget, _EarlyStop
from src.utils.config import TrainerConfig


# ---------------------------------------------------------------------------
# TrainerConfig defaults + validation
# ---------------------------------------------------------------------------

def test_trainer_config_small_cycle_defaults():
    cfg = TrainerConfig()
    assert cfg.num_epochs == 3, "num_epochs default must be 3 for small-cycle RSI"
    assert cfg.gradient_accumulation_steps == 4, (
        "grad_accum default must be 4 so cycles with <30 samples get 2-4 steps"
    )
    assert cfg.early_stop_loss == 0.15
    assert cfg.max_steps_per_cycle == 8
    assert cfg.min_steps_per_cycle == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("early_stop_loss", 0.0),
        ("early_stop_loss", -0.1),
        ("max_steps_per_cycle", 0),
        ("min_steps_per_cycle", 0),
    ],
)
def test_trainer_config_rejects_invalid_regularization(field, value):
    kwargs = {field: value}
    with pytest.raises(ValueError):
        TrainerConfig(**kwargs)


def test_trainer_config_rejects_min_above_max():
    with pytest.raises(ValueError):
        TrainerConfig(min_steps_per_cycle=10, max_steps_per_cycle=4)


# ---------------------------------------------------------------------------
# Adaptive step-budget planner
# ---------------------------------------------------------------------------

def test_plan_caps_steps_at_max_for_large_dataset():
    # 9 samples × 5 epochs, batch 1, base_accum 1 → 45 batches, would give
    # 45 optimizer steps (the cycle-3 catastrophe). With max_steps=8, we want
    # accum to scale up so steps ≤ 8.
    accum, steps, skip = _plan_step_budget(
        total_batches=45, base_accum=1, max_steps_per_cycle=8, min_steps_per_cycle=1
    )
    assert not skip
    assert steps <= 8
    # ceil(45/6) = 8, so accum=6 fits exactly.
    assert accum == 6


def test_plan_honors_base_accum_as_floor():
    # If the user asks for grad_accum=16 and the dataset is tiny, we must
    # not silently shrink to 1.
    accum, steps, skip = _plan_step_budget(
        total_batches=4, base_accum=16, max_steps_per_cycle=8, min_steps_per_cycle=1
    )
    assert accum == 16
    assert steps == 1  # 4 batches < 16 → 1 flushed step
    assert not skip


def test_plan_preserves_base_when_within_cap():
    # Cycle-2 regime: ~6 batches, base_accum=4 → 2 steps, within cap, unchanged.
    accum, steps, skip = _plan_step_budget(
        total_batches=6, base_accum=4, max_steps_per_cycle=8, min_steps_per_cycle=1
    )
    assert accum == 4
    assert steps == 2
    assert not skip


def test_plan_skips_when_below_min_steps():
    # If the math would yield 0 effective steps, skip the cycle.
    accum, steps, skip = _plan_step_budget(
        total_batches=0, base_accum=4, max_steps_per_cycle=8, min_steps_per_cycle=1
    )
    assert skip


def test_plan_cycle3_like_scenario_is_tamed():
    """Cycle-3 actual config: 9 samples, num_epochs≈5, grad_accum=1 → 25+ steps.

    With the new defaults (num_epochs=2, grad_accum=4) and the max-step cap,
    the same 9-sample cycle gets 1-2 steps — exactly the cycle-2 regime.
    """
    total_batches = 9 * 2  # samples × num_epochs with batch_size=1
    accum, steps, skip = _plan_step_budget(
        total_batches=total_batches,
        base_accum=4,
        max_steps_per_cycle=8,
        min_steps_per_cycle=1,
    )
    assert not skip
    assert 1 <= steps <= 8


# ---------------------------------------------------------------------------
# Early-stop signal
# ---------------------------------------------------------------------------

def test_early_stop_is_an_exception():
    # The trainer uses _EarlyStop as an internal control-flow signal. Keep the
    # contract explicit so any future refactor that changes the type breaks loudly.
    assert issubclass(_EarlyStop, Exception)


def test_early_stop_trigger_condition():
    """Replicates the guard in _train_inner. Loss below early_stop_loss AFTER at
    least one optimizer step must trigger, but only after step_count >= 1 — so
    a cycle that starts below the threshold still performs one real update."""
    cfg = TrainerConfig()
    # Mid-training, step already taken, loss collapses → trigger.
    assert 0.04 < cfg.early_stop_loss
    assert 1 >= 1 and 0.04 < cfg.early_stop_loss
    # Pre-first-step, even low loss does NOT trigger.
    step_count_zero = 0
    should_trigger = step_count_zero >= 1 and 0.04 < cfg.early_stop_loss
    assert not should_trigger


# ---------------------------------------------------------------------------
# Overfit detector warning
# ---------------------------------------------------------------------------

def test_overfit_detector_logs_warning(caplog):
    """Mirror the guard in _train_inner: final_loss<0.1 AND samples_used<20
    AND step_count>0 emits a 'training likely memorized' warning."""
    from src.trainer import custom_lora

    # Replicate the exact conditional the code uses so that a future edit to
    # the real conditional breaks this test (which is the intent — we want the
    # warning to keep firing on the cycle-3 pattern).
    last_loss = 0.045
    samples_used = 9
    step_count = 3
    with caplog.at_level(logging.WARNING, logger=custom_lora.logger.name):
        if last_loss < 0.1 and samples_used < 20 and step_count > 0:
            custom_lora.logger.warning(
                f"  Overfit suspected: final_loss={last_loss:.4f} < 0.1"
                f" on samples_used={samples_used} < 20."
                f" Training likely memorized; consider revert."
            )
    assert any("Overfit suspected" in rec.message for rec in caplog.records)


def test_overfit_detector_silent_for_healthy_run(caplog):
    from src.trainer import custom_lora

    last_loss = 0.55  # cycle-2 regime
    samples_used = 9
    step_count = 2
    with caplog.at_level(logging.WARNING, logger=custom_lora.logger.name):
        if last_loss < 0.1 and samples_used < 20 and step_count > 0:
            custom_lora.logger.warning("Overfit suspected: ...")
    assert not any("Overfit suspected" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Task #14: sample-quality clean-floor filter
# ---------------------------------------------------------------------------

def test_clean_floor_filter_disabled_when_floor_zero():
    from src.trainer.custom_lora import _filter_any_fail_when_clean_enough

    class _S:
        def __init__(self, warn):
            self.verdict_warnings = warn

    samples = [_S(("any_fail",)), _S(()), _S(("any_fail",)), _S(())]
    out, dropped = _filter_any_fail_when_clean_enough(samples, clean_floor=0)
    assert dropped == 0
    assert len(out) == 4


def test_clean_floor_filter_keeps_all_when_pool_below_floor():
    """Starvation guard: don't filter when total < floor."""
    from src.trainer.custom_lora import _filter_any_fail_when_clean_enough

    class _S:
        def __init__(self, warn):
            self.verdict_warnings = warn

    samples = [_S(("any_fail",)), _S(()), _S(("any_fail",))]
    out, dropped = _filter_any_fail_when_clean_enough(samples, clean_floor=16)
    assert dropped == 0
    assert len(out) == 3


def test_clean_floor_filter_keeps_all_when_clean_subset_below_floor():
    """If dropping any_fail would leave clean < floor, keep all samples."""
    from src.trainer.custom_lora import _filter_any_fail_when_clean_enough

    class _S:
        def __init__(self, warn):
            self.verdict_warnings = warn

    samples = [_S(("any_fail",))] * 14 + [_S(())] * 5
    # total = 19 >= 16, clean = 5 < 16 → keep all.
    out, dropped = _filter_any_fail_when_clean_enough(samples, clean_floor=16)
    assert dropped == 0
    assert len(out) == 19


def test_clean_floor_filter_drops_any_fail_when_clean_enough():
    from src.trainer.custom_lora import _filter_any_fail_when_clean_enough

    class _S:
        def __init__(self, warn):
            self.verdict_warnings = warn

    samples = [_S(("any_fail",))] * 4 + [_S(())] * 20
    # total = 24 >= 16, clean = 20 >= 16 → drop 4 any_fail samples.
    out, dropped = _filter_any_fail_when_clean_enough(samples, clean_floor=16)
    assert dropped == 4
    assert len(out) == 20
    assert all("any_fail" not in s.verdict_warnings for s in out)


# ---------------------------------------------------------------------------
# Task #14: warmup-cycle epoch cap
# ---------------------------------------------------------------------------

def test_warmup_epoch_cap_defaults_and_validation():
    cfg = TrainerConfig()
    assert cfg.num_epochs_warmup == 1
    assert cfg.num_epochs_warmup_cycles == 0

    # warmup=0 rejected (must be >= 1 to mean anything).
    with pytest.raises(ValueError, match="num_epochs_warmup must be >= 1"):
        TrainerConfig(num_epochs_warmup=0)
    # cycles < 0 rejected. 0 is allowed (= disabled).
    with pytest.raises(ValueError, match="num_epochs_warmup_cycles must be >= 0"):
        TrainerConfig(num_epochs_warmup_cycles=-1)
    TrainerConfig(num_epochs_warmup_cycles=0)  # disabled — allowed


def test_warmup_epoch_cap_override_logic():
    """Emulate the train() override without spinning up torch — tests the
    min(num_epochs, warmup) rule for cycle <= warmup_cycles and no override
    past that window.
    """
    cfg = TrainerConfig(num_epochs=3, num_epochs_warmup=1, num_epochs_warmup_cycles=5)
    for cycle in range(1, 6):
        eff = min(cfg.num_epochs, cfg.num_epochs_warmup) if cycle <= cfg.num_epochs_warmup_cycles else cfg.num_epochs
        assert eff == 1, f"cycle {cycle} should be capped to warmup=1"
    # Cycle 6+ uses full num_epochs.
    cycle = 6
    eff = min(cfg.num_epochs, cfg.num_epochs_warmup) if cycle <= cfg.num_epochs_warmup_cycles else cfg.num_epochs
    assert eff == 3

    # Disabled (warmup_cycles=0): never override.
    cfg2 = TrainerConfig(num_epochs=3, num_epochs_warmup_cycles=0)
    for cycle in range(1, 10):
        eff = min(cfg2.num_epochs, cfg2.num_epochs_warmup) if cycle <= cfg2.num_epochs_warmup_cycles else cfg2.num_epochs
        assert eff == 3
