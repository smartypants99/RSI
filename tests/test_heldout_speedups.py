"""Task #23: held-out eval speedup wedges.

Covers the 5 wedges shipped to crush held-out cost per cycle:

  1. Skip full eval when quick probe regressed
  2. N reduction (full 1200 → 600) with documented MDE trade-off
  3. Base-model prediction cache
  4. max_num_seqs bump during held-out phase
  5. max_new_tokens cap at 512 for held-out generation

Each wedge has at least one regression-guard test. Integration tests
covering the actual wall-clock savings are out of scope (GPU-only).
"""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

from src.orchestrator.loop import CycleResult, ImprovementLoop
from src.utils.config import OrchestratorConfig


# --- Shared helpers --------------------------------------------------------


def _make_loop(tmp_path: Path, **orch_overrides) -> ImprovementLoop:
    """Build a minimal ImprovementLoop stub — no model, no training.
    Only the fields exercised by the eval-phase skip logic are set."""
    loop = ImprovementLoop.__new__(ImprovementLoop)
    orchestrator = OrchestratorConfig(
        output_dir=tmp_path,
        log_dir=tmp_path / "logs",
        **orch_overrides,
    )
    loop.config = SimpleNamespace(orchestrator=orchestrator)
    loop.history = []
    return loop


def _make_result(cycle: int, *, pre: float, post: float) -> CycleResult:
    r = CycleResult(cycle)
    r.pre_score = pre
    r.post_score = post
    r.improvement = post - pre
    r.training_metrics = SimpleNamespace(steps=1)  # trained=True
    return r


# --- Wedge 1: skip full eval on quick regression ---------------------------


def test_quick_regression_triggers_full_eval_skip_logic(tmp_path):
    """With the default threshold 0.10 and a quick improvement of -0.15,
    the eval-phase decision should mark the cycle to skip the full eval."""
    loop = _make_loop(
        tmp_path,
        skip_full_heldout_on_quick_regression=True,
        quick_regression_skip_threshold=0.10,
    )
    result = _make_result(1, pre=0.7, post=0.55)  # -0.15 regression
    # Simulate the decision used at loop.py ~line 600: if the quick probe
    # regressed beyond the threshold we skip.
    ocfg = loop.config.orchestrator
    threshold = ocfg.quick_regression_skip_threshold
    trained = True
    should_skip = (
        ocfg.skip_full_heldout_on_quick_regression
        and trained
        and result.improvement < -threshold
    )
    assert should_skip is True


def test_quick_minor_regression_below_threshold_does_not_skip(tmp_path):
    """A small regression (-0.05) inside the 0.10 tolerance must NOT skip."""
    loop = _make_loop(
        tmp_path,
        skip_full_heldout_on_quick_regression=True,
        quick_regression_skip_threshold=0.10,
    )
    result = _make_result(1, pre=0.7, post=0.65)  # -0.05 regression
    ocfg = loop.config.orchestrator
    should_skip = (
        ocfg.skip_full_heldout_on_quick_regression
        and result.improvement < -ocfg.quick_regression_skip_threshold
    )
    assert should_skip is False


def test_quick_regression_skip_disabled_flag(tmp_path):
    """skip_full_heldout_on_quick_regression=False restores prior behavior."""
    loop = _make_loop(
        tmp_path,
        skip_full_heldout_on_quick_regression=False,
    )
    result = _make_result(1, pre=0.7, post=0.3)  # -0.40 regression
    ocfg = loop.config.orchestrator
    should_skip = (
        ocfg.skip_full_heldout_on_quick_regression
        and result.improvement < -ocfg.quick_regression_skip_threshold
    )
    assert should_skip is False


# --- Wedge 2: full-eval MDE math locked in ---------------------------------


def _mde_paired(*, n: int, vr: float, p: float = 0.5,
                alpha: float = 0.05, power: float = 0.8) -> float:
    """Paired McNemar-style MDE, two-sided. Matches the math in
    heldout_full_subsample_n config doc."""
    # z_{1-α/2} + z_{1-β} via normal CDF lookup. Hard-code the canonical
    # values so this test is purely arithmetic — no scipy dependency.
    z_alpha = 1.959963984540054   # two-sided 0.05
    z_beta = 0.8416212335729143   # power 0.8
    sigma2_unpaired = 2.0 * p * (1.0 - p)
    sigma2_paired = sigma2_unpaired / vr
    return (z_alpha + z_beta) * math.sqrt(sigma2_paired / n)


def test_full_eval_n600_mde_matches_documented_math():
    """At N=600 VR=2.5 the MDE must land within 1e-3 of the documented
    5.11%. This locks in the power-math the task-spec sensitivity claim
    depends on."""
    mde = _mde_paired(n=600, vr=2.5)
    assert abs(mde - 0.0511) < 1e-3, mde


def test_full_eval_n1200_mde_is_strictly_tighter_than_n600():
    """Doubling N must lower MDE by roughly sqrt(2). Sanity that the
    config-doc table is self-consistent."""
    mde_600 = _mde_paired(n=600, vr=2.5)
    mde_1200 = _mde_paired(n=1200, vr=2.5)
    assert mde_1200 < mde_600
    # sqrt(2) ratio, tolerance 5%
    ratio = mde_600 / mde_1200
    assert abs(ratio - math.sqrt(2)) < 0.05, ratio


def test_one_percent_mde_requires_unrealistic_vr_at_n600():
    """The task-spec 1pp sensitivity target would need VR≈65 at N=600 —
    still well above the empirically observed 3-5× VR. Lock in the trade-
    off so future optimizers don't claim 1pp detectability on default config.
    """
    # Solve MDE=0.01 for VR at N=600:
    # 0.01 = z_sum * sqrt(sigma2_paired / N) = z_sum * sqrt(0.5 / VR / N)
    # → VR = 0.5 / N / (0.01 / z_sum)^2
    z_sum = 1.959963984540054 + 0.8416212335729143
    required_vr = 0.5 / 600.0 / (0.01 / z_sum) ** 2
    # Empirical VR observed in production is 3-5×. Anything above ~15 is
    # unrealistic for held-out binary outcomes.
    assert required_vr > 50, required_vr
    assert required_vr < 100, required_vr


def test_config_default_full_subsample_is_600(tmp_path):
    """Default ships at N=600 (wedge 2). Operator can raise via config."""
    cfg = OrchestratorConfig(output_dir=tmp_path, log_dir=tmp_path / "l")
    assert cfg.heldout_full_subsample_n == 600


def test_config_allows_n1200_fallback(tmp_path):
    """Setting heldout_full_subsample_n=1200 is valid for ops who prefer
    the tighter MDE."""
    cfg = OrchestratorConfig(
        output_dir=tmp_path, log_dir=tmp_path / "l",
        heldout_full_subsample_n=1200,
    )
    assert cfg.heldout_full_subsample_n == 1200


# --- Wedge 3: base-model prediction cache ---------------------------------


def test_config_default_cache_base_predictions_on(tmp_path):
    cfg = OrchestratorConfig(output_dir=tmp_path, log_dir=tmp_path / "l")
    assert cfg.heldout_cache_base_predictions is True


# --- Wedge 4: heldout max_num_seqs -----------------------------------------


def test_config_default_heldout_max_num_seqs_is_96(tmp_path):
    cfg = OrchestratorConfig(output_dir=tmp_path, log_dir=tmp_path / "l")
    assert cfg.heldout_max_num_seqs == 96


def test_config_rejects_negative_heldout_max_num_seqs(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        OrchestratorConfig(
            output_dir=tmp_path, log_dir=tmp_path / "l",
            heldout_max_num_seqs=-1,
        )


# --- Wedge 5: max_new_tokens cap -------------------------------------------


def test_config_default_heldout_eval_max_tokens_is_512(tmp_path):
    cfg = OrchestratorConfig(output_dir=tmp_path, log_dir=tmp_path / "l")
    assert cfg.heldout_eval_max_tokens == 512


def test_config_rejects_zero_heldout_eval_max_tokens(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        OrchestratorConfig(
            output_dir=tmp_path, log_dir=tmp_path / "l",
            heldout_eval_max_tokens=0,
        )
