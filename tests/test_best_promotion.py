"""Task #2: lagged-confirmation best-promotion + eligibility gate.

The overnight run pinned cycle 1's 1-sample/2-step eval=0.624 as the reference
bank; cycles 2-6 all reverted to it and the loop produced zero forward progress
for 6 hours. These tests lock in the fix:

  - A cycle with samples_verified below the min floor cannot become best
    even if its score beats the current best.
  - A cycle flagged verifier_capture_alarm cannot become best.
  - A candidate new-best must be held for best_confirm_cycles consecutive
    eligible cycles before it replaces the reference.
  - An interrupting lower / ineligible cycle resets the pending streak.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.orchestrator.loop import CycleResult, ImprovementLoop
from src.utils.config import OrchestratorConfig


def _make_loop(tmp_path: Path, *, confirm_n: int = 2, min_samples: int = 8):
    loop = ImprovementLoop.__new__(ImprovementLoop)
    orchestrator = OrchestratorConfig(
        output_dir=tmp_path,
        log_dir=tmp_path / "logs",
        best_confirm_cycles=confirm_n,
        best_min_samples_verified=min_samples,
    )
    loop.config = SimpleNamespace(orchestrator=orchestrator)
    loop.history = []
    loop._best_score = 0.0
    loop._best_checkpoint_cycle = None
    loop._degradation_count = 0
    loop._pending_best_score = 0.0
    loop._pending_best_cycle = None
    loop._pending_best_streak = 0
    loop._capture_alarm_consecutive = 0
    return loop


def _make_result(cycle: int, *, eval_score: float, samples_verified: int = 32,
                 capture_alarm: bool = False,
                 mode_collapse: bool = False) -> CycleResult:
    r = CycleResult(cycle)
    r.eval_score = eval_score
    r.samples_verified = samples_verified
    r.verifier_capture_alarm = capture_alarm
    r.mode_collapse_detected = mode_collapse
    return r


def test_tiny_sample_cycle_cannot_become_best(tmp_path):
    """The core overnight bug: 1-sample/2-step cycle 1 pinned the reference."""
    loop = _make_loop(tmp_path, confirm_n=2, min_samples=8)
    result = _make_result(1, eval_score=0.624, samples_verified=1)
    loop.history.append(result)
    loop._check_early_stopping(1, result)
    assert loop._best_checkpoint_cycle is None
    assert loop._best_score == 0.0
    # And crucially — the pending streak should NOT be started by an ineligible
    # cycle. Otherwise two consecutive 1-sample flukes would confirm each other.
    assert loop._pending_best_streak == 0


def test_capture_alarm_cycle_cannot_become_best(tmp_path):
    loop = _make_loop(tmp_path, confirm_n=2, min_samples=8)
    result = _make_result(1, eval_score=0.8, samples_verified=64,
                          capture_alarm=True)
    loop.history.append(result)
    loop._check_early_stopping(1, result)
    assert loop._best_checkpoint_cycle is None
    assert loop._best_score == 0.0


def test_promotion_requires_two_consecutive_confirmations(tmp_path):
    loop = _make_loop(tmp_path, confirm_n=2, min_samples=8)
    # Cycle 1 proposes high-water 0.7 — pending, not promoted.
    r1 = _make_result(1, eval_score=0.7, samples_verified=32)
    loop.history.append(r1)
    loop._check_early_stopping(1, r1)
    assert loop._best_checkpoint_cycle is None
    assert loop._pending_best_streak == 1
    assert loop._pending_best_cycle == 1
    # Cycle 2 confirms at-or-above → promote, with checkpoint pointer at
    # the EARLIER cycle whose weights actually produced the score.
    r2 = _make_result(2, eval_score=0.71, samples_verified=32)
    loop.history.append(r2)
    loop._check_early_stopping(2, r2)
    assert loop._best_checkpoint_cycle == 1
    assert loop._best_score == 0.7


def test_confirmation_requires_consecutive_at_or_above(tmp_path):
    """Confirmation counts only consecutive cycles at-or-above pending mark.

    Task #11 concern #3: a cycle that REGRESSES vs. a prior cycle's eval
    score must NOT advance or re-open the pending-best streak, even if
    its score strictly exceeds the (possibly 0.0) confirmed _best_score.
    Before this fix, overnight cycle 2 logged "streak=1/2 — awaiting
    confirmation" despite being reverted vs. reference.
    """
    loop = _make_loop(tmp_path, confirm_n=3, min_samples=8)
    r1 = _make_result(1, eval_score=0.7, samples_verified=32)
    loop.history.append(r1)
    loop._check_early_stopping(1, r1)
    assert loop._pending_best_streak == 1
    # Cycle 2 regresses (0.55 < 0.7 - tol). Despite 0.55 > _best_score (0.0),
    # the regression guard must refuse to open a new pending streak.
    r2 = _make_result(2, eval_score=0.55, samples_verified=32)
    loop.history.append(r2)
    loop._check_early_stopping(2, r2)
    assert loop._pending_best_streak == 0
    assert loop._pending_best_cycle is None
    assert loop._pending_best_score == 0.0
    # Cycle 3 ALSO regresses vs. cycle 1 (0.56 < 0.7 - tol). Still no streak.
    r3 = _make_result(3, eval_score=0.56, samples_verified=32)
    loop.history.append(r3)
    loop._check_early_stopping(3, r3)
    assert loop._pending_best_streak == 0
    # Cycle 4 recovers AT-OR-ABOVE the prior max (0.71 >= 0.7 - tol). A fresh
    # pending streak opens — this is a legitimate new high-water candidate.
    r4 = _make_result(4, eval_score=0.71, samples_verified=32)
    loop.history.append(r4)
    loop._check_early_stopping(4, r4)
    assert loop._pending_best_streak == 1
    assert loop._pending_best_score == 0.71
    assert loop._pending_best_cycle == 4


def test_ineligible_cycle_sets_high_water_mark(tmp_path):
    """Task #29: an ineligible cycle's eval IS a valid high-water mark.
    The eval is of the frozen model state; how many training samples
    produced that state has no bearing on whether the eval number is
    trustworthy. Future cycles must beat that high-water to promote.
    Live bug: cycle 1 (0.562, 1 sample) ineligible; cycle 2 (0.539, 15
    samples) eligible; cycle 3 confirmed cycle 2 as best — but 0.539 < 0.562.
    """
    loop = _make_loop(tmp_path, confirm_n=2, min_samples=8)
    # Cycle 1: eval high but samples low → ineligible for promotion, but
    # its eval score sets the high-water bar.
    r1 = _make_result(1, eval_score=0.75, samples_verified=2)
    loop.history.append(r1)
    loop._check_early_stopping(1, r1)
    assert loop._pending_best_streak == 0  # ineligible, no streak
    # Cycle 2: eligible samples, but score BELOW cycle-1 high-water.
    # Must NOT advance streak despite beating _best_score=0.
    r2 = _make_result(2, eval_score=0.7, samples_verified=32)
    loop.history.append(r2)
    loop._check_early_stopping(2, r2)
    assert loop._pending_best_streak == 0
    assert loop._best_checkpoint_cycle is None
    # Cycle 3: also below high-water — still no streak.
    r3 = _make_result(3, eval_score=0.72, samples_verified=32)
    loop.history.append(r3)
    loop._check_early_stopping(3, r3)
    assert loop._pending_best_streak == 0
    # Cycle 4: finally beats high-water (within tolerance) → streak opens.
    r4 = _make_result(4, eval_score=0.75, samples_verified=32)
    loop.history.append(r4)
    loop._check_early_stopping(4, r4)
    assert loop._pending_best_streak == 1
    assert loop._pending_best_score == 0.75


def test_confirm_one_restores_legacy_behavior(tmp_path):
    """confirm_n=1 makes promotion happen on first eligible beat, like before."""
    loop = _make_loop(tmp_path, confirm_n=1, min_samples=8)
    r1 = _make_result(1, eval_score=0.7, samples_verified=32)
    loop.history.append(r1)
    loop._check_early_stopping(1, r1)
    assert loop._best_checkpoint_cycle == 1
    assert loop._best_score == 0.7


def test_config_rejects_bad_values():
    import pytest
    with pytest.raises(ValueError):
        OrchestratorConfig(best_confirm_cycles=0)
    with pytest.raises(ValueError):
        OrchestratorConfig(best_min_samples_verified=-1)
    with pytest.raises(ValueError):
        OrchestratorConfig(verifier_capture_halt_consecutive=0)


# --- Task #11 concern #3: regression guard for streak-advance ---


def test_streak_does_not_advance_on_regression_vs_prior_eligible(tmp_path):
    """Cycle 2 with score < cycle 1's score (both eligible) must NOT log
    'streak=1/2' — that was the overnight-run false-positive where cycle 2
    ticked streak=1 despite being reverted vs. reference."""
    loop = _make_loop(tmp_path, confirm_n=2, min_samples=8)
    r1 = _make_result(1, eval_score=0.70, samples_verified=32)
    loop.history.append(r1)
    loop._check_early_stopping(1, r1)
    assert loop._pending_best_streak == 1
    # Cycle 2: regressed vs. cycle 1 (0.60 < 0.70 - 0.005). Must NOT advance.
    r2 = _make_result(2, eval_score=0.60, samples_verified=32)
    loop.history.append(r2)
    loop._check_early_stopping(2, r2)
    assert loop._pending_best_streak == 0
    assert loop._pending_best_cycle is None
    assert loop._pending_best_score == 0.0
    # _best_score stays unpromoted — regression can't accidentally confirm.
    assert loop._best_score == 0.0
    assert loop._best_checkpoint_cycle is None


def test_streak_advances_on_recovery_to_prior_high(tmp_path):
    """After a regression, a recovery cycle at-or-above the prior high must
    open a fresh pending streak (not be blocked by the regressed cycle)."""
    loop = _make_loop(tmp_path, confirm_n=2, min_samples=8)
    r1 = _make_result(1, eval_score=0.70, samples_verified=32)
    loop.history.append(r1)
    loop._check_early_stopping(1, r1)
    r2 = _make_result(2, eval_score=0.60, samples_verified=32)
    loop.history.append(r2)
    loop._check_early_stopping(2, r2)
    assert loop._pending_best_streak == 0
    # Cycle 3 recovers to 0.72 > prior_max 0.70 → fresh pending streak.
    r3 = _make_result(3, eval_score=0.72, samples_verified=32)
    loop.history.append(r3)
    loop._check_early_stopping(3, r3)
    assert loop._pending_best_streak == 1
    assert loop._pending_best_score == 0.72


def test_regression_guard_honors_suspect_eval_exclusions(tmp_path):
    """Task #29 semantics: eval_score of ANY prior cycle is a valid high-water
    mark UNLESS that eval itself is suspect. Low training-sample count does NOT
    make the eval suspect (eval is on the frozen held-out set independent of
    training); but capture_alarm and mode_collapse DO make the eval suspect
    (the model is gaming or producing degenerate outputs). This test pins
    that only capture-alarm / mode-collapse priors are excluded, not tiny-
    sample priors."""
    loop = _make_loop(tmp_path, confirm_n=2, min_samples=8)
    # Cycle 1: capture alarm tripped — eval is suspect, excluded from high-water.
    r1 = _make_result(1, eval_score=0.90, samples_verified=32,
                      capture_alarm=True)
    loop.history.append(r1)
    loop._check_early_stopping(1, r1)
    # Cycle 2: legitimate 0.70 with no suspicion → opens pending streak
    # despite cycle 1's suspect 0.90.
    r2 = _make_result(2, eval_score=0.70, samples_verified=32)
    loop.history.append(r2)
    loop._check_early_stopping(2, r2)
    assert loop._pending_best_streak == 1
    assert loop._pending_best_score == 0.70


# --- Task #11 concern #2: mode-collapse ineligibility ---


def test_mode_collapse_cycle_cannot_become_best(tmp_path):
    """A cycle with mode_collapse_detected=True is ineligible for best-
    promotion even if its samples count and score are otherwise strong."""
    loop = _make_loop(tmp_path, confirm_n=2, min_samples=8)
    r1 = _make_result(1, eval_score=0.80, samples_verified=64,
                      mode_collapse=True)
    loop.history.append(r1)
    loop._check_early_stopping(1, r1)
    assert loop._pending_best_streak == 0
    assert loop._best_checkpoint_cycle is None
    assert loop._best_score == 0.0
