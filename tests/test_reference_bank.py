"""Reference-bank + capture-alarm invariants (orchestrator-auditor, task #2).

Pins the behaviour that prevents the cycle-1 outlier lock-in we observed on
the 7-cycle overnight run (cycle-1 held-out 0.624 from a 1-sample/2-step
cycle became an unassailable reference bank; cycles 2-6 all reverted to it).

Invariants under test:

1. An INELIGIBLE high score (samples_verified < best_min_samples_verified)
   never gets promoted, and does not start a pending-best streak.
2. An ELIGIBLE high score requires N≥best_confirm_cycles consecutive eligible
   cycles at or above the candidate before it is promoted to best.
3. A verifier_capture_alarm cycle is ineligible for best-promotion, AND
   resets any pending-best streak (so a captured cycle can never become
   the reference bank even in isolation).
4. Until the first best is confirmed, _best_score stays 0 (no outlier
   reference in the revert path).

The trainer / model are fully mocked — these tests exercise only the
state-machine in _check_early_stopping.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.orchestrator.loop import CycleResult, ImprovementLoop
from src.utils.config import OrchestratorConfig


def _bare_loop(tmp_path: Path, *, min_samples: int = 8, confirm_n: int = 2) -> ImprovementLoop:
    loop = ImprovementLoop.__new__(ImprovementLoop)
    loop.config = SimpleNamespace(
        orchestrator=OrchestratorConfig(
            output_dir=tmp_path,
            log_dir=tmp_path / "logs",
            best_min_samples_verified=min_samples,
            best_confirm_cycles=confirm_n,
        )
    )
    loop.model_loader = SimpleNamespace(model_path="base")
    loop._use_vllm = False
    loop._best_score = 0.0
    loop._best_checkpoint_cycle = None
    loop._degradation_count = 0
    loop._pending_best_score = 0.0
    loop._pending_best_cycle = None
    loop._pending_best_streak = 0
    loop.history = []
    return loop


def _result(cycle: int, *, eval_score: float, samples_verified: int = 20,
            capture_alarm: bool = False, regression_reverted: bool = False) -> CycleResult:
    r = CycleResult(cycle)
    r.eval_score = eval_score
    r.post_score = eval_score
    r.samples_verified = samples_verified
    r.verifier_capture_alarm = capture_alarm
    r.regression_reverted = regression_reverted
    return r


# ---------------------------------------------------------------------------
# Outlier-reference bug (cycle-1 1-sample lock-in) — regression guard.
# ---------------------------------------------------------------------------

def test_tiny_sample_outlier_is_not_promoted(tmp_path):
    """The exact overnight-run scenario: cycle 1 eval=0.624, samples_verified=1."""
    loop = _bare_loop(tmp_path, min_samples=8, confirm_n=2)
    r = _result(1, eval_score=0.624, samples_verified=1)
    loop.history.append(r)
    loop._check_early_stopping(1, r)
    assert loop._best_score == 0.0
    assert loop._best_checkpoint_cycle is None
    assert loop._pending_best_streak == 0  # ineligible resets, not starts


def test_tiny_sample_outlier_repeated_still_not_promoted(tmp_path):
    """Even if cycles 2..N repeat the 0.624, none are eligible → no promotion."""
    loop = _bare_loop(tmp_path, min_samples=8, confirm_n=2)
    for c in range(1, 6):
        r = _result(c, eval_score=0.624, samples_verified=1)
        loop.history.append(r)
        loop._check_early_stopping(c, r)
    assert loop._best_score == 0.0
    assert loop._best_checkpoint_cycle is None


# ---------------------------------------------------------------------------
# Confirmation-gate: N consecutive eligible cycles required.
# ---------------------------------------------------------------------------

def test_single_eligible_cycle_does_not_promote(tmp_path):
    loop = _bare_loop(tmp_path, min_samples=8, confirm_n=2)
    r = _result(1, eval_score=0.55, samples_verified=20)
    loop.history.append(r)
    loop._check_early_stopping(1, r)
    # pending but not promoted
    assert loop._best_score == 0.0
    assert loop._pending_best_cycle == 1
    assert loop._pending_best_streak == 1


def test_two_consecutive_eligible_cycles_promote(tmp_path):
    loop = _bare_loop(tmp_path, min_samples=8, confirm_n=2)
    r1 = _result(1, eval_score=0.55, samples_verified=20)
    loop.history.append(r1)
    loop._check_early_stopping(1, r1)
    r2 = _result(2, eval_score=0.56, samples_verified=20)
    loop.history.append(r2)
    loop._check_early_stopping(2, r2)
    # After 2 consecutive eligible cycles, first eligible cycle is promoted.
    assert loop._best_score == pytest.approx(0.55)
    assert loop._best_checkpoint_cycle == 1


def test_drop_below_candidate_resets_streak(tmp_path):
    loop = _bare_loop(tmp_path, min_samples=8, confirm_n=3)
    for c, s in [(1, 0.60), (2, 0.58), (3, 0.61)]:
        r = _result(c, eval_score=s, samples_verified=20)
        loop.history.append(r)
        loop._check_early_stopping(c, r)
    # Cycle 2 dropped below the cycle-1 candidate → streak reset; cycle 3
    # starts a new candidate. confirm_n=3 and only 1 in streak → no best yet.
    assert loop._best_score == 0.0


# ---------------------------------------------------------------------------
# Capture-alarm response: ineligible + resets streak.
# ---------------------------------------------------------------------------

def test_capture_alarm_cycle_is_ineligible_for_promotion(tmp_path):
    """Cycle 7 scenario: internal up, anchor down → capture alarm fired."""
    loop = _bare_loop(tmp_path, min_samples=8, confirm_n=2)
    # Build up a pending candidate first.
    r1 = _result(1, eval_score=0.55, samples_verified=20)
    loop.history.append(r1)
    loop._check_early_stopping(1, r1)
    assert loop._pending_best_streak == 1
    # Capture-alarm cycle scores higher but is ineligible.
    r2 = _result(2, eval_score=0.62, samples_verified=20, capture_alarm=True)
    loop.history.append(r2)
    loop._check_early_stopping(2, r2)
    # Streak reset, not promoted.
    assert loop._best_score == 0.0
    assert loop._pending_best_streak == 0
    assert loop._best_checkpoint_cycle is None


def test_capture_alarm_cannot_become_reference_ever(tmp_path):
    """Even with confirm_n=1, a capture-alarm cycle cannot become best."""
    loop = _bare_loop(tmp_path, min_samples=0, confirm_n=1)
    r = _result(1, eval_score=0.90, samples_verified=1, capture_alarm=True)
    loop.history.append(r)
    loop._check_early_stopping(1, r)
    assert loop._best_score == 0.0
    assert loop._best_checkpoint_cycle is None


# ---------------------------------------------------------------------------
# Revert-reference: falls back to pre_score when no confirmed best exists.
# ---------------------------------------------------------------------------

def test_revert_reference_prefers_trimmed_mean_when_best_is_stale(tmp_path):
    """Once local level drifts down, the revert reference should drop with it.
    A stale confirmed best must not trap us into perpetual reverts."""
    loop = _bare_loop(tmp_path, min_samples=8, confirm_n=2)
    loop._best_score = 0.624  # stale historical high
    # Last K eval scores have drifted to ~0.30.
    for c, s in enumerate([0.32, 0.28, 0.31, 0.29, 0.30], start=1):
        r = _result(c, eval_score=s, samples_verified=20)
        loop.history.append(r)
    cur = _result(6, eval_score=0.25, samples_verified=20)
    ref = loop._revert_reference(cur)
    # Trimmed mean of {0.32,0.28,0.31,0.29,0.30} = mean of middle 3 = 0.30.
    # min(0.624, 0.30) == 0.30.
    assert ref == pytest.approx(0.30, abs=0.01)


def test_revert_reference_uses_best_when_fresh(tmp_path):
    loop = _bare_loop(tmp_path, min_samples=8, confirm_n=2)
    loop._best_score = 0.60
    # Recent cycles all around 0.62 — trimmed mean > best.
    for c, s in enumerate([0.61, 0.62, 0.63, 0.62, 0.63], start=1):
        r = _result(c, eval_score=s, samples_verified=20)
        loop.history.append(r)
    cur = _result(6, eval_score=0.30, samples_verified=20)
    ref = loop._revert_reference(cur)
    # min(0.60, trimmed_mean~0.62) == 0.60.
    assert ref == pytest.approx(0.60, abs=0.01)


def test_revert_reference_excludes_capture_alarm_cycles(tmp_path):
    loop = _bare_loop(tmp_path, min_samples=8, confirm_n=2)
    loop._best_score = 0.0
    # One captured cycle at 0.80 + four clean cycles at 0.40.
    r1 = _result(1, eval_score=0.80, samples_verified=20, capture_alarm=True)
    loop.history.append(r1)
    for c, s in enumerate([0.40, 0.41, 0.39, 0.40], start=2):
        r = _result(c, eval_score=s, samples_verified=20)
        loop.history.append(r)
    cur = _result(6, eval_score=0.25, samples_verified=20)
    ref = loop._revert_reference(cur)
    # Captured cycle excluded → trimmed mean over {0.40,0.41,0.39,0.40}.
    assert ref == pytest.approx(0.40, abs=0.02)


def test_revert_reference_fallback_to_pre_score(tmp_path):
    loop = _bare_loop(tmp_path, min_samples=8, confirm_n=2)
    # Empty history, no confirmed best.
    cur = _result(1, eval_score=0.30, samples_verified=20)
    cur.pre_score = 0.50
    ref = loop._revert_reference(cur)
    assert ref == pytest.approx(0.50)


def test_no_confirmed_best_means_reference_is_not_outlier(tmp_path):
    """The revert-guard uses self._best_score as reference. Until a best is
    confirmed, _best_score stays 0 — so the guard falls back to pre_score,
    which is the cycle's own pre-training baseline (robust, per-cycle)."""
    loop = _bare_loop(tmp_path, min_samples=8, confirm_n=2)
    # Several ineligible-high cycles shouldn't leak into _best_score.
    for c in range(1, 5):
        r = _result(c, eval_score=0.80, samples_verified=2)
        loop.history.append(r)
        loop._check_early_stopping(c, r)
    assert loop._best_score == 0.0


# ---------------------------------------------------------------------------
# Task #15: regression-revert bars streak advancement.
# ---------------------------------------------------------------------------

def test_regression_reverted_cycle_does_not_advance_streak(tmp_path):
    """Live bug: cycle 2 held-out=0.478 vs reference=0.633 was reverted,
    yet the next log line was 'streak=1/2 — awaiting confirmation'."""
    loop = _bare_loop(tmp_path, min_samples=8, confirm_n=2)
    # Cycle 1: clean, builds pending streak=1 at 0.633.
    r1 = _result(1, eval_score=0.633, samples_verified=20)
    loop.history.append(r1)
    loop._check_early_stopping(1, r1)
    assert loop._pending_best_streak == 1
    assert loop._pending_best_cycle == 1

    # Cycle 2: regression-revert fires (eval 0.478 vs ref 0.633).
    r2 = _result(2, eval_score=0.478, samples_verified=20, regression_reverted=True)
    loop.history.append(r2)
    loop._check_early_stopping(2, r2)

    # Streak must NOT be at 1/2; pending state must be cleared.
    assert loop._pending_best_streak == 0
    assert loop._pending_best_cycle is None
    assert loop._pending_best_score == 0.0
    assert loop._best_score == 0.0
    assert loop._best_checkpoint_cycle is None


def test_regression_reverted_blocks_even_highest_score(tmp_path):
    """Even if the reverted cycle's score exceeds all prior, revert wins."""
    loop = _bare_loop(tmp_path, min_samples=0, confirm_n=1)
    r = _result(1, eval_score=0.99, samples_verified=20, regression_reverted=True)
    loop.history.append(r)
    loop._check_early_stopping(1, r)
    assert loop._best_score == 0.0
    assert loop._pending_best_streak == 0


def test_regression_vs_best_blocks_streak_advance(tmp_path):
    """Post-promotion: current_score < _best_score - 0.005 must not advance."""
    loop = _bare_loop(tmp_path, min_samples=8, confirm_n=2)
    loop._best_score = 0.60
    loop._best_checkpoint_cycle = 1
    # Cycle scoring 0.55 (0.05 below best) — must not advance streak.
    r = _result(2, eval_score=0.55, samples_verified=20)
    loop.history.append(r)
    loop._check_early_stopping(2, r)
    assert loop._pending_best_streak == 0
    assert loop._best_score == 0.60  # unchanged


def test_within_tolerance_of_best_does_advance(tmp_path):
    """current_score = best_score - 0.003 is within 0.005 tolerance — should
    be treated as not-regressed-vs-best, though it still needs > best to
    enter the candidate branch. This pins the tolerance boundary."""
    loop = _bare_loop(tmp_path, min_samples=8, confirm_n=2)
    loop._best_score = 0.60
    loop._best_checkpoint_cycle = 1
    # current_score > best_score: enter candidate branch.
    r = _result(2, eval_score=0.605, samples_verified=20)
    loop.history.append(r)
    loop._check_early_stopping(2, r)
    # Pending started (streak=1). Not promoted yet.
    assert loop._pending_best_streak == 1
    assert loop._pending_best_cycle == 2
