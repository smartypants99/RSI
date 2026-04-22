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
                 capture_alarm: bool = False) -> CycleResult:
    r = CycleResult(cycle)
    r.eval_score = eval_score
    r.samples_verified = samples_verified
    r.verifier_capture_alarm = capture_alarm
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
    """Confirmation counts only consecutive cycles at-or-above pending mark."""
    loop = _make_loop(tmp_path, confirm_n=3, min_samples=8)
    r1 = _make_result(1, eval_score=0.7, samples_verified=32)
    loop.history.append(r1)
    loop._check_early_stopping(1, r1)
    assert loop._pending_best_streak == 1
    # Cycle 2 below pending → the pending mark is dropped (new candidate).
    r2 = _make_result(2, eval_score=0.55, samples_verified=32)
    loop.history.append(r2)
    loop._check_early_stopping(2, r2)
    # 0.55 > 0 (_best_score still 0) so it opens a fresh pending streak.
    assert loop._pending_best_score == 0.55
    assert loop._pending_best_streak == 1
    # Cycle 3 confirms 0.55 → streak advances to 2 of 3.
    r3 = _make_result(3, eval_score=0.56, samples_verified=32)
    loop.history.append(r3)
    loop._check_early_stopping(3, r3)
    assert loop._pending_best_streak == 2
    # Cycle 4 regresses BELOW _best_score=0 is impossible, but 0.55 pending
    # → a score below pending but above best should NOT continue the streak.
    r4 = _make_result(4, eval_score=0.54, samples_verified=32)
    loop.history.append(r4)
    loop._check_early_stopping(4, r4)
    assert loop._pending_best_streak == 1
    assert loop._pending_best_score == 0.54


def test_interrupting_ineligible_cycle_resets_pending(tmp_path):
    """An ineligible cycle between two eligible highs must NOT bridge them."""
    loop = _make_loop(tmp_path, confirm_n=2, min_samples=8)
    r1 = _make_result(1, eval_score=0.7, samples_verified=32)
    loop.history.append(r1)
    loop._check_early_stopping(1, r1)
    assert loop._pending_best_streak == 1
    # Cycle 2: would beat the bar but samples too low → ineligible, resets.
    r2 = _make_result(2, eval_score=0.75, samples_verified=2)
    loop.history.append(r2)
    loop._check_early_stopping(2, r2)
    assert loop._pending_best_streak == 0
    # Cycle 3: back to eligible at the bar — starts a fresh streak, no promote.
    r3 = _make_result(3, eval_score=0.7, samples_verified=32)
    loop.history.append(r3)
    loop._check_early_stopping(3, r3)
    assert loop._best_checkpoint_cycle is None
    assert loop._pending_best_streak == 1


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
