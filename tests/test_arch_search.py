"""Tests for src/trainer/arch_search.py — motif proposal loop."""

from __future__ import annotations

import torch.nn as nn

from src.trainer.arch_search import (
    ArchSearchResult,
    DEFAULT_MOTIFS,
    motif_sparse_moe_ffn,
    run_arch_search,
)
from src.trainer.growth import GrowthConfig


class _Skel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)


def _cfg(enabled: bool = True, min_delta: float = 0.005) -> GrowthConfig:
    return GrowthConfig(
        arch_search_enabled=enabled,
        arch_search_every=30,
        arch_search_min_delta=min_delta,
    )


def test_run_arch_search_disabled_returns_empty():
    result = run_arch_search(
        _Skel(),
        _cfg(enabled=False),
        train_fn=lambda m: m,
        eval_fn=lambda m: 1.0,
    )
    assert result == ArchSearchResult()
    assert result.accepted_motif is None


def test_run_arch_search_accepts_first_improving_motif():
    scores = {"baseline": 0.50, "sparse_moe_ffn": 0.60}
    # Only sparse_moe clears the bar; others return baseline.
    def eval_fn(m):
        motif = getattr(m, "_arch_motif", "baseline")
        return scores.get(motif, 0.50)

    result = run_arch_search(
        _Skel(),
        _cfg(min_delta=0.05),
        train_fn=lambda m: m,
        eval_fn=eval_fn,
    )
    assert result.accepted_motif == "sparse_moe_ffn"
    assert result.best_delta > 0.05
    assert result.accepted_model is not None


def test_run_arch_search_rejects_when_below_min_delta():
    # Every motif only gives 0.001 delta; min_delta=0.005 → all rejected.
    def eval_fn(m):
        return 0.50 + (0.001 if hasattr(m, "_arch_motif") else 0.0)

    result = run_arch_search(
        _Skel(),
        _cfg(min_delta=0.005),
        train_fn=lambda m: m,
        eval_fn=eval_fn,
    )
    assert result.accepted_motif is None
    assert len(result.trials) == len(DEFAULT_MOTIFS)
    assert all(not t.accepted for t in result.trials)


def test_motif_sparse_moe_ffn_tags_but_does_not_mutate_teacher():
    teacher = _Skel()
    student = motif_sparse_moe_ffn(teacher)
    assert student is not teacher
    assert student._arch_motif == "sparse_moe_ffn"
    assert not hasattr(teacher, "_arch_motif")


def test_train_fn_is_called_on_baseline_and_each_motif():
    calls = []
    def train_fn(m):
        calls.append(getattr(m, "_arch_motif", "baseline"))
        return m
    run_arch_search(
        _Skel(),
        _cfg(min_delta=99.0),  # nothing accepted → all motifs tried
        train_fn=train_fn,
        eval_fn=lambda m: 0.0,
    )
    assert calls[0] == "baseline"
    assert set(calls[1:]) == {name for name, _ in DEFAULT_MOTIFS}
