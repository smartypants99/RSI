"""Tests for src/trainer/stability.py — gradient-norm tracker + divergence detector."""

from __future__ import annotations

import pytest

from src.trainer.stability import (
    DEFAULT_BLOWUP_RATIO,
    DEFAULT_COLLAPSE_FLOOR,
    DEFAULT_CV_LIMIT,
    DEFAULT_HISTORY_WINDOW,
    TRIGGER_GRAD_BLOWUP,
    TRIGGER_GRAD_COLLAPSE,
    TRIGGER_GRAD_INSTABILITY,
    TRIGGER_NONE,
    CycleGradSummary,
    GradientNormTracker,
    detect_gradient_divergence,
    summarize_grad_norms,
)


def test_summarize_empty_grad_norms_is_safe():
    s = summarize_grad_norms(cycle=1, grad_norms=[], lora_weight_delta_norm=0.0)
    assert s.n_steps == 0
    assert s.grad_norm_mean == 0.0
    assert s.lora_weight_delta_norm == 0.0


def test_summarize_populates_quantiles():
    s = summarize_grad_norms(cycle=2, grad_norms=[1.0, 2.0, 3.0, 4.0, 5.0],
                             lora_weight_delta_norm=0.25)
    assert s.n_steps == 5
    assert s.grad_norm_min == 1.0
    assert s.grad_norm_max == 5.0
    assert s.grad_norm_median == 3.0
    assert 0.0 < s.grad_norm_p10 <= s.grad_norm_p90
    assert s.lora_weight_delta_norm == 0.25


def test_summary_coefficient_of_variation():
    s = summarize_grad_norms(cycle=0, grad_norms=[1.0, 1.0, 1.0], lora_weight_delta_norm=0.0)
    assert s.coefficient_of_variation == 0.0
    s2 = summarize_grad_norms(cycle=0, grad_norms=[0.1, 10.0, 0.1, 10.0],
                              lora_weight_delta_norm=0.0)
    assert s2.coefficient_of_variation > 0.5


def _mk(cycle, med, std=0.0, mean=None):
    mean = med if mean is None else mean
    return CycleGradSummary(
        cycle=cycle, n_steps=10,
        grad_norm_mean=mean, grad_norm_median=med, grad_norm_std=std,
        grad_norm_max=med + std, grad_norm_min=max(0.0, med - std),
        grad_norm_p10=max(0.0, med - std), grad_norm_p90=med + std,
        lora_weight_delta_norm=0.01,
    )


def test_detect_returns_none_when_insufficient_history():
    assert detect_gradient_divergence([_mk(0, 1.0)]) == TRIGGER_NONE
    # Fewer cycles than window → TRIGGER_NONE.
    assert detect_gradient_divergence([_mk(i, 1.0) for i in range(DEFAULT_HISTORY_WINDOW - 1)]) == TRIGGER_NONE


def test_detect_blowup():
    hist = [_mk(i, 1.0) for i in range(DEFAULT_HISTORY_WINDOW - 1)]
    hist.append(_mk(99, DEFAULT_BLOWUP_RATIO * 1.0 + 0.5))
    assert detect_gradient_divergence(hist) == TRIGGER_GRAD_BLOWUP


def test_detect_collapse():
    hist = [_mk(i, DEFAULT_COLLAPSE_FLOOR / 10.0) for i in range(DEFAULT_HISTORY_WINDOW)]
    assert detect_gradient_divergence(hist) == TRIGGER_GRAD_COLLAPSE


def test_detect_instability_cv():
    hist = [_mk(i, 1.0, std=0.1) for i in range(DEFAULT_HISTORY_WINDOW - 1)]
    # Last cycle: huge std relative to mean.
    hist.append(_mk(99, 1.0, std=5.0, mean=1.0))
    assert detect_gradient_divergence(hist) == TRIGGER_GRAD_INSTABILITY


def test_detect_healthy_returns_none():
    hist = [_mk(i, 1.0, std=0.2) for i in range(DEFAULT_HISTORY_WINDOW)]
    assert detect_gradient_divergence(hist) == TRIGGER_NONE


def test_detect_accepts_dict_history():
    hist = [_mk(i, 1.0).to_dict() for i in range(DEFAULT_HISTORY_WINDOW - 1)]
    hist.append(_mk(99, DEFAULT_BLOWUP_RATIO * 1.0 + 0.5).to_dict())
    assert detect_gradient_divergence(hist) == TRIGGER_GRAD_BLOWUP


def test_window_must_be_at_least_2():
    with pytest.raises(ValueError):
        detect_gradient_divergence([_mk(0, 1.0)], window=1)


def test_tracker_record_and_end_without_torch():
    t = GradientNormTracker()
    t.begin_cycle(cycle=3)
    for g in [0.5, 1.0, 1.5]:
        t.record_step_norm(g)
    s = t.end_cycle()
    assert s.cycle == 3
    assert s.n_steps == 3
    assert s.grad_norm_median == 1.0
    # No params tracked → delta norm is zero.
    assert s.lora_weight_delta_norm == 0.0


def test_tracker_install_on_optimizer_requires_torch():
    torch = pytest.importorskip("torch")
    t = GradientNormTracker()
    p = torch.nn.Parameter(torch.zeros(4, requires_grad=True))
    p.grad = torch.ones(4)
    opt = torch.optim.SGD([p], lr=0.01)
    t.begin_cycle(cycle=5, lora_params=[p])
    t.install_on_optimizer(opt)
    try:
        opt.step()
    finally:
        t.uninstall_on_optimizer(opt)
    s = t.end_cycle(lora_params=[p])
    assert s.n_steps == 1
    assert s.grad_norm_median == pytest.approx(2.0, rel=1e-4)  # sqrt(4 * 1^2)
    assert s.lora_weight_delta_norm > 0.0
