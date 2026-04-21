"""Unit tests for src/trainer/grpo.py.

Covers the three pure/lightweight pieces (no GPU):
  - OOD_bonus reward shaping
  - Plateau auto-switch detector
  - KLDivergenceGuard install/uninstall contract + stats plumbing
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pytest

from src.trainer.grpo import (
    DEFAULT_MAX_KL,
    DEFAULT_OOD_ALPHA,
    DEFAULT_PLATEAU_MIN_GAIN,
    DEFAULT_PLATEAU_WINDOW,
    KLDivergenceGuard,
    install_kl_guard,
    make_ood_bonus_reward_fn,
    make_property_quorum_reward_fn,
    should_switch_to_grpo,
)


@dataclass
class _FakeSample:
    prompt: str = "p"
    response: str = "r"
    domain: str = "code"
    expected_answer: str = ""
    category_novelty: float = 0.0


# ---------------------------------------------------------------------------
# Reward factories
# ---------------------------------------------------------------------------

def test_quorum_reward_without_ood_is_pass_rate():
    r = make_property_quorum_reward_fn(lambda p, c, s: 0.8, ood_alpha=0.5)
    s = _FakeSample(category_novelty=0.0)
    assert r("p", "c", s) == pytest.approx(0.8)


def test_quorum_reward_with_full_ood_bonus():
    r = make_property_quorum_reward_fn(lambda p, c, s: 1.0, ood_alpha=0.5)
    s = _FakeSample(category_novelty=1.0)
    assert r("p", "c", s) == pytest.approx(1.5)


def test_quorum_reward_clamps_bonus_to_unit_interval():
    r = make_property_quorum_reward_fn(lambda p, c, s: 1.0, ood_alpha=0.5)
    assert r("p", "c", _FakeSample(category_novelty=5.0)) == pytest.approx(1.5)
    assert r("p", "c", _FakeSample(category_novelty=-1.0)) == pytest.approx(1.0)


def test_quorum_reward_handles_quorum_fn_exception():
    def _blow(*args, **kwargs):
        raise RuntimeError("boom")
    r = make_property_quorum_reward_fn(_blow)
    assert r("p", "c", _FakeSample()) == 0.0


def test_quorum_reward_handles_nonfinite():
    r = make_property_quorum_reward_fn(lambda p, c, s: float("nan"))
    assert r("p", "c", _FakeSample()) == 0.0


def test_ood_wrap_over_arbitrary_base_reward():
    base = lambda p, c, s: 0.4
    wrapped = make_ood_bonus_reward_fn(base, ood_alpha=0.5)
    assert wrapped("p", "c", _FakeSample(category_novelty=0.0)) == pytest.approx(0.4)
    assert wrapped("p", "c", _FakeSample(category_novelty=1.0)) == pytest.approx(0.6)


def test_ood_reads_meta_dict_fallback():
    @dataclass
    class _MetaSample:
        prompt: str = ""
        response: str = ""
        domain: str = ""
        meta: dict = field(default_factory=lambda: {"category_novelty": 0.4})
    r = make_property_quorum_reward_fn(lambda p, c, s: 1.0, ood_alpha=0.5)
    # 1.0 * (1 + 0.5*0.4) = 1.2
    assert r("p", "c", _MetaSample()) == pytest.approx(1.2)


def test_invalid_alpha_rejected():
    with pytest.raises(ValueError):
        make_property_quorum_reward_fn(lambda p, c, s: 1.0, ood_alpha=-0.1)
    with pytest.raises(ValueError):
        make_ood_bonus_reward_fn(lambda p, c, s: 1.0, ood_alpha=-0.1)


# ---------------------------------------------------------------------------
# Plateau detector
# ---------------------------------------------------------------------------

def test_plateau_needs_full_window():
    # only 4 cycles of history — not enough to declare plateau at default window=5
    hist = [0.0, 0.0, 0.0, 0.0]
    assert should_switch_to_grpo(hist) is False


def test_plateau_fires_when_all_gains_below_threshold():
    hist = [0.002, 0.001, 0.0, -0.001, 0.002]
    assert should_switch_to_grpo(hist) is True


def test_plateau_does_not_fire_if_any_gain_meets_threshold():
    hist = [0.002, 0.001, 0.005, 0.001, 0.001]  # middle cycle broke plateau
    assert should_switch_to_grpo(hist) is False


def test_plateau_only_inspects_tail_window():
    # old large gains shouldn't matter — only the trailing `window` cycles do
    hist = [0.05, 0.05, 0.001, 0.001, 0.001, 0.001, 0.001]
    assert should_switch_to_grpo(hist, window=5) is True


def test_plateau_custom_window_and_threshold():
    hist = [0.01, 0.01, 0.01]
    assert should_switch_to_grpo(hist, window=3, min_gain=0.02) is True
    assert should_switch_to_grpo(hist, window=3, min_gain=0.005) is False


def test_plateau_window_must_be_positive():
    with pytest.raises(ValueError):
        should_switch_to_grpo([0.0], window=0)


# ---------------------------------------------------------------------------
# KL guard install/uninstall
# ---------------------------------------------------------------------------

class _StubTrainer:
    def __init__(self):
        self._train_grpo_calls = []
        self._build_calls = []

    def _train_grpo(self, verified_samples, cycle):
        self._train_grpo_calls.append((verified_samples, cycle))
        return "metrics"

    def _grpo_build_batch(self, rollouts_for_prompt):
        self._build_calls.append(rollouts_for_prompt)
        return {"stub": True}


def test_kl_guard_install_uninstall_restores_methods():
    t = _StubTrainer()
    g = KLDivergenceGuard(t, max_kl=0.1)
    g.install()
    # While installed, dispatching _train_grpo runs through the wrapper.
    # We can't easily verify by identity (bound methods rebind on each access)
    # but we can verify that _grpo_build_batch is the wrapped plain function.
    assert callable(t._train_grpo)
    assert t._grpo_build_batch.__name__ == "_wrapped_build"
    g.uninstall()
    # After uninstall, the instance attribute overrides are gone and the
    # original bound methods resolve from the class.
    assert "_train_grpo" not in t.__dict__
    assert "_grpo_build_batch" not in t.__dict__


def test_kl_guard_rejects_bad_max_kl():
    with pytest.raises(ValueError):
        KLDivergenceGuard(_StubTrainer(), max_kl=0)


def test_kl_guard_requires_train_grpo_method():
    class _Empty:
        pass
    g = KLDivergenceGuard(_Empty(), max_kl=0.1)
    with pytest.raises(AttributeError):
        g.install()


def test_install_kl_guard_helper_returns_installed_instance():
    t = _StubTrainer()
    g = install_kl_guard(t, max_kl=0.05)
    try:
        assert g._installed is True
        assert g.max_kl == 0.05
    finally:
        g.uninstall()
    assert g._installed is False
