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
    make_code_quorum_pass_fn,
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


# ---------------------------------------------------------------------------
# make_code_quorum_pass_fn — task #9: activates GRPO on code-domain rollouts.
#   Without this the default reward returns 0.0 on code completions (no
#   canonical answer) → zero advantage → no GRPO update. The test below
#   proves the "GRPO is a no-op on code" bug is fixed by running a real
#   sandboxed passes_provided_tests check against a rollout completion.
# ---------------------------------------------------------------------------

@dataclass
class _CodeSample:
    prompt: str = "Write add(a, b) that returns a+b"
    response: str = ""
    domain: str = "code"
    expected_answer: str = ""
    problem_id: str = ""
    category_novelty: float = 0.0


def test_code_quorum_pass_fn_runs_real_properties_on_rollout(tmp_path, monkeypatch):
    """End-to-end test: stash a code problem's ctx, call pass_fn on a
    completion that should PASS the provided tests, assert non-zero reward.
    This directly disproves "GRPO on code is a no-op" — a binary canonical
    grade would have returned 0.0 here.
    """
    from src.verifier.property_engine import stash_problem_ctx, clear_problem_ctx

    pid = "t9_unittest_problem"
    stash_problem_ctx(
        pid,
        {
            "tests": ["add(2, 3) == 5", "add(0, 0) == 0"],
            "reference": "def add(a, b):\n    return a + b\n",
            "entry_point": "add",
            "edge_inputs": ["(2, 3)", "(0, 0)"],
        },
    )
    try:
        pass_fn = make_code_quorum_pass_fn()
        sample = _CodeSample(problem_id=pid)
        good_completion = "```python\ndef add(a, b):\n    return a + b\n```"
        r = pass_fn("p", good_completion, sample)
        # With 2 live properties and a correct completion, expect 1.0
        # (or at least > 0 if one of the sandbox checks bails on this host).
        assert r > 0.0, (
            "code_quorum_pass_fn returned 0.0 on a correct completion — "
            "GRPO code-domain would be a no-op"
        )
        # Sanity: a broken completion should fail the tests.
        bad_completion = "```python\ndef add(a, b):\n    return a - b\n```"
        r_bad = pass_fn("p", bad_completion, sample)
        assert r_bad < r, "bad completion should score lower than good"
    finally:
        clear_problem_ctx(pid)


def test_code_quorum_pass_fn_zero_without_problem_id():
    pass_fn = make_code_quorum_pass_fn()
    sample = _CodeSample(problem_id="")  # empty pid
    assert pass_fn("p", "any completion", sample) == 0.0


def test_code_quorum_pass_fn_zero_on_non_code_domain():
    from src.verifier.property_engine import stash_problem_ctx, clear_problem_ctx
    pid = "t9_math_problem"
    stash_problem_ctx(pid, {"tests": ["2+2==4"]})
    try:
        pass_fn = make_code_quorum_pass_fn()
        sample = _CodeSample(problem_id=pid, domain="math")
        assert pass_fn("p", "x", sample) == 0.0
    finally:
        clear_problem_ctx(pid)


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


def test_ood_reads_meta_domain_maturity_contract():
    """Source-of-truth contract from curriculum-ood's OODSeedBatch.metadata_for."""
    @dataclass
    class _OODSample:
        prompt: str = ""
        response: str = ""
        domain: str = ""
        meta: dict = field(default_factory=lambda: {
            "ood": True, "ood_domain": "graph_theory",
            "domain_maturity": 0.0, "cycle_proposed": 12,
        })
    # Brand-new domain: maturity=0 → novelty=1 → bonus=0.5 → reward=1.5
    r = make_property_quorum_reward_fn(lambda p, c, s: 1.0, ood_alpha=0.5)
    assert r("p", "c", _OODSample()) == pytest.approx(1.5)

    # Mainstream domain: maturity=1 → novelty=0 → no bonus
    mainstream = _OODSample()
    mainstream.meta = {"ood": True, "domain_maturity": 1.0}
    assert r("p", "c", mainstream) == pytest.approx(1.0)


def test_ood_meta_without_ood_true_is_treated_as_indist():
    """meta dict lacking ood=True should not trigger novelty bonus."""
    @dataclass
    class _PlainSample:
        prompt: str = ""
        response: str = ""
        domain: str = ""
        meta: dict = field(default_factory=lambda: {
            "ood": False, "domain_maturity": 0.0,  # would be 1.0 novelty if ood=True
        })
    r = make_property_quorum_reward_fn(lambda p, c, s: 1.0, ood_alpha=0.5)
    assert r("p", "c", _PlainSample()) == pytest.approx(1.0)


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
# Paired-held-out plateau spec (v2)
# ---------------------------------------------------------------------------

def _paired(delta, se, n=200):
    return {"paired_delta": delta, "paired_se": se, "n": n}


def test_plateau_paired_fires_when_all_flat():
    hist = [_paired(0.001, 0.0005) for _ in range(5)]
    # delta < min_gain=0.003 and |z|=2.0 is at the boundary; use smaller se to
    # be safely below z_max.
    hist = [_paired(0.0005, 0.001) for _ in range(5)]
    assert should_switch_to_grpo(hist) is True


def test_plateau_paired_blocks_when_tight_se_regression():
    # delta is below min_gain but it's a statistically-significant regression:
    # delta = -0.01, se=0.001 → |z| = 10. That's NOT flat, it's regression —
    # should refuse to fire the plateau alarm.
    hist = [_paired(-0.01, 0.001) for _ in range(5)]
    assert should_switch_to_grpo(hist) is False


def test_plateau_paired_blocked_by_single_real_gain():
    hist = [_paired(0.001, 0.002) for _ in range(4)]
    hist.append(_paired(0.02, 0.003))  # real gain — breaks plateau
    assert should_switch_to_grpo(hist) is False


def test_plateau_paired_degrades_when_se_missing():
    # n<2 / missing SE → falls back to point-estimate criterion.
    hist = [{"paired_delta": 0.001, "paired_se": 0.0, "n": 1} for _ in range(5)]
    assert should_switch_to_grpo(hist) is True


def test_plateau_paired_missing_delta_is_not_flat():
    # A record with no paired_delta cannot be declared flat.
    hist = [_paired(0.001, 0.002) for _ in range(4)]
    hist.append({"n": 0})  # skipped / missing
    assert should_switch_to_grpo(hist) is False


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
