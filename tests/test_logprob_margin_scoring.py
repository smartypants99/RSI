"""Task #7: log-prob margin scoring tests.

Math locked in here:
  margin_t  = lp_gold_t − max(lp_v for v != gold)
  score     = sigmoid(mean_t margin_t)
  paired Δ  = E[score_post − score_pre] tracks E[margin_post − margin_pre]
              monotonically under fixed-prompt training (training pushes
              gold token probability UP vs best competitor).

Verification vs team-lead's ask:
  * Margin > 0 when gold is the argmax; < 0 when gold is not argmax.
  * Config rejects invalid score_method values.
  * Synthetic training shift X in log-prob space → paired Δ(score) moves
    monotonically with X (same-sign as X, bounded by sigmoid slope).
"""

from __future__ import annotations

import math

import pytest

from src.diagnostics.ground_truth import (
    GroundTruthQuestion,
    _score_logprob_margin,
    grade_ground_truth_score_ex,
)
from src.utils.config import DiagnosticsConfig


# ──────────────── Fake vLLM plumbing for margin extraction ────────────────


class _FakeLogprob:
    def __init__(self, logprob: float):
        self.logprob = float(logprob)


class _FakeGenOutput:
    def __init__(self, prompt_logprobs):
        self.prompt_logprobs = prompt_logprobs


class _FakeLLM:
    """Returns configured prompt_logprobs on every generate() call."""
    def __init__(self, plp_by_text: dict[str, list]):
        self._plp = plp_by_text

    def generate(self, prompts, params):
        # single-prompt path (the scorer only submits one at a time)
        p = prompts[0]
        return [_FakeGenOutput(self._plp.get(p, []))]


class _FakeSamplingParams:
    def __init__(self, **kw):
        self.kw = kw


class _FakeTokenizer:
    """Maps chars to ascii ids; prompt/full tokenisation by length."""
    def __init__(self):
        pass

    def __call__(self, s, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in s]}

    def encode(self, s, add_special_tokens=False):
        return [ord(c) for c in s]


class _FakeModelLoader:
    def __init__(self, plp_by_text):
        self._llm = _FakeLLM(plp_by_text)
        self._sampling_params_cls = _FakeSamplingParams
        self._tokenizer = _FakeTokenizer()


# ──────────────── margin scoring: direction is correct ────────────────


def _build_plp(prompt: str, gold: str, gold_lp: float, nongold_lp: float):
    """Return (loader, full_text) for the fake vLLM.

    The prompt_logprobs list has len(full_ids) entries, the tail n_gold
    of which the scorer reads. Each entry is a dict with 2 keys: the
    gold token id (mapping to gold_lp) and a non-gold id (nongold_lp).
    """
    full_text = prompt.rstrip() + "\n" + gold
    prompt_ids = [ord(c) for c in (prompt.rstrip() + "\n")]
    full_ids = [ord(c) for c in full_text]
    n_gold = len(full_ids) - len(prompt_ids)
    assert n_gold > 0
    plp: list = [None] * len(prompt_ids)
    for i in range(n_gold):
        gold_id = full_ids[len(prompt_ids) + i]
        nongold_id = (gold_id + 1) % 256  # any different id
        plp.append({
            gold_id: _FakeLogprob(gold_lp),
            nongold_id: _FakeLogprob(nongold_lp),
        })
    return _FakeModelLoader({full_text: plp}), full_text


def test_margin_positive_when_gold_dominates():
    loader, _ = _build_plp("Q:", "A", gold_lp=-0.1, nongold_lp=-2.0)
    score, raw = _score_logprob_margin(loader, "Q:", "A")
    assert not math.isnan(raw)
    assert raw == pytest.approx(1.9, abs=1e-6)
    # sigmoid(1.9) ≈ 0.870
    assert 0.86 < score < 0.88


def test_margin_negative_when_gold_not_argmax():
    loader, _ = _build_plp("Q:", "A", gold_lp=-3.0, nongold_lp=-0.5)
    score, raw = _score_logprob_margin(loader, "Q:", "A")
    assert raw == pytest.approx(-2.5, abs=1e-6)
    # sigmoid(-2.5) ≈ 0.076
    assert 0.07 < score < 0.09


def test_margin_returns_nan_without_vllm_backend():
    # no llm → nan
    class BareLoader:
        _llm = None
        _sampling_params_cls = None
        _tokenizer = None
    score, raw = _score_logprob_margin(BareLoader(), "Q:", "A")
    assert math.isnan(score) and math.isnan(raw)


def test_margin_multi_token_averages_not_sums():
    # 5-char gold → n_gold=5. Use constant per-token margin of +1.0
    # so mean(margin) == 1.0 regardless of gold length.
    loader, _ = _build_plp("Q:", "hello", gold_lp=-0.5, nongold_lp=-1.5)
    _, raw = _score_logprob_margin(loader, "Q:", "hello")
    assert raw == pytest.approx(1.0, abs=1e-6)


# ──────────────── paired Δ tracks training-induced margin shift ────────────────


def test_paired_delta_on_margin_tracks_known_shift():
    """Under a synthetic 'training shift' of +1.5 nat per token (training
    pushed gold up by 1.5 vs the competitor), paired Δ on sigmoid(margin)
    must be positive and monotone in the shift."""
    from src.diagnostics.continuous_paired_eval import continuous_paired_delta

    rng_seed = 7
    import random
    r = random.Random(rng_seed)

    # Build N=60 synthetic items. For each, pre margin ~ N(0, 0.8),
    # post margin = pre + 1.5 (training shift). Convert to sigmoid scores.
    pre_recs, post_recs = [], []
    N = 60
    shift = 1.5
    for i in range(N):
        pre_margin = r.gauss(0.0, 0.8)
        post_margin = pre_margin + shift
        sig = lambda x: 1.0 / (1.0 + math.exp(-x))
        pre_recs.append({
            "prompt": f"q{i}", "expected": "E",
            "score": sig(pre_margin), "correct": pre_margin > 0,
        })
        post_recs.append({
            "prompt": f"q{i}", "expected": "E",
            "score": sig(post_margin), "correct": post_margin > 0,
        })
    res = continuous_paired_delta(pre_recs, post_recs)
    assert res is not None
    # Positive shift → positive delta, far above MDE80.
    assert res.delta > 0.0
    assert res.z > 3.0, f"expected strong signal, got z={res.z}"
    # And the direction is monotone: shift=0 would give near-zero delta.
    pre_only = [{**r, "score": r["score"]} for r in pre_recs]
    no_shift_post = [{**r, "score": r["score"]} for r in pre_recs]
    res0 = continuous_paired_delta(pre_only, no_shift_post)
    assert res0 is not None
    assert abs(res0.delta) < 1e-9


# ──────────────── grade_ground_truth_score_ex dispatch ────────────────


def test_grader_binary_method_skips_vllm():
    q = GroundTruthQuestion(
        prompt="2+2=", canonical_answer="4", check_method="numeric_exact",
        domain="math", subdomain="arithmetic",
    )
    ok, score, aux = grade_ground_truth_score_ex(
        q, response="4", model_loader=None, score_method="binary",
    )
    assert ok is True and score == 1.0
    assert aux["score_margin_raw"] is None
    assert aux["score_logprob_gold_raw"] is None


def test_grader_margin_method_propagates_raw_margin():
    q = GroundTruthQuestion(
        prompt="Q:", canonical_answer="A", check_method="numeric_exact",
        domain="math", subdomain="arithmetic",
    )
    # wrong-answer response but gold margin is strongly positive — score
    # lands on sigmoid(margin), raw margin is in aux for observability.
    loader, _ = _build_plp("Q:", "A", gold_lp=-0.2, nongold_lp=-1.7)
    ok, score, aux = grade_ground_truth_score_ex(
        q, response="WRONG", model_loader=loader, score_method="margin",
    )
    # numeric_exact("WRONG", "A") is False.
    assert ok is False
    assert aux["score_margin_raw"] == pytest.approx(1.5, abs=1e-6)
    # sigmoid(1.5) ≈ 0.818
    assert 0.80 < score < 0.84


# ──────────────── config validation ────────────────


def test_config_rejects_unknown_score_method():
    with pytest.raises(ValueError, match="heldout_score_method"):
        DiagnosticsConfig(heldout_score_method="nope")


def test_config_defaults_to_margin():
    cfg = DiagnosticsConfig()
    assert cfg.heldout_score_method == "margin"
