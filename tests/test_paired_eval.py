"""Tests for paired-sample variance reduction on held-out deltas."""

from __future__ import annotations

import random

from src.diagnostics.paired_eval import paired_delta


def _q(prompt: str, correct: bool) -> dict:
    return {"prompt": prompt, "expected": "ans", "correct": correct}


def test_paired_delta_matches_hand_computed():
    """3 questions: pre=[0,1,0], post=[1,1,0] -> delta=+1/3, paired SE small."""
    pre = [_q("p0", False), _q("p1", True), _q("p2", False)]
    post = [_q("p0", True), _q("p1", True), _q("p2", False)]
    r = paired_delta(pre, post)
    assert r is not None
    assert r.n == 3
    assert abs(r.delta - 1 / 3) < 1e-9
    # d_values = [1, 0, 0], mean=1/3, var=((1-1/3)^2 + 2*(0-1/3)^2)/2 = 1/3
    assert r.delta_se > 0


def test_paired_delta_none_when_no_overlap():
    pre = [_q("a", True)]
    post = [_q("b", True)]
    assert paired_delta(pre, post) is None


def test_paired_se_smaller_than_unpaired_when_questions_correlate():
    """On correlated data, paired SE should beat unpaired SE."""
    rng = random.Random(42)
    pre: list[dict] = []
    post: list[dict] = []
    for i in range(200):
        # Easy questions almost always correct pre & post; hard almost never.
        # Post has a small improvement on medium questions.
        difficulty = rng.random()
        if difficulty < 0.3:
            pre_c, post_c = True, True
        elif difficulty < 0.7:
            pre_c = rng.random() < 0.5
            post_c = pre_c or (rng.random() < 0.1)  # small improvement
        else:
            pre_c, post_c = False, False
        pre.append(_q(f"q{i}", pre_c))
        post.append(_q(f"q{i}", post_c))
    r = paired_delta(pre, post)
    assert r is not None
    assert r.n == 200
    # Paired SE should be meaningfully smaller since question difficulty is
    # highly correlated pre/post. Require ≥2× variance reduction here; 3-5×
    # is typical but we leave slack for RNG.
    assert r.variance_reduction >= 2.0, f"vr={r.variance_reduction}"


def test_paired_delta_zero_when_no_change():
    """If post == pre on every question, delta is 0 and SE is 0."""
    pre = [_q(f"q{i}", i % 2 == 0) for i in range(50)]
    post = [_q(f"q{i}", i % 2 == 0) for i in range(50)]
    r = paired_delta(pre, post)
    assert r is not None
    assert r.delta == 0.0
    assert r.delta_se == 0.0
