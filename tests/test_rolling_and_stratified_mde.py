"""Task #3: rolling-window paired z and domain-stratified CUPED.

Math locked in here:
  • Rolling: pooling K cycles of i.i.d. paired diffs amplifies N by K,
    shrinking SE by √K.  At K=5, MDE80 drops ~2.24× vs. single-cycle.
  • Stratified CUPED: per-domain ANCOVA pooled with n_d/N weights has
    Var(δ̂_strat) = Σ (n_d/N)² · se_d²  ≤  pooled-CUPED variance when
    domain means differ.

Gemini cross-check: 2026-04-23.
"""

from __future__ import annotations

import math
import random

from src.diagnostics.continuous_paired_eval import (
    paired_diffs,
    regression_adjusted_delta,
    rolling_paired_delta_from_diffs,
    stratified_regression_adjusted_delta,
)


def _mk_recs(n: int, domain: str, pre_mean: float, post_mean: float,
             rho: float, sigma: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    pre_recs: list[dict] = []
    post_recs: list[dict] = []
    for i in range(n):
        base = rng.gauss(0.0, sigma)
        noise = rng.gauss(0.0, sigma * math.sqrt(max(0.0, 1.0 - rho ** 2)))
        pre = pre_mean + base
        post = post_mean + rho * base + noise
        key = f"q{domain}-{i}"
        pre_recs.append({"prompt": key, "expected": "E", "score": pre,
                         "domain": domain})
        post_recs.append({"prompt": key, "expected": "E", "score": post,
                          "domain": domain})
    return pre_recs, post_recs


# ────────────────────── rolling paired ──────────────────────


def test_rolling_pooling_shrinks_se_by_sqrt_k():
    # Five cycles of diffs with known variance => pooled SE = √(var/NK)
    rng = random.Random(0xC0FFEE)
    sigma_d = 0.1
    per_cycle_n = 120
    cycle_diffs: list[list[float]] = []
    for _ in range(5):
        cycle_diffs.append([rng.gauss(0.0, sigma_d) for _ in range(per_cycle_n)])

    single = rolling_paired_delta_from_diffs(cycle_diffs[-1:], window=1)
    pooled = rolling_paired_delta_from_diffs(cycle_diffs, window=5)
    assert single is not None and pooled is not None
    assert pooled.k_windows == 5
    assert pooled.n_total == 5 * per_cycle_n
    # SE ratio should be ~√5 with some sampling noise (±20%).
    ratio = single.delta_se / pooled.delta_se
    assert 0.8 * math.sqrt(5) < ratio < 1.2 * math.sqrt(5), ratio
    # MDE ratio equals SE ratio (same z).
    assert abs((single.mde_80 / pooled.mde_80) - ratio) < 1e-9


def test_rolling_returns_none_on_empty():
    assert rolling_paired_delta_from_diffs([], window=3) is None
    assert rolling_paired_delta_from_diffs([[]], window=3) is None


def test_paired_diffs_then_rolling_roundtrip():
    pre, post = _mk_recs(50, "math", 0.5, 0.52, rho=0.9, sigma=0.15, seed=1)
    d = paired_diffs(pre, post)
    assert len(d) == 50
    roll = rolling_paired_delta_from_diffs([d], window=1)
    assert roll is not None
    assert roll.k_windows == 1 and roll.n_total == 50


# ────────────────────── stratified CUPED ──────────────────────


def test_stratified_cuped_reduces_variance_when_domain_means_differ():
    # Two domains with very different means but identical within-domain
    # pre/post correlation. Aggregate CUPED sees inflated σ_Y² from the
    # between-domain spread; stratified CUPED does not.
    pre_a, post_a = _mk_recs(100, "A", pre_mean=0.2, post_mean=0.22,
                             rho=0.8, sigma=0.1, seed=11)
    pre_b, post_b = _mk_recs(100, "B", pre_mean=0.8, post_mean=0.82,
                             rho=0.8, sigma=0.1, seed=22)
    pre = pre_a + pre_b
    post = post_a + post_b

    agg = regression_adjusted_delta(pre, post)
    strat = stratified_regression_adjusted_delta(pre, post)
    assert agg is not None and strat is not None
    assert strat.domains == 2 and strat.n == 200
    # Key invariant: stratified SE < aggregate CUPED SE when domain means
    # differ (domain fixed effects eat the between-domain variance).
    assert strat.delta_se < agg.delta_se, (strat.delta_se, agg.delta_se)
    assert strat.mde_80 < agg.mde_80


def test_stratified_cuped_none_when_no_domain_has_pairs():
    assert stratified_regression_adjusted_delta([], []) is None
