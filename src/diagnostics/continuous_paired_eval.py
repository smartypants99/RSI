"""Continuous-score paired delta with regression adjustment (task #25).

Binary {0,1} correctness has Var ≈ p(1-p) = 0.25 at p=0.5, which
produces a paired-delta SE floor that forces MDE ≈ 3-5% even at N=1200.
Switching to a CONTINUOUS per-question score — e.g. log-prob of the
gold completion, a judge rating in [0,1], or BLEU — typically gives
Var(s) ≈ 0.02-0.04, a 6-10× variance reduction before pairing.

Wedge 1 (continuous_paired_delta):
    SE = sqrt(2·σ_s²·(1-ρ) / N)       [paired SE of continuous scores]
    MDE = z·SE                         [z = 1.96+0.842 ≈ 2.802]

Wedge 2 (regression_adjusted_delta):
    Var(δ̂) = σ²_Y (1-ρ²) / N          [CUPED/ANCOVA estimator]
    MDE = z·sqrt(σ_Y²(1-ρ²)/N)

Worked example (cross-checked vs. gemini 2026-04-22):
    σ_s² = 0.025, ρ = 0.9, N = 600
    → paired SE ≈ 0.00289, MDE ≈ 0.81%
    → regression-adjusted SE ≈ 0.00281, MDE ≈ 0.79%
Both clear the ≤1% sensitivity target. See tests/test_continuous_paired_eval.py
for the arithmetic locked in as regression guards.

This module deliberately takes abstract float scores, not binary
correctness, so the caller chooses how to score: log-prob, judge
rating, continuous grader, BLEU — any per-question score in a bounded
range works.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# Standard critical-value sum for two-sided α=0.05 at power=0.8.
# z_{1-α/2} + z_{1-β} = 1.959963984540054 + 0.8416212335729143
_Z_ALPHA_BETA = 1.959963984540054 + 0.8416212335729143


@dataclass(frozen=True)
class ContinuousPairedDelta:
    """Result of a continuous-score paired-delta analysis.

    Fields:
        n:          number of matched (pre, post) pairs
        mean_pre:   sample mean of pre-scores
        mean_post:  sample mean of post-scores
        delta:      mean_post - mean_pre
        delta_se:   SE of the paired mean (continuous)
        rho:        sample Pearson correlation between pre and post
        var_s:      sample variance of pre-scores (used for MDE math)
        z:          delta / delta_se
        mde_80:     minimum detectable effect at α=0.05, power=0.8
    """
    n: int
    mean_pre: float
    mean_post: float
    delta: float
    delta_se: float
    rho: float
    var_s: float
    z: float
    mde_80: float


@dataclass(frozen=True)
class RegressionAdjustedDelta:
    """CUPED/ANCOVA-style regression-adjusted delta.

    δ̂ = ȳ - β̂·x̄  (where β̂ = Cov(pre, post) / Var(pre))

    Asymptotic Var(δ̂) ≈ σ_Y² · (1-ρ²) / N — strictly smaller than the
    paired-difference variance 2σ²(1-ρ)/N whenever ρ > 0.5 (see gemini
    cross-check 2026-04-22 and test_regression_adjusted_variance_ratio).
    """
    n: int
    delta_adjusted: float
    delta_se: float
    rho: float
    beta: float
    mde_80: float


# ────────────────────────── helpers ──────────────────────────


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _var(xs: list[float], mean: float) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    return sum((x - mean) ** 2 for x in xs) / (n - 1)


def _cov(xs: list[float], mx: float, ys: list[float], my: float) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n - 1)


def _score_map(per_q_records: list[dict], score_key: str = "score") -> dict[str, float]:
    """Extract {question_key: float_score} from per_question records.

    Prefers `score_key` (default 'score') when present and numeric, falls
    back to 0/1 from `correct` so the module is drop-in for records that
    only carry binary correctness. Callers running a continuous grader
    should populate rec['score'] alongside rec['correct'].
    """
    out: dict[str, float] = {}
    for rec in per_q_records or []:
        key = f"{rec.get('prompt', rec.get('question', ''))}|{rec.get('expected', '')}"
        raw = rec.get(score_key)
        if raw is None:
            # Binary fallback: keep backward compatibility.
            out[key] = 1.0 if bool(rec.get("correct", False)) else 0.0
            continue
        try:
            out[key] = float(raw)
        except (TypeError, ValueError):
            out[key] = 1.0 if bool(rec.get("correct", False)) else 0.0
    return out


# ────────────────────────── primary API ──────────────────────────


def continuous_paired_delta(
    pre_per_q: list[dict],
    post_per_q: list[dict],
    *,
    score_key: str = "score",
) -> ContinuousPairedDelta | None:
    """Compute a paired delta on continuous per-question scores.

    Formula: SE = sqrt(2·σ²·(1-ρ) / N). Returns None if n < 2.
    """
    pre_map = _score_map(pre_per_q, score_key=score_key)
    post_map = _score_map(post_per_q, score_key=score_key)
    keys = sorted(pre_map.keys() & post_map.keys())
    n = len(keys)
    if n < 2:
        return None

    pre_vals = [pre_map[k] for k in keys]
    post_vals = [post_map[k] for k in keys]
    mean_pre = _mean(pre_vals)
    mean_post = _mean(post_vals)
    mean_d = mean_post - mean_pre

    d_vals = [post_map[k] - pre_map[k] for k in keys]
    var_d = _var(d_vals, _mean(d_vals))
    paired_se = math.sqrt(var_d / n) if var_d > 0 else 0.0

    var_pre = _var(pre_vals, mean_pre)
    var_post = _var(post_vals, mean_post)
    cov = _cov(pre_vals, mean_pre, post_vals, mean_post)
    # Use the geometric mean of pre and post variances as σ_s² for MDE
    # reporting; the actual delta_se comes from var_d (unbiased, n-1).
    var_s = math.sqrt(var_pre * var_post) if var_pre > 0 and var_post > 0 else max(var_pre, var_post)
    rho = cov / math.sqrt(var_pre * var_post) if var_pre > 0 and var_post > 0 else 0.0

    z = mean_d / paired_se if paired_se > 0 else 0.0
    mde = _Z_ALPHA_BETA * paired_se

    return ContinuousPairedDelta(
        n=n,
        mean_pre=mean_pre,
        mean_post=mean_post,
        delta=mean_d,
        delta_se=paired_se,
        rho=rho,
        var_s=var_s,
        z=z,
        mde_80=mde,
    )


def regression_adjusted_delta(
    pre_per_q: list[dict],
    post_per_q: list[dict],
    *,
    score_key: str = "score",
) -> RegressionAdjustedDelta | None:
    """Compute a CUPED/ANCOVA-style regression-adjusted delta.

    Model: post_i = α + β·pre_i + δ + ε_i. The OLS estimator
        δ̂ = ȳ - β̂·x̄     where     β̂ = Cov(pre, post) / Var(pre)
    has asymptotic Var(δ̂) ≈ σ_Y²(1-ρ²)/N.

    Returns None if n < 2 or Var(pre) == 0 (no correlation structure to
    exploit; fall back to continuous_paired_delta upstream).
    """
    pre_map = _score_map(pre_per_q, score_key=score_key)
    post_map = _score_map(post_per_q, score_key=score_key)
    keys = sorted(pre_map.keys() & post_map.keys())
    n = len(keys)
    if n < 2:
        return None

    pre_vals = [pre_map[k] for k in keys]
    post_vals = [post_map[k] for k in keys]
    mx = _mean(pre_vals)
    my = _mean(post_vals)
    var_pre = _var(pre_vals, mx)
    var_post = _var(post_vals, my)
    if var_pre <= 0 or var_post <= 0:
        return None
    cov = _cov(pre_vals, mx, post_vals, my)
    beta = cov / var_pre
    rho = cov / math.sqrt(var_pre * var_post)
    # Residual variance of post regressed on pre: σ²_Y(1-ρ²)
    residual_var = var_post * (1.0 - rho * rho)
    # Adjusted delta: ȳ - β̂·x̄ (CUPED-style). Equivalent to the intercept
    # of post regressed on pre when pre is centered; we keep the
    # uncentered form because x̄ is the paired pre-mean of interest.
    delta_adj = my - beta * mx
    delta_se = math.sqrt(residual_var / n) if residual_var > 0 else 0.0
    mde = _Z_ALPHA_BETA * delta_se
    return RegressionAdjustedDelta(
        n=n,
        delta_adjusted=delta_adj,
        delta_se=delta_se,
        rho=rho,
        beta=beta,
        mde_80=mde,
    )


# ────────────────────────── theoretical MDE calculators ──────────────────────────


def theoretical_paired_mde(
    *,
    n: int,
    var_s: float,
    rho: float,
    alpha_beta_z: float = _Z_ALPHA_BETA,
) -> float:
    """MDE of the paired mean δ at given N, per-item σ², and correlation ρ.

    MDE = z · sqrt(2·σ²·(1-ρ) / N)

    Cross-check: σ²=0.025, ρ=0.9, N=600 → MDE ≈ 0.00809 (gemini-verified).
    """
    if n <= 0 or var_s < 0 or not (-1.0 <= rho <= 1.0):
        raise ValueError(f"invalid args: n={n}, var_s={var_s}, rho={rho}")
    se = math.sqrt(2.0 * var_s * (1.0 - rho) / n)
    return alpha_beta_z * se


def theoretical_regression_adjusted_mde(
    *,
    n: int,
    var_y: float,
    rho: float,
    alpha_beta_z: float = _Z_ALPHA_BETA,
) -> float:
    """MDE of the CUPED/ANCOVA-adjusted δ̂ at given N, σ²_Y, and ρ.

    MDE = z · sqrt(σ²_Y·(1-ρ²) / N)

    Cross-check: σ²_Y=0.025, ρ=0.9, N=600 → MDE ≈ 0.00788 (gemini-verified).
    """
    if n <= 0 or var_y < 0 or not (-1.0 <= rho <= 1.0):
        raise ValueError(f"invalid args: n={n}, var_y={var_y}, rho={rho}")
    se = math.sqrt(var_y * (1.0 - rho * rho) / n)
    return alpha_beta_z * se


__all__ = [
    "ContinuousPairedDelta",
    "RegressionAdjustedDelta",
    "continuous_paired_delta",
    "regression_adjusted_delta",
    "theoretical_paired_mde",
    "theoretical_regression_adjusted_mde",
]
