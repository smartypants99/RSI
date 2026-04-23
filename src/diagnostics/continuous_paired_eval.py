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


# ────────────────────────── multi-cycle rolling + stratified ──────────────────────────


@dataclass(frozen=True)
class RollingPairedDelta:
    """Rolling-window paired delta pooled over the last K cycles.

    Under H0 (constant underlying delta, stationary paired-difference
    noise), pooling K cycles of per-question paired differences into one
    concatenated vector multiplies the effective N by K. Since SE scales
    as 1/√N, the rolling MDE shrinks by √K relative to single-cycle MDE.
    At K=5 this is a 2.236× reduction (gemini-verified 2026-04-23).

    Fields:
        k_windows: number of cycles actually pooled (≤ requested window)
        n_total:   Σ N_k (concatenated-pair count across pooled cycles)
        delta:     weighted pooled mean of per-question (post − pre)
        delta_se:  √(Var_pool / n_total)   where Var_pool is concatenated-vector variance
        z:         delta / delta_se
        mde_80:    2.802 · delta_se
    """
    k_windows: int
    n_total: int
    delta: float
    delta_se: float
    z: float
    mde_80: float


@dataclass(frozen=True)
class StratifiedRegressionAdjustedDelta:
    """Domain-stratified CUPED/ANCOVA delta (per-domain fixed effects).

    Residual variance is the within-domain-weighted ANCOVA variance:
        Var(δ̂_strat) ≈ (1/N) · Σ_d (n_d/N) · σ_{Y,d}² · (1 − ρ_d²)
    which is ≤ pooled-CUPED variance σ_Y²(1−ρ²)/N whenever between-domain
    mean differences or heterogeneous ρ_d are present (gemini-verified
    2026-04-23). Strictly equal when domain means and ρ_d are identical.
    """
    n: int
    domains: int
    delta_adjusted: float
    delta_se: float
    mde_80: float


def rolling_paired_delta_from_diffs(
    cycle_diffs: list[list[float]],
    *,
    window: int = 5,
) -> RollingPairedDelta | None:
    """Pool the last `window` cycles' paired-difference vectors and compute
    the concatenated-sample paired delta.

    `cycle_diffs` is an ordered list of per-cycle per-question (post − pre)
    difference vectors, OLDEST FIRST. The last `window` entries are used.
    Returns None if fewer than 2 differences are available total.
    """
    if not cycle_diffs:
        return None
    pooled_windows = [d for d in cycle_diffs[-window:] if d]
    if not pooled_windows:
        return None
    all_diffs: list[float] = []
    for d in pooled_windows:
        all_diffs.extend(d)
    n = len(all_diffs)
    if n < 2:
        return None
    m = _mean(all_diffs)
    v = _var(all_diffs, m)
    se = math.sqrt(v / n) if v > 0 else 0.0
    z = m / se if se > 0 else 0.0
    mde = _Z_ALPHA_BETA * se
    return RollingPairedDelta(
        k_windows=len(pooled_windows),
        n_total=n,
        delta=m,
        delta_se=se,
        z=z,
        mde_80=mde,
    )


def paired_diffs(
    pre_per_q: list[dict],
    post_per_q: list[dict],
    *,
    score_key: str = "score",
) -> list[float]:
    """Extract the per-question (post − pre) difference vector used by
    rolling_paired_delta_from_diffs. Returns [] if fewer than 2 matches."""
    pre_map = _score_map(pre_per_q, score_key=score_key)
    post_map = _score_map(post_per_q, score_key=score_key)
    keys = sorted(pre_map.keys() & post_map.keys())
    if len(keys) < 2:
        return []
    return [post_map[k] - pre_map[k] for k in keys]


def _domain_of(rec: dict) -> str:
    return str(
        rec.get("domain")
        or rec.get("subject")
        or rec.get("category")
        or "_"
    )


def stratified_regression_adjusted_delta(
    pre_per_q: list[dict],
    post_per_q: list[dict],
    *,
    score_key: str = "score",
    min_per_domain: int = 3,
) -> StratifiedRegressionAdjustedDelta | None:
    """Domain-stratified CUPED: run regression_adjusted_delta within each
    domain, then pool the per-domain δ̂_d and Var(δ̂_d) by domain weight.

    Domains with fewer than `min_per_domain` matched pairs fall back to
    the pooled continuous paired delta (no adjustment) for that bucket,
    so they can't dominate with spurious near-zero residual variance.
    Returns None if no domain has ≥2 matched pairs.
    """
    pre_by_key: dict[str, dict] = {}
    for r in pre_per_q or []:
        pre_by_key[f"{r.get('prompt', r.get('question',''))}|{r.get('expected','')}"] = r
    post_by_key: dict[str, dict] = {}
    for r in post_per_q or []:
        post_by_key[f"{r.get('prompt', r.get('question',''))}|{r.get('expected','')}"] = r
    shared = sorted(pre_by_key.keys() & post_by_key.keys())
    if len(shared) < 2:
        return None

    by_domain: dict[str, tuple[list[dict], list[dict]]] = {}
    for k in shared:
        pr = pre_by_key[k]
        po = post_by_key[k]
        d = _domain_of(pr) or _domain_of(po)
        pre_b, post_b = by_domain.setdefault(d, ([], []))
        pre_b.append(pr)
        post_b.append(po)

    total_n = 0
    weighted_delta = 0.0
    weighted_var_times_n_sq = 0.0
    domains_used = 0
    for d, (pre_b, post_b) in by_domain.items():
        n_d = len(pre_b)
        if n_d < 2:
            continue
        if n_d >= min_per_domain:
            res = regression_adjusted_delta(pre_b, post_b, score_key=score_key)
            if res is None:
                res_c = continuous_paired_delta(pre_b, post_b, score_key=score_key)
                if res_c is None:
                    continue
                delta_d = res_c.delta
                se_d = res_c.delta_se
            else:
                delta_d = res.delta_adjusted
                se_d = res.delta_se
        else:
            res_c = continuous_paired_delta(pre_b, post_b, score_key=score_key)
            if res_c is None:
                continue
            delta_d = res_c.delta
            se_d = res_c.delta_se
        total_n += n_d
        weighted_delta += n_d * delta_d
        weighted_var_times_n_sq += (n_d ** 2) * (se_d ** 2)
        domains_used += 1

    if total_n < 2 or domains_used == 0:
        return None

    delta_adj = weighted_delta / total_n
    # Var(Σ w_d · δ̂_d) with w_d = n_d/N and independent-domain assumption:
    #   Var(δ̂_strat) = Σ w_d² · Var(δ̂_d) = Σ (n_d/N)² · se_d²
    var_strat = weighted_var_times_n_sq / (total_n ** 2)
    se_strat = math.sqrt(var_strat) if var_strat > 0 else 0.0
    mde = _Z_ALPHA_BETA * se_strat
    return StratifiedRegressionAdjustedDelta(
        n=total_n,
        domains=domains_used,
        delta_adjusted=delta_adj,
        delta_se=se_strat,
        mde_80=mde,
    )


__all__ = [
    "ContinuousPairedDelta",
    "RegressionAdjustedDelta",
    "RollingPairedDelta",
    "StratifiedRegressionAdjustedDelta",
    "continuous_paired_delta",
    "regression_adjusted_delta",
    "rolling_paired_delta_from_diffs",
    "paired_diffs",
    "stratified_regression_adjusted_delta",
    "theoretical_paired_mde",
    "theoretical_regression_adjusted_mde",
]
