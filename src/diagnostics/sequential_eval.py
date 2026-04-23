"""Group-sequential testing for held-out eval (task #25 wedge 3).

Full held-out eval is ~20 min at N=600 post-wedge-2. Many cycles show
clear signals well before the full N is generated (trained cycle that
improved ~5pp on a continuous score has z >> 3 at N/3). Group-
sequential testing stops early on those clear signals, amortizing
wall-clock down.

O'Brien-Fleming design with K=3 equally-spaced looks at two-sided
α=0.05 yields critical values (derived c_k = c/√t_k, c=2.004, Jennison-
Turnbull; cross-checked vs. gemini 2026-04-22):

    look 1 (N/3 complete)  → |z| ≥ 3.471 to stop
    look 2 (2N/3 complete) → |z| ≥ 2.454 to stop
    look 3 (N complete)    → |z| ≥ 2.004 (standard α=0.05 boundary)

Under a realistic 5pp-delta training run with MDE ≈ 0.8%, expected
effective z at look 1 is ≈ 0.05 / (SE·√3) ≈ 10 — enormous, so look 1
fires almost always, cutting eval cost by ~66%. Under the null, OBF
spending at look 1 is ≈ 0.0005, so the rate of false early-stops is
< 0.1%.

This module is a pure computation layer: a caller runs partial eval,
calls sprt_decide() with the partial stats, and stops or continues.
The loop integration (break out of the rep loop when decision='stop')
is deferred to orchestrator-auditor.

Safety gates: early-stop affects NOTHING about promotion gates. A
cycle that fires early-stop at look 1 is still subject to all
existing eligibility gates (samples_verified, capture alarm, mode
collapse, regression guard). Early-stop is purely a throughput
optimization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


Decision = Literal["stop_reject_null", "continue", "stop_accept_null"]


# O'Brien-Fleming critical values for K=3 equally-spaced looks at two-sided α=0.05.
# c_k = c / sqrt(k/K) with c=2.004 (Jennison-Turnbull). Verified vs gemini.
_OBF_K3_ALPHA_05: tuple[float, float, float] = (3.471, 2.454, 2.004)

# OBF K=4 α=0.05 two-sided. Jennison-Turnbull Table 2.3: c=2.024, so
# c_k = 2.024 / sqrt(k/4). Used by the intra-rep chunked-SPRT wedge
# (chunk_size=150, N≈600 → K=4 looks at 25/50/75/100%).
_OBF_K4_ALPHA_05: tuple[float, float, float, float] = (4.049, 2.863, 2.337, 2.024)


@dataclass(frozen=True)
class SequentialDecision:
    """Result of a group-sequential look at the running paired-delta statistic.

    Fields:
        look:         1-indexed look number (1, 2, or 3 for K=3)
        n_so_far:     number of paired samples included in this look
        z:            running test statistic z = delta / SE at this look
        critical:     OBF critical value for this look
        decision:     "stop_reject_null" if |z| ≥ critical (effect confirmed),
                      "continue" otherwise
    """
    look: int
    n_so_far: int
    z: float
    critical: float
    decision: Decision


def obf_critical_values(K: int = 3, alpha: float = 0.05) -> tuple[float, ...]:
    """Return O'Brien-Fleming critical values for K equally-spaced looks.

    Currently only K=3, α=0.05 (two-sided) is supported with the canonical
    cross-checked values. Other K / alpha requires the Jennison-Turnbull
    iterative solver which we deliberately do not reimplement here.
    """
    if K == 3 and abs(alpha - 0.05) < 1e-9:
        return _OBF_K3_ALPHA_05
    if K == 4 and abs(alpha - 0.05) < 1e-9:
        return _OBF_K4_ALPHA_05
    raise NotImplementedError(
        f"obf_critical_values currently ships only K∈{{3,4}} α=0.05 "
        f"(got K={K}, α={alpha}). Add the Jennison-Turnbull solver "
        f"or import a pre-tabulated c value."
    )


def sprt_decide(
    *,
    look: int,
    n_so_far: int,
    delta: float,
    delta_se: float,
    K: int = 3,
    alpha: float = 0.05,
    futility_z: float | None = None,
) -> SequentialDecision:
    """Decide whether to stop or continue at interim look `look` (1-indexed).

    Under OBF K=3 α=0.05:
        - |z| ≥ 3.471 at look 1 → stop early (reject null).
        - |z| ≥ 2.454 at look 2 → stop early.
        - |z| ≥ 2.004 at look 3 → stop (final boundary).
        - else → continue.

    This is a two-sided boundary that rejects on EITHER direction of
    delta. Callers that want to stop only on improvement can filter
    decision == 'stop_reject_null' AND delta > 0 upstream.

    Note: this implementation does NOT include a futility boundary
    (stop_accept_null). OBF futility requires a beta-spending function
    symmetric to alpha-spending and adds complexity without wall-clock
    gain on the training runs we care about (rare null cycles). Left
    as an extension if operator wants it.
    """
    if not (1 <= look <= K):
        raise ValueError(f"look must be in 1..{K}, got {look}")
    if n_so_far < 1:
        raise ValueError(f"n_so_far must be >= 1, got {n_so_far}")
    if delta_se <= 0:
        raise ValueError(f"delta_se must be > 0, got {delta_se}")

    criticals = obf_critical_values(K=K, alpha=alpha)
    critical = criticals[look - 1]
    z = delta / delta_se
    if abs(z) >= critical:
        decision: Decision = "stop_reject_null"
    elif (
        futility_z is not None
        and look < K
        and abs(z) < futility_z
    ):
        # Optional futility (accept-null) boundary — only fires before the
        # final look. Symmetric single-threshold form; no spending function.
        # futility_z=None (default) preserves prior behavior bit-for-bit.
        decision = "stop_accept_null"
    else:
        decision = "continue"
    return SequentialDecision(
        look=look,
        n_so_far=n_so_far,
        z=z,
        critical=critical,
        decision=decision,
    )


def expected_wall_clock_savings_under_alternative(
    *,
    true_delta: float,
    true_se_at_full_n: float,
    K: int = 3,
) -> float:
    """Heuristic: expected fraction of full-N cost under the alternative.

    Under the alternative hypothesis (true_delta > 0), the probability
    the OBF boundary fires at look k increases with k. This function
    returns the expected fraction of full-N actually evaluated,
    E[k/K | true_delta], assuming the partial-look statistics at look k
    have SE scaled by √(K/k) and mean = true_delta.

    Used by the savings estimator in test_sequential_eval to lock in
    the ≥30% savings claim under realistic run conditions.
    """
    criticals = obf_critical_values(K=K)
    se_full = true_se_at_full_n
    # Expected fraction of N evaluated = sum_k (k/K) · P(stop at look k).
    # Stop at look k requires continue at looks 1..k-1 and boundary hit at k.
    # We compute P(stop at look k) under a normal approximation.
    try:
        from statistics import NormalDist
    except ImportError:
        return 1.0  # pragma: no cover — statistics always ships
    nd = NormalDist(mu=0.0, sigma=1.0)

    p_continue_prev = 1.0
    expected_fraction = 0.0
    for k in range(1, K + 1):
        t_k = k / K
        se_k = se_full / math.sqrt(t_k)
        # Z under true_delta at look k: mean = true_delta / se_k, sd = 1.
        mu_z = true_delta / se_k if se_k > 0 else 0.0
        # P(|Z| >= c_k) under N(mu_z, 1) one-sided on the positive tail
        # (true_delta > 0 so lower-tail prob is negligible).
        c_k = criticals[k - 1]
        p_hit = 1.0 - nd.cdf(c_k - mu_z)
        # P(stop at look k) = P(hit k) · P(didn't hit 1..k-1 independently).
        # This ignores the correlation between interim statistics; the
        # estimate is a floor on savings (true savings ≥ estimate).
        p_stop_k = p_continue_prev * p_hit
        expected_fraction += t_k * p_stop_k
        p_continue_prev *= 1.0 - p_hit
    # Remaining mass goes to the final look (or equivalently "never
    # stopped" = go to K).
    expected_fraction += 1.0 * p_continue_prev
    return expected_fraction


__all__ = [
    "Decision",
    "SequentialDecision",
    "obf_critical_values",
    "sprt_decide",
    "expected_wall_clock_savings_under_alternative",
]
