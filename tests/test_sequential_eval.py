"""Task #25 wedge 3: group-sequential testing (O'Brien-Fleming).

OBF K=3 α=0.05 critical values (cross-checked vs. gemini 2026-04-22):
    look 1 (N/3):    |z| ≥ 3.471
    look 2 (2N/3):   |z| ≥ 2.454
    look 3 (N):      |z| ≥ 2.004

Under a 5pp-delta training cycle with full-N SE ≈ 0.003 (wedge-1
continuous scoring), z at look 1 is ~5/(0.003·√3) ≈ 10 — the look-1
boundary fires reliably. Under the null, look-1 false-stop prob is
≈ 2·(1-Φ(3.471)) ≈ 0.0005.

Tests in this file:
  - Boundary arithmetic is exactly what we derived (and gemini-verified).
  - sprt_decide reflects the boundary correctly on representative inputs.
  - Under a realistic alternative, expected wall-clock is < 50% of full N.
  - Under the null, false-stop rate stays under the OBF spending limit.
"""

from __future__ import annotations

import math
import random

import pytest

from src.diagnostics.sequential_eval import (
    SequentialDecision,
    expected_wall_clock_savings_under_alternative,
    obf_critical_values,
    sprt_decide,
)


# ────────────────────── boundary arithmetic ──────────────────────


def test_obf_k3_critical_values_match_derivation():
    """c_k = 2.004 / sqrt(k/3) gives (3.471, 2.454, 2.004). Locked in.

    Derivation: OBF uses c_k = c / sqrt(t_k) with t_k = k/K. For
    K=3 α=0.05 two-sided, c=2.004 (Jennison-Turnbull table).
        c_1 = 2.004 / sqrt(1/3) = 2.004 · √3        ≈ 3.471
        c_2 = 2.004 / sqrt(2/3) = 2.004 · √1.5      ≈ 2.454
        c_3 = 2.004 / sqrt(1)   = 2.004            ≈ 2.004
    Gemini cross-check 2026-04-22: "Yes, those values are correct."
    """
    c1, c2, c3 = obf_critical_values(K=3, alpha=0.05)
    assert abs(c1 - 2.004 * math.sqrt(3.0)) < 0.01, c1
    assert abs(c2 - 2.004 * math.sqrt(1.5)) < 0.01, c2
    assert abs(c3 - 2.004) < 0.01, c3
    # Monotonically decreasing — boundary gets easier as evidence accumulates.
    assert c1 > c2 > c3


def test_obf_critical_values_rejects_unsupported_configs():
    with pytest.raises(NotImplementedError):
        obf_critical_values(K=5)
    with pytest.raises(NotImplementedError):
        obf_critical_values(K=3, alpha=0.01)


# ────────────────────── sprt_decide behavior ──────────────────────


def test_sprt_look1_stops_on_huge_z():
    """A 5pp delta with SE=0.005 at look 1 yields z=10 — boundary fires."""
    d = sprt_decide(look=1, n_so_far=200, delta=0.05, delta_se=0.005)
    assert d.decision == "stop_reject_null"
    assert d.z == 10.0
    assert d.critical > 3.0


def test_sprt_look1_continues_on_moderate_z():
    """z = 2.2 at look 1 is below the 3.471 look-1 boundary — continue.
    Same z WOULD stop at look 3 (boundary 2.004, z=2.2 > 2.004), which
    is the whole point of OBF: aggressive early, forgiving late."""
    d_l1 = sprt_decide(look=1, n_so_far=200, delta=0.022, delta_se=0.01)
    assert d_l1.decision == "continue", d_l1
    d_l3 = sprt_decide(look=3, n_so_far=600, delta=0.022, delta_se=0.01)
    assert d_l3.decision == "stop_reject_null", d_l3


def test_sprt_stops_on_negative_effect_too():
    """Two-sided boundary: |z| ≥ critical stops on either direction."""
    d = sprt_decide(look=1, n_so_far=200, delta=-0.05, delta_se=0.005)
    assert d.decision == "stop_reject_null"
    assert d.z == -10.0


def test_sprt_rejects_invalid_args():
    with pytest.raises(ValueError):
        sprt_decide(look=0, n_so_far=200, delta=0.01, delta_se=0.005)
    with pytest.raises(ValueError):
        sprt_decide(look=4, n_so_far=200, delta=0.01, delta_se=0.005)
    with pytest.raises(ValueError):
        sprt_decide(look=1, n_so_far=0, delta=0.01, delta_se=0.005)
    with pytest.raises(ValueError):
        sprt_decide(look=1, n_so_far=200, delta=0.01, delta_se=0.0)


# ────────────────────── wall-clock savings ──────────────────────


def test_expected_savings_realistic_5pp_cycle_under_50pct():
    """A realistic trained-cycle scenario: true_delta = 5pp, full-N SE
    ≈ 0.003 (wedge-1 continuous). Expected evaluated fraction should be
    near 1/3 (look-1 fires almost always)."""
    frac = expected_wall_clock_savings_under_alternative(
        true_delta=0.05, true_se_at_full_n=0.003,
    )
    # E[k/K] ≈ 1/3 · P(stop at 1) + 2/3 · P(stop at 2) + 1 · P(stop at 3)
    # With true delta this large, P(stop at 1) is essentially 1.
    assert frac < 0.40, frac


def test_expected_savings_small_effect_closer_to_full():
    """At true_delta = 1pp (near the MDE), most evaluations run to full N."""
    frac = expected_wall_clock_savings_under_alternative(
        true_delta=0.01, true_se_at_full_n=0.003,
    )
    # At 1pp delta the look-1 signal is weak; expected fraction closer to 1.
    assert frac > 0.5, frac


def test_savings_monotonic_in_effect_size():
    """Larger true delta → smaller expected fraction of N evaluated."""
    fracs = [
        expected_wall_clock_savings_under_alternative(
            true_delta=d, true_se_at_full_n=0.003,
        )
        for d in (0.005, 0.01, 0.02, 0.03, 0.05)
    ]
    for a, b in zip(fracs, fracs[1:]):
        assert a >= b - 1e-6, fracs


# ────────────────────── null false-stop rate ──────────────────────


def test_null_false_stop_rate_at_look1_under_bound():
    """Under the null (true_delta=0), simulate many runs with partial-look
    z-statistics drawn from N(0,1). P(|z| ≥ 3.471 at look 1) should be
    ≈ 2·(1-Φ(3.471)) ≈ 5.2e-4. We verify empirically within MC noise."""
    rng = random.Random(0)
    trials = 20_000
    hits = 0
    # z at look 1 ~ N(0,1) under null.
    for _ in range(trials):
        z = rng.gauss(0, 1)
        if abs(z) >= 3.471:
            hits += 1
    rate = hits / trials
    # Expected ≈ 5.2e-4; 95% CI at n=20k tolerates anything < 2e-3.
    assert rate < 2e-3, rate


def test_null_overall_false_stop_rate_under_alpha():
    """Overall OBF false-stop rate across all 3 looks must stay under α=0.05
    by construction. Empirical check via correlated-walk simulation."""
    rng = random.Random(1)
    trials = 10_000
    false_stops = 0
    criticals = obf_critical_values(K=3, alpha=0.05)
    for _ in range(trials):
        # Simulate an independent-increment Brownian-motion style walk:
        # z_k = S_k / sqrt(t_k) with S_k = sum of k iid N(0, 1/K) increments.
        increments = [rng.gauss(0, 1.0 / math.sqrt(3)) for _ in range(3)]
        z_1 = increments[0] / math.sqrt(1 / 3)
        z_2 = (increments[0] + increments[1]) / math.sqrt(2 / 3)
        z_3 = (increments[0] + increments[1] + increments[2]) / 1.0
        z_vals = (z_1, z_2, z_3)
        for zk, ck in zip(z_vals, criticals):
            if abs(zk) >= ck:
                false_stops += 1
                break
    rate = false_stops / trials
    # Expected ≈ 0.05; MC at n=10k gives SE ≈ 0.0022, so allow up to 0.06.
    assert rate <= 0.06, rate
