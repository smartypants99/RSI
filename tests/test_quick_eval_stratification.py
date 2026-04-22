"""Task #16: Phase-5b QUICK-eval per-domain stratification.

Live-run cycle 2 configured heldout_quick_subsample_n=128 but logged
n=207 (per-domain=117). Root cause: the prior per-domain target was
ceil(target_n / N_domains), which compounded with HELD_OUT_ONLY
retention variance and overshot by ~60%.

Fix verified here: _quick_eval_stratified_targets stratifies WITHIN
target_n by equal domain weight (1/N), so the expected post-filter
total is round(target_n / N) * N — bounded at ±(N_domains/2) from
the configured target. For target=128, N=4 the expected total lands
in [126, 130]; the assertion uses the spec's [124, 132] tolerance.
"""
from __future__ import annotations

from src.orchestrator.loop import _quick_eval_stratified_targets


def test_quick_eval_target_128_four_domains_lands_within_tolerance():
    """Headline case: heldout_quick_subsample_n=128, 4 domains.
    Expected total must be in [124, 132] per spec."""
    pre, expected = _quick_eval_stratified_targets(
        target_n=128, n_domains=4,
    )
    assert 124 <= expected <= 132, (pre, expected)


def test_quick_eval_target_96_four_domains_lands_within_tolerance():
    """Task #22 speed-round-2 default: heldout_quick_subsample_n=96, 4 domains.
    24 per-domain → expected total = 96; spec tolerance ±(N_domains/2)=±2."""
    pre, expected = _quick_eval_stratified_targets(
        target_n=96, n_domains=4,
    )
    assert 94 <= expected <= 98, (pre, expected)


def test_quick_eval_target_128_three_domains_is_proportional():
    """With 3 domains the expected total is round(128/3)*3 = 43*3 = 129,
    also within the [124, 132] band."""
    _pre, expected = _quick_eval_stratified_targets(
        target_n=128, n_domains=3,
    )
    assert 124 <= expected <= 132, expected


def test_quick_eval_pre_filter_per_domain_accounts_for_retention():
    """pre_filter_per_domain must be ~round(post_filter / 0.37). With
    N=4 domains and target=128, post-filter per-domain = 32, so
    pre-filter per-domain ≈ 32 / 0.37 ≈ 86."""
    pre, _expected = _quick_eval_stratified_targets(
        target_n=128, n_domains=4,
    )
    assert 80 <= pre <= 92, pre


def test_quick_eval_min_per_domain_floor_respected():
    """min_per_domain acts as a lower bound on pre_filter_per_domain."""
    pre, _expected = _quick_eval_stratified_targets(
        target_n=128, n_domains=4, min_per_domain=500,
    )
    assert pre == 500


def test_quick_eval_target_zero_returns_zero_total():
    pre, expected = _quick_eval_stratified_targets(target_n=0, n_domains=4)
    assert expected == 0
    assert pre == 0


def test_quick_eval_single_domain_expected_equals_target():
    """With N=1, stratification is trivial: expected_total == target_n."""
    _pre, expected = _quick_eval_stratified_targets(
        target_n=128, n_domains=1,
    )
    assert expected == 128


def test_quick_eval_rounding_bounded_by_half_ndomains():
    """For a range of (target_n, n_domains), assert |expected - target_n|
    is bounded by ceil(n_domains / 2). This is the property the user
    specified in the task and is the invariant that fixes the overnight-
    run 128→207 drift."""
    import math
    for target_n in (64, 96, 128, 160, 200, 256):
        for n_domains in (1, 2, 3, 4, 5, 6, 8):
            _pre, expected = _quick_eval_stratified_targets(
                target_n=target_n, n_domains=n_domains,
            )
            bound = math.ceil(n_domains / 2)
            assert abs(expected - target_n) <= bound, (
                target_n, n_domains, expected, bound,
            )


def test_quick_eval_regression_128_does_not_blow_up_to_207():
    """Regression guard: the overnight-run bug produced total=207 when
    target=128. Verify we are firmly under 200 for that configuration."""
    _pre, expected = _quick_eval_stratified_targets(
        target_n=128, n_domains=4,
    )
    assert expected < 200
