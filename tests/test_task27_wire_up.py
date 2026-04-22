"""Task #27: end-to-end wire-up of continuous paired-delta + SPRT into
the orchestrator eval path.

The pure-math layers (continuous_paired_delta, sprt_decide) already have
exhaustive unit coverage in test_continuous_paired_eval.py and
test_sequential_eval.py. These tests exercise the LOOP WIRING:

  - OrchestratorConfig.heldout_eval_mode defaults to "continuous" and
    validates to "binary" or "continuous".
  - _compute_paired_delta_with_base_cache dispatches on the mode and
    populates the schema shape required by downstream consumers
    (delta, delta_se, n, z).
  - _sprt_interim_decision returns sprt_decide results against the
    same reference the final report uses.
  - End-to-end: a synthetic held-out with δ=0.01, ρ=0.9, N=600 should
    have MDE ≤ 1pp under the continuous estimator, and sprt_decide
    at look 1 (N/3 effective ≥200) should "stop_reject_null" once the
    synthetic z clears the OBF critical.
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.diagnostics.continuous_paired_eval import (
    continuous_paired_delta,
    theoretical_paired_mde,
)
from src.diagnostics.sequential_eval import sprt_decide
from src.orchestrator.loop import CycleResult, ImprovementLoop
from src.utils.config import OrchestratorConfig


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------

def test_config_default_mode_is_continuous(tmp_path):
    cfg = OrchestratorConfig(output_dir=tmp_path, log_dir=tmp_path / "l")
    assert cfg.heldout_eval_mode == "continuous"
    assert cfg.sprt_early_stop_enabled is True


def test_config_rejects_unknown_mode(tmp_path):
    with pytest.raises(ValueError):
        OrchestratorConfig(
            output_dir=tmp_path, log_dir=tmp_path / "l",
            heldout_eval_mode="mcnemar",
        )


def test_config_binary_mode_accepted_for_legacy(tmp_path):
    cfg = OrchestratorConfig(
        output_dir=tmp_path, log_dir=tmp_path / "l",
        heldout_eval_mode="binary",
    )
    assert cfg.heldout_eval_mode == "binary"


# ---------------------------------------------------------------------------
# _compute_paired_delta_with_base_cache dispatch
# ---------------------------------------------------------------------------

def _bare_loop(tmp_path: Path, *, mode: str = "continuous") -> ImprovementLoop:
    from src.diagnostics.heldout_base_cache import BaseHeldoutCache
    loop = ImprovementLoop.__new__(ImprovementLoop)
    orchestrator = OrchestratorConfig(
        output_dir=tmp_path,
        log_dir=tmp_path / "logs",
        paired_eval_enabled=True,
        heldout_eval_mode=mode,
    )
    loop.config = SimpleNamespace(orchestrator=orchestrator)
    loop.model_loader = SimpleNamespace(model_path="base/m")
    loop.history = []
    loop._heldout_base_cache = BaseHeldoutCache.load_or_new(
        path=tmp_path / "heldout_base_cache.jsonl",
        model_id="base/m",
    )
    return loop


def _make_per_q(n: int, *, seed: int, delta: float, rho: float,
                score_key: str = "score") -> tuple[list[dict], list[dict]]:
    """Generate pre/post continuous-score per-question records with known
    delta and correlation. Pre ~ Uniform(0,1) conceptually; post = α·pre +
    noise so that Var and Cov realize the target ρ. Both sides carry the
    same (prompt, expected) keys for paired matching."""
    rng = random.Random(seed)
    # Target variance and correlation: choose pre ~ N(0.5, σ²), post = pre +
    # δ + η where η independent so Cov(pre,post)=Var(pre) and ρ→1. To get
    # ρ=0.9 exactly, post = ρ·(pre-0.5)/σ_pre · σ_post + (1-ρ²)^0.5·noise + μ_post.
    sigma = 0.158  # so σ² ≈ 0.025 (matches module cross-check math)
    pre_vals = [0.5 + rng.gauss(0, sigma) for _ in range(n)]
    # Clip to [0,1] to keep it realistic for a "continuous score".
    pre_vals = [max(0.0, min(1.0, v)) for v in pre_vals]
    post_vals = []
    for p in pre_vals:
        indep_noise = rng.gauss(0, sigma) * math.sqrt(1 - rho * rho)
        v = (p - 0.5) * rho + 0.5 + indep_noise + delta
        post_vals.append(max(0.0, min(1.0, v)))
    pre_recs = [
        {"prompt": f"Q{i}", "expected": "E", score_key: pre_vals[i],
         "correct": pre_vals[i] >= 0.5}
        for i in range(n)
    ]
    post_recs = [
        {"prompt": f"Q{i}", "expected": "E", score_key: post_vals[i],
         "correct": post_vals[i] >= 0.5}
        for i in range(n)
    ]
    return pre_recs, post_recs


def test_dispatch_continuous_populates_schema_and_continuous_fields(tmp_path):
    loop = _bare_loop(tmp_path, mode="continuous")
    pre, post = _make_per_q(n=300, seed=1, delta=0.05, rho=0.9)
    from src.diagnostics.heldout_base_cache import populate_from_eval
    populate_from_eval(loop._heldout_base_cache, pre)

    r = CycleResult(1)
    loop._compute_paired_delta_with_base_cache(r, post)
    # Schema shape (required for downstream regression_revert_threshold).
    assert r.paired_delta is not None
    assert r.paired_delta_se is not None
    assert r.paired_delta_n == 300
    # Continuous-only fields.
    assert r.paired_delta_mode == "continuous"
    assert r.paired_delta_mde_80 is not None
    assert r.paired_delta_rho is not None
    assert 0.85 <= r.paired_delta_rho <= 0.95  # synthetic target


def test_dispatch_binary_mode_preserves_legacy_schema(tmp_path):
    loop = _bare_loop(tmp_path, mode="binary")
    pre, post = _make_per_q(n=200, seed=2, delta=0.1, rho=0.9)
    from src.diagnostics.heldout_base_cache import populate_from_eval
    populate_from_eval(loop._heldout_base_cache, pre)

    r = CycleResult(1)
    loop._compute_paired_delta_with_base_cache(r, post)
    assert r.paired_delta is not None
    assert r.paired_delta_se is not None
    assert r.paired_delta_n == 200
    assert r.paired_delta_mode == "binary"
    assert r.paired_delta_mde_80 is None
    assert r.paired_delta_rho is None


# ---------------------------------------------------------------------------
# _sprt_interim_decision wiring
# ---------------------------------------------------------------------------

def test_sprt_interim_uses_base_cache_reference_on_cycle1(tmp_path):
    loop = _bare_loop(tmp_path, mode="continuous")
    pre, post = _make_per_q(n=200, seed=3, delta=0.05, rho=0.9)
    from src.diagnostics.heldout_base_cache import populate_from_eval
    populate_from_eval(loop._heldout_base_cache, pre)

    dec = loop._sprt_interim_decision(
        per_rep_per_question=[post], look=1, K=3,
    )
    assert dec is not None
    assert dec.look == 1
    # Large synthetic signal should clear look-1 OBF critical (3.471).
    assert dec.decision == "stop_reject_null"
    assert abs(dec.z) >= 3.471


def test_sprt_interim_returns_none_without_reference(tmp_path):
    """Cycle 1, empty base cache, no history → no reference → no decision."""
    loop = _bare_loop(tmp_path, mode="continuous")
    # Cache is empty and history is empty.
    _, post = _make_per_q(n=200, seed=4, delta=0.05, rho=0.9)
    dec = loop._sprt_interim_decision(
        per_rep_per_question=[post], look=1, K=3,
    )
    assert dec is None


def test_sprt_interim_continues_on_null_signal(tmp_path):
    loop = _bare_loop(tmp_path, mode="continuous")
    # Delta=0, signal is null — z should be small, SPRT should continue.
    pre, post = _make_per_q(n=200, seed=5, delta=0.0, rho=0.9)
    from src.diagnostics.heldout_base_cache import populate_from_eval
    populate_from_eval(loop._heldout_base_cache, pre)
    dec = loop._sprt_interim_decision(
        per_rep_per_question=[post], look=1, K=3,
    )
    assert dec is not None
    assert dec.decision == "continue"
    assert abs(dec.z) < 3.471


# ---------------------------------------------------------------------------
# End-to-end: synthetic 1% delta, ρ=0.9, N=600 → MDE ≤ 1%
# ---------------------------------------------------------------------------

def test_e2e_n600_delta_1pct_rho_09_mde_le_1pct(tmp_path):
    """Cross-reviewer's headline claim: 1% MDE is reachable on the
    continuous estimator at N=600, ρ=0.9, σ²≈0.025. This is the
    theoretical MDE from the math module (gemini cross-checked), which
    the wire-up must preserve end-to-end."""
    # Theoretical math layer:
    mde_theo = theoretical_paired_mde(n=600, var_s=0.025, rho=0.9)
    assert mde_theo <= 0.01, f"theoretical MDE {mde_theo:.4f} > 1pp"

    # End-to-end through the loop dispatcher:
    loop = _bare_loop(tmp_path, mode="continuous")
    # Synthetic pre/post with matched keys. We target δ=0.01, ρ=0.9.
    pre, post = _make_per_q(n=600, seed=42, delta=0.01, rho=0.9)
    from src.diagnostics.heldout_base_cache import populate_from_eval
    populate_from_eval(loop._heldout_base_cache, pre)
    r = CycleResult(1)
    loop._compute_paired_delta_with_base_cache(r, post)
    assert r.paired_delta_n == 600
    assert r.paired_delta_mde_80 is not None
    # Empirical MDE on the observed sample variance should be within a
    # small factor of the theoretical number. Loose bound (1.1×) accounts
    # for sampling noise in σ̂² at N=600.
    assert r.paired_delta_mde_80 <= 0.011, (
        f"observed MDE80 {r.paired_delta_mde_80:.4f} > 1.1pp "
        f"(theoretical={mde_theo:.4f})"
    )


def test_e2e_sprt_stops_before_full_n_on_real_signal(tmp_path):
    """Synthetic large signal (δ=5pp, ρ=0.9): sprt_decide at the first
    look (N/3 ≈ 200) fires stop_reject_null, so the loop would break
    before the full-N eval ran. Asserts the loop's interim decision
    path actually triggers — the win the task spec promises."""
    loop = _bare_loop(tmp_path, mode="continuous")
    pre, post_look1 = _make_per_q(n=200, seed=7, delta=0.05, rho=0.9)
    from src.diagnostics.heldout_base_cache import populate_from_eval
    populate_from_eval(loop._heldout_base_cache, pre)

    dec = loop._sprt_interim_decision(
        per_rep_per_question=[post_look1], look=1, K=3,
    )
    assert dec is not None
    # N=200 at look 1 is well below a full-N=600 eval; if SPRT fires here
    # the loop breaks, saving the remaining 2/3 of wall-clock.
    assert dec.n_so_far == 200
    assert dec.decision == "stop_reject_null"
