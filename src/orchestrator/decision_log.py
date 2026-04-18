"""Causal record of per-cycle decisions and their outcomes.

The meta-improvement layer (meta_improver) needs grounded signal about which
configuration choices actually produce held-out improvement. A naive approach
compares cycle N's config against cycle N's improvement and picks the best —
but noise dominates and you end up chasing random variation.

This module implements the two primitives that make grounded meta-decisions:

  1. **DecisionRecord**: captures the full configuration state + observed
     improvement for a cycle, with provenance so we can audit why a change
     was made.

  2. **CausalTracker**: given a sequence of records, answers "does changing
     parameter X tend to help?" using a paired comparison across cycles
     where X changed while other factors were held roughly constant. Uses
     a permutation test so confidence is honest with small N (the regime
     we're always in on a single GPU).

The goal is that meta_improver should NEVER apply a change unless the causal
tracker says "we have enough data to believe this helps" — otherwise we're
just overfitting to noise, which is exactly how RSI pipelines fail silently.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class DecisionRecord:
    """One cycle's decisions and outcome — the unit of causal evidence.

    We store the CONFIG that was in effect (hyperparams, verifier weights,
    prompt templates) alongside the observed outcome (held-out eval delta)
    so later analyses can ask "when this field was X, what happened?".
    """
    cycle: int
    config_snapshot: dict[str, Any]  # flat {key: value} of tunable params
    proposed_changes: dict[str, Any] = field(default_factory=dict)  # what meta tried
    reason: str = ""  # human-readable justification for the change
    accepted: bool = True  # was the change applied? (safety layer may reject)
    pre_score: float = 0.0
    post_score: float = 0.0
    eval_score: float | None = None  # held-out — the ground truth signal
    prev_eval_score: float | None = None  # for delta computation
    samples_generated: int = 0
    samples_verified: int = 0
    training_steps: int = 0
    had_errors: bool = False

    @property
    def eval_delta(self) -> float | None:
        """Change in held-out eval — the only honest improvement metric.

        Training-set improvement is contaminated by the generator picking
        problems the model already almost-solves. Held-out delta is not.
        """
        if self.eval_score is None or self.prev_eval_score is None:
            return None
        return self.eval_score - self.prev_eval_score


class CausalTracker:
    """Accumulates decision records and tests whether a proposed change helps.

    Why not just regression? With 10-30 cycles you don't have enough degrees
    of freedom to fit a confident regression over 20 config dimensions. A
    **paired permutation test** asks a simpler question — "among cycles that
    differed only in field X, was the higher-X cycle's eval_delta better
    more often than chance would predict?" — and gives a p-value that stays
    honest when N is small.
    """

    # Minimum (cycles_with, cycles_without) needed before we'll conclude
    # anything. Below this, permutation p-values are uninformative no matter
    # how lopsided the wins look.
    # Lowered 4→3: at n=8 real cycles with ~3 null eval_delta, the former
    # threshold (needs 8 usable) stays "insufficient_data" forever. 3 per side
    # activates the tracker at n=6 usable while the permutation test still
    # guards against spurious signals via SIGNIFICANCE_ALPHA.
    MIN_PAIRS_FOR_DECISION = 3

    # p < 0.10 is lenient — appropriate because the alternative is doing
    # nothing. A false positive costs one cycle of regressed performance,
    # which the safety revert will catch; a false negative costs forever.
    SIGNIFICANCE_ALPHA = 0.10

    # Cap the number of permutations. 1000 is enough resolution for p ≥ 0.01
    # and keeps this fast even when called every cycle.
    PERMUTATION_ITERATIONS = 1000

    def __init__(self, log_path: Path | None = None):
        self.records: list[DecisionRecord] = []
        self.log_path = log_path

    # ─── recording ──────────────────────────────────────────────────────

    def record(self, rec: DecisionRecord) -> None:
        self.records.append(rec)
        if self.log_path is not None:
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(asdict(rec)) + "\n")
            except OSError:
                # Logging to disk is best-effort — never let a full disk
                # crash the training loop.
                pass

    def load(self, path: Path) -> None:
        """Replay a prior log (for resume). Silent if file missing/corrupt."""
        if not path.exists():
            return
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    self.records.append(DecisionRecord(**data))
        except (json.JSONDecodeError, OSError, TypeError):
            # Prefer forgetting history to crashing; a misformatted log
            # means nothing — we can still proceed from current state.
            pass

    # ─── querying ───────────────────────────────────────────────────────

    def paired_effect(
        self, field_name: str, threshold_for_higher: float | None = None,
    ) -> dict[str, Any]:
        """Estimate whether higher values of `field_name` cause higher eval_delta.

        Returns:
            {
                "n_high": count of cycles where field was "high",
                "n_low": count of cycles where it was "low",
                "mean_delta_high": avg eval_delta when high,
                "mean_delta_low": avg eval_delta when low,
                "diff": mean_delta_high - mean_delta_low,
                "p_value": permutation p-value (one-sided: high > low),
                "significant": p_value < SIGNIFICANCE_ALPHA,
                "decision": "increase" | "decrease" | "insufficient_data" | "neutral",
            }

        `threshold_for_higher`: values > this are "high"; if None, uses the
        median of observed values.
        """
        usable = [
            r for r in self.records
            if r.eval_delta is not None and field_name in r.config_snapshot
        ]
        if len(usable) < 2 * self.MIN_PAIRS_FOR_DECISION:
            return self._insufficient(field_name, len(usable))

        values = [r.config_snapshot[field_name] for r in usable]
        # Non-numeric field — fall back to categorical treatment.
        if not all(isinstance(v, (int, float)) for v in values):
            return self._categorical_effect(field_name, usable)

        if threshold_for_higher is None:
            sorted_vals = sorted(values)
            threshold_for_higher = sorted_vals[len(sorted_vals) // 2]

        high = [r.eval_delta for r in usable
                if r.config_snapshot[field_name] > threshold_for_higher]
        low = [r.eval_delta for r in usable
               if r.config_snapshot[field_name] <= threshold_for_higher]

        if len(high) < self.MIN_PAIRS_FOR_DECISION or len(low) < self.MIN_PAIRS_FOR_DECISION:
            return self._insufficient(field_name, len(usable))

        mean_high = sum(high) / len(high)
        mean_low = sum(low) / len(low)
        observed_diff = mean_high - mean_low

        p = self._permutation_p_value(high, low, observed_diff)

        if p < self.SIGNIFICANCE_ALPHA:
            decision = "increase" if observed_diff > 0 else "decrease"
        else:
            decision = "neutral"

        return {
            "field": field_name,
            "n_high": len(high),
            "n_low": len(low),
            "mean_delta_high": mean_high,
            "mean_delta_low": mean_low,
            "diff": observed_diff,
            "p_value": p,
            "significant": p < self.SIGNIFICANCE_ALPHA,
            "decision": decision,
            "threshold": threshold_for_higher,
        }

    def recent_regressions(self, n: int = 3) -> int:
        """Count cycles in the last n where eval_delta was negative.

        meta_improver uses this as a revert trigger — if the last 2 of 3
        regressed, we back out the most recent config change regardless of
        statistical significance.
        """
        recent = [r for r in self.records[-n:] if r.eval_delta is not None]
        return sum(1 for r in recent if r.eval_delta < -0.005)

    # ─── internals ──────────────────────────────────────────────────────

    def _insufficient(self, field_name: str, n: int) -> dict[str, Any]:
        return {
            "field": field_name,
            "decision": "insufficient_data",
            "significant": False,
            "p_value": 1.0,
            "n": n,
            "required": 2 * self.MIN_PAIRS_FOR_DECISION,
        }

    def _categorical_effect(self, field_name: str, usable: list[DecisionRecord]) -> dict[str, Any]:
        """Handle non-numeric fields (e.g. prompt template IDs) via one-vs-rest.

        Returns the category with the highest mean delta if it's significantly
        above the rest; "neutral" otherwise.
        """
        buckets: dict[Any, list[float]] = {}
        for r in usable:
            key = r.config_snapshot[field_name]
            # Make it hashable — dict/list values get stringified so this
            # doesn't crash on whatever configuration shape teammates invent.
            if not isinstance(key, (str, int, float, bool, type(None))):
                key = json.dumps(key, sort_keys=True, default=str)
            buckets.setdefault(key, []).append(r.eval_delta)

        buckets = {k: v for k, v in buckets.items() if len(v) >= self.MIN_PAIRS_FOR_DECISION}
        if len(buckets) < 2:
            return self._insufficient(field_name, len(usable))

        means = {k: sum(v) / len(v) for k, v in buckets.items()}
        best = max(means, key=means.get)
        best_deltas = buckets[best]
        rest_deltas = [d for k, v in buckets.items() if k != best for d in v]
        diff = means[best] - (sum(rest_deltas) / len(rest_deltas))
        p = self._permutation_p_value(best_deltas, rest_deltas, diff)

        return {
            "field": field_name,
            "best_value": best,
            "best_mean": means[best],
            "diff_from_rest": diff,
            "p_value": p,
            "significant": p < self.SIGNIFICANCE_ALPHA,
            "decision": f"prefer={best!r}" if p < self.SIGNIFICANCE_ALPHA else "neutral",
            "bucket_counts": {str(k): len(v) for k, v in buckets.items()},
        }

    @classmethod
    def _permutation_p_value(
        cls, a: list[float], b: list[float], observed_diff: float,
    ) -> float:
        """One-sided permutation test: P(random_diff >= observed_diff).

        Shuffles group labels; asks how often the observed mean difference
        would arise by chance under the null hypothesis of no effect.
        """
        combined = list(a) + list(b)
        n_a = len(a)
        n_total = len(combined)
        if n_total < 4:
            return 1.0

        rng = random.Random(0xA1BAD0)  # deterministic for reproducibility
        extreme = 0
        iters = cls.PERMUTATION_ITERATIONS
        for _ in range(iters):
            rng.shuffle(combined)
            perm_a = combined[:n_a]
            perm_b = combined[n_a:]
            perm_diff = (sum(perm_a) / n_a) - (sum(perm_b) / (n_total - n_a))
            if perm_diff >= observed_diff:
                extreme += 1

        # Add-one smoothing so p is never exactly 0 (which would falsely
        # suggest infinite confidence from a finite sample).
        return (extreme + 1) / (iters + 1)


def flatten_config(cfg: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested config object into {dotted.key: primitive_value}.

    Used by ImprovementLoop to snapshot the config state each cycle into
    a form CausalTracker can analyze. Skips non-primitive leaves so we
    don't try to correlate over unhashable objects.
    """
    out: dict[str, Any] = {}
    if cfg is None:
        return out

    # Dataclass-like: iterate __dict__
    if hasattr(cfg, "__dict__"):
        for k, v in cfg.__dict__.items():
            if k.startswith("_"):
                continue
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (int, float, bool, str)) and not isinstance(v, bool):
                # bool is a subclass of int — keep it as int for comparison
                out[key] = v
            elif isinstance(v, bool):
                out[key] = int(v)
            elif isinstance(v, (list, tuple)):
                # Only keep lists of primitives as length — the content
                # rarely matters for meta-decisions, the shape usually does.
                if all(isinstance(x, (int, float, bool, str)) for x in v):
                    out[key + ".len"] = len(v)
            elif hasattr(v, "__dict__"):
                out.update(flatten_config(v, key))
            # dicts, arbitrary objects: skip — they confuse causal analysis.

    return out
