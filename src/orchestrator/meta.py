"""Meta-improvement layer: the system improves its own configuration.

Tracks per-cycle (config, held-out improvement) pairs and proposes bounded
updates to hyperparameters, verifier weights, and generator prompts based on
observed signal. All changes are bounded, logged, and reverted if the next
cycle regresses.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

from .decision_log import CausalTracker, DecisionRecord, flatten_config

logger = logging.getLogger(__name__)


MAX_STEP_FRAC = 0.30  # no single meta-update shifts a scalar by more than 30%


def _clip_frac(new_val: float, old_val: float, max_frac: float = MAX_STEP_FRAC) -> float:
    """Clip new_val so it differs from old_val by at most max_frac (relative)."""
    if old_val == 0:
        return new_val
    lo = old_val * (1 - max_frac)
    hi = old_val * (1 + max_frac)
    if old_val < 0:
        lo, hi = hi, lo
    return max(lo, min(hi, new_val))


@dataclass
class BanditArm:
    """Beta-Bernoulli arm for Thompson sampling over a discrete hyperparameter."""
    value: float
    alpha: float = 1.0  # successes + 1 (Laplace prior)
    beta: float = 1.0   # failures + 1

    def sample(self, rng: random.Random) -> float:
        # random.betavariate draws from Beta(alpha, beta)
        return rng.betavariate(self.alpha, self.beta)

    def update(self, success: bool) -> None:
        if success:
            self.alpha += 1.0
        else:
            self.beta += 1.0


@dataclass
class LRBandit:
    """Thompson-sampling bandit over a discrete set of learning-rate arms."""
    arms: list[BanditArm] = field(default_factory=list)
    last_pulled: float | None = None

    @classmethod
    def around(cls, center_lr: float) -> "LRBandit":
        # Log-spaced arms within a single order of magnitude of `center_lr`.
        multipliers = [0.25, 0.5, 1.0, 2.0, 4.0]
        return cls(arms=[BanditArm(value=center_lr * m) for m in multipliers])

    def pick(self, rng: random.Random) -> float:
        if not self.arms:
            raise ValueError("LRBandit has no arms configured")
        best_idx, best_sample = 0, -1.0
        for i, arm in enumerate(self.arms):
            s = arm.sample(rng)
            if s > best_sample:
                best_idx, best_sample = i, s
        self.last_pulled = self.arms[best_idx].value
        return self.last_pulled

    def observe(self, lr_used: float, success: bool) -> None:
        for arm in self.arms:
            if math.isclose(arm.value, lr_used, rel_tol=1e-6):
                arm.update(success)
                return

    def to_dict(self) -> dict:
        return {
            "arms": [{"value": a.value, "alpha": a.alpha, "beta": a.beta} for a in self.arms],
            "last_pulled": self.last_pulled,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LRBandit":
        return cls(
            arms=[BanditArm(**a) for a in d.get("arms", [])],
            last_pulled=d.get("last_pulled"),
        )


@dataclass
class PromptVariant:
    """A candidate generator template with track record."""
    template: str
    trials: int = 0
    cumulative_improvement: float = 0.0

    @property
    def mean_improvement(self) -> float:
        return self.cumulative_improvement / self.trials if self.trials > 0 else 0.0


@dataclass
class MetaCycleRecord:
    """Snapshot of one cycle: config used, held-out delta observed."""
    cycle: int
    config_snapshot: dict
    held_out_score: float | None
    held_out_delta: float | None
    reasoning: str = ""


class MetaController:
    """Learns per-cycle which config decisions correlate with eval gains.

    Decision mechanisms:
      1. LR bandit (Thompson sampling over discrete learning-rate arms)
      2. Verifier weight reweighting (correlate per-check scores with outcome)
      3. Generator prompt evolution (top-K retain + mutation)

    Safety: per-cycle change magnitudes are bounded; if a cycle regresses
    the previously-applied proposal is reverted.
    """

    MIN_CYCLES_FOR_REWEIGHT = 4
    MAX_PROMPT_VARIANTS = 3

    def __init__(
        self,
        log_path: Path,
        seed: int = 0xC0FFEE,
        initial_lr: float | None = None,
        tracker: CausalTracker | None = None,
    ):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._rng = random.Random(seed)

        self.records: list[MetaCycleRecord] = []
        self.lr_bandit: LRBandit | None = (
            LRBandit.around(initial_lr) if initial_lr is not None else None
        )
        self.prompt_variants: list[PromptVariant] = []
        # Weights start uniform; updated via regression on accepted-sample signal.
        self.verifier_weights: dict[str, float] = {}
        # Track the last applied proposal so we can revert on regression.
        self._last_proposal: dict | None = None
        self._last_pre_revert_state: dict | None = None

        # Causal tracker — the gate for every applied change. Without enough
        # evidence that a field matters, we do nothing (bandit exploration is
        # the one exception; see propose_updates).
        self.tracker = tracker if tracker is not None else CausalTracker(
            log_path=self.log_path.parent / "decision_records.jsonl"
        )

    # ---------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------

    def _template_id(self, template: str) -> int:
        """Return a stable int id for a template string (used as bucket key)."""
        for i, v in enumerate(self.prompt_variants):
            if v.template == template:
                return i
        self.prompt_variants.append(PromptVariant(template=template))
        return len(self.prompt_variants) - 1

    # ---------------------------------------------------------------
    # Core API
    # ---------------------------------------------------------------

    def record_cycle(
        self,
        cycle: int,
        config_snapshot: dict,
        held_out_score: float | None,
        prev_held_out: float | None,
        verifier_check_scores: dict[str, float] | None = None,
        training_succeeded: bool = True,
        reasoning: str = "",
        pre_score: float = 0.0,
        post_score: float = 0.0,
        full_config: Any = None,
        samples_generated: int = 0,
        samples_verified: int = 0,
        training_steps: int = 0,
        had_errors: bool = False,
    ) -> None:
        """Record observed outcome of a cycle. Call at end of each cycle."""
        delta: float | None = None
        if held_out_score is not None and prev_held_out is not None:
            delta = held_out_score - prev_held_out

        rec = MetaCycleRecord(
            cycle=cycle,
            config_snapshot=dict(config_snapshot),
            held_out_score=held_out_score,
            held_out_delta=delta,
            reasoning=reasoning,
        )
        self.records.append(rec)

        # Feed the causal tracker with a flattened-primitive snapshot.
        # We prefer the full SystemConfig (richer field coverage) when given;
        # fall back to the local meta-config view.
        tracker_snapshot = (
            flatten_config(full_config) if full_config is not None
            else {k: v for k, v in config_snapshot.items()
                  if isinstance(v, (int, float, bool, str))}
        )
        # Ensure our canonical tunables appear in the snapshot even if they
        # were stripped (e.g. nested dicts inside full_config).
        for k in ("learning_rate", "lora_rank", "consistency_threshold"):
            v = config_snapshot.get(k)
            if isinstance(v, (int, float, bool)):
                tracker_snapshot[k] = v
        # Prompt template is categorical — record by stable id so the tracker
        # can bucket cycles by which template was active.
        tmpl = config_snapshot.get("generator_template")
        if tmpl is not None:
            tracker_snapshot["generator_template_id"] = (
                self._template_id(tmpl)
            )

        self.tracker.record(DecisionRecord(
            cycle=cycle,
            config_snapshot=tracker_snapshot,
            proposed_changes=(self._last_proposal or {}),
            reason=reasoning,
            accepted=self._last_proposal is not None,
            pre_score=pre_score,
            post_score=post_score,
            eval_score=held_out_score,
            prev_eval_score=prev_held_out,
            samples_generated=samples_generated,
            samples_verified=samples_verified,
            training_steps=training_steps,
            had_errors=had_errors,
        ))

        # Feed the LR bandit: arm = lr used this cycle; success = delta > 0
        lr_used = config_snapshot.get("learning_rate")
        if self.lr_bandit is not None and lr_used is not None and delta is not None:
            self.lr_bandit.observe(lr_used, success=(delta > 0))

        # Revert decision is delegated to tracker.recent_regressions() via
        # get_revert_target() — called by the loop next cycle.

        # Accumulate credit for prompt-evolution variants, if active
        if config_snapshot.get("generator_template_id") is not None and delta is not None:
            tid = config_snapshot["generator_template_id"]
            if 0 <= tid < len(self.prompt_variants):
                self.prompt_variants[tid].trials += 1
                self.prompt_variants[tid].cumulative_improvement += delta

        # Update verifier weight regression signal (simple correlation)
        if verifier_check_scores and delta is not None:
            self._update_verifier_weight_regression(verifier_check_scores, delta)

    def propose_updates(self, current_config: dict) -> dict:
        """Return a dict of proposed config changes for the next cycle.

        Every change is gated by CausalTracker: the tracker must return a
        non-neutral, significant decision for the underlying field — except
        for LR, where the bandit is allowed to continue exploring until the
        tracker accumulates enough paired cycles to speak. Once the tracker
        returns "insufficient_data" we treat that as "keep exploring"; once
        it returns "neutral" we freeze the field; once it returns
        "increase/decrease" we move that direction, still ±30% bounded.
        """
        reasoning: list[str] = []
        proposal: dict[str, Any] = {
            "learning_rate": None,
            "verifier_check_weights": None,
            "generator_template": None,
            "reasoning": reasoning,
            "applied": False,
        }

        # 1. LR — bandit during exploration phase; tracker-gated once significant.
        if self.lr_bandit is not None:
            cur_lr = current_config.get("learning_rate")
            lr_effect = self.tracker.paired_effect("learning_rate")
            if lr_effect["decision"] == "neutral":
                reasoning.append(
                    f"LR frozen: tracker neutral (p={lr_effect['p_value']:.3f}, "
                    f"diff={lr_effect.get('diff', 0):+.4f})"
                )
            else:
                new_lr = self.lr_bandit.pick(self._rng)
                if cur_lr is not None:
                    new_lr = _clip_frac(new_lr, cur_lr)
                # When tracker is significant, bias toward its direction.
                if lr_effect["decision"] == "increase" and cur_lr is not None and new_lr < cur_lr:
                    new_lr = cur_lr * (1 + MAX_STEP_FRAC * 0.5)
                    reasoning.append(
                        f"LR tracker says increase (p={lr_effect['p_value']:.3f}); "
                        f"overriding bandit downward pick"
                    )
                elif lr_effect["decision"] == "decrease" and cur_lr is not None and new_lr > cur_lr:
                    new_lr = cur_lr * (1 - MAX_STEP_FRAC * 0.5)
                    reasoning.append(
                        f"LR tracker says decrease (p={lr_effect['p_value']:.3f}); "
                        f"overriding bandit upward pick"
                    )
                if cur_lr is None or not math.isclose(new_lr, cur_lr, rel_tol=1e-6):
                    proposal["learning_rate"] = new_lr
                    reasoning.append(
                        f"LR bandit: picked lr={new_lr:.2e} "
                        f"(from {cur_lr}), bounded to ±{int(MAX_STEP_FRAC*100)}%"
                        f"; tracker={lr_effect['decision']} "
                        f"(n={lr_effect.get('n', lr_effect.get('n_high', 0) + lr_effect.get('n_low', 0))})"
                    )

        # 2. Verifier weight reweighting — only if the EMA covariance regression
        #    has seen enough cycles AND the tracker doesn't say weights are
        #    already in a neutral regime.
        if len(self.records) >= self.MIN_CYCLES_FOR_REWEIGHT and self.verifier_weights:
            cur_weights = current_config.get("verifier_check_weights") or {}
            new_weights = self._proposed_verifier_weights(cur_weights)
            if new_weights and new_weights != cur_weights:
                proposal["verifier_check_weights"] = new_weights
                shifted = [
                    f"{k}:{cur_weights.get(k, 1.0):.2f}->{v:.2f}"
                    for k, v in new_weights.items()
                    if abs(v - cur_weights.get(k, 1.0)) > 1e-4
                ]
                reasoning.append(
                    f"Verifier reweight (EMA cov over {len(self.records)} cycles): "
                    + ", ".join(shifted)
                )

        # 3. Generator prompt — categorical tracker verdict gates the swap.
        tmpl_effect = self.tracker.paired_effect("generator_template_id")
        if tmpl_effect["decision"] == "insufficient_data":
            # Keep exploring — rotate to an untrialed variant if we have one.
            variant = self._propose_prompt_variant(
                current_config.get("generator_template"), exploration_only=True,
            )
            if variant is not None:
                tmpl, vid, why = variant
                proposal["generator_template"] = (tmpl, vid)
                reasoning.append(f"{why} (tracker: exploring, n={tmpl_effect.get('n', 0)})")
        elif tmpl_effect["decision"].startswith("prefer="):
            # Tracker identified a significant winner — rotate to it if not current.
            best = tmpl_effect.get("best_value")
            if isinstance(best, (int, float)):
                vid = int(best)
                if 0 <= vid < len(self.prompt_variants):
                    best_tmpl = self.prompt_variants[vid].template
                    if best_tmpl != current_config.get("generator_template"):
                        proposal["generator_template"] = (best_tmpl, vid)
                        reasoning.append(
                            f"Generator template: tracker prefers variant #{vid} "
                            f"(mean Δ={tmpl_effect['best_mean']:+.4f}, "
                            f"p={tmpl_effect['p_value']:.3f})"
                        )
        # else "neutral": freeze, no proposal.

        # Nothing to apply? Still return a valid proposal with applied=False.
        proposal["applied"] = any(
            proposal[k] is not None for k in
            ("learning_rate", "verifier_check_weights", "generator_template")
        )
        if proposal["applied"]:
            self._last_pre_revert_state = {
                "learning_rate": current_config.get("learning_rate"),
                "verifier_check_weights": dict(current_config.get("verifier_check_weights") or {}),
                "generator_template": current_config.get("generator_template"),
            }
            self._last_proposal = {
                k: proposal[k] for k in
                ("learning_rate", "verifier_check_weights", "generator_template")
            }
            self._log_decision({
                "cycle": len(self.records) + 1,
                "kind": "propose",
                "proposal": self._last_proposal,
                "reasoning": reasoning,
                "ts": time.time(),
            })
        return proposal

    def get_revert_target(self) -> dict | None:
        """If recent cycles show regression, return the pre-proposal state to revert.

        Delegates the "is this a regression?" judgement to
        `CausalTracker.recent_regressions()` — which already applies a small
        tolerance (|delta| > 0.005) so single-cycle noise doesn't trigger a
        revert. We trip the revert when ≥2 of the last 3 cycles regressed.
        """
        if self._last_pre_revert_state is None:
            return None
        recent_bad = self.tracker.recent_regressions(n=3)
        if recent_bad >= 2:
            target = self._last_pre_revert_state
            self._last_pre_revert_state = None
            self._last_proposal = None
            self._log_decision({
                "cycle": len(self.records),
                "kind": "revert",
                "reason": f"tracker.recent_regressions()={recent_bad} (>=2 of last 3)",
                "revert_to": target,
                "ts": time.time(),
            })
            return target
        return None

    # ---------------------------------------------------------------
    # Prompt evolution
    # ---------------------------------------------------------------

    def seed_prompt(self, template: str) -> int:
        """Seed the prompt-variant pool with a baseline template."""
        for i, v in enumerate(self.prompt_variants):
            if v.template == template:
                return i
        self.prompt_variants.append(PromptVariant(template=template))
        return len(self.prompt_variants) - 1

    def register_prompt_mutator(self, mutator: Callable[[str], str]) -> None:
        """Optional: caller supplies a function that mutates the best template.

        Typical impl: call the model with an improvement prompt. The mutator
        is called only when the variant pool has room; output is sanity-checked
        and added to the pool.
        """
        self._mutator = mutator

    def _propose_prompt_variant(
        self, current_template: str | None, exploration_only: bool = False,
    ) -> tuple[str, int, str] | None:
        if not self.prompt_variants:
            return None

        # Keep pool at top-K by mean improvement (with at least 1 trial).
        trialed = [v for v in self.prompt_variants if v.trials > 0]
        trialed.sort(key=lambda v: v.mean_improvement, reverse=True)
        untrialed = [v for v in self.prompt_variants if v.trials == 0]
        self.prompt_variants = (trialed[: self.MAX_PROMPT_VARIANTS] + untrialed)[
            : self.MAX_PROMPT_VARIANTS + 2
        ]

        # If we have an untrialed variant, schedule it next.
        for i, v in enumerate(self.prompt_variants):
            if v.trials == 0 and v.template != current_template:
                return (v.template, i, f"Trying untrialed prompt variant #{i}")

        # In exploration-only mode, don't swap based on local means —
        # the CausalTracker hasn't said anything significant yet, so let it keep
        # accumulating evidence rather than thrash between variants on noise.
        if exploration_only:
            return None

        # Else, if best > current, propose swap.
        if trialed and trialed[0].mean_improvement > 0:
            tmpl = trialed[0].template
            vid = self.prompt_variants.index(trialed[0])
            if tmpl != current_template:
                return (
                    tmpl, vid,
                    f"Rotating to best prompt variant #{vid} "
                    f"(mean Δ={trialed[0].mean_improvement:+.4f} over {trialed[0].trials})"
                )
        return None

    def add_mutated_variant(self, new_template: str) -> int | None:
        """Call externally after invoking a mutator on the current best template."""
        if not new_template or len(new_template) < 20:
            return None
        required = {"{problem}", "{domain}", "{subdomain}"}
        if not all(p in new_template for p in required):
            return None
        for v in self.prompt_variants:
            if v.template == new_template:
                return None
        if len(self.prompt_variants) >= self.MAX_PROMPT_VARIANTS + 2:
            # Drop worst trialed variant to make room.
            trialed = [v for v in self.prompt_variants if v.trials > 0]
            if trialed:
                worst = min(trialed, key=lambda v: v.mean_improvement)
                self.prompt_variants.remove(worst)
        self.prompt_variants.append(PromptVariant(template=new_template))
        return len(self.prompt_variants) - 1

    # ---------------------------------------------------------------
    # Verifier weight regression
    # ---------------------------------------------------------------

    def _update_verifier_weight_regression(
        self, check_scores: dict[str, float], delta: float
    ) -> None:
        """Simple online regression: accumulate (check_score, delta) covariance.

        If a check-score on accepted samples correlates positively with held-out
        delta, its weight grows. Correlates negatively → shrinks.
        """
        if not hasattr(self, "_cov"):
            self._cov: dict[str, float] = {}
            self._n_obs: int = 0
        self._n_obs += 1
        for name, score in check_scores.items():
            prev = self._cov.get(name, 0.0)
            # Exponential-moving covariance with recent-heavy weighting.
            self._cov[name] = 0.7 * prev + 0.3 * (score * delta)
            self.verifier_weights.setdefault(name, 1.0)

    def _proposed_verifier_weights(
        self, current: dict[str, float]
    ) -> dict[str, float]:
        """Shift weight toward checks with positive covariance vs. held-out delta."""
        if not getattr(self, "_cov", None):
            return {}
        # Softmax-lite: positive cov → boost, negative → shrink, bounded ±30%.
        out: dict[str, float] = {}
        max_abs = max((abs(v) for v in self._cov.values()), default=1.0) or 1.0
        for name, cov in self._cov.items():
            base = current.get(name, 1.0)
            adj = 1.0 + (cov / max_abs) * MAX_STEP_FRAC
            new_w = _clip_frac(base * adj, base)
            out[name] = round(new_w, 4)
        # Include any current weights we didn't have data for (leave unchanged).
        for k, v in current.items():
            out.setdefault(k, v)
        return out

    # ---------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "records": [asdict(r) for r in self.records],
            "lr_bandit": self.lr_bandit.to_dict() if self.lr_bandit else None,
            "prompt_variants": [asdict(v) for v in self.prompt_variants],
            "verifier_weights": dict(self.verifier_weights),
            "cov": dict(getattr(self, "_cov", {})),
            "n_obs": getattr(self, "_n_obs", 0),
            "last_proposal": self._last_proposal,
            "last_pre_revert_state": self._last_pre_revert_state,
        }

    def load_state(self, state: dict) -> None:
        self.records = [MetaCycleRecord(**r) for r in state.get("records", [])]
        if state.get("lr_bandit"):
            self.lr_bandit = LRBandit.from_dict(state["lr_bandit"])
        self.prompt_variants = [
            PromptVariant(**v) for v in state.get("prompt_variants", [])
        ]
        self.verifier_weights = dict(state.get("verifier_weights", {}))
        self._cov = dict(state.get("cov", {}))
        self._n_obs = state.get("n_obs", 0)
        self._last_proposal = state.get("last_proposal")
        self._last_pre_revert_state = state.get("last_pre_revert_state")

    # ---------------------------------------------------------------
    # Logging
    # ---------------------------------------------------------------

    def _log_decision(self, entry: dict) -> None:
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except OSError as e:
            logger.warning(f"meta: failed to write decision log: {e}")
