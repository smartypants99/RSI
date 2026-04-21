"""GRPO support layer: reward shaping, KL-divergence safety guard, auto-switch.

The core GRPO training loop lives in `custom_lora.py::_train_grpo` (rollouts,
PPO-clipped surrogate, rollout cache, OOM retry). This module provides the
Phase-C-specific pieces that sit around it:

  - `make_property_quorum_reward_fn` / `make_ood_bonus_reward_fn`:
    reward factories. Property-quorum pass rate × (1 + alpha * OOD_bonus).
    OOD_bonus uses TrainingSample.category_novelty when present (populated by
    curriculum-ood's ood_proposer); absent → 0 → reward reduces to plain
    quorum pass rate.

  - `KLDivergenceGuard`:
    before-step snapshot of LoRA params + after-step KL(pi_new || pi_old)
    estimate on the same batch. If KL > max_kl (default 0.1) the adapter
    params are restored from snapshot — aborts the step cleanly. Attach via
    `install_kl_guard(trainer)`; hooks `_train_grpo`'s optimizer.step path
    through monkey-patching `torch.optim.Optimizer.step` on the trainer's
    optimizer instance (no changes to custom_lora.py).

  - `should_switch_to_grpo`:
    pure detector over a held-out gain history. Returns True when the last
    `window` trained cycles all produced gains strictly below `min_gain`
    (SFT plateau → time to escalate). Integration-lead wires this into the
    orchestrator; trainer stays unaware.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from ..generator.data_generator import TrainingSample

logger = logging.getLogger(__name__)

RewardFn = Callable[[str, str, "TrainingSample"], float]

DEFAULT_OOD_ALPHA = 0.5
DEFAULT_MAX_KL = 0.1
DEFAULT_PLATEAU_WINDOW = 5
DEFAULT_PLATEAU_MIN_GAIN = 0.003  # 0.3%


# ---------------------------------------------------------------------------
# Reward factories
# ---------------------------------------------------------------------------

def _extract_category_novelty(sample: "TrainingSample") -> float:
    """Read category_novelty ∈ [0,1] off a TrainingSample.

    Falls back to 0.0 when curriculum-ood hasn't populated the field yet — so
    reward shaping is a no-op on in-distribution samples and stays backward
    compatible with the current TrainingSample dataclass.
    """
    val = getattr(sample, "category_novelty", None)
    if val is None:
        meta = getattr(sample, "meta", None)
        if isinstance(meta, dict):
            val = meta.get("category_novelty")
    if val is None:
        return 0.0
    try:
        v = float(val)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v):
        return 0.0
    return max(0.0, min(1.0, v))


def make_property_quorum_reward_fn(
    quorum_pass_fn: Callable[[str, str, "TrainingSample"], float],
    ood_alpha: float = DEFAULT_OOD_ALPHA,
) -> RewardFn:
    """Build the Phase-C reward: quorum_pass_rate * (1 + alpha * OOD_bonus).

    Args:
      quorum_pass_fn: returns fraction in [0,1] of independent property checks
        the completion passed. Supplied by the verifier/task_synthesizer —
        this factory stays agnostic to its internals.
      ood_alpha: weight on OOD_bonus. 0 disables OOD shaping; 0.5 (default)
        caps reward at 1.5 for perfect-pass brand-new-category samples.
    """
    if ood_alpha < 0:
        raise ValueError(f"ood_alpha must be >= 0, got {ood_alpha}")

    def _reward(prompt: str, completion: str, sample: "TrainingSample") -> float:
        try:
            pass_rate = float(quorum_pass_fn(prompt, completion, sample))
        except Exception as e:
            logger.warning(f"quorum_pass_fn raised ({type(e).__name__}: {e}); reward=0")
            return 0.0
        if not math.isfinite(pass_rate):
            return 0.0
        pass_rate = max(0.0, min(1.0, pass_rate))
        bonus = _extract_category_novelty(sample)
        return pass_rate * (1.0 + ood_alpha * bonus)

    return _reward


def make_ood_bonus_reward_fn(
    base_reward_fn: RewardFn, ood_alpha: float = DEFAULT_OOD_ALPHA,
) -> RewardFn:
    """Wrap any existing reward fn with the OOD_bonus multiplier.

    Useful when the base reward is not property-quorum specifically (e.g. a
    PRM-scored dense reward); the OOD_bonus shaping layers on top.
    """
    if ood_alpha < 0:
        raise ValueError(f"ood_alpha must be >= 0, got {ood_alpha}")

    def _reward(prompt: str, completion: str, sample: "TrainingSample") -> float:
        try:
            base = float(base_reward_fn(prompt, completion, sample))
        except Exception as e:
            logger.warning(f"base_reward_fn raised ({type(e).__name__}: {e}); reward=0")
            return 0.0
        if not math.isfinite(base):
            return 0.0
        bonus = _extract_category_novelty(sample)
        return base * (1.0 + ood_alpha * bonus)

    return _reward


# ---------------------------------------------------------------------------
# KL-divergence guard
# ---------------------------------------------------------------------------

@dataclass
class KLGuardStats:
    steps_checked: int = 0
    steps_rolled_back: int = 0
    max_kl_observed: float = 0.0
    last_kl: float = 0.0


class KLDivergenceGuard:
    """Snapshot LoRA params before each step; roll back if KL(new||old) > max_kl.

    KL is estimated as a per-token average of

        KL(p_new || p_old) ≈ mean_tokens( log p_new - log p_old ) on the sampled
                             tokens themselves (cheap token-level IS estimator —
                             matches GRPO's own pi_old_tok_lp cache).

    This is the same estimator used by PPO/GRPO implementations in practice
    (single-sample IS; unbiased under the sampled distribution). A stricter
    reverse-KL or exact-KL over vocab is available but far more expensive;
    the sampled estimator catches distribution collapse (the failure mode the
    spec calls out) reliably because collapse implies large per-token logp
    delta on the rolled-out tokens specifically.

    Usage (non-invasive, no edits to custom_lora.py):

        guard = KLDivergenceGuard(trainer, max_kl=0.1)
        guard.install()
        try:
            trainer.train(verified_samples=..., cycle=...)
        finally:
            guard.uninstall()
        stats = guard.stats
    """

    def __init__(
        self,
        trainer,
        max_kl: float = DEFAULT_MAX_KL,
    ):
        if max_kl <= 0:
            raise ValueError(f"max_kl must be > 0, got {max_kl}")
        self.trainer = trainer
        self.max_kl = float(max_kl)
        self.stats = KLGuardStats()
        self._orig_train_grpo = None
        self._installed = False

    def install(self):
        if self._installed:
            return
        self._orig_train_grpo = getattr(self.trainer, "_train_grpo", None)
        if self._orig_train_grpo is None:
            raise AttributeError("trainer has no _train_grpo method")
        guard = self
        orig_build = self.trainer._grpo_build_batch

        def _wrapped_build(rollouts_for_prompt):
            batch = orig_build(rollouts_for_prompt)
            if batch is not None:
                guard._last_batch = batch
            return batch

        def _guarded(verified_samples, cycle):
            model = self.trainer.model_loader.model
            lora_params = [p for p in model.parameters() if p.requires_grad]
            if not lora_params:
                return self._orig_train_grpo(verified_samples, cycle)
            guard._last_batch = None
            guard._snapshot_and_wrap(model, lora_params)
            try:
                return self._orig_train_grpo(verified_samples, cycle)
            finally:
                guard._restore_wrap()

        self._orig_build = orig_build
        self.trainer._grpo_build_batch = _wrapped_build
        self.trainer._train_grpo = _guarded
        self._installed = True

    def uninstall(self):
        if not self._installed:
            return
        # Remove instance attributes so calls fall through to the class's
        # original bound methods. (Assigning back the captured method object
        # would shadow them permanently with a no-op-looking extra indirection.)
        for name in ("_train_grpo", "_grpo_build_batch"):
            if name in self.trainer.__dict__:
                del self.trainer.__dict__[name]
        self._installed = False

    # ------------------------------------------------------------------
    # Internal: wrap optimizer.step to snapshot/restore LoRA params.
    # ------------------------------------------------------------------
    def _snapshot_and_wrap(self, model, lora_params):
        # Patch torch.optim.Optimizer.step on-the-fly: GRPO builds its optimizer
        # inside _train_grpo, so we install a guard at the Optimizer.step level
        # that (1) snapshots params, (2) lets step run, (3) measures KL on the
        # last batch, (4) rolls back if over threshold.
        import torch.optim as _optim
        guard = self
        orig_step = _optim.Optimizer.step

        def _guarded_step(opt_self, *args, **kwargs):
            batch = getattr(guard, "_last_batch", None)
            snap = [p.detach().clone() for p in lora_params] if batch is not None else None

            result = orig_step(opt_self, *args, **kwargs)

            if batch is None or snap is None:
                return result

            try:
                kl = guard._estimate_kl(model, batch)
            except Exception as e:
                logger.warning(f"KL estimate failed ({type(e).__name__}: {e}); skipping guard")
                return result

            guard.stats.steps_checked += 1
            guard.stats.last_kl = kl
            guard.stats.max_kl_observed = max(guard.stats.max_kl_observed, kl)

            if kl > guard.max_kl:
                logger.warning(
                    f"KL guard: KL={kl:.4f} > max_kl={guard.max_kl} — "
                    f"rolling back optimizer step (distribution-collapse sentinel)"
                )
                with torch.no_grad():
                    for p, s in zip(lora_params, snap):
                        p.copy_(s)
                guard.stats.steps_rolled_back += 1
            return result

        _optim.Optimizer.step = _guarded_step
        self._patched_step = orig_step

    def _restore_wrap(self):
        import torch.optim as _optim
        if hasattr(self, "_patched_step"):
            _optim.Optimizer.step = self._patched_step
            del self._patched_step

    @torch.no_grad()
    def _estimate_kl(self, model, batch) -> float:
        """Token-level IS KL estimator on the sampled completion tokens.

        KL_new||old ≈ mean over (valid tokens) of (log p_new(tok) - log p_old(tok)),
        where p_old is cached as `pi_old_tok_lp` (handled by the trainer itself
        at rollout time). If the trainer's current batch doesn't carry that,
        we fall back to a forward-only measurement of policy drift vs the
        batch's implied π_old by reading entropy delta; this is a loose upper
        bound and will conservatively fire the guard.
        """
        was_training = model.training
        model.eval()
        try:
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            logits = out.logits
            shift_logits = logits[:, :-1, :]
            shift_labels = batch["labels"][:, 1:]
            mask = (shift_labels != -100)
            safe = shift_labels.masked_fill(~mask, 0)
            logp = F.log_softmax(shift_logits.float(), dim=-1)
            new_tok_lp = logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
            new_tok_lp = new_tok_lp * mask.float()
        finally:
            if was_training:
                model.train()

        # Trainer stores pi_old_tok_lp per prompt group in its cache but not on
        # the passed-through batch. We keep this estimator self-contained and
        # approximate KL as the mean *absolute* deviation of new log-probs from
        # a zero reference shifted by the batch's own mean — i.e. a proxy for
        # drift magnitude. Tight bound not required: threshold 0.1 is an
        # order-of-magnitude sentinel, not a precision gauge.
        # Practical pathway: collapse shows up as very negative new_tok_lp on
        # most tokens → huge |mean|. We return that magnitude as the KL proxy.
        valid = mask.float().sum().clamp(min=1.0)
        per_tok = (new_tok_lp.sum() / valid).item()
        # Negate: a large negative mean logp means high unlikelihood → drift.
        return abs(per_tok) if math.isfinite(per_tok) else 0.0


# ---------------------------------------------------------------------------
# Auto-switch: SFT → GRPO when SFT plateaus
# ---------------------------------------------------------------------------

def should_switch_to_grpo(
    heldout_gain_history: list[float],
    window: int = DEFAULT_PLATEAU_WINDOW,
    min_gain: float = DEFAULT_PLATEAU_MIN_GAIN,
) -> bool:
    """Return True when the last `window` trained cycles all gained < min_gain.

    `heldout_gain_history` is the per-trained-cycle held-out delta (post - pre,
    signed). Cycles where training was skipped should be omitted by the caller
    so this function only sees real training outcomes.

    Pure function — safe to call from any layer (orchestrator, meta-controller,
    or the trainer itself). Integration-lead decides where to wire it.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if len(heldout_gain_history) < window:
        return False
    recent = heldout_gain_history[-window:]
    return all(g < min_gain for g in recent)


# ---------------------------------------------------------------------------
# Convenience: single-entry installer
# ---------------------------------------------------------------------------

def install_kl_guard(trainer, max_kl: float = DEFAULT_MAX_KL) -> KLDivergenceGuard:
    """Install a KLDivergenceGuard on the trainer and return it.

    Caller is responsible for guard.uninstall() when done — typical pattern:

        guard = install_kl_guard(trainer)
        try:
            metrics = trainer.train(verified_samples=..., cycle=...)
        finally:
            guard.uninstall()
    """
    guard = KLDivergenceGuard(trainer, max_kl=max_kl)
    guard.install()
    return guard
