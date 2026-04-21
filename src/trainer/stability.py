"""Training stability diagnostics: gradient norms + LoRA weight deltas.

Eval regression is a lagging indicator — by the time paired held-out
delta goes red, the adapter may have already eaten several steps of bad
updates. Gradient-norm divergence is a leading indicator: the step
*size* of updates blows up (or collapses to zero) before the measured
capability does.

This module provides two light-weight tools:

1. ``GradientNormTracker``: per-step record of the global gradient
   L2-norm over trainable (LoRA) params + a per-cycle snapshot of the
   LoRA weight-delta norm (||W_after - W_before||) for the cycle.
   Aggregates into a summary that the orchestrator logs into
   ``cycle_metrics`` alongside eval scores.

2. ``detect_gradient_divergence``: pure detector over a history of
   per-cycle summaries. Fires when recent gradient-norm behavior looks
   non-stationary in a failure-mode-specific way — either the median
   grad-norm climbs monotonically (drift toward blowup) or the
   coefficient of variation explodes (instability). Returns a trigger
   code the caller matches, rather than a bare bool, so the
   orchestrator can distinguish "blowup" from "collapse".

Integration is non-invasive: the tracker hooks ``optimizer.step`` the
same way ``grpo.KLDivergenceGuard`` does, so no edits to the training
loops in ``custom_lora.py`` are required.
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# Alarm trigger codes. Orchestrator maps these to actions.
TRIGGER_NONE = "none"
TRIGGER_GRAD_BLOWUP = "grad_blowup"          # grad norm monotonically rising
TRIGGER_GRAD_COLLAPSE = "grad_collapse"       # grad norm → 0 (stuck)
TRIGGER_GRAD_INSTABILITY = "grad_instability"  # CV explosion


# Default thresholds. Tuned loose — these are sentinels, not gauges.
# The orchestrator's paired-held-out-delta is the precision signal; this
# module is the early-warning alarm.
DEFAULT_BLOWUP_RATIO = 3.0       # median grad norm > 3x the rolling baseline
DEFAULT_COLLAPSE_FLOOR = 1e-6    # median grad norm below this = stuck
DEFAULT_CV_LIMIT = 2.0           # std / mean over cycle > 2.0 = unstable
DEFAULT_HISTORY_WINDOW = 5       # cycles of history required to fire


@dataclass
class CycleGradSummary:
    """One cycle's worth of gradient + LoRA-delta statistics.

    All fields are JSON-safe so this drops straight into ``cycle_metrics``.
    """
    cycle: int
    n_steps: int
    grad_norm_mean: float
    grad_norm_median: float
    grad_norm_std: float
    grad_norm_max: float
    grad_norm_min: float
    lora_weight_delta_norm: float  # ||W_after - W_before|| end-to-end
    # Distribution percentiles for log-scale alarms.
    grad_norm_p10: float
    grad_norm_p90: float

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @property
    def coefficient_of_variation(self) -> float:
        mu = self.grad_norm_mean
        if mu <= 0 or not math.isfinite(mu):
            return float("inf")
        return self.grad_norm_std / mu


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * pct
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return sorted_vals[int(k)]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def summarize_grad_norms(
    cycle: int,
    grad_norms: list[float],
    lora_weight_delta_norm: float,
) -> CycleGradSummary:
    """Reduce a list of per-step grad norms into a cycle summary."""
    finite = [g for g in grad_norms if math.isfinite(g)]
    n = len(finite)
    if n == 0:
        return CycleGradSummary(
            cycle=cycle, n_steps=0,
            grad_norm_mean=0.0, grad_norm_median=0.0, grad_norm_std=0.0,
            grad_norm_max=0.0, grad_norm_min=0.0,
            grad_norm_p10=0.0, grad_norm_p90=0.0,
            lora_weight_delta_norm=float(lora_weight_delta_norm),
        )
    s = sorted(finite)
    return CycleGradSummary(
        cycle=cycle,
        n_steps=n,
        grad_norm_mean=float(statistics.fmean(finite)),
        grad_norm_median=float(statistics.median(finite)),
        grad_norm_std=float(statistics.pstdev(finite)) if n > 1 else 0.0,
        grad_norm_max=float(s[-1]),
        grad_norm_min=float(s[0]),
        grad_norm_p10=float(_percentile(s, 0.10)),
        grad_norm_p90=float(_percentile(s, 0.90)),
        lora_weight_delta_norm=float(lora_weight_delta_norm),
    )


# ---------------------------------------------------------------------------
# Tracker — hooks optimizer.step to record gradient L2-norm per step.
# ---------------------------------------------------------------------------

class GradientNormTracker:
    """Per-step gradient-norm collector with optional ``torch`` integration.

    Usage (end-to-end cycle integration):

        import torch
        tracker = GradientNormTracker()
        tracker.begin_cycle(cycle=7, lora_params=lora_params)
        tracker.install_on_optimizer(optimizer)
        try:
            trainer.train(...)
        finally:
            tracker.uninstall_on_optimizer(optimizer)
        summary = tracker.end_cycle(lora_params)
        cycle_metrics["grad_stability"] = summary.to_dict()

    The module deliberately avoids a hard dependency on torch at import
    time so tests can exercise ``summarize_grad_norms`` /
    ``detect_gradient_divergence`` in torch-free environments. The
    optimizer-hook path imports torch lazily.
    """

    def __init__(self):
        self._per_step_norms: list[float] = []
        self._cycle: Optional[int] = None
        self._orig_step = None
        self._tracked_params = None
        self._pre_cycle_snapshot = None

    def begin_cycle(self, cycle: int, lora_params=None) -> None:
        self._per_step_norms = []
        self._cycle = int(cycle)
        self._tracked_params = list(lora_params) if lora_params is not None else None
        if self._tracked_params:
            try:
                import torch
                self._pre_cycle_snapshot = [p.detach().clone() for p in self._tracked_params]
            except ImportError:
                self._pre_cycle_snapshot = None
        else:
            self._pre_cycle_snapshot = None

    def record_step_norm(self, norm: float) -> None:
        if math.isfinite(norm):
            self._per_step_norms.append(float(norm))

    def install_on_optimizer(self, optimizer) -> None:
        """Wrap ``optimizer.step`` to record the current grad-norm each call."""
        if self._orig_step is not None or getattr(self, "_hook_handle", None) is not None:
            return  # already installed
        try:
            import torch
        except ImportError:
            logger.debug("torch unavailable; tracker records only via record_step_norm")
            return
        tracker = self
        params = list(self._tracked_params or [p for g in optimizer.param_groups for p in g["params"]])

        def _pre_step_hook(opt, args, kwargs):
            # Compute norm BEFORE step (step consumes grads via optimizer update).
            total_sq = 0.0
            for p in params:
                g = getattr(p, "grad", None)
                if g is None:
                    continue
                total_sq += float(g.detach().pow(2).sum().item())
            tracker.record_step_norm(total_sq ** 0.5)

        # Use PyTorch's native step-pre-hook API instead of monkey-patching
        # optimizer.step. Monkey-patching replaces the bound method with a
        # plain function, which breaks `optimizer.step.__func__` lookups that
        # transformers' training loop / accelerate internals perform.
        try:
            handle = optimizer.register_step_pre_hook(_pre_step_hook)
            self._hook_handle = handle
            self._hooked_optimizer = optimizer
        except AttributeError:
            # Older torch (<2.0) doesn't have register_step_pre_hook.
            # Fall back to manual method-wrap via types.MethodType so
            # the replaced attribute remains a bound method (preserves .__func__).
            import types
            orig_step = optimizer.step

            def _wrapped(opt_self, *args, **kwargs):
                total_sq = 0.0
                for p in params:
                    g = getattr(p, "grad", None)
                    if g is None:
                        continue
                    total_sq += float(g.detach().pow(2).sum().item())
                tracker.record_step_norm(total_sq ** 0.5)
                return orig_step(*args, **kwargs)

            optimizer.step = types.MethodType(_wrapped, optimizer)
            self._orig_step = orig_step
            self._hooked_optimizer = optimizer

    def uninstall_on_optimizer(self, optimizer=None) -> None:
        opt = optimizer if optimizer is not None else getattr(self, "_hooked_optimizer", None)
        handle = getattr(self, "_hook_handle", None)
        if handle is not None:
            try:
                handle.remove()
            except Exception:
                pass
            self._hook_handle = None
        if self._orig_step is not None:
            if opt is not None:
                opt.step = self._orig_step
            self._orig_step = None
        self._hooked_optimizer = None

    def end_cycle(self, lora_params=None) -> CycleGradSummary:
        delta_norm = 0.0
        if self._pre_cycle_snapshot is not None:
            current = list(lora_params) if lora_params is not None else self._tracked_params
            if current is not None and len(current) == len(self._pre_cycle_snapshot):
                try:
                    total = 0.0
                    for p_now, p_pre in zip(current, self._pre_cycle_snapshot):
                        diff = (p_now.detach() - p_pre).pow(2).sum().item()
                        total += float(diff)
                    delta_norm = total ** 0.5
                except Exception as e:
                    logger.debug("delta-norm calc failed: %s", e)
        cycle = self._cycle if self._cycle is not None else -1
        summary = summarize_grad_norms(cycle, self._per_step_norms, delta_norm)
        # Free snapshot refs.
        self._pre_cycle_snapshot = None
        self._tracked_params = None
        return summary


# ---------------------------------------------------------------------------
# Cross-cycle divergence detector.
# ---------------------------------------------------------------------------

def detect_gradient_divergence(
    history: list[CycleGradSummary | dict],
    *,
    window: int = DEFAULT_HISTORY_WINDOW,
    blowup_ratio: float = DEFAULT_BLOWUP_RATIO,
    collapse_floor: float = DEFAULT_COLLAPSE_FLOOR,
    cv_limit: float = DEFAULT_CV_LIMIT,
) -> str:
    """Return a trigger code — ``TRIGGER_NONE`` if healthy, else the alarm.

    Semantics:

    - ``TRIGGER_GRAD_BLOWUP`` — median grad-norm of the last cycle
      exceeds ``blowup_ratio`` times the median of the rolling baseline
      (all prior cycles in ``history[-window:-1]``). Catches the
      pre-divergence step-size ramp that foreshadows loss-goes-NaN.
    - ``TRIGGER_GRAD_COLLAPSE`` — every cycle in the window has median
      grad-norm below ``collapse_floor``. Catches the silent-dead-adapter
      failure where training runs but parameters aren't meaningfully
      updated (effective learning rate ≈ 0).
    - ``TRIGGER_GRAD_INSTABILITY`` — coefficient of variation of the
      last cycle exceeds ``cv_limit``. Catches the case where mean grad
      norm looks fine but within-cycle variance is pathological (sparse
      blowup steps averaged out by many near-zero steps).

    The detector is a pure function — orchestrator-agnostic, trivially
    testable, no side effects. Caller decides the action (rollback,
    shrink LR, escalate to GRPO, abort).
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if len(history) < window:
        return TRIGGER_NONE

    recent = [_as_summary(h) for h in history[-window:]]
    last = recent[-1]
    prior = recent[:-1]
    medians = [h.grad_norm_median for h in prior]

    # Collapse — entire window is flatlined.
    if all(h.grad_norm_median < collapse_floor for h in recent):
        return TRIGGER_GRAD_COLLAPSE

    # Instability — intra-cycle CV of the last cycle exploded.
    if last.coefficient_of_variation > cv_limit:
        return TRIGGER_GRAD_INSTABILITY

    # Blowup — last median is >= blowup_ratio × prior median.
    if medians:
        baseline = statistics.median(medians)
        if baseline > 0 and last.grad_norm_median > blowup_ratio * baseline:
            return TRIGGER_GRAD_BLOWUP

    return TRIGGER_NONE


def _as_summary(h) -> CycleGradSummary:
    if isinstance(h, CycleGradSummary):
        return h
    if isinstance(h, dict):
        # Accept cycle_metrics["grad_stability"] passthrough.
        return CycleGradSummary(
            cycle=int(h.get("cycle", -1)),
            n_steps=int(h.get("n_steps", 0)),
            grad_norm_mean=float(h.get("grad_norm_mean", 0.0)),
            grad_norm_median=float(h.get("grad_norm_median", 0.0)),
            grad_norm_std=float(h.get("grad_norm_std", 0.0)),
            grad_norm_max=float(h.get("grad_norm_max", 0.0)),
            grad_norm_min=float(h.get("grad_norm_min", 0.0)),
            grad_norm_p10=float(h.get("grad_norm_p10", 0.0)),
            grad_norm_p90=float(h.get("grad_norm_p90", 0.0)),
            lora_weight_delta_norm=float(h.get("lora_weight_delta_norm", 0.0)),
        )
    raise TypeError(f"cannot coerce {type(h)!r} to CycleGradSummary")


__all__ = [
    "CycleGradSummary",
    "GradientNormTracker",
    "TRIGGER_NONE",
    "TRIGGER_GRAD_BLOWUP",
    "TRIGGER_GRAD_COLLAPSE",
    "TRIGGER_GRAD_INSTABILITY",
    "DEFAULT_BLOWUP_RATIO",
    "DEFAULT_COLLAPSE_FLOOR",
    "DEFAULT_CV_LIMIT",
    "DEFAULT_HISTORY_WINDOW",
    "summarize_grad_norms",
    "detect_gradient_divergence",
]
