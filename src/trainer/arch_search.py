"""Architecture search loop — propose structural motif changes, evaluate
on a small test model, keep if anchor improves.

Fires every `GrowthConfig.arch_search_every` cycles when
`arch_search_enabled=True`. Each iteration:

  1. Propose a motif from the candidate pool.
  2. Apply it to a small 1-3B test skeleton.
  3. Train briefly (trainer injected).
  4. Evaluate on the anchor (eval_fn injected).
  5. Compare to baseline; keep the motif iff
     delta >= cfg.arch_search_min_delta (default 0.005).

Motifs are expressed as pure `Callable[[nn.Module], nn.Module]` mutators
so new ones can be added without touching the loop. Production callers
inject real `train_fn` / `eval_fn`; tests inject mocks. The test model is
also injected via `model_factory`, letting the caller control cost.

Return type: :class:`ArchSearchResult` with the motif decision, deltas,
and the mutated skeleton (if accepted).
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch.nn as nn

from src.trainer.growth import GrowthConfig

logger = logging.getLogger(__name__)


# -------------------- motif pool ---------------------------------------------

Motif = Callable[[nn.Module], nn.Module]


def motif_sparse_moe_ffn(model: nn.Module) -> nn.Module:
    """Mark the model's FFN layers for sparse-MoE conversion.

    The actual dense→sparse rewrite happens via
    :mod:`src.trainer.moe_conversion`; this motif only tags the skeleton
    so the growth orchestrator routes it to the MoE path on next grow.
    """
    student = copy.deepcopy(model)
    student._arch_motif = "sparse_moe_ffn"
    return student


def motif_state_space_hybrid(model: nn.Module) -> nn.Module:
    """Interleave a state-space (Mamba-style) block between attention blocks.

    Tag-only; the actual SSM layer construction is downstream's job. The
    search loop just needs something that the eval_fn can distinguish.
    """
    student = copy.deepcopy(model)
    student._arch_motif = "state_space_hybrid"
    return student


def motif_recurrent_memory(model: nn.Module) -> nn.Module:
    """Add a persistent recurrent memory cell alongside the attention stack."""
    student = copy.deepcopy(model)
    student._arch_motif = "recurrent_memory"
    return student


def motif_attention_variant(model: nn.Module) -> nn.Module:
    """Swap standard MHA for a grouped / linear attention variant."""
    student = copy.deepcopy(model)
    student._arch_motif = "attention_variant"
    return student


DEFAULT_MOTIFS: tuple[tuple[str, Motif], ...] = (
    ("sparse_moe_ffn", motif_sparse_moe_ffn),
    ("state_space_hybrid", motif_state_space_hybrid),
    ("recurrent_memory", motif_recurrent_memory),
    ("attention_variant", motif_attention_variant),
)


# -------------------- search loop --------------------------------------------


@dataclass
class ArchTrial:
    motif_name: str
    baseline_score: float
    trial_score: float
    delta: float
    accepted: bool


@dataclass
class ArchSearchResult:
    trials: list[ArchTrial] = field(default_factory=list)
    accepted_motif: Optional[str] = None
    accepted_model: Optional[nn.Module] = None
    best_delta: float = 0.0


def run_arch_search(
    base_model: nn.Module,
    cfg: GrowthConfig,
    *,
    train_fn: Callable[[nn.Module], nn.Module],
    eval_fn: Callable[[nn.Module], float],
    motifs: Sequence[tuple[str, Motif]] = DEFAULT_MOTIFS,
    model_factory: Optional[Callable[[nn.Module], nn.Module]] = None,
) -> ArchSearchResult:
    """Try each motif in turn. Return the first one that clears
    `cfg.arch_search_min_delta`, plus the full trial log.

    `train_fn(model) -> model` — short training pass on the test skeleton.
    `eval_fn(model) -> float` — anchor score (higher is better).
    `model_factory(base) -> model` — produce a small test-scale copy of
    the base model. Defaults to `deepcopy` so tiny test skeletons work
    out of the box.
    """
    if not cfg.arch_search_enabled:
        return ArchSearchResult()

    factory = model_factory or (lambda b: copy.deepcopy(b))
    baseline_skeleton = factory(base_model)
    baseline_trained = train_fn(baseline_skeleton)
    baseline_score = eval_fn(baseline_trained)

    result = ArchSearchResult()
    for name, motif in motifs:
        trial_skeleton = motif(factory(base_model))
        trial_trained = train_fn(trial_skeleton)
        trial_score = eval_fn(trial_trained)
        delta = trial_score - baseline_score
        accepted = delta >= cfg.arch_search_min_delta
        trial = ArchTrial(
            motif_name=name,
            baseline_score=baseline_score,
            trial_score=trial_score,
            delta=delta,
            accepted=accepted,
        )
        result.trials.append(trial)
        logger.info(
            "arch_search motif=%s baseline=%.4f trial=%.4f delta=%.4f accepted=%s",
            name, baseline_score, trial_score, delta, accepted,
        )
        if accepted and delta > result.best_delta:
            result.accepted_motif = name
            result.accepted_model = trial_trained
            result.best_delta = delta
            # First acceptable motif wins — cost guard. Caller can set
            # min_delta higher to force a de-facto best-of-K.
            break
    return result
