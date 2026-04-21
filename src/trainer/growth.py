"""Upward distillation — grow N → 1.5N parameters via depth expansion.

Every `grow_every` trained cycles, the orchestrator can call `grow_and_distill`
to instantiate a larger student architecture (more layers) initialised from
the current trained model (teacher), then distill the teacher's behaviour
into it on the self-generated sample pool.

Initialisation strategy (configurable):

  - "duplicate"   — duplicate nearest teacher layer + small gaussian noise.
                    Default; referenced in the task spec. Stable, preserves
                    teacher signal through structural doubling.
  - "identity"    — copy nearest teacher layer but zero the residual output
                    projections (attention `o_proj` and MLP `down_proj`) so
                    the new layer is a no-op at t=0. Function-preserving
                    (Net2Net / bert2BERT); gemini consult favours this for
                    fidelity. Exposed as an alternative but not default.

Safety guard: after distillation, compare held-out on teacher vs student. If
the student is worse by > `abort_if_worse_by`, discard the student and return
the teacher untouched. The orchestrator must treat `grow_and_distill` as
transactional — the returned model is either strictly-not-worse or the
original.

All heavy operations (real AutoModel load, config construction) go through
thin wrappers so tests can inject a fake skeleton.
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class GrowthConfig:
    grow_every: int = 15
    growth_factor: float = 1.5
    init_method: str = "duplicate"  # "duplicate" | "identity"
    duplicate_noise_std: float = 0.01
    distill_epochs: int = 1
    distill_lr: float = 1e-5
    distill_kl_temperature: float = 2.0
    distill_kl_weight: float = 0.5
    abort_if_worse_by: float = 0.01
    max_seq_length: int = 1024
    # Storage backend for grown weights.
    #   "vram"       — keep grown model resident in VRAM (default; fits up to
    #                  ~64B params at 4bit on a 48GB GPU).
    #   "tdq_stream" — when grown fp16 size would exceed the VRAM budget,
    #                  serialize to HF dir, compress to .tdq, and load via
    #                  TDQModelLoader for streaming inference thereafter.
    #   "auto"       — pick vram if budget permits, else tdq_stream.
    storage_backend: str = "vram"
    vram_safety_margin_gb: float = 4.0
    # If None at runtime, read from torch.cuda.get_device_properties(0).total_memory.
    vram_budget_gb: Optional[float] = None
    tdq_config: str = "A"

    def __post_init__(self):
        if self.grow_every < 1:
            raise ValueError(f"grow_every must be >= 1, got {self.grow_every}")
        if not (1.0 < self.growth_factor <= 4.0):
            raise ValueError(
                f"growth_factor must be in (1.0, 4.0], got {self.growth_factor}"
            )
        if self.init_method not in ("duplicate", "identity"):
            raise ValueError(
                f"init_method must be 'duplicate' or 'identity', "
                f"got {self.init_method!r}"
            )
        if self.duplicate_noise_std < 0:
            raise ValueError(
                f"duplicate_noise_std must be >= 0, got {self.duplicate_noise_std}"
            )
        if self.distill_epochs < 1:
            raise ValueError(f"distill_epochs must be >= 1, got {self.distill_epochs}")
        if self.distill_lr <= 0:
            raise ValueError(f"distill_lr must be > 0, got {self.distill_lr}")
        if self.distill_kl_temperature <= 0:
            raise ValueError("distill_kl_temperature must be > 0")
        if not (0.0 <= self.distill_kl_weight <= 1.0):
            raise ValueError(
                f"distill_kl_weight must be in [0, 1], got {self.distill_kl_weight}"
            )
        if self.abort_if_worse_by < 0:
            raise ValueError(
                f"abort_if_worse_by must be >= 0, got {self.abort_if_worse_by}"
            )
        if self.storage_backend not in ("vram", "tdq_stream", "auto"):
            raise ValueError(
                f"storage_backend must be 'vram' | 'tdq_stream' | 'auto', "
                f"got {self.storage_backend!r}"
            )
        if self.vram_safety_margin_gb < 0:
            raise ValueError("vram_safety_margin_gb must be >= 0")
        if self.vram_budget_gb is not None and self.vram_budget_gb <= 0:
            raise ValueError("vram_budget_gb must be > 0 when set")


def plan_target_layers(teacher_num_layers: int, growth_factor: float) -> int:
    """Compute the target student layer count for a given growth factor.

    Rounded to nearest int and clamped to strictly > teacher.
    """
    if teacher_num_layers < 1:
        raise ValueError("teacher_num_layers must be >= 1")
    target = max(teacher_num_layers + 1, round(teacher_num_layers * growth_factor))
    return target


def layer_mapping(teacher_num_layers: int, student_num_layers: int) -> list[int]:
    """Return a list[int] of length `student_num_layers` where entry i is
    the teacher-layer index to initialise student layer i from.

    Interleaves source layers so duplicated copies live adjacent to their
    original — the inductive bias most papers (LiGO, Staged-Training) found
    useful vs naïve stacking.
    """
    if student_num_layers <= teacher_num_layers:
        raise ValueError(
            f"student_num_layers ({student_num_layers}) must be > "
            f"teacher ({teacher_num_layers})"
        )
    # For each student slot i in [0, S), pick teacher layer floor(i * T / S).
    # This distributes the T teacher layers across S student slots, with
    # contiguous blocks of student slots sharing a teacher source.
    return [
        min(teacher_num_layers - 1, (i * teacher_num_layers) // student_num_layers)
        for i in range(student_num_layers)
    ]


def _zero_residual_projections(layer: nn.Module) -> None:
    """For identity-init: zero the output projections so the residual
    contribution of this layer is ~0 at t=0 and it acts as a pass-through.

    Looks for conventional names: ``o_proj`` (attn output), ``down_proj``
    (MLP output, Llama/Qwen), ``dense`` (BERT-style attn output). Silent
    no-op for layers that don't match — caller is responsible for checking.
    """
    targets = ("o_proj", "down_proj", "out_proj", "dense")
    for name, module in layer.named_modules():
        leaf = name.split(".")[-1]
        if leaf in targets and hasattr(module, "weight"):
            with torch.no_grad():
                module.weight.zero_()
                if getattr(module, "bias", None) is not None:
                    module.bias.zero_()


def _perturb_(module: nn.Module, std: float) -> None:
    """Add N(0, std) noise to all trainable parameters, in-place."""
    if std == 0:
        return
    with torch.no_grad():
        for p in module.parameters():
            p.add_(torch.randn_like(p) * std)


def build_student_layers(
    teacher_layers: Sequence[nn.Module],
    student_num_layers: int,
    cfg: GrowthConfig,
) -> nn.ModuleList:
    """Construct `student_num_layers` new layers by deep-copying from
    teacher according to the mapping, then applying the chosen init tweak.

    For ``identity`` init, only *inserted* duplicates (mapping repeats) are
    zeroed; the first student slot pointing at a teacher layer keeps that
    layer's weights, so at t=0 the student computes the same function as
    the teacher. For ``duplicate`` init, every student layer (including
    originals) gets the gaussian perturbation so all are slightly on the
    move — matching the task spec.

    Framework-specific integration (swapping them into an AutoModelForCausalLM)
    lives in :func:`grow_model`.
    """
    mapping = layer_mapping(len(teacher_layers), student_num_layers)
    new_layers: list[nn.Module] = []
    seen_src: set[int] = set()
    for src_idx in mapping:
        layer = copy.deepcopy(teacher_layers[src_idx])
        is_duplicate = src_idx in seen_src
        if cfg.init_method == "duplicate":
            _perturb_(layer, cfg.duplicate_noise_std)
        elif cfg.init_method == "identity" and is_duplicate:
            _zero_residual_projections(layer)
        seen_src.add(src_idx)
        new_layers.append(layer)
    return nn.ModuleList(new_layers)


def _locate_layers(model: nn.Module) -> tuple[nn.Module, str]:
    """Return (parent, attr_name) for the ModuleList holding transformer layers.

    Tries common conventions: ``model.model.layers`` (Llama/Qwen),
    ``model.transformer.h`` (GPT-2/Qwen-1), ``model.layers`` (bare stack).
    Raises AttributeError if none match.
    """
    candidates = [
        ("model", "layers"),           # Llama/Qwen/Mistral
        ("transformer", "h"),          # GPT-2/GPT-J
        (None, "layers"),              # bare skeleton (used in tests)
        (None, "h"),
    ]
    for parent_attr, child_attr in candidates:
        parent = getattr(model, parent_attr) if parent_attr else model
        if parent is None:
            continue
        if hasattr(parent, child_attr):
            child = getattr(parent, child_attr)
            if isinstance(child, (nn.ModuleList, nn.Sequential)):
                return parent, child_attr
    raise AttributeError(
        "Could not locate a transformer layer stack on this model — "
        "expected one of .model.layers / .transformer.h / .layers"
    )


def grow_model(teacher: nn.Module, cfg: GrowthConfig) -> nn.Module:
    """Return a deep-copied model with its transformer layer stack grown
    from T layers to round(T * growth_factor) layers, initialised per cfg.

    Does not touch the teacher. The student shares no parameter tensors
    with the teacher (safe to train independently). Config object on the
    returned model is NOT updated — caller should re-save the config if
    they want HF-compatible persistence (optional; the live model object
    runs fine regardless).
    """
    student = copy.deepcopy(teacher)
    parent, attr = _locate_layers(student)
    teacher_layers = list(getattr(parent, attr))
    target = plan_target_layers(len(teacher_layers), cfg.growth_factor)
    new_stack = build_student_layers(teacher_layers, target, cfg)
    setattr(parent, attr, new_stack)

    # Best-effort: update the HF-style config if present, so downstream code
    # (e.g. loss masks, attention masks) reads the correct depth.
    hf_config = getattr(student, "config", None)
    if hf_config is not None:
        for key in ("num_hidden_layers", "n_layer"):
            if hasattr(hf_config, key):
                setattr(hf_config, key, target)
    return student


# -------------------- distillation -------------------------------------------


def _kl_distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Standard soft-target KL on logits, temperature-scaled (Hinton 2015).

    Shapes: (..., V). Reduces to a scalar via mean over non-V dims.
    """
    t = temperature
    s = F.log_softmax(student_logits / t, dim=-1)
    q = F.softmax(teacher_logits / t, dim=-1)
    # KL(q || s) summed over vocab, mean over batch*seq
    kl = (q * (q.add(1e-12).log() - s)).sum(dim=-1)
    return (t * t) * kl.mean()


def distill_step(
    student: nn.Module,
    teacher: nn.Module,
    input_ids: torch.Tensor,
    labels: Optional[torch.Tensor],
    cfg: GrowthConfig,
) -> torch.Tensor:
    """Compute the combined CE + KL distillation loss for one batch.

    The teacher is run under ``torch.no_grad``. Caller is responsible for
    backward/optimizer.step.
    """
    student.train()
    teacher.eval()
    s_out = student(input_ids)
    s_logits = s_out.logits if hasattr(s_out, "logits") else s_out
    with torch.no_grad():
        t_out = teacher(input_ids)
        t_logits = t_out.logits if hasattr(t_out, "logits") else t_out
    kl = _kl_distill_loss(s_logits, t_logits, cfg.distill_kl_temperature)
    if labels is None:
        return kl
    ce = F.cross_entropy(
        s_logits.reshape(-1, s_logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
    )
    return (1.0 - cfg.distill_kl_weight) * ce + cfg.distill_kl_weight * kl


# -------------------- top-level orchestration entry point --------------------


# -------------------- VRAM budget / streaming path ---------------------------


def _detect_vram_budget_bytes(cfg: GrowthConfig) -> int:
    """Return the usable VRAM budget (bytes), minus safety margin.

    Prefers ``cfg.vram_budget_gb`` when set; otherwise queries CUDA. Returns
    0 when CUDA is unavailable and no explicit budget is set — callers
    should treat 0 as "unknown, assume fits".
    """
    margin = int(cfg.vram_safety_margin_gb * (1024 ** 3))
    if cfg.vram_budget_gb is not None:
        return max(0, int(cfg.vram_budget_gb * (1024 ** 3)) - margin)
    if not torch.cuda.is_available():
        return 0
    total = torch.cuda.get_device_properties(0).total_memory
    return max(0, total - margin)


def _estimate_grown_bytes(teacher: nn.Module, cfg: GrowthConfig) -> int:
    """Estimate fp16 bytes of the grown model.

    Assumes growth multiplies the transformer stack's parameter count by
    `growth_factor` and leaves the rest (embedding, lm_head) unchanged.
    Good enough for a budget decision; we over-estimate rather than
    under-estimate to stay conservative.
    """
    parent, attr = _locate_layers(teacher)
    stack = getattr(parent, attr)
    stack_params = sum(p.numel() for p in stack.parameters())
    total_params = sum(p.numel() for p in teacher.parameters())
    other_params = total_params - stack_params
    target_layers = plan_target_layers(len(stack), cfg.growth_factor)
    scale = target_layers / max(1, len(stack))
    grown_stack_params = int(stack_params * scale)
    # fp16 = 2 bytes/param
    return (grown_stack_params + other_params) * 2


def _decide_backend(teacher: nn.Module, cfg: GrowthConfig) -> str:
    """Resolve 'auto' to 'vram' or 'tdq_stream' based on budget; pass-through
    for explicit modes."""
    if cfg.storage_backend != "auto":
        return cfg.storage_backend
    budget = _detect_vram_budget_bytes(cfg)
    if budget == 0:
        # Unknown budget → conservative default: keep in vram.
        return "vram"
    need = _estimate_grown_bytes(teacher, cfg)
    return "tdq_stream" if need > budget else "vram"


@dataclass
class StreamGrowthResult:
    grew: bool
    target_layers: int
    teacher_layers: int
    estimated_bytes: int
    vram_budget_bytes: int
    tdq_path: Optional[str] = None
    backend: str = "vram"
    abort_reason: Optional[str] = None


def grow_and_stream(
    teacher: nn.Module,
    cfg: GrowthConfig,
    *,
    output_dir,
    tokenizer=None,
    save_fn: Optional[Callable] = None,
    compress_fn: Optional[Callable] = None,
) -> StreamGrowthResult:
    """Grow the teacher but route the grown weights through TDQ compression
    rather than keeping them resident in VRAM.

    Activated when `cfg.storage_backend == "tdq_stream"` or when 'auto'
    resolves to tdq_stream because the estimated grown fp16 size exceeds
    the VRAM budget.

    Flow:
      1. Compute grown fp16 size and compare to budget. (Skip check when
         caller forced 'tdq_stream'.)
      2. Build the grown skeleton on CPU via :func:`grow_model`.
      3. Save to `output_dir/hf_stage` via `save_fn` (default: save_pretrained).
      4. Compress to `output_dir/grown.tdq` via `compress_fn`
         (default: src.utils.tdq_bridge.compress_model_dir_to_tdq).
      5. Return paths — caller loads the .tdq via TDQModelLoader.

    `save_fn(model, tokenizer, hf_dir) -> Path` and
    `compress_fn(hf_dir, tdq_path, config) -> Path` are injected so tests
    can mock them without touching the external compressor.
    """
    from pathlib import Path as _Path
    output_dir = _Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parent, attr = _locate_layers(teacher)
    teacher_layers = len(getattr(parent, attr))
    target_layers = plan_target_layers(teacher_layers, cfg.growth_factor)
    estimated = _estimate_grown_bytes(teacher, cfg)
    budget = _detect_vram_budget_bytes(cfg)

    if cfg.storage_backend == "vram":
        return StreamGrowthResult(
            grew=False,
            target_layers=target_layers,
            teacher_layers=teacher_layers,
            estimated_bytes=estimated,
            vram_budget_bytes=budget,
            backend="vram",
            abort_reason="storage_backend=vram — grow_and_stream is a no-op",
        )

    backend = _decide_backend(teacher, cfg)
    if backend == "vram":
        return StreamGrowthResult(
            grew=False,
            target_layers=target_layers,
            teacher_layers=teacher_layers,
            estimated_bytes=estimated,
            vram_budget_bytes=budget,
            backend="vram",
            abort_reason="auto-resolved to vram: estimated size within budget",
        )

    student = grow_model(teacher, cfg)

    if save_fn is None:
        from src.utils.tdq_bridge import save_model_to_hf_dir as save_fn
    if compress_fn is None:
        from src.utils.tdq_bridge import compress_model_dir_to_tdq as compress_fn

    hf_dir = output_dir / "hf_stage"
    tdq_path = output_dir / "grown.tdq"
    save_fn(student, tokenizer, hf_dir)
    compress_fn(hf_dir, tdq_path, config=cfg.tdq_config)

    logger.info(
        "growth streamed: %d→%d layers, est %d bytes > budget %d, "
        "tdq=%s",
        teacher_layers, target_layers, estimated, budget, tdq_path,
    )
    return StreamGrowthResult(
        grew=True,
        target_layers=target_layers,
        teacher_layers=teacher_layers,
        estimated_bytes=estimated,
        vram_budget_bytes=budget,
        tdq_path=str(tdq_path),
        backend="tdq_stream",
    )


@dataclass
class GrowthResult:
    grew: bool
    target_layers: int
    teacher_layers: int
    teacher_heldout: Optional[float] = None
    student_heldout: Optional[float] = None
    abort_reason: Optional[str] = None


def grow_and_distill(
    teacher: nn.Module,
    sample_batches: Iterable,
    cfg: GrowthConfig,
    *,
    heldout_eval: Callable[[nn.Module], float],
    optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer] = (
        lambda m: torch.optim.AdamW(m.parameters(), lr=1e-5)
    ),
) -> tuple[nn.Module, GrowthResult]:
    """End-to-end: grow → distill → guard → return.

    Returns (model, result). On abort (worse held-out), `model is teacher`
    and `result.grew is False` — the caller should treat this as a no-op.

    `sample_batches` must be an iterable of dicts with keys
    ``input_ids`` and optional ``labels`` (tensor). Kept deliberately
    simple — the trainer already owns dataset/DataLoader concerns.
    """
    teacher_layers = len(getattr(_locate_layers(teacher)[0], _locate_layers(teacher)[1]))
    target_layers = plan_target_layers(teacher_layers, cfg.growth_factor)

    teacher_heldout = heldout_eval(teacher)
    student = grow_model(teacher, cfg)
    optimizer = optimizer_factory(student)

    # Use whichever device the teacher is on; caller is responsible for
    # ensuring the batches match.
    for _ in range(cfg.distill_epochs):
        for batch in sample_batches:
            optimizer.zero_grad(set_to_none=True)
            loss = distill_step(
                student,
                teacher,
                batch["input_ids"],
                batch.get("labels"),
                cfg,
            )
            loss.backward()
            optimizer.step()

    student_heldout = heldout_eval(student)
    delta = student_heldout - teacher_heldout
    if delta < -cfg.abort_if_worse_by:
        logger.warning(
            "growth aborted: student held-out %.4f vs teacher %.4f (delta %.4f)",
            student_heldout, teacher_heldout, delta,
        )
        return teacher, GrowthResult(
            grew=False,
            target_layers=target_layers,
            teacher_layers=teacher_layers,
            teacher_heldout=teacher_heldout,
            student_heldout=student_heldout,
            abort_reason=f"held-out regression {delta:.4f} < -{cfg.abort_if_worse_by}",
        )

    logger.info(
        "growth accepted: %d→%d layers, held-out %.4f→%.4f",
        teacher_layers, target_layers, teacher_heldout, student_heldout,
    )
    return student, GrowthResult(
        grew=True,
        target_layers=target_layers,
        teacher_layers=teacher_layers,
        teacher_heldout=teacher_heldout,
        student_heldout=student_heldout,
    )
