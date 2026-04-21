"""Fast-student distillation for the propose/solve generation phase.

Premise: the teacher (big model, currently 32B-R1) is what we want to *train*,
but its generation cost is the dominant per-cycle time sink for propose/solve.
Periodically distill the teacher's generation behaviour on RSI-shaped prompts
into a small student (1-3B) and route future propose/solve calls through the
student's vLLM until the next re-distill. Cycles get faster as the student
gets better — this is the lever for cycle-time reduction.

Design constraints:

  - Verification MUST still run against real ground truth (property quorum is
    unchanged). The student only accelerates *generation*; correctness comes
    from the builtin-based verifier, not the student.
  - Training of the teacher is untouched — only the propose/solve calls in the
    RSI tick route through the student's vLLM.
  - Re-distillation fires every F trained cycles. The student is re-initialised
    from a fresh base each time and fine-tuned on (prompt, teacher_completion)
    pairs harvested from the most recent teacher generation batch.
  - No GPU code runs at import time; vLLM is optional and lazily resolved so
    unit tests can exercise the manager without a GPU.
  - Checkpoints live under outputs/fast_student/ckpt_K/ where K is the cycle
    at which the distill fired.

Integration seam: the orchestrator constructs a `FastStudentManager` once,
calls `manager.record_teacher_generation(prompts, completions)` from inside
the propose/solve path to accumulate distillation data, and asks the manager
for a generate_fn via `manager.generate_fn_for_cycle(cycle)`. When the student
is active that fn batches prompts through the student's vLLM; otherwise it
returns None and the caller falls back to the teacher.

Peer reviewer: verifier-sim.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)


DEFAULT_FAST_STUDENT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


@dataclass
class FastStudentConfig:
    """User-facing fast-student configuration.

    Mirrored on OrchestratorConfig via `use_fast_student` / `fast_student_model_name`
    (integration-lead wires the flags). Kept as its own dataclass so the manager
    stays testable without touching SystemConfig.
    """

    enabled: bool = False
    model_name: str = DEFAULT_FAST_STUDENT_MODEL
    # Re-distill every F *trained* cycles. Cycle 0 always uses the teacher;
    # after the first F trained cycles we harvest teacher generations and
    # distill. F=5 balances staleness (student drifts from current teacher)
    # against distill cost.
    redistill_every: int = 5
    # Minimum (prompt, completion) pairs required before the first distill
    # fires. Below this we keep generating with the teacher — distilling on
    # too little data produces a student that hurts more than it helps.
    min_pairs_for_distill: int = 64
    # Upper bound on pairs used per distill. Pairs are a rolling buffer
    # (most-recent wins), so more-than-cap pairs are discarded FIFO.
    max_pairs_for_distill: int = 2048
    # LoRA rank for the student distill. Small students tolerate higher rank
    # than the 32B teacher because their absolute param count is tiny.
    student_lora_rank: int = 16
    student_lora_alpha: int = 32
    student_learning_rate: float = 2e-5
    student_num_epochs: int = 1
    student_batch_size: int = 8
    # Student vLLM sizing. Small student + 24GB of unused KV cache is fine.
    student_max_model_len: int = 4096
    student_gpu_memory_utilization: float = 0.30
    # Where checkpoints go. ckpt_{cycle}/ subdir per distill.
    checkpoint_root: Path = field(default_factory=lambda: Path("outputs/fast_student"))

    def __post_init__(self) -> None:
        if self.redistill_every < 1:
            raise ValueError(f"redistill_every must be >= 1, got {self.redistill_every}")
        if self.min_pairs_for_distill < 1:
            raise ValueError(
                f"min_pairs_for_distill must be >= 1, got {self.min_pairs_for_distill}"
            )
        if self.max_pairs_for_distill < self.min_pairs_for_distill:
            raise ValueError(
                f"max_pairs_for_distill ({self.max_pairs_for_distill}) < "
                f"min_pairs_for_distill ({self.min_pairs_for_distill})"
            )
        if self.student_lora_rank < 1:
            raise ValueError(f"student_lora_rank must be >= 1, got {self.student_lora_rank}")
        if self.student_learning_rate <= 0:
            raise ValueError(
                f"student_learning_rate must be > 0, got {self.student_learning_rate}"
            )
        if self.student_num_epochs < 1:
            raise ValueError(f"student_num_epochs must be >= 1, got {self.student_num_epochs}")
        if self.student_batch_size < 1:
            raise ValueError(
                f"student_batch_size must be >= 1, got {self.student_batch_size}"
            )
        if not (0.0 < self.student_gpu_memory_utilization <= 1.0):
            raise ValueError(
                "student_gpu_memory_utilization must be in (0, 1], "
                f"got {self.student_gpu_memory_utilization}"
            )
        if isinstance(self.checkpoint_root, str):
            self.checkpoint_root = Path(self.checkpoint_root)


@dataclass
class DistillPair:
    prompt: str
    completion: str
    cycle: int


@dataclass
class StudentCheckpoint:
    """Handle for a distilled student checkpoint. The actual weights live on
    disk under `path`; the vLLM engine is loaded lazily on first use so
    import of this module stays GPU-free."""

    cycle: int
    path: Path
    model_name: str
    num_pairs: int
    distill_seconds: float
    _engine: object | None = None  # vLLM engine if live

    def to_metadata(self) -> dict:
        return {
            "cycle": self.cycle,
            "path": str(self.path),
            "model_name": self.model_name,
            "num_pairs": self.num_pairs,
            "distill_seconds": self.distill_seconds,
        }


class FastStudentManager:
    """Owns the distillation buffer, the current student checkpoint, and the
    decision of whether to route generation through the student.

    Usage pattern inside the orchestrator loop:

        manager = FastStudentManager(cfg)
        ...
        # In propose/solve, before choosing who generates:
        fn = manager.generate_fn_for_cycle(cycle)
        if fn is None:
            completions = teacher_generate(prompts)
            manager.record_teacher_generation(prompts, completions, cycle=cycle)
        else:
            completions = fn(prompts)

        # After training lands in a cycle:
        manager.on_trained_cycle(cycle)
    """

    def __init__(
        self,
        cfg: FastStudentConfig,
        *,
        distill_fn: Optional[Callable[[Sequence[DistillPair], Path, "FastStudentConfig"], StudentCheckpoint]] = None,
        student_loader: Optional[Callable[[StudentCheckpoint], Callable[[Sequence[str]], list[str]]]] = None,
    ) -> None:
        self.cfg = cfg
        self._pairs: list[DistillPair] = []
        self._trained_cycles_since_distill = 0
        self._current: Optional[StudentCheckpoint] = None
        # Injectable hooks so tests never hit vLLM / GPU. Production paths
        # resolve to the vLLM-backed implementations below.
        self._distill_fn = distill_fn or _default_distill_fn
        self._student_loader = student_loader or _default_student_loader

    # --- data collection ------------------------------------------------

    def record_teacher_generation(
        self,
        prompts: Iterable[str],
        completions: Iterable[str],
        *,
        cycle: int,
    ) -> None:
        """Append (prompt, completion) pairs harvested from a teacher generate
        call. Silently skips empty completions — distilling "model produced no
        output" just teaches the student to also produce nothing."""
        if not self.cfg.enabled:
            return
        for p, c in zip(prompts, completions):
            if not c or not c.strip():
                continue
            self._pairs.append(DistillPair(prompt=p, completion=c, cycle=cycle))
        # Rolling FIFO buffer — keep only the most-recent max_pairs_for_distill.
        excess = len(self._pairs) - self.cfg.max_pairs_for_distill
        if excess > 0:
            del self._pairs[:excess]

    def buffer_size(self) -> int:
        return len(self._pairs)

    # --- cycle hooks ----------------------------------------------------

    def on_trained_cycle(self, cycle: int) -> None:
        """Called by the orchestrator after training lands in a cycle.
        Increments the distill counter and, if the threshold is reached
        *and* we have enough pairs, fires a re-distill."""
        if not self.cfg.enabled:
            return
        self._trained_cycles_since_distill += 1
        if self._trained_cycles_since_distill < self.cfg.redistill_every:
            return
        if len(self._pairs) < self.cfg.min_pairs_for_distill:
            logger.info(
                "fast_student: skip distill at cycle %d — only %d pairs (need %d)",
                cycle,
                len(self._pairs),
                self.cfg.min_pairs_for_distill,
            )
            return
        self._distill(cycle)
        self._trained_cycles_since_distill = 0

    def _distill(self, cycle: int) -> None:
        ckpt_dir = self.cfg.checkpoint_root / f"ckpt_{cycle}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        try:
            ckpt = self._distill_fn(list(self._pairs), ckpt_dir, self.cfg)
        except Exception as exc:  # pragma: no cover — logged, surfaced via status
            logger.warning("fast_student: distill failed at cycle %d: %s", cycle, exc)
            return
        ckpt.cycle = cycle
        ckpt.distill_seconds = time.time() - started
        # Release previous engine (if any) before taking the new checkpoint live.
        self._release_current()
        self._current = ckpt
        (ckpt_dir / "metadata.json").write_text(json.dumps(ckpt.to_metadata(), indent=2))
        logger.info(
            "fast_student: distilled cycle=%d pairs=%d elapsed=%.1fs path=%s",
            cycle,
            ckpt.num_pairs,
            ckpt.distill_seconds,
            ckpt.path,
        )

    def _release_current(self) -> None:
        if self._current is None:
            return
        eng = self._current._engine
        if eng is not None and hasattr(eng, "shutdown"):
            try:
                eng.shutdown()
            except Exception:  # pragma: no cover
                logger.debug("fast_student: engine shutdown raised", exc_info=True)
        self._current._engine = None

    # --- generation routing --------------------------------------------

    def generate_fn_for_cycle(
        self, cycle: int
    ) -> Optional[Callable[[Sequence[str]], list[str]]]:
        """Return a batched generate_fn backed by the current student, or
        None if the teacher should be used. The fn takes a list of prompts
        and returns a list of completions the same length."""
        if not self.cfg.enabled or self._current is None:
            return None
        return self._student_loader(self._current)

    def current_checkpoint(self) -> Optional[StudentCheckpoint]:
        return self._current

    def status(self) -> dict:
        return {
            "enabled": self.cfg.enabled,
            "buffer_size": len(self._pairs),
            "trained_cycles_since_distill": self._trained_cycles_since_distill,
            "current_checkpoint": (
                self._current.to_metadata() if self._current is not None else None
            ),
        }


# ---------------------------------------------------------------------------
# Production distill / loader hooks. Kept at module scope (not methods) so
# tests can cleanly swap them. They import heavy deps lazily.
# ---------------------------------------------------------------------------


def _default_distill_fn(
    pairs: Sequence[DistillPair], out_dir: Path, cfg: "FastStudentConfig"
) -> StudentCheckpoint:
    """Default distill: LoRA SFT of the small student on (prompt, completion)
    pairs. Uses HuggingFace + PEFT. Writes merged weights to `out_dir`.

    Kept minimal — the student's job is to mimic the teacher's generation
    distribution on RSI prompts, not to be a general-purpose fine-tune.
    """
    # Heavy imports deferred so test runs without a transformers install work.
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    base_model_name = cfg.model_name
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    try:
        from peft import LoraConfig, get_peft_model  # type: ignore

        peft_cfg = LoraConfig(
            r=cfg.student_lora_rank,
            lora_alpha=cfg.student_lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)
    except ImportError:
        logger.warning("fast_student: peft not installed — falling back to full fine-tune")

    model.train()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.student_learning_rate
    )

    for _ in range(cfg.student_num_epochs):
        for pair in pairs:
            text = pair.prompt + pair.completion
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
            input_ids = enc["input_ids"].to(device)
            labels = input_ids.clone()
            # Mask prompt tokens so the student only learns completion continuation.
            prompt_len = len(
                tokenizer(pair.prompt, truncation=True, max_length=2048)["input_ids"]
            )
            labels[:, :prompt_len] = -100
            out = model(input_ids=input_ids, labels=labels)
            out.loss.backward()
            optim.step()
            optim.zero_grad()

    # Save merged weights so vLLM can load a plain HF checkpoint.
    try:
        merged = model.merge_and_unload()  # peft path
    except AttributeError:
        merged = model
    merged.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    return StudentCheckpoint(
        cycle=-1,  # real cycle is stamped by FastStudentManager._distill
        path=out_dir,
        model_name=base_model_name,
        num_pairs=len(pairs),
        distill_seconds=0.0,
    )


def _default_student_loader(
    ckpt: StudentCheckpoint,
) -> Callable[[Sequence[str]], list[str]]:
    """Return a batched generate_fn backed by a vLLM engine over the
    student's saved weights. The engine is cached on the checkpoint so
    repeated calls in the same cycle don't re-load."""
    if ckpt._engine is None:
        from vllm import LLM  # type: ignore

        ckpt._engine = LLM(
            model=str(ckpt.path),
            dtype="bfloat16",
            gpu_memory_utilization=0.30,
            trust_remote_code=True,
        )

    def _gen(prompts: Sequence[str]) -> list[str]:
        from vllm import SamplingParams  # type: ignore

        sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1024)
        outs = ckpt._engine.generate(list(prompts), sp)
        return [o.outputs[0].text for o in outs]

    return _gen
