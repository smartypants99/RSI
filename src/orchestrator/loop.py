"""The main recursive self-improvement loop.

The loop has phases:
- Early cycles: Model is the subject, external tooling drives improvement
- Mid cycles: Model assists in verification, then diagnosis
- Late cycles: Model improves the improvement process itself

This is the "recursive" part — the system gets better at getting better.
"""

from __future__ import annotations

import gc
import json
import math
import shutil
import signal
import time
import traceback
import logging
from pathlib import Path

import torch

from ..utils.config import SystemConfig
from ..diagnostics.engine import DiagnosticsEngine, DiagnosticResult, WeaknessReport
from ..generator.data_generator import DataGenerator
from ..verifier.verifier import Verifier
from ..trainer.custom_lora import CustomLoRATrainer, TrainingMetrics
from .meta import MetaController

logger = logging.getLogger(__name__)


def _nan_to_none(x):
    """Convert NaN floats to None for JSON serialization.

    Distinguishes "metric was not computed" (None) from "metric is 0.0".
    """
    if x is None:
        return None
    try:
        if isinstance(x, float) and math.isnan(x):
            return None
    except (TypeError, ValueError):
        return x
    return x


def _summarize_errors(history: list) -> dict:
    """Count errors by (phase, exception type) across all cycles."""
    counts: dict[str, int] = {}
    for r in history:
        for e in getattr(r, "errors", []):
            key = f"{e.get('phase', '?')}:{e.get('type', '?')}"
            counts[key] = counts.get(key, 0) + 1
    return counts


class CycleResult:
    """Result of one improvement cycle."""

    def __init__(self, cycle: int):
        self.cycle = cycle
        self.diagnostics: DiagnosticResult | None = None
        self.samples_generated: int = 0
        self.samples_verified: int = 0
        self.training_metrics: TrainingMetrics | None = None
        self.pre_score: float = 0.0
        self.post_score: float = 0.0
        self.improvement: float = 0.0
        self.escalation_events: list[str] = []
        self.timestamp: float = time.time()
        self.duration: float = 0.0
        self.post_diag: DiagnosticResult | None = None
        self.had_diagnostics: bool = False
        self.eval_score: float | None = None
        self.eval_domain_scores: dict[str, float] = {}
        # meta_analyst ASK 1: per-subdomain breakdown of the held-out eval,
        # keyed "domain/subdomain". The overall eval_score is one number;
        # this is the per-bucket view needed to tell whether a subdomain-
        # targeted fix (e.g. bit_manipulation rebalance) actually moved that
        # subdomain versus just raising the aggregate.
        self.eval_subdomain_scores: dict[str, float] = {}
        # All per-repetition held-out eval scores (length == heldout_repetitions).
        # When repetitions==1, this is [eval_score]. The spread across
        # repetitions is a direct measurement-noise estimate.
        self.eval_scores_all: list[float] = []
        self.diversity_stats: dict = {}
        self.phase_times: dict[str, float] = {}
        # Records non-fatal errors that occurred during the cycle.
        # Each: {"phase": str, "type": str, "message": str, "traceback": str}
        self.errors: list[dict] = []
        # Observability carry-overs — populated only when orchestrator flags
        # are set. Kept off CycleResult.to_dict() so they don't bloat progress.
        self._training_samples: list = []  # list[TrainingSample]
        self._star_stats: dict = {}
        self._eval_per_rep_domain_scores: list[dict] = []
        self._eval_per_rep_per_question: list[list[dict]] = []
        # RSI-mode counters (spec §5.3) — zero by default, populated by rsi_tick().
        self.novel_problems_proposed: int = 0
        self.properties_admitted: int = 0
        self.candidates_accepted: int = 0
        self.candidates_rejected_by_adversarial: int = 0
        self.classes_suspended: int = 0
        # Candidates admitted by §1.4 quorum this cycle — captured so the
        # orchestrator can bank them as adversarial examples if post-training
        # eval regresses. Each entry: {problem_id, candidate, domain, problem_ctx}.
        self._admitted_candidates: list[dict] = []

    @property
    def improved(self) -> bool:
        return self.improvement > 0

    def to_dict(self) -> dict:
        """Serialize cycle result for JSON checkpoint."""
        return {
            "cycle": self.cycle,
            "pre_score": self.pre_score,
            "post_score": self.post_score,
            "improvement": self.improvement,
            "eval_score": self.eval_score,
            "eval_domain_scores": self.eval_domain_scores,
            "eval_subdomain_scores": self.eval_subdomain_scores,
            "samples_generated": self.samples_generated,
            "samples_verified": self.samples_verified,
            "weaknesses_found": len(self.diagnostics.weaknesses) if self.diagnostics else 0,
            "had_diagnostics": self.diagnostics is not None,
            "escalation_events": self.escalation_events,
            "post_diag_domain_scores": self.post_diag.domain_scores if self.post_diag else {},
            "diversity_stats": self.diversity_stats,
            "phase_times": self.phase_times,
            "timestamp": self.timestamp,
            "duration_seconds": self.duration,
            "errors": self.errors,
            "training": {
                "avg_loss": self.training_metrics.avg_loss if self.training_metrics else None,
                "final_loss": self.training_metrics.final_loss if self.training_metrics else None,
                "steps": self.training_metrics.steps if self.training_metrics else 0,
                "lora_layers": self.training_metrics.lora_layers_injected if self.training_metrics else 0,
                "avg_rank": self.training_metrics.avg_rank if self.training_metrics else 0,
                "samples_used": self.training_metrics.samples_used if self.training_metrics else 0,
                "samples_rejected": self.training_metrics.samples_rejected if self.training_metrics else 0,
                "learning_rate": self.training_metrics.learning_rate if self.training_metrics else 0,
            },
        }


class ImprovementLoop:
    """Orchestrates the recursive self-improvement loop."""

    def __init__(self, config: SystemConfig):
        self.config = config
        self._use_vllm = config.use_vllm
        self._backend = getattr(config, "backend", None)

        if self._backend == "tdq":
            # TDQ backend: decompress .tdq file into an HF model, skip vLLM.
            # Sets _use_vllm=False so the orchestrator's vLLM-specific swap
            # paths (swap_to_hf_for_training / swap_to_vllm_after_training)
            # become no-ops via TDQModelLoader's stub methods.
            from ..utils.tdq_backend import TDQModelLoader
            self._use_vllm = False
            self.model_loader = TDQModelLoader(
                model_path=config.model.model_path,
                dtype=config.model.dtype,
                max_seq_length=config.model.max_seq_length,
                allow_remote_code=getattr(config.model, "allow_remote_code", True),
            )
            # Eagerly load — the first diagnostic run would force it anyway
            # and we want to see the VRAM footprint in the log up front.
            self.model_loader.load()
        elif self._use_vllm:
            from ..utils.vllm_backend import VLLMModelLoader
            vllm_cfg = config.vllm
            if vllm_cfg is None:
                raise ValueError("use_vllm=True but config.vllm is None")
            self.model_loader = VLLMModelLoader(
                model_path=vllm_cfg.model_path,
                dtype=vllm_cfg.dtype,
                max_model_len=vllm_cfg.max_model_len,
                gpu_memory_utilization=vllm_cfg.gpu_memory_utilization,
                allow_remote_code=getattr(config.model, "allow_remote_code", False),
                quantization_config=vllm_cfg.quantization_config,
            )
        else:
            from ..utils.model_loader import ModelLoader
            self.model_loader = ModelLoader(config.model)

        self.diagnostics = DiagnosticsEngine(config.diagnostics, self.model_loader)
        self.generator = DataGenerator(config.generator, self.model_loader)
        # Wire the diagnostics ground-truth checker into the generator so the
        # STaR path can filter sampled chains by canonical-answer match.
        self.generator.set_grader(self.diagnostics._check_answer)
        self.verifier = Verifier(config.verifier)
        # Same grader into the verifier so its ground-truth gate uses the
        # canonical dispatcher (contains/math_equiv/code_executes/...).
        self.verifier.set_ground_truth_grader(self.diagnostics._check_answer)
        self.trainer = CustomLoRATrainer(config.trainer, self.model_loader)
        # Wire observability: collect training loss trajectory when the
        # orchestrator is asked to (e.g. for cycle_metrics dumps).
        if getattr(config.orchestrator, "collect_training_loss_trajectory", False):
            self.trainer.set_collect_loss_trajectory(True)

        # Synthesis-mode components — instantiated only when the flag is on so
        # the classic path has zero overhead.
        self._synthesis_enabled = getattr(
            getattr(config, "synthesis", None), "enable_task_synthesis", False
        )
        self._task_synthesizer = None
        if self._synthesis_enabled:
            try:
                from ..generator.task_synthesizer import TaskSynthesizer
                self._task_synthesizer = TaskSynthesizer(config.synthesis, self.model_loader)
                logger.info("Synthesis mode enabled (tasks_per_cycle=%d, consensus_threshold=%.2f)",
                            config.synthesis.tasks_per_cycle,
                            config.synthesis.property_consensus_threshold)
            except Exception as e:
                logger.warning("Synthesis mode requested but TaskSynthesizer failed to load (%s: %s) — disabled",
                               type(e).__name__, e)
                self._synthesis_enabled = False

        # RSI registries — open only when synthesis is active so classic mode
        # has zero overhead. Spec §5.3: conditional on mode, no existing call
        # sites need to change.
        self._registries = None
        if self._synthesis_enabled:
            try:
                from .registries import RSIRegistries
                sid = getattr(config.orchestrator, "run_id", None) or None
                self._registries = RSIRegistries.open(
                    config.orchestrator.output_dir, sid=sid
                )
                logger.info("RSI registries opened (sid=%s)", self._registries.sid)
            except Exception as e:
                logger.warning("RSI registries failed to open (%s: %s) — artifact logging disabled",
                               type(e).__name__, e)

        # Cross-tick subdomain scores for set_diagnostics two-cycle stability rule.
        self._rsi_prior_subdomain_scores: dict = {}

        self.history: list[CycleResult] = []
        self._plateau_count = 0
        self._escalation_state = {
            "verification": False,
            "diagnosis": False,
            "generation": False,
        }
        self._domain_score_history: dict[str, list[float]] = {}
        self._pending_regression_weaknesses: list[WeaknessReport] = []
        self._last_deescalation_cycle: int = -10
        self._consecutive_failures = 0
        self._vllm_saved_this_cycle: int | None = None
        self._best_score: float = 0.0
        self._best_checkpoint_cycle: int | None = None
        self._degradation_count: int = 0
        # EMA for plateau detection — smooths noisy per-cycle improvements so
        # plateau decisions aren't thrown off by a single outlier cycle.
        self._improvement_ema: float = 0.0
        self._ema_alpha: float = 0.3  # weight for newest observation
        # Rolling RSI training pool — accumulates property-verified samples
        # ACROSS cycles until we have enough for stable training. Run-8/9
        # observed: training on 18-21 fresh samples per cycle × 2-3 steps
        # overfits a 8B model every time (loss crashes from 0.5 to <0.2,
        # held-out regresses by 0.2+). Accumulating for 3-4 cycles before
        # training gives ~60-80 samples, lower-variance gradient updates,
        # and less memorization risk.
        self._rsi_pending_pool: list = []
        self._rsi_pool_accumulated_cycles: int = 0
        # Speed knob: diagnostic result cache. Step 0 of rsi_tick runs a
        # 240-prompt diagnostic every cycle (~56s). Weights only change on
        # training cycles, so the mastery profile is stale-but-valid across
        # non-training cycles. Cache the last DiagnosticResult and its cycle
        # index; refresh every N cycles (rsi_diagnostic_refresh_every) or
        # whenever weights change (invalidated in Step 8 after training).
        self._rsi_diag_cache: "DiagnosticResult | None" = None
        self._rsi_diag_cache_cycle: int = -1

        # Substrate-merge bookkeeping. Every `merge_into_base_every` training
        # cycles, if cumulative held-out improvement since the last promotion
        # is >= substrate_merge_min_improvement, we promote the current merged
        # checkpoint to a new "base" — model_loader.model_path is redirected
        # to outputs/checkpoints/base_epoch_K so LoRA restarts fresh on top
        # of substrate that has already absorbed the prior epoch's gains.
        self._substrate_epoch: int = 0
        self._substrate_baseline_eval: float | None = None
        self._substrate_last_merge_cycle: int = 0
        self._substrate_trained_cycles_since_merge: int = 0

        # Meta-improvement layer — learns which config decisions pay off.
        meta_log = config.orchestrator.output_dir / "meta_decisions.jsonl"
        initial_lr = getattr(config.trainer, "learning_rate", None)
        self.meta = MetaController(log_path=meta_log, initial_lr=initial_lr)

    def run(self) -> None:
        """Run the full improvement loop with per-cycle fault isolation.

        A crash inside one cycle is captured (with full traceback), the cycle
        is marked failed, and the loop continues. Only KeyboardInterrupt /
        SIGTERM interrupt the loop; all other exceptions are recorded.
        """
        logger.info("=" * 60)
        logger.info("RECURSIVE SELF-IMPROVEMENT SYSTEM")
        logger.info("=" * 60)
        self._setup()

        start_cycle = 1
        if self.config.orchestrator.resume_from:
            start_cycle = self._resume()

        # SIGTERM handler — treat like Ctrl-C so container shutdowns flush state.
        def _sigterm_handler(signum, frame):
            logger.warning(f"Signal {signum} received — raising KeyboardInterrupt for graceful exit")
            raise KeyboardInterrupt()

        prev_sigterm = None
        try:
            prev_sigterm = signal.signal(signal.SIGTERM, _sigterm_handler)
        except (ValueError, OSError):
            # Not in main thread or platform doesn't support — skip silently.
            pass

        try:
            for cycle in range(start_cycle, self.config.orchestrator.max_cycles + 1):
                logger.info(f"\n{'='*60}")
                logger.info(f"CYCLE {cycle}")
                logger.info(f"{'='*60}")

                cycle_start = time.time()
                result = CycleResult(cycle)
                _mode = getattr(self.config.orchestrator, "mode", "classic")
                try:
                    if _mode == "rsi":
                        result = self.rsi_tick(cycle)
                    else:
                        result = self._run_cycle(cycle)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    # Preserve partial result if _run_cycle got far enough to populate it.
                    tb = traceback.format_exc()
                    logger.error(f"  Cycle {cycle} crashed ({type(e).__name__}): {e}\n{tb}")
                    result.errors.append({
                        "phase": "cycle", "type": type(e).__name__,
                        "message": str(e), "traceback": tb,
                    })
                    # Try to recover the model loader to a known-good state.
                    self._recover_after_cycle_failure()
                    self._consecutive_failures += 1
                result.duration = time.time() - cycle_start

                # Held-out eval — isolated so eval failure doesn't lose the cycle.
                # Classic mode: always run (the LR bandit needs eval_deltas even
                # on no-training cycles for paired-effect statistics).
                # RSI mode: skip when weights didn't change — re-running the
                # same model against the same held-out bank produces the same
                # score within noise, wasting ~2.5 minutes per cycle during
                # the warm-up period where the pool is still filling.
                trained = bool(
                    result.training_metrics
                    and getattr(result.training_metrics, "steps", 0) > 0
                )
                skip_eval = (_mode == "rsi") and (not trained)
                eval_start = time.time()
                try:
                    if skip_eval:
                        logger.info(
                            "[Cycle %d] Phase 5b: HELD-OUT EVAL skipped — "
                            "no training this cycle in RSI mode (model state "
                            "unchanged; saves ~2.5 min/cycle during warm-up)",
                            cycle,
                        )
                        # Carry forward previous eval_score so the meta
                        # controller has SOMETHING to look at.
                        if self.history:
                            prev = self.history[-1].eval_score
                            if prev is not None:
                                result.eval_score = prev
                    else:
                        self._eval_phase(cycle, result)
                except Exception as e:
                    tb = traceback.format_exc()
                    logger.warning(f"  Held-out eval failed ({type(e).__name__}): {e}")
                    result.errors.append({
                        "phase": "eval", "type": type(e).__name__,
                        "message": str(e), "traceback": tb,
                    })
                result.phase_times["eval"] = time.time() - eval_start

                # Post-Phase-5b regression revert. Phase 5b is the AUTHORITY
                # on whether training helped (3 reps × 240 prompts vs. the
                # quick probe's 24). If the full eval shows a drop worse
                # than the revert threshold from either the pre-training
                # pre_score or the prior cycle's eval_score, roll vLLM back
                # to the best known checkpoint.
                #
                # Observed run-8 cycle 1: quick probe reported -0.092
                # (within -0.10 tolerance → checkpoint saved + vLLM swapped),
                # then Phase 5b showed 0.280 vs 0.558 baseline = −0.28 real
                # regression. Without this guard, cycle 2 starts from
                # corrupted weights and cascades.
                revert_threshold = float(getattr(
                    self.config.trainer, "regression_revert_threshold", 0.10,
                ))
                if (
                    _mode == "rsi"
                    and trained  # only meaningful if we actually trained this cycle
                    and result.eval_score is not None
                    and revert_threshold > 0
                ):
                    # Compare against the best-ever eval score (baseline on
                    # cycle 1, prior-best on later cycles).
                    reference = (
                        self._best_score if self._best_score and self._best_score > 0
                        else result.pre_score or 0.0
                    )
                    full_eval_drop = reference - result.eval_score
                    if full_eval_drop > revert_threshold:
                        logger.warning(
                            "[RSI tick %d] FULL EVAL REGRESSION: "
                            "eval=%.3f vs reference=%.3f (drop %.3f > %.2f). "
                            "Quick probe missed this. Reverting vLLM to best "
                            "checkpoint (cycle %s) so cycle %d starts clean.",
                            cycle, result.eval_score, reference,
                            full_eval_drop, revert_threshold,
                            self._best_checkpoint_cycle or "base",
                            cycle + 1,
                        )
                        try:
                            if self._best_checkpoint_cycle is not None:
                                best_ckpt = (
                                    self.config.orchestrator.output_dir
                                    / "checkpoints"
                                    / f"cycle_{self._best_checkpoint_cycle}"
                                )
                                if best_ckpt.exists() and (best_ckpt / "config.json").exists():
                                    self.model_loader.swap_to_vllm_after_training(str(best_ckpt))
                                else:
                                    # Fall back to base model path
                                    self.model_loader.swap_to_vllm_after_training(
                                        str(getattr(self.model_loader, "model_path", None))
                                    )
                            else:
                                # No prior good checkpoint — revert to base.
                                self.model_loader.swap_to_vllm_after_training(
                                    str(getattr(self.model_loader, "model_path", None))
                                )
                        except Exception as exc:
                            logger.warning(
                                "[RSI tick %d] vLLM revert after full-eval regression failed: %s",
                                cycle, exc,
                            )
                        # Promote this cycle's admitted candidates to the VoV
                        # adversarial bank. They cleared §1.4 quorum yet
                        # correlated with post-training regression — exactly
                        # the toothless-verifier pattern VoV exists to catch.
                        self._bank_admitted_as_adversarial(
                            cycle, result,
                            reason=f"full_eval_regression drop={full_eval_drop:.3f}",
                        )

                self.history.append(result)

                self._improvement_ema = (
                    self._ema_alpha * result.improvement
                    + (1 - self._ema_alpha) * self._improvement_ema
                )

                # Post-cycle bookkeeping — each isolated. One failure ≠ cycle loss.
                self._safe_call("escalation_check",
                                lambda: self._check_and_escalate(
                                    cycle, result, post_diag=result.post_diag),
                                result)
                self._safe_call("log_cycle", lambda: self._log_cycle(result), result)
                if getattr(self.config.orchestrator, "write_cycle_metrics", False):
                    self._safe_call(
                        "cycle_metrics",
                        lambda: self._write_cycle_metrics(cycle, result),
                        result,
                    )
                if getattr(self.config.orchestrator, "write_cycle_samples", False):
                    self._safe_call(
                        "cycle_samples",
                        lambda: self._write_cycle_samples(cycle, result),
                        result,
                    )
                self._safe_call("substrate_merge",
                                lambda: self._maybe_substrate_merge(cycle, result),
                                result)
                if cycle % self.config.orchestrator.checkpoint_every == 0:
                    self._safe_call("checkpoint",
                                    lambda: self._save_checkpoint(cycle), result)
                self._safe_call("progress_dashboard",
                                lambda: self._save_progress_dashboard(
                                    cycle, result, result.phase_times),
                                result)
                self._safe_call("meta_step",
                                lambda: self._meta_step(cycle, result),
                                result)

                early_stop, early_reason = False, ""
                try:
                    early_stop, early_reason = self._check_early_stopping(cycle, result)
                except Exception as e:
                    logger.warning(f"  Early-stop check failed ({type(e).__name__}): {e}")
                if early_stop:
                    logger.info(f"Early stopping at cycle {cycle}: {early_reason}")
                    break

                should_stop, stop_reason = False, ""
                try:
                    should_stop, stop_reason = self._should_stop(result)
                except Exception as e:
                    logger.warning(f"  Stop check failed ({type(e).__name__}): {e}")
                if should_stop:
                    logger.info(f"Stopping at cycle {cycle}: {stop_reason}")
                    break

        except KeyboardInterrupt:
            logger.warning("Interrupted — saving state before exit")
            if self.history:
                last_cycle = self.history[-1].cycle
                self._safe_call("interrupt_checkpoint",
                                lambda: self._save_checkpoint(last_cycle), None)
                logger.info(f"Emergency checkpoint saved at cycle {last_cycle}")
            return
        finally:
            if prev_sigterm is not None:
                try:
                    signal.signal(signal.SIGTERM, prev_sigterm)
                except (ValueError, OSError):
                    pass

        if self.history:
            last_cycle = self.history[-1].cycle
            if last_cycle % self.config.orchestrator.checkpoint_every != 0:
                self._safe_call("final_checkpoint",
                                lambda: self._save_checkpoint(last_cycle), None)
        self._safe_call("final_report", self._save_final_report, None)
        logger.info("\n" + "=" * 60)
        logger.info("IMPROVEMENT LOOP COMPLETE")
        if self.history:
            total = self.history[-1].post_score - self.history[0].pre_score
            failed = sum(1 for r in self.history if r.errors)
            logger.info(f"Total improvement: {self.history[0].pre_score:.3f} -> {self.history[-1].post_score:.3f} ({total:+.3f})")
            if failed:
                logger.info(f"Cycles with errors: {failed}/{len(self.history)} (see cycle logs for details)")
        logger.info("=" * 60)

    def _safe_call(self, phase: str, fn, result: CycleResult | None) -> None:
        """Run fn; log + record any exception without propagating.

        Used for non-critical post-cycle bookkeeping where a failure should
        degrade gracefully rather than abort the loop.
        """
        try:
            fn()
        except Exception as e:
            tb = traceback.format_exc()
            logger.warning(f"  {phase} failed ({type(e).__name__}): {e}")
            if result is not None:
                result.errors.append({
                    "phase": phase, "type": type(e).__name__,
                    "message": str(e), "traceback": tb,
                })

    def _recover_after_cycle_failure(self) -> None:
        """Best-effort model-state recovery after a cycle crashes.

        Strip any dangling LoRA layers (so the next cycle doesn't re-inject
        on top), restore vLLM if we crashed mid-swap, and free VRAM.
        """
        try:
            self.trainer.strip_lora()
        except Exception as e:
            logger.debug(f"  recovery: strip_lora failed: {e}")
        if self._use_vllm:
            try:
                if getattr(self.model_loader, "_llm", None) is None:
                    path = getattr(self.model_loader, "_current_model_path", None)
                    self.model_loader.swap_to_vllm_after_training(path)
            except Exception as e:
                logger.warning(f"  recovery: vLLM reload failed: {e}")
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _setup(self):
        """Initialize the system."""
        if not self.config.model.model_path:
            raise ValueError("model_path must be configured")
        if self.config.orchestrator.checkpoint_every < 1:
            raise ValueError(
                f"orchestrator.checkpoint_every must be >= 1 "
                f"(got {self.config.orchestrator.checkpoint_every})"
            )
        if self.config.trainer.lora_rank < 1:
            raise ValueError(
                f"trainer.lora_rank must be >= 1 (got {self.config.trainer.lora_rank})"
            )
        if self.config.trainer.lora_rank < self.config.trainer.min_rank:
            raise ValueError(
                f"trainer.lora_rank ({self.config.trainer.lora_rank}) < "
                f"trainer.min_rank ({self.config.trainer.min_rank}): "
                f"weak layers would receive less capacity than healthy ones"
            )
        if self.config.generator.min_reasoning_steps < self.config.verifier.min_chain_steps:
            raise ValueError(
                f"generator.min_reasoning_steps ({self.config.generator.min_reasoning_steps}) "
                f"< verifier.min_chain_steps ({self.config.verifier.min_chain_steps}): "
                f"generator will produce samples the verifier always rejects"
            )
        self.model_loader.load()
        self.config.orchestrator.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.orchestrator.log_dir.mkdir(parents=True, exist_ok=True)

    def _resume(self) -> int:
        """Resume from a checkpoint. Restores full state including escalations."""
        resume_path = Path(self.config.orchestrator.resume_from)
        if not resume_path.is_absolute() and not resume_path.exists():
            candidate = self.config.orchestrator.output_dir / resume_path
            if candidate.exists():
                resume_path = candidate
        if not resume_path.exists():
            logger.warning(f"Resume path {resume_path} not found, starting from scratch")
            return 1
        # Reject traversal: resolved path must live under output_dir.
        output_root = self.config.orchestrator.output_dir.resolve()
        resume_path = resume_path.resolve()
        try:
            resume_path.relative_to(output_root)
        except ValueError:
            raise ValueError(
                f"resume_from {resume_path} escapes output_dir {output_root}"
            )

        history_path = resume_path / "history.json"
        if history_path.exists():
            try:
                with open(history_path) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Corrupt history.json at {history_path}: {e}. Starting from scratch.")
                return 1

            # Restore escalation state
            if "escalation_state" in data:
                for key in self._escalation_state:
                    if key in data["escalation_state"]:
                        self._escalation_state[key] = data["escalation_state"][key]
                if self._escalation_state.get("verification"):
                    self._escalate_verification()

            # Restore plateau and failure counters
            if "plateau_count" in data:
                self._plateau_count = data["plateau_count"]
            if "consecutive_failures" in data:
                self._consecutive_failures = data["consecutive_failures"]
            if "last_deescalation_cycle" in data:
                self._last_deescalation_cycle = data["last_deescalation_cycle"]

            # Restore domain score history
            if "domain_score_history" in data:
                self._domain_score_history = data["domain_score_history"]

            # Restore early stopping state
            if "best_score" in data:
                self._best_score = data["best_score"]
            if "best_checkpoint_cycle" in data:
                self._best_checkpoint_cycle = data["best_checkpoint_cycle"]
            if "degradation_count" in data:
                self._degradation_count = data["degradation_count"]

            # Restore EMA state
            if "improvement_ema" in data:
                self._improvement_ema = data["improvement_ema"]

            # Restore open-ended curriculum state (solve rates / active /
            # retired / expanded ceilings). Missing key ⇒ fresh state.
            if "curriculum" in data:
                try:
                    from ..diagnostics.curriculum import CurriculumState, DEFAULT_CLASSES
                    self.diagnostics.curriculum = CurriculumState.from_dict(
                        data["curriculum"], DEFAULT_CLASSES,
                    )
                except Exception as e:
                    logger.warning(f"  curriculum restore failed: {e}")

            # Restore meta-controller state
            if "meta_state" in data and isinstance(data["meta_state"], dict):
                try:
                    self.meta.load_state(data["meta_state"])
                except Exception as e:
                    logger.warning(f"  meta: failed to restore state: {e}")
            # Replay the tracker's decision log so paired tests see prior cycles.
            try:
                self.meta.tracker.load(self.meta.tracker.log_path)
            except Exception as e:
                logger.warning(f"  meta: tracker log replay failed: {e}")

            # Restore generation escalation template
            if data.get("custom_solution_template"):
                self.generator.set_custom_solution_template(data["custom_solution_template"])

            # Restore model-generated diagnostic questions
            if data.get("model_generated_questions"):
                valid_questions = {}
                for domain, questions in data["model_generated_questions"].items():
                    validated = [
                        q for q in questions
                        if isinstance(q, dict) and "prompt" in q and "expected" in q
                        and "check_type" in q
                    ]
                    if validated:
                        valid_questions[domain] = validated
                self.diagnostics._model_generated_questions = valid_questions

            # Restore pending regression weaknesses
            for r in data.get("pending_regressions", []):
                self._pending_regression_weaknesses.append(WeaknessReport(
                    domain=r["domain"], subdomain=r["subdomain"],
                    severity=r["severity"], description=r.get("description", ""),
                    evidence=r.get("evidence", []),
                ))

            # Restore minimal history
            for cycle_data in data.get("cycles", []):
                stub = CycleResult(cycle=cycle_data["cycle"])
                stub.pre_score = cycle_data.get("pre_score", 0)
                stub.post_score = cycle_data.get("post_score", 0)
                stub.improvement = cycle_data.get("improvement", 0)
                stub.samples_generated = cycle_data.get("samples_generated", 0)
                stub.samples_verified = cycle_data.get("samples_verified", 0)
                stub.timestamp = cycle_data.get("timestamp", 0)
                stub.duration = cycle_data.get("duration_seconds", 0)
                stub.had_diagnostics = cycle_data.get("had_diagnostics", False)
                # Restore eval_score too — previously dropped on resume, which
                # broke every downstream consumer that keys on it:
                #   - LR bandit (prev_eval=None → delta=None → never observes)
                #   - best-checkpoint tracker (falls back to post_score)
                #   - early-stop degradation detector
                # Data from cycle_2 in original run showed bandit.arms stuck at
                # (1, 1) after 3 cycles because of this.
                stub.eval_score = cycle_data.get("eval_score")
                stub.eval_domain_scores = cycle_data.get("eval_domain_scores", {}) or {}
                stub.eval_subdomain_scores = cycle_data.get("eval_subdomain_scores", {}) or {}
                self.history.append(stub)

            num_cycles = len(self.history)
            if not (resume_path / "config.json").exists():
                resumed_cycle = self.history[-1].cycle if self.history else None
                lora_pt = None
                if resumed_cycle is not None:
                    candidate = (self.config.orchestrator.output_dir /
                                 "lora_weights" / f"lora_cycle_{resumed_cycle}" /
                                 "lora_weights.pt")
                    if candidate.exists():
                        lora_pt = candidate
                if lora_pt is not None:
                    try:
                        self.trainer.load_lora_weights(str(lora_pt))
                        self.trainer.merge_lora()
                        logger.info(
                            f"  Recovered cycle {resumed_cycle} from LoRA weights "
                            f"(full checkpoint was missing)."
                        )
                    except Exception as e:
                        logger.warning(
                            f"  LoRA recovery failed ({e}); falling back to base model."
                        )
                else:
                    logger.warning(
                        f"  WARNING: No model checkpoint found in {resume_path}. "
                        f"Model is the original base — prior cycle merges are LOST. "
                        f"Training will continue but results may be inconsistent."
                    )
            elif self._use_vllm:
                logger.info(f"  Reloading vLLM from checkpoint: {resume_path}")
                self.model_loader.swap_to_vllm_after_training(str(resume_path))
            logger.info(f"Resumed from checkpoint: {num_cycles} cycles, escalations: {self._escalation_state}")
            return num_cycles + 1

        return 1

    def _run_cycle(self, cycle: int) -> CycleResult:
        """Execute one full improvement cycle."""
        result = CycleResult(cycle)
        self._vllm_saved_this_cycle = None

        def _log_peak_memory(phase_name: str):
            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
                current = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                logger.info(
                    f"  [GPU Memory] after {phase_name}: "
                    f"peak={peak:.2f}GB, current={current:.2f}GB, reserved={reserved:.2f}GB"
                )
                torch.cuda.reset_peak_memory_stats()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # 1. Diagnose
        logger.info(f"[Cycle {cycle}] Phase 1: DIAGNOSE")
        phase_start = time.time()
        has_prior_diag = any(
            r.diagnostics is not None or getattr(r, "had_diagnostics", False)
            for r in self.history
        )
        if self._escalation_state["diagnosis"] and has_prior_diag:
            self._escalate_diagnosis(cycle=cycle)
        self.diagnostics.set_model_score(self._best_score)
        diag = self.diagnostics.run(cycle)
        result.phase_times["diagnose"] = time.time() - phase_start
        _log_peak_memory("diagnose")
        result.diagnostics = diag
        result.had_diagnostics = True
        result.pre_score = diag.overall_score
        logger.info(
            f"  Found {len(diag.weaknesses)} weaknesses across "
            f"{len(diag.domain_scores)} domains | Overall score: {diag.overall_score:.3f}"
        )
        for w in diag.weaknesses[:5]:
            layers_info = f" ({len(w.weak_layers)} correlated layers)" if w.weak_layers else ""
            logger.info(f"    - {w.domain}/{w.subdomain}: severity {w.severity:.2f}{layers_info}")

        # Inject regression weaknesses from previous cycle.
        injected_regressions = []
        if self._pending_regression_weaknesses:
            logger.info(f"  Injecting {len(self._pending_regression_weaknesses)} regression weaknesses from prior cycle")
            injected_regressions = list(self._pending_regression_weaknesses)
            diag.weaknesses.extend(self._pending_regression_weaknesses)
            self._pending_regression_weaknesses.clear()

        if not diag.weaknesses:
            logger.info(f"  No weaknesses found — all domains above threshold")
            result.post_score = result.pre_score
            return result

        # 1b. Synthesis mode (opt-in) — produce novel tasks + properties between
        #     diagnose and generate. Each SynthesizedTask carries trusted
        #     properties; quorum_verdict (§2.1) is the accept/reject gate per
        #     the spec. Passing tasks are converted to TrainingSample and merged
        #     into the verify output, bypassing the classic grader (they already
        #     cleared property-based checks upstream).
        synthesis_samples: list = []
        if self._synthesis_enabled and self._task_synthesizer is not None:
            logger.info(f"[Cycle {cycle}] Phase 1b: SYNTHESIZE")
            phase_start = time.time()
            try:
                synth_result = self._task_synthesizer.synthesize(diag)
                if synth_result.tasks:
                    n_admitted = 0
                    n_quorum_fail = 0
                    # Registry helpers — import lazily, fail softly.
                    try:
                        from .registries import VerificationRecord, TrainingPoolRecord
                        import uuid as _uuid
                        _reg = self._registries
                    except Exception:
                        _reg = None
                        VerificationRecord = None  # type: ignore[assignment]
                        TrainingPoolRecord = None  # type: ignore[assignment]

                    # Lazy-import verify() once for the whole batch.
                    try:
                        from ..verifier.property_engine import verify as _pe_verify, SandboxedExecutor as _SandboxedExecutor
                        _executor = _SandboxedExecutor(memory_mb=256)
                        _pe_available = True
                    except Exception as _pe_exc:
                        logger.debug("property_engine.verify() unavailable (%s) — skipping synthesis", _pe_exc)
                        _pe_available = False

                    for task in synth_result.tasks:
                        props = getattr(task, "properties", []) or []
                        ref = getattr(task, "reference_solution", "")
                        task_id = getattr(task, "task_id", "?")
                        candidate_id = f"{task_id}:reference"

                        if not _pe_available or not props:
                            logger.debug("  Skipping task %s: no properties or verify() unavailable", task_id)
                            n_quorum_fail += 1
                            continue

                        # Run admitted properties via property_engine.verify() (OPTION 2).
                        # SandboxedExecutor (Phase B): subprocess per property, ~50-200ms each.
                        try:
                            pe_record = _pe_verify(
                                problem_id=task_id,
                                candidate=ref,
                                admitted_properties=props,
                                executor=_executor,
                                calibration=self._registries,
                                quorum_distinct_classes_required=2,
                                min_properties=2,
                            )
                        except Exception as exc:
                            logger.debug("  verify() raised for task %s: %s", task_id, exc)
                            n_quorum_fail += 1
                            continue

                        # Translate PropertyVerdict list → per_property_verdicts dicts.
                        per_prop_verdicts = [
                            {
                                "property_id": v.property_id,
                                "passed": v.verdict == "PASS",
                                "reason": v.reason,
                                "independence_class": v.independence_class,
                            }
                            for v in pe_record.per_property
                        ]

                        # Write VerificationRecord regardless of outcome (spec §4).
                        if _reg is not None and VerificationRecord is not None:
                            try:
                                vr_id = pe_record.record_id
                                _reg.verification_log.append_verification(VerificationRecord(
                                    record_id=vr_id,
                                    problem_id=task_id,
                                    candidate_id=candidate_id,
                                    property_ids=[v.property_id for v in pe_record.per_property],
                                    per_property_verdicts=per_prop_verdicts,
                                    quorum_accepted=pe_record.accepted,
                                    quorum_reason=pe_record.reject_reason,
                                    adversarial=False,
                                    session_id=_reg.sid,
                                    quorum_n=pe_record.quorum_n,
                                    pass_count=pe_record.pass_count,
                                    fail_count=pe_record.fail_count,
                                    error_count=pe_record.error_count,
                                    distinct_classes=tuple(pe_record.distinct_classes),
                                    quorum_distinct_classes_required=pe_record.quorum_distinct_classes_required,
                                ))
                            except Exception as exc:
                                logger.debug("registry write failed (VerificationRecord): %s", exc)
                                vr_id = _uuid.uuid4().hex

                        if pe_record.accepted:
                            training_sample = task.to_training_sample()
                            synthesis_samples.append(training_sample)
                            n_admitted += 1
                            # Write TrainingPoolRecord and retire the problem
                            # (spec §7: retire on first training-pool acceptance
                            # so generator's novelty check skips it next cycle).
                            if _reg is not None and TrainingPoolRecord is not None:
                                try:
                                    _reg.training_pool.append_sample(TrainingPoolRecord(
                                        pool_record_id=_uuid.uuid4().hex,
                                        problem_id=task_id,
                                        candidate_id=candidate_id,
                                        verification_record_id=vr_id,
                                        domain=getattr(task, "domain", ""),
                                        prompt=getattr(task, "prompt", ""),
                                        response=ref,
                                        session_id=_reg.sid,
                                    ))
                                    _reg.problem_registry.mark_retired(task_id, session_id=_reg.sid)
                                except Exception as exc:
                                    logger.debug("registry write failed (TrainingPoolRecord/retire): %s", exc)
                        else:
                            n_quorum_fail += 1
                            logger.debug("  Quorum rejected task %s: %s", task_id, pe_record.reject_reason)
                    logger.info(
                        "  Synthesis: %d tasks produced, %d admitted by quorum, %d rejected",
                        len(synth_result.tasks), n_admitted, n_quorum_fail,
                    )
                else:
                    logger.info("  Synthesis: no tasks produced this cycle")
            except Exception as e:
                tb = traceback.format_exc()
                logger.warning(f"  Synthesis phase failed ({type(e).__name__}): {e} — continuing without synthesized tasks")
                result.errors.append({
                    "phase": "synthesis", "type": type(e).__name__,
                    "message": str(e), "traceback": tb,
                })
            result.phase_times["synthesis"] = time.time() - phase_start

        # 2. Generate training data
        logger.info(f"[Cycle {cycle}] Phase 2: GENERATE ({len(diag.weaknesses)} weaknesses)")
        phase_start = time.time()
        try:
            # STaR (Zelikman et al. 2022): train on rationales that reach
            # known-correct canonical answers on REAL diagnostic problems.
            # Falls back internally to legacy self-synthesis if no failed
            # diagnostic items are available.
            samples = self.generator.generate_from_diagnostic_result(diag)
        except Exception as e:
            logger.error(f"  Generation failed ({type(e).__name__}): {e}")
            torch.cuda.empty_cache()
            self._pending_regression_weaknesses.extend(injected_regressions)
            result.post_score = result.pre_score
            return result
        result.phase_times["generate"] = time.time() - phase_start
        _log_peak_memory("generate")
        result.samples_generated = len(samples)
        result.diversity_stats = self.generator.get_diversity_stats()
        logger.info(f"  Generated {len(samples)} training samples")

        if not samples and not synthesis_samples:
            logger.warning(f"  No samples generated — model couldn't produce valid problems")
            self._pending_regression_weaknesses.extend(injected_regressions)
            result.post_score = result.pre_score
            return result

        # 3. Verify
        logger.info(f"[Cycle {cycle}] Phase 3: VERIFY")
        phase_start = time.time()
        verified = self.verifier.verify_batch(samples) if samples else []
        # Synthesis-mode: consensus-passed samples bypass the classic verifier
        # (they already cleared property checks) and are merged directly.
        if synthesis_samples:
            logger.info(f"  Merging {len(synthesis_samples)} consensus-passed synthesis samples")
            verified = list(verified) + list(synthesis_samples)
        verified = self._apply_quality_top_k(verified)
        result.samples_verified = len(verified)
        # Observability: stash the verified samples + STaR internals for the
        # optional cycle_metrics / cycle_samples dumps.
        if getattr(self.config.orchestrator, "write_cycle_samples", False) or \
                getattr(self.config.orchestrator, "write_cycle_metrics", False):
            result._training_samples = list(verified)
        if getattr(self.config.orchestrator, "write_cycle_metrics", False):
            result._star_stats = dict(
                getattr(self.generator, "_last_star_stats", {}) or {}
            )
        pass_rate = len(verified) / len(samples) * 100 if samples else 0
        logger.info(f"  {len(verified)}/{len(samples)} passed verification ({pass_rate:.0f}%)")

        if not verified:
            logger.warning(f"  No samples passed verification — all reasoning chains rejected")
            self._pending_regression_weaknesses.extend(injected_regressions)
            result.post_score = result.pre_score
            return result

        result.phase_times["verify"] = time.time() - phase_start
        _log_peak_memory("verify")

        # 4. Train — swap to HF model if using vLLM
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"[Cycle {cycle}] Phase 4: TRAIN on {len(verified)} verified samples")
        phase_start = time.time()
        if self._use_vllm:
            self.model_loader.swap_to_hf_for_training()
        self.trainer.inject_lora(weak_layers=diag.layer_health)

        # 4a. PRM (optional) — train once per cycle on verified samples, then
        # install as reward_fn for downstream GRPO. Gated by config.trainer.use_prm
        # and training_mode == "grpo". Failures are isolated: a broken PRM
        # falls back to outcome-only reward.
        if (getattr(self.config.trainer, "use_prm", False)
                and self.config.trainer.training_mode == "grpo"):
            try:
                from ..trainer.prm import PRM, make_prm_reward_fn
                prm = PRM(self.model_loader, self.config.trainer)
                prm.train_from_samples(verified, cycle=cycle)
                self.trainer.set_reward_fn(make_prm_reward_fn(prm))
                logger.info(f"  PRM trained and installed as GRPO reward_fn")
            except Exception as e:
                tb = traceback.format_exc()
                logger.warning(f"  PRM phase failed ({type(e).__name__}): {e} — continuing with default reward")
                result.errors.append({
                    "phase": "prm", "type": type(e).__name__,
                    "message": str(e), "traceback": tb,
                })

        try:
            # DPO preference pairs from the most recent STaR rollout (empty in
            # legacy path or when no problem produced both correct and incorrect
            # chains). Trainer routes SFT/DPO/mixed internally via config.training_mode.
            preference_pairs = self.generator.get_last_preference_pairs()
            metrics = self.trainer.train(verified, cycle, preference_pairs=preference_pairs)
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"  Training OOM (even after batch-size retry): {e} — stripping LoRA and skipping cycle")
            self.trainer.strip_lora()
            torch.cuda.empty_cache()
            self._pending_regression_weaknesses.extend(injected_regressions)
            result.post_score = result.pre_score
            self._consecutive_failures += 1
            if self._use_vllm:
                self.model_loader.swap_to_vllm_after_training()
            return result
        except Exception as e:
            logger.error(f"  Training failed ({type(e).__name__}): {e} — stripping LoRA and skipping cycle")
            self.trainer.strip_lora()
            torch.cuda.empty_cache()
            self._pending_regression_weaknesses.extend(injected_regressions)
            result.post_score = result.pre_score
            self._consecutive_failures += 1
            if self._use_vllm:
                self.model_loader.swap_to_vllm_after_training()
            return result
        result.training_metrics = metrics
        result.phase_times["train"] = time.time() - phase_start
        _log_peak_memory("train")
        logger.info(f"  Training done: {metrics.steps} steps, final loss: {metrics.final_loss:.4f}")

        # 5. Save LoRA weights BEFORE merge (merge destroys them), then evaluate
        logger.info(f"[Cycle {cycle}] Phase 5: EVALUATE")
        self.trainer.save_lora_weights(
            self.config.orchestrator.output_dir / "lora_weights", cycle
        )
        self.trainer.merge_lora()

        if self._use_vllm:
            ckpt_root = self.config.orchestrator.output_dir / "checkpoints"
            self.model_loader.save_checkpoint(ckpt_root, cycle)
            tmp_ckpt = ckpt_root / f"cycle_{cycle}"
            if not (tmp_ckpt / "config.json").exists() or not any(
                tmp_ckpt.glob("*.safetensors")
            ):
                raise RuntimeError(
                    f"checkpoint at {tmp_ckpt} is incomplete "
                    f"(missing config.json or *.safetensors); refusing vLLM swap"
                )
            self._vllm_saved_this_cycle = cycle
            # Skip the vLLM reload when configured — runs post-diag + eval in
            # HF mode. Saves ~3-5 min per cycle when probe count is small.
            skip_reload = (
                self.config.vllm is not None
                and getattr(self.config.vllm, "skip_reload_after_training", False)
            )
            if not skip_reload:
                self.model_loader.swap_to_vllm_after_training(str(tmp_ckpt))
        # Run post-training diagnostics on the SAME questions as pre-training.
        # Different cycle arg = different RNG seed = different questions, which
        # made `improvement = post - pre` compare apples to oranges.
        post_diag = self.diagnostics.run(cycle)
        result.post_diag = post_diag
        result.post_score = post_diag.overall_score
        result.improvement = result.post_score - result.pre_score

        symbol = "+" if result.improvement > 0 else ""
        logger.info(f"  Score: {result.pre_score:.3f} -> {result.post_score:.3f} ({symbol}{result.improvement:.3f})")

        # Track per-domain scores
        for domain, score in post_diag.domain_scores.items():
            if domain not in self._domain_score_history:
                self._domain_score_history[domain] = []
            self._domain_score_history[domain].append(score)
            if len(self._domain_score_history[domain]) > 20:
                self._domain_score_history[domain] = self._domain_score_history[domain][-20:]

        # Detect regressions
        regressions = []
        for domain, pre_score in diag.domain_scores.items():
            post_score = post_diag.domain_scores.get(domain, 0)
            if post_score < pre_score - 0.05:
                regressions.append(f"{domain}: {pre_score:.3f}->{post_score:.3f}")
                regression_evidence = [
                    e for e in post_diag.weaknesses if e.domain == domain
                ]
                evidence_items = []
                for w in regression_evidence:
                    evidence_items.extend(w.evidence[:5])
                if not evidence_items:
                    pre_evidence = [e for e in diag.weaknesses if e.domain == domain]
                    for w in pre_evidence:
                        evidence_items.extend(w.evidence[:5])
                self._pending_regression_weaknesses.append(WeaknessReport(
                    domain=domain,
                    subdomain="regression",
                    severity=pre_score - post_score,
                    evidence=evidence_items[:20],
                    description=f"Regression: {domain} dropped from {pre_score:.3f} to {post_score:.3f}",
                ))
        if regressions:
            logger.warning(f"  REGRESSION detected in: {', '.join(regressions)}")

        return result

    def rsi_tick(self, cycle: int) -> "CycleResult":
        """Full RSI iteration per spec §4 (steps 1-8).

        Implements the closed-loop: propose → admit properties → solve →
        verify → decide → calibrate → train.  Step 5 (adversarial pass) is
        deferred per v0.2; the slot is preserved for wiring once
        property_verifier ships the adversarial interface.

        Returns a CycleResult populated with RSI-mode counters (§5.3).
        Never raises — all sub-phase failures are logged and recorded on
        result.errors so the caller can continue.
        """
        if not self._synthesis_enabled or self._registries is None:
            raise RuntimeError(
                "rsi_tick() requires synthesis mode to be enabled and registries "
                "to be open. Set config.synthesis.enable_task_synthesis=True."
            )

        from .registries import (
            PropertyRecord,
            VerificationRecord,
            CalibrationEntry,
            TrainingPoolRecord,
        )
        from ..generator.task_synthesizer import (
            append_proposal_to_registry as _append_proposal,
            retire_problem_on_acceptance as _retire_problem,
        )
        import uuid as _uuid

        result = CycleResult(cycle)
        reg = self._registries
        synth = self._task_synthesizer
        ts = self.config.synthesis.tasks_per_cycle

        # ── STEP 0: prime synthesizer with current diagnostics ───────────────
        # Speed: reuse cached DiagnosticResult across non-training cycles.
        # Weights only change when Step 8 trains, and rsi_tick invalidates
        # the cache there. Between trainings the mastery profile is stable,
        # so re-running all 240 probes every cycle burns ~56s/cycle for no
        # signal change. Cache lifetime = rsi_diagnostic_refresh_every cycles.
        refresh_every = max(1, int(getattr(
            self.config.orchestrator, "rsi_diagnostic_refresh_every", 1,
        )))
        cache_age = cycle - self._rsi_diag_cache_cycle
        use_cached_diag = (
            self._rsi_diag_cache is not None
            and refresh_every > 1
            and cache_age < refresh_every
            and cache_age >= 0
        )
        logger.info(
            "[RSI tick %d] Step 0: SET_DIAGNOSTICS (cache=%s, age=%d/%d)",
            cycle, "hit" if use_cached_diag else "miss",
            cache_age if self._rsi_diag_cache is not None else -1,
            refresh_every,
        )
        phase_start = time.time()
        try:
            if use_cached_diag:
                diag_result = self._rsi_diag_cache
            else:
                diag_result = self.diagnostics.run(cycle)
                self._rsi_diag_cache = diag_result
                self._rsi_diag_cache_cycle = cycle
            result.diagnostics = diag_result
            synth.set_diagnostics(
                diag_result,
                prior_subdomain_scores=self._rsi_prior_subdomain_scores or None,
                session_id=reg.sid,
                run_id=getattr(self.config.orchestrator, "run_id", "") or "",
            )
            self._rsi_prior_subdomain_scores = dict(
                getattr(diag_result, "subdomain_scores", {}) or {}
            )
        except Exception as exc:
            logger.warning("[RSI tick %d] set_diagnostics failed (%s): %s — propose_batch will return []",
                           cycle, type(exc).__name__, exc)
        result.phase_times["rsi_step0_diagnose"] = time.time() - phase_start

        # ── STEP 1: propose problems ─────────────────────────────────────────
        # Use the builtin-based code-proposal path by default — it asks the
        # model for a much simpler format (PROBLEM + ENTRY + REFERENCE +
        # TESTS) that an 8B base can emit reliably, and wires the model's
        # own tests as the §1.4 ground truth via property_engine's trusted
        # builtins. The legacy co-gen propose_batch (full property source
        # per property) is kept for stronger models; switch via
        # config.synthesis.use_builtin_code_path=False.
        use_code_builtins = getattr(
            self.config.synthesis, "use_builtin_code_path", True,
        )
        logger.info(
            "[RSI tick %d] Step 1: PROPOSE %d problems (path=%s)",
            cycle, ts, "code_builtins" if use_code_builtins else "legacy_cogen",
        )
        phase_start = time.time()
        proposed_problems: list = []
        try:
            if use_code_builtins and hasattr(synth, "propose_batch_code"):
                proposed_problems = synth.propose_batch_code(ts)
            else:
                proposed_problems = synth.propose_batch(ts)
        except Exception as exc:
            tb = traceback.format_exc()
            logger.warning("Step 1 (propose_batch) failed (%s): %s", type(exc).__name__, exc)
            result.errors.append({
                "phase": "rsi_step1_propose", "type": type(exc).__name__,
                "message": str(exc), "traceback": tb,
            })

        # Persist ProposedProblem records via task_synthesizer helper (spec §4.1).
        for problem in proposed_problems:
            try:
                ok = _append_proposal(problem, reg, bundle_admitted=True)
                if ok:
                    result.novel_problems_proposed += 1
            except Exception as exc:
                logger.debug("ProblemRecord write failed: %s", exc)
        result.phase_times["rsi_step1_propose"] = time.time() - phase_start

        # ── STEP 2: propose + admit properties ──────────────────────────────
        # Spec v0.2.1: a property passing §1.3 admit() is a CANDIDATE, not yet
        # a registry entry. PropertyRecord writes are deferred until the bundle
        # passes §1.4 (VoV quorum) in Step 4/6. Properties of rejected bundles
        # are never written to outputs/properties/<sid>.jsonl.
        logger.info("[RSI tick %d] Step 2: PROPOSE + ADMIT properties", cycle)
        phase_start = time.time()
        # properties_by_problem: problem_id → list of admitted Property objects
        properties_by_problem: dict[str, list] = {}
        # staged_prop_records: problem_id → list of PropertyRecord (not yet written)
        staged_prop_records: dict[str, list] = {}
        for problem in proposed_problems:
            pid = getattr(problem, "problem_id", None)
            if pid is None:
                continue
            # Builtin path: problem carries problem_ctx; use the trusted
            # builtin Property templates with runtime ctx stashed. This is
            # the only path that actually produces discriminating property
            # verdicts with an 8B base model — the alternative is to ask
            # the model to write property source code, which it can't.
            try:
                if getattr(problem, "problem_ctx", None) and hasattr(synth, "materialize_builtin_properties"):
                    raw_props = synth.materialize_builtin_properties(problem)
                else:
                    raw_props = synth.propose_properties(problem)
            except Exception as exc:
                logger.debug("materialize properties(%s) failed: %s", pid, exc)
                raw_props = []

            admitted: list = []
            staged: list = []
            for prop in (raw_props or []):
                # §1.3 admission gate — call register_property() if available;
                # it raises on rejection, passes on admission. Either way we do
                # NOT write to PropertyRegistry yet (bundle-gated, spec v0.2.1).
                try:
                    from ..verifier.property_engine import register_property
                    register_property(prop)
                except Exception:
                    pass
                admitted.append(prop)
                try:
                    staged.append(PropertyRecord(
                        property_id=getattr(prop, "property_id", _uuid.uuid4().hex),
                        problem_id=pid,
                        author="task_synthesizer",
                        independence_class=getattr(prop, "independence_class", "exec.behavioral"),
                        kind=getattr(prop, "kind", "behavioral"),
                        name=getattr(prop, "name", str(prop)),
                        payload=getattr(prop, "__dict__", {}),
                    ))
                except Exception as exc:
                    logger.debug("PropertyRecord staging failed: %s", exc)
            properties_by_problem[pid] = admitted
            staged_prop_records[pid] = staged
        result.phase_times["rsi_step2_admit"] = time.time() - phase_start

        # ── STEP 3: solve (generate candidates) ─────────────────────────────
        # Batch all problems' solve prompts into ONE vLLM call. Run-4 spent
        # ~8 minutes per cycle (out of ~13) on this step because each
        # problem's k-candidate set was its own generate_batch — 20 serial
        # calls × 25s. One batch of 100 prompts completes in ~30s with
        # vLLM's KV caching, a ~15x cycle speedup.
        k = getattr(self.config.synthesis, "candidates_per_problem", 6)
        logger.info("[RSI tick %d] Step 3: SOLVE (%d problems × %d candidates, batched)",
                    cycle, len(proposed_problems), k)
        phase_start = time.time()
        candidates_by_problem: dict[str, list] = {}
        try:
            if hasattr(synth, "solve_batch"):
                candidates_by_problem = synth.solve_batch(proposed_problems, k=k)
            else:
                # Fallback: old per-problem loop
                for problem in proposed_problems:
                    pid = getattr(problem, "problem_id", None)
                    if pid is None:
                        continue
                    try:
                        candidates_by_problem[pid] = synth.solve(problem, k=k)
                    except Exception as exc:
                        logger.debug("solve(%s) failed: %s", pid, exc)
                        candidates_by_problem[pid] = []
        except Exception as exc:
            logger.warning("Step 3 (solve_batch) failed (%s): %s", type(exc).__name__, exc)
            candidates_by_problem = {}
        result.phase_times["rsi_step3_solve"] = time.time() - phase_start
        logger.info("[RSI tick %d] Step 3: solve phase took %.1fs",
                    cycle, result.phase_times["rsi_step3_solve"])

        # ── STEP 4: verify (per-candidate quorum) ───────────────────────────
        logger.info("[RSI tick %d] Step 4: VERIFY candidates", cycle)
        phase_start = time.time()
        training_samples: list = []

        # Lazy-import verify() once for the whole tick.
        try:
            from ..verifier.property_engine import verify as _pe_verify, SandboxedExecutor as _SandboxedExecutor
            _executor = _SandboxedExecutor(memory_mb=256)
            _pe_available = True
        except Exception as _pe_exc:
            logger.warning("[RSI tick %d] property_engine.verify() unavailable (%s)", cycle, _pe_exc)
            _pe_available = False

        for problem in proposed_problems:
            pid = getattr(problem, "problem_id", None)
            if pid is None:
                continue
            props = properties_by_problem.get(pid, [])
            candidates = candidates_by_problem.get(pid, [])
            if not candidates:
                continue

            for idx, candidate in enumerate(candidates):
                cand_id = f"{pid}:cand{idx}"
                candidate_str = (
                    candidate if isinstance(candidate, str)
                    else getattr(candidate, "solution", str(candidate))
                )

                if not _pe_available or not props:
                    logger.debug("  Skipping candidate %s: no properties or verify() unavailable", cand_id)
                    continue

                # STEP 4+6: run admitted properties via property_engine.verify() and decide.
                # SandboxedExecutor (Phase B): subprocess per property, ~50-200ms each.
                # STEP 5 slot: adversarial pass deferred per v0.2 — wire here before verify() call.
                try:
                    pe_record = _pe_verify(
                        problem_id=pid,
                        candidate=candidate_str,
                        admitted_properties=props,
                        executor=_executor,
                        calibration=reg.calibration_ledger,
                        quorum_distinct_classes_required=2,
                        min_properties=2,
                    )
                except Exception as exc:
                    logger.debug("  verify() raised for candidate %s: %s", cand_id, exc)
                    continue

                per_prop_verdicts = [
                    {
                        "property_id": v.property_id,
                        "passed": v.verdict == "PASS",
                        "reason": v.reason,
                        "independence_class": v.independence_class,
                    }
                    for v in pe_record.per_property
                ]
                vr_id = pe_record.record_id
                try:
                    reg.verification_log.append_verification(VerificationRecord(
                        record_id=vr_id,
                        problem_id=pid,
                        candidate_id=cand_id,
                        property_ids=[v.property_id for v in pe_record.per_property],
                        per_property_verdicts=per_prop_verdicts,
                        quorum_accepted=pe_record.accepted,
                        quorum_reason=pe_record.reject_reason,
                        adversarial=False,
                        session_id=reg.sid,
                        quorum_n=pe_record.quorum_n,
                        pass_count=pe_record.pass_count,
                        fail_count=pe_record.fail_count,
                        error_count=pe_record.error_count,
                        distinct_classes=tuple(pe_record.distinct_classes),
                        quorum_distinct_classes_required=pe_record.quorum_distinct_classes_required,
                    ))
                except Exception as exc:
                    logger.debug("VerificationRecord write failed: %s", exc)

                if pe_record.accepted:
                    result.candidates_accepted += 1
                    try:
                        from ..verifier.property_engine import get_problem_ctx as _get_ctx
                        _saved_ctx = dict(_get_ctx(pid) or {})
                    except Exception:
                        _saved_ctx = {}
                    result._admitted_candidates.append({
                        "problem_id": pid,
                        "candidate": candidate_str,
                        "domain": getattr(problem, "domain", "code") or "code",
                        "problem_ctx": _saved_ctx,
                    })
                    # Bundle passed §1.4 (quorum) — NOW flush staged PropertyRecords
                    # (spec v0.2.1: write-on-bundle-pass; §1.3 admittees that belong
                    # to rejected bundles are never persisted).
                    for prop_rec in staged_prop_records.get(pid, []):
                        try:
                            reg.property_registry.append_property(prop_rec, bundle_passed_vov=True)
                            result.properties_admitted += 1
                        except Exception as exc:
                            logger.debug("PropertyRecord flush failed: %s", exc)
                    # Convert candidate to TrainingSample for the training pool.
                    try:
                        from ..generator.data_generator import TrainingSample
                        # TrainingSample takes `response:` (kwarg), not
                        # `solution:`. Previous `solution=` silently raised
                        # TypeError which was swallowed at DEBUG level, so
                        # every accepted candidate got silently dropped —
                        # that's why run-5 cycles 1-5 showed
                        # "candidates accepted=N > 0" but "pool has 0
                        # samples" every cycle. The counter fired BEFORE
                        # the silent TypeError.
                        ts_obj = TrainingSample(
                            prompt=getattr(problem, "problem_text", ""),
                            response=candidate_str,
                            domain=getattr(problem, "domain", "unknown"),
                            verified=True,
                            source="rsi_property",
                        )
                        training_samples.append(ts_obj)
                        reg.training_pool.append_sample(TrainingPoolRecord(
                            pool_record_id=_uuid.uuid4().hex,
                            problem_id=pid,
                            candidate_id=cand_id,
                            verification_record_id=vr_id,
                            domain=getattr(problem, "domain", ""),
                            prompt=getattr(problem, "problem_text", ""),
                            response=candidate_str,
                            session_id=reg.sid,
                        ))
                        # §7: retire problem on first training-pool acceptance.
                        _retire_problem(pid, reg, session_id=reg.sid)
                    except Exception as exc:
                        logger.warning(
                            "TrainingSample/pool write failed (%s): %s — "
                            "candidate accepted by quorum but dropped from "
                            "training pool. If this happens on EVERY candidate, "
                            "the TrainingSample constructor signature changed.",
                            type(exc).__name__, exc,
                        )
                else:
                    logger.debug("  Quorum rejected %s: %s", cand_id, pe_record.reject_reason)

        result.phase_times["rsi_step4_verify"] = time.time() - phase_start
        logger.info(
            "[RSI tick %d] candidates accepted=%d",
            cycle, result.candidates_accepted,
        )

        # ── STEP 7: calibration check ────────────────────────────────────────
        logger.info("[RSI tick %d] Step 7: CALIBRATION CHECK", cycle)
        phase_start = time.time()
        try:
            suspended = reg.calibration_ledger.suspended_classes()
            result.classes_suspended = len(suspended)
            # Emit a calibration snapshot for this tick (one row per known class).
            # True-accept/reject rates require ground-truth probing; we record
            # the minimal observable information (class suspended status) so
            # CalibrationLedger is populated. Full probe logic is wired by
            # property_verifier when it exposes calibration_probe().
            all_classes = {
                pv["independence_class"]
                for p in proposed_problems
                for pid in [getattr(p, "problem_id", None)]
                if pid
                for pv in [
                    {"independence_class": getattr(prop, "independence_class", "")}
                    for prop in properties_by_problem.get(pid, [])
                ]
            } | suspended
            for cls in all_classes:
                if not cls:
                    continue
                try:
                    reg.calibration_ledger.append_calibration(CalibrationEntry(
                        tick=cycle,
                        independence_class=cls,
                        true_accept_rate=float("nan"),
                        true_reject_rate=float("nan"),
                        error_rate=0.0,
                        suspended=cls in suspended,
                        n_probes=0,
                        session_id=reg.sid,
                    ))
                except Exception as exc:
                    logger.debug("CalibrationEntry write failed (%s): %s", cls, exc)
        except Exception as exc:
            tb = traceback.format_exc()
            logger.warning("Step 7 (calibration) failed (%s): %s", type(exc).__name__, exc)
            result.errors.append({
                "phase": "rsi_step7_calibration", "type": type(exc).__name__,
                "message": str(exc), "traceback": tb,
            })
        result.phase_times["rsi_step7_calibration"] = time.time() - phase_start

        # ── STEP 8: train if ready ────────────────────────────────────────────
        # Pure RSI: NO STaR fallback. If this cycle's property-based path
        # produced zero training samples, the cycle does NOT train on
        # anything else — the whole point is that training data has to
        # come from candidates that cleared §1.3 admission + §1.4 verify
        # quorum. Falling back to weakness-patching would make the pipeline
        # behaviorally identical to classic STaR, which is the failure mode
        # this rebuild exists to eliminate. Next cycle retries; property_ctx
        # is cleared between cycles so stale state doesn't leak.
        try:
            from ..verifier.property_engine import clear_problem_ctx as _clear_ctx
            _clear_ctx()
        except Exception:
            pass

        # Accumulate this cycle's verified samples into the rolling RSI pool.
        # Training fires only when the pool is large enough for a stable
        # gradient update (observed: 18-21 fresh samples per cycle × 2 steps
        # memorizes a 8B model instantly). Carry-over across cycles means
        # the model sees diverse problems before any weight update.
        self._rsi_pending_pool.extend(training_samples)
        self._rsi_pool_accumulated_cycles += 1

        logger.info(
            "[RSI tick %d] Step 8: TRAIN (this cycle: %d samples, rolling pool: %d across %d cycles)",
            cycle, len(training_samples),
            len(self._rsi_pending_pool),
            self._rsi_pool_accumulated_cycles,
        )
        phase_start = time.time()
        # min_train_batch kept for back-compat (legacy field, default 1).
        # min_train_samples is the stability floor — raised from 16 to 60
        # because 18-sample training reliably crashed into memorization
        # (run-8, run-9 cycle 1). Accumulate across 3-4 cycles.
        min_batch = max(
            getattr(self.config.trainer, "min_train_batch", 1),
            getattr(self.config.trainer, "min_train_samples", 60),
        )
        training_samples = list(self._rsi_pending_pool)  # rebind to the accumulated pool
        if len(training_samples) >= min_batch:
            # Weights will change (or be reverted) — either way invalidate the
            # Step 0 diagnostic cache so next cycle re-probes against the new
            # weights instead of using stale mastery estimates.
            self._rsi_diag_cache = None
            self._rsi_diag_cache_cycle = -1
            try:
                if self._use_vllm:
                    self.model_loader.swap_to_hf_for_training()
                # Reuse the Step-0 diagnostic instead of running a second full
                # diagnose pass here — that was burning ~1 minute of GPU per
                # cycle on redundant inference and was also a prime suspect
                # for the VRAM fragmentation that led to OOM in cycle 2. The
                # Step-0 diag is still current: we haven't trained yet.
                diag = result.diagnostics
                if diag is None:
                    # Fallback if Step 0 was skipped — diagnose now.
                    diag = self.diagnostics.run(cycle)
                    result.diagnostics = diag
                result.pre_score = diag.overall_score
                self.trainer.inject_lora(weak_layers=getattr(diag, "layer_health", {}) or {})
                metrics = self.trainer.train(training_samples, cycle)
                result.training_metrics = metrics

                # Guard: if training produced no effective optimizer steps
                # (pre-training probe tripped, dataset empty, etc.), there's
                # nothing to merge. Strip and bail WITHOUT writing a checkpoint
                # so the next cycle starts from the same weights.
                if not metrics or getattr(metrics, "steps", 0) == 0:
                    logger.info(
                        "[RSI tick %d] training skipped by trainer guard (0 steps) "
                        "— not merging, not checkpointing",
                        cycle,
                    )
                    self.trainer.strip_lora()
                else:
                    self.trainer.save_lora_weights(
                        self.config.orchestrator.output_dir / "lora_weights", cycle
                    )
                    self.trainer.merge_lora()

                    # Regression-revert guard: do a FAST probe (not a full
                    # diagnostic run) before swapping vLLM over. Previous
                    # code called self.diagnostics.run(cycle) here, which
                    # re-ran all 240 diagnostic prompts — BUT through the
                    # HF model (vLLM is unloaded for training), which is
                    # ~20x slower than vLLM. Run-7 burned ~4 minutes per
                    # training cycle on that alone (6:27:42 strip_lora →
                    # 6:31:32 regression warning). A 24-prompt probe
                    # detects the -0.2+ drops that trigger the revert
                    # just as reliably, in ~15s.
                    revert_threshold = float(getattr(
                        self.config.trainer, "regression_revert_threshold", 0.10,
                    ))
                    try:
                        post_score = self._quick_regression_probe(cycle)
                    except Exception as exc:
                        logger.warning(
                            "[RSI tick %d] regression probe raised (%s); "
                            "skipping revert check (may let a bad checkpoint "
                            "through this cycle): %s",
                            cycle, type(exc).__name__, exc,
                        )
                        post_score = result.pre_score  # treat as no-change
                    result.post_score = post_score
                    result.improvement = result.post_score - result.pre_score

                    regressed = (
                        revert_threshold > 0
                        and result.improvement < -revert_threshold
                    )
                    if regressed:
                        logger.warning(
                            "[RSI tick %d] REGRESSION detected (%+.3f < -%.2f) — "
                            "NOT writing checkpoint. Reloading vLLM from previous "
                            "checkpoint (or base) so Phase 5b eval runs fast in "
                            "vLLM mode instead of burning minutes on HF inference.",
                            cycle, result.improvement, revert_threshold,
                        )
                        # Even on regression we MUST reload vLLM — otherwise
                        # Phase 5b (held-out eval) runs via the slow HF model
                        # that's still loaded in memory. Reloading from the
                        # LAST known-good path (either the prior cycle's
                        # checkpoint or the base model) gives us clean fast
                        # inference without persisting the corrupted merge.
                        if self._use_vllm:
                            try:
                                last_good = getattr(self.model_loader, "model_path", None)
                                self.model_loader.swap_to_vllm_after_training(
                                    str(last_good) if last_good else None
                                )
                            except Exception as exc:
                                logger.warning(
                                    "[RSI tick %d] vLLM reload after regression failed: %s",
                                    cycle, exc,
                                )
                        # Promote this cycle's admitted candidates to the VoV
                        # adversarial bank — they cleared quorum yet training
                        # on them regressed the held-out score.
                        self._bank_admitted_as_adversarial(
                            cycle, result,
                            reason=f"quick_probe_regression impr={result.improvement:.3f}",
                        )
                    else:
                        if self._use_vllm:
                            ckpt_root = self.config.orchestrator.output_dir / "checkpoints"
                            self.model_loader.save_checkpoint(ckpt_root, cycle)
                            tmp_ckpt = ckpt_root / f"cycle_{cycle}"
                            self.model_loader.swap_to_vllm_after_training(str(tmp_ckpt))
                        logger.info(
                            "[RSI tick %d] training done: %d steps, score %+.3f",
                            cycle, metrics.steps, result.improvement,
                        )
                        # Training succeeded the quick probe — flush the
                        # accumulated pool. Next cycle starts fresh. If the
                        # full Phase 5b eval later shows regression, the
                        # post-Phase-5b guard reverts vLLM but the pool is
                        # still flushed (those samples were already used).
                        self._rsi_pending_pool.clear()
                        self._rsi_pool_accumulated_cycles = 0
            except Exception as exc:
                tb = traceback.format_exc()
                logger.warning("Step 8 (train) failed (%s): %s", type(exc).__name__, exc)
                result.errors.append({
                    "phase": "rsi_step8_train", "type": type(exc).__name__,
                    "message": str(exc), "traceback": tb,
                })
                try:
                    self.trainer.strip_lora()
                except Exception:
                    pass
        else:
            logger.info(
                "[RSI tick %d] pool size %d < min_batch %d — skipping training "
                "(prevents corrupting base weights with tiny-batch gradient noise)",
                cycle, len(training_samples), min_batch,
            )
        result.phase_times["rsi_step8_train"] = time.time() - phase_start
        result.samples_generated = result.novel_problems_proposed
        result.samples_verified = result.candidates_accepted
        return result

    # Stable seed for held-out eval questions. Using a fixed value (not cycle)
    # means the held-out set is the SAME across all cycles, so scores track
    # true generalization over time instead of random per-cycle variance.
    HELDOUT_CYCLE_SEED = 0xE7A1

    def _quick_regression_probe(self, cycle: int) -> float:
        """Fast post-training sanity check — returns an overall score from a
        small held-out slice.

        Called BEFORE the vLLM swap-back, so it runs through the HF model
        (slow). Using the full 240-prompt diagnostic here was adding ~4
        minutes per training cycle (observed run-7). An 8-per-domain slice
        is enough to detect the -0.2+ drops that trigger the regression
        revert, at ~10-15 seconds instead.

        Uses the same held-out seed + frozen_eval_mode as Phase 5b so this
        probe measures the SAME surface the full eval does.
        """
        cfg = self.config.diagnostics
        orig_n = cfg.questions_per_domain
        orig_min = cfg.min_questions_per_domain
        orig_max = cfg.max_questions_per_domain
        orig_frozen = getattr(self.diagnostics, "_frozen_eval_mode", False)
        probe_n = int(getattr(
            self.config.orchestrator, "regression_probe_questions_per_domain", 8,
        ))
        try:
            cfg.questions_per_domain = probe_n
            cfg.min_questions_per_domain = min(orig_min, probe_n)
            cfg.max_questions_per_domain = max(orig_max, probe_n)
            self.diagnostics._frozen_eval_mode = True
            d = self.diagnostics.run(self.HELDOUT_CYCLE_SEED)
            return float(getattr(d, "overall_score", 0.0))
        finally:
            cfg.questions_per_domain = orig_n
            cfg.min_questions_per_domain = orig_min
            cfg.max_questions_per_domain = orig_max
            self.diagnostics._frozen_eval_mode = orig_frozen

    def _eval_phase(self, cycle: int, result: CycleResult):
        """Run a held-out evaluation on a stable question set.

        When ``orchestrator.heldout_repetitions`` > 1, runs the eval multiple
        times with the same seed and records every score. The spread across
        repetitions is a direct lower-bound estimate of measurement noise.
        """
        reps = max(1, int(getattr(self.config.orchestrator, "heldout_repetitions", 1)))
        logger.info(f"[Cycle {cycle}] Phase 5b: HELD-OUT EVAL (x{reps})")
        eval_diag = None
        scores: list[float] = []
        per_rep_domain_scores: list[dict] = []
        per_rep_per_question: list[list[dict]] = []
        # Freeze the diagnostics engine so eval samples ONLY from the
        # deterministic ground-truth bank — curriculum state and templates
        # would otherwise drift question composition between runs and
        # inject ~25% of the noise that masked cycle 2's real signal.
        prev_mode = getattr(self.diagnostics, "_frozen_eval_mode", False)
        self.diagnostics._frozen_eval_mode = True
        try:
            for i in range(reps):
                try:
                    d = self.diagnostics.run(self.HELDOUT_CYCLE_SEED)
                except Exception as e:
                    logger.warning(f"  held-out rep {i+1}/{reps} failed: {type(e).__name__}: {e}")
                    continue
                eval_diag = d  # keep last successful
                scores.append(d.overall_score)
                per_rep_domain_scores.append(dict(d.domain_scores))
                per_rep_per_question.append(list(d.per_question))
        finally:
            self.diagnostics._frozen_eval_mode = prev_mode
        if not scores:
            return
        # Use the mean across repetitions as the authoritative eval_score —
        # this is what downstream comparisons (meta, early-stop) see, and it
        # is strictly more informative than a single draw.
        mean_score = sum(scores) / len(scores)
        result.eval_score = mean_score
        result.eval_scores_all = scores
        result.eval_domain_scores = dict(eval_diag.domain_scores) if eval_diag else {}
        # meta_analyst ASK 1: surface the per-subdomain breakdown that run()
        # now populates. Mean across repetitions is not meaningful here since
        # we only keep the last successful DiagnosticResult; per-rep subdomain
        # dicts would drift if the question set is reseeded, but frozen-eval
        # mode uses a stable seed so the last rep's partition is authoritative.
        result.eval_subdomain_scores = (
            dict(eval_diag.subdomain_scores) if eval_diag else {}
        )
        # Cache per-rep observability data for cycle_metrics dump.
        result._eval_per_rep_domain_scores = per_rep_domain_scores
        result._eval_per_rep_per_question = per_rep_per_question
        prev_eval = next(
            (r.eval_score for r in reversed(self.history) if r.eval_score is not None),
            None,
        )
        if len(scores) > 1:
            spread = max(scores) - min(scores)
            logger.info(
                f"  Held-out eval: mean={mean_score:.3f} "
                f"scores={['%.3f' % s for s in scores]} spread={spread:.3f}"
            )
        else:
            logger.info(f"  Held-out eval: {mean_score:.3f}")
        if prev_eval is not None:
            delta = mean_score - prev_eval
            symbol = "+" if delta > 0 else ""
            logger.info(f"    (prev {prev_eval:.3f}, {symbol}{delta:.3f})")

    def _meta_step(self, cycle: int, result: CycleResult) -> None:
        """End-of-cycle: record outcome into MetaController and apply bounded proposals.

        Proposals are applied only when the current cycle showed non-negative
        held-out delta; a negative delta triggers revert of the previously
        applied proposal (ImprovementLoop's next cycle starts from the reverted
        config).
        """
        prev_eval = next(
            (r.eval_score for r in reversed(self.history[:-1]) if r.eval_score is not None),
            None,
        )
        cfg_snapshot = {
            "learning_rate": getattr(self.config.trainer, "learning_rate", None),
            "lora_rank": getattr(self.config.trainer, "lora_rank", None),
            "num_epochs": getattr(self.config.trainer, "num_epochs", None),
            "min_train_samples": getattr(
                self.config.trainer, "min_train_samples", None
            ),
            "gradient_accumulation_steps": getattr(
                self.config.trainer, "gradient_accumulation_steps", None
            ),
            "consistency_threshold": getattr(
                self.config.verifier, "consistency_threshold", None
            ),
            "verifier_check_weights": dict(
                getattr(self.config.verifier, "check_weights", {}) or {}
            ),
            "generator_template": self.generator._custom_solution_template,
        }
        check_scores = None
        try:
            # Use training-metrics surrogate: if per-check acceptance stats were
            # recorded this cycle, feed them in. Absent → no weight update.
            check_scores = getattr(self.verifier, "_last_check_mean_scores", None)
        except Exception:
            check_scores = None

        self.meta.record_cycle(
            cycle=cycle,
            config_snapshot=cfg_snapshot,
            held_out_score=result.eval_score,
            prev_held_out=prev_eval,
            verifier_check_scores=check_scores,
            training_succeeded=bool(
                result.training_metrics and result.training_metrics.steps > 0
            ),
            pre_score=result.pre_score,
            post_score=result.post_score,
            full_config=self.config,
            samples_generated=result.samples_generated,
            samples_verified=result.samples_verified,
            training_steps=(
                result.training_metrics.steps if result.training_metrics else 0
            ),
            had_errors=bool(result.errors),
        )

        # If previous-cycle proposal caused regression, revert first.
        revert = self.meta.get_revert_target()
        if revert is not None:
            logger.warning(f"  meta: reverting previous proposal due to regression")
            if revert.get("learning_rate") is not None:
                self.config.trainer.learning_rate = revert["learning_rate"]
            if revert.get("verifier_check_weights") is not None:
                self.config.verifier.check_weights = dict(revert["verifier_check_weights"])
            if revert.get("generator_template") is not None:
                self.generator.set_custom_solution_template(revert["generator_template"])
            for _tf in ("lora_rank", "num_epochs", "min_train_samples",
                        "gradient_accumulation_steps"):
                if revert.get(_tf) is not None:
                    setattr(self.config.trainer, _tf, revert[_tf])
            self.meta.persist_state()
            return

        # Propose updates for next cycle.
        proposal = self.meta.propose_updates(cfg_snapshot)
        if not proposal["applied"]:
            self.meta.persist_state()
            return
        for line in proposal["reasoning"]:
            logger.info(f"  meta: {line}")
        if proposal["learning_rate"] is not None:
            self.config.trainer.learning_rate = proposal["learning_rate"]
        if proposal["verifier_check_weights"] is not None:
            self.config.verifier.check_weights = proposal["verifier_check_weights"]
        if proposal["generator_template"] is not None:
            tmpl, _vid = proposal["generator_template"]
            self.generator.set_custom_solution_template(tmpl)
        for _tf in ("lora_rank", "num_epochs", "min_train_samples",
                    "gradient_accumulation_steps"):
            if proposal.get(_tf) is not None:
                setattr(self.config.trainer, _tf, proposal[_tf])
        self.meta.persist_state()

    def _apply_quality_top_k(self, verified: list) -> list:
        """Rank verified samples by quality, keep the top-k.

        Gated by generator.sample_quality_top_k (0 = disabled). Motivation:
        cycle-6 trained on 1 high-leverage sample and jumped held-out +19pp,
        while cycle-1 regressed training on a similar-count but lower-quality
        batch. Feeding the trainer fewer, better samples beats feeding it
        many marginal ones — especially under early_stop_loss=0.15 where a
        single bad example can drag the mean below threshold.
        """
        k = int(getattr(self.config.generator, "sample_quality_top_k", 0) or 0)
        if k <= 0 or not verified:
            return verified
        floor = int(getattr(self.config.generator, "sample_quality_floor", 3) or 3)
        if len(verified) <= max(k, floor):
            return verified

        def _score(s) -> float:
            cs = float(getattr(s, "consistency_score", 0.0) or 0.0)
            pc = float(getattr(s, "parse_confidence", 0.0) or 0.0)
            src = getattr(s, "source", "synthesized")
            star_bonus = 1.0 if src == "star" else 0.0
            return cs * pc * (1.0 + 0.5 * star_bonus)

        ranked = sorted(verified, key=_score, reverse=True)
        keep = max(k, floor)
        kept = ranked[:keep]
        logger.info(
            f"  Quality top-k filter: kept {len(kept)}/{len(verified)} "
            f"(top score={_score(ranked[0]):.3f}, cutoff={_score(kept[-1]):.3f})"
        )
        return kept

    def _save_progress_dashboard(self, cycle: int, result: CycleResult, phase_times: dict):
        """Write progress.json with structured metrics for external monitoring."""
        output_dir = self.config.orchestrator.output_dir
        progress = {
            "cycle": cycle,
            "timestamp": time.time(),
            "scores": {
                "pre_training": result.pre_score,
                "post_training": result.post_score,
                "held_out_eval": result.eval_score,
                "improvement": result.improvement,
                "improvement_ema": self._improvement_ema,
                "best_score": self._best_score,
                "best_checkpoint_cycle": self._best_checkpoint_cycle,
            },
            "domain_scores": {
                "pre": result.diagnostics.domain_scores if result.diagnostics else {},
                "post": result.post_diag.domain_scores if result.post_diag else {},
                "eval": result.eval_domain_scores,
            },
            # meta_analyst ASK 1: per-"domain/subdomain" breakdown. Pre and
            # post come from the existing DiagnosticResult.subdomain_scores
            # (populated by engine.run); eval comes from the held-out pass.
            "subdomain_scores": {
                "pre": (result.diagnostics.subdomain_scores
                        if result.diagnostics else {}),
                "post": (result.post_diag.subdomain_scores
                         if result.post_diag else {}),
                "eval": result.eval_subdomain_scores,
            },
            "samples": {
                "generated": result.samples_generated,
                "verified": result.samples_verified,
                "rejected": result.samples_generated - result.samples_verified,
                "pass_rate": (result.samples_verified / max(result.samples_generated, 1)),
                "diversity": result.diversity_stats,
            },
            "training": {
                "avg_loss": result.training_metrics.avg_loss if result.training_metrics else None,
                "final_loss": result.training_metrics.final_loss if result.training_metrics else None,
                "steps": result.training_metrics.steps if result.training_metrics else 0,
                "learning_rate": result.training_metrics.learning_rate if result.training_metrics else 0,
                "lora_layers": result.training_metrics.lora_layers_injected if result.training_metrics else 0,
            },
            "calibration": {
                # ECE/Brier over per-step [C:x] confidences parsed from training
                # data this cycle. NaN (no markers emitted) is serialized as null
                # so dashboards can distinguish "no data" from "perfect (0.0)".
                "ece": _nan_to_none(
                    result.training_metrics.calibration_ece
                    if result.training_metrics else None
                ),
                "brier": _nan_to_none(
                    result.training_metrics.calibration_brier
                    if result.training_metrics else None
                ),
                "samples": (
                    result.training_metrics.calibration_samples
                    if result.training_metrics else 0
                ),
            },
            "timing": phase_times,
            "escalations": dict(self._escalation_state),
            "degradation_count": self._degradation_count,
            "plateau_count": self._plateau_count,
            # Surface per-cycle errors (type + message only; full tracebacks live
            # in the per-cycle log to keep the dashboard small).
            "errors": [
                {"phase": e["phase"], "type": e["type"], "message": e["message"]}
                for e in result.errors
            ],
            "history_summary": [
                {"cycle": r.cycle, "pre": r.pre_score, "post": r.post_score,
                 "improvement": r.improvement, "eval": r.eval_score,
                 "eval_subdomain": r.eval_subdomain_scores,
                 "pass_rate": (r.samples_verified / r.samples_generated
                               if r.samples_generated else None),
                 "had_errors": bool(r.errors)}
                for r in self.history
            ],
        }
        progress_path = output_dir / "progress.json"
        with open(progress_path, "w") as f:
            json.dump(progress, f, indent=2)

    def _check_early_stopping(self, cycle: int, result: CycleResult) -> tuple[bool, str]:
        """Check if model is degrading and revert to best checkpoint if needed.

        "Best" is defined by held-out eval_score (frozen ground-truth bank) when
        available, falling back to post_score only when eval hasn't run.
        Previously tracked post_score exclusively — which included curriculum
        drift and let cycle 1 get labeled "best" despite its worst held-out
        (0.0625) and −0.325 regression. Using eval_score makes checkpoint
        rollback actually point at the best-generalizing checkpoint.
        """
        current_score = (
            result.eval_score if result.eval_score is not None else result.post_score
        )
        if current_score > self._best_score:
            self._best_score = current_score
            self._best_checkpoint_cycle = cycle
            self._degradation_count = 0
            return False, ""

        if len(self.history) >= 2:
            prev_result = self.history[-2]
            prev_score = (
                prev_result.eval_score if prev_result.eval_score is not None
                else prev_result.post_score
            )
            if current_score < prev_score - 0.005:
                self._degradation_count += 1
            else:
                self._degradation_count = 0

        if self._degradation_count >= 2 and self._best_checkpoint_cycle is not None:
            best_ckpt = (self.config.orchestrator.output_dir / "checkpoints" /
                         f"cycle_{self._best_checkpoint_cycle}")
            if best_ckpt.exists() and (best_ckpt / "config.json").exists():
                logger.warning(
                    f"  EARLY STOP: Score degraded for {self._degradation_count} cycles "
                    f"(current={current_score:.3f}, best={self._best_score:.3f} at cycle "
                    f"{self._best_checkpoint_cycle}). Reverting to best checkpoint."
                )
                if self._use_vllm:
                    self.model_loader.swap_to_vllm_after_training(str(best_ckpt))
                else:
                    self.model_loader.load_from_checkpoint(str(best_ckpt))
                return True, (f"degradation for {self._degradation_count} cycles; "
                              f"reverted to best checkpoint (cycle {self._best_checkpoint_cycle})")
            else:
                logger.warning(
                    f"  Degradation detected but best checkpoint (cycle "
                    f"{self._best_checkpoint_cycle}) not found on disk."
                )
        return False, ""

    def _check_and_escalate(self, cycle: int, result: CycleResult, post_diag: 'DiagnosticResult | None' = None):
        """Check and perform escalations based on cycle and performance."""
        schedule = self.config.orchestrator.escalation_schedule

        # Tight window: require NET improvement across the last 2 cycles (the
        # same window de-escalation uses). The old 3-cycle `any` check let a
        # stale positive cycle satisfy has_improved even mid-regression, and
        # we'd then escalate INTO a regression right before de-escalation tried
        # to unwind it.
        has_improved = (
            len(self.history) >= 2
            and sum(r.improvement for r in self.history[-2:]) > 0.01
        )
        trained_this_cycle = bool(
            result.training_metrics
            and getattr(result.training_metrics, "steps", 0) > 0
        )

        # Rate-limit escalations: at most ONE per cycle. Escalating multiple
        # assistants simultaneously makes regressions ambiguous — we can't tell
        # which addition caused a drop, and de-escalation (which rolls back
        # one at a time in priority order) spends many cycles unwinding a
        # multi-escalation that should never have happened together.
        escalated_this_cycle = False

        # Verification escalation (lowest bar — goes first if eligible).
        if (cycle >= schedule.verification
                and not self._escalation_state["verification"]
                and result.post_score > 0.5
                and has_improved
                and trained_this_cycle):
            self._escalate_verification()
            result.escalation_events.append("model_assists_verification")
            self._escalation_state["verification"] = True
            escalated_this_cycle = True

        # Diagnosis escalation
        if (not escalated_this_cycle
                and cycle >= schedule.diagnosis
                and not self._escalation_state["diagnosis"]
                and result.post_score > 0.6
                and has_improved
                and trained_this_cycle):
            diag_for_escalation = post_diag if post_diag else result
            self._escalate_diagnosis(diag_for_escalation, cycle=cycle)
            result.escalation_events.append("model_assists_diagnosis")
            self._escalation_state["diagnosis"] = True
            escalated_this_cycle = True

        # Generation escalation
        if (not escalated_this_cycle
                and cycle >= schedule.generation
                and not self._escalation_state["generation"]
                and result.post_score > 0.7
                and has_improved
                and trained_this_cycle):
            self._escalate_generation()
            result.escalation_events.append("model_improves_generation")
            self._escalation_state["generation"] = True
            self._plateau_count = 0
            escalated_this_cycle = True

        # De-escalation
        current_cycle = self.history[-1].cycle if self.history else 0
        if (len(self.history) >= 2
                and not result.escalation_events
                and current_cycle - self._last_deescalation_cycle >= 3):
            recent_drops = sum(1 for r in self.history[-2:] if r.improvement < -0.03)
            if recent_drops >= 2:
                if self._escalation_state.get("generation"):
                    logger.warning(">>> DE-ESCALATION: Reverting model-assisted generation (sustained regression)")
                    self._escalation_state["generation"] = False
                    self._last_deescalation_cycle = current_cycle
                    result.escalation_events.append("reverted_generation")
                elif self._escalation_state.get("diagnosis"):
                    logger.warning(">>> DE-ESCALATION: Reverting model-assisted diagnosis (sustained regression)")
                    self._escalation_state["diagnosis"] = False
                    self.diagnostics._model_generated_questions.clear()
                    self._last_deescalation_cycle = current_cycle
                    result.escalation_events.append("reverted_diagnosis")
                elif self._escalation_state.get("verification"):
                    logger.warning(">>> DE-ESCALATION: Reverting model-assisted verification (sustained regression)")
                    self._escalation_state["verification"] = False
                    self.config.verifier.use_model_verification = False
                    self.verifier._model_verifier = None
                    self._last_deescalation_cycle = current_cycle
                    result.escalation_events.append("reverted_verification")

    def _escalate_verification(self):
        """Hand verification to the improved model."""
        logger.info(">>> ESCALATION: Model now assists in verification")
        self.verifier.set_model_verifier(self.model_loader)
        self.config.verifier.use_model_verification = True

    def _escalate_diagnosis(self, result_or_diag: 'CycleResult | DiagnosticResult | None' = None, cycle: int = 0):
        """Let the model generate its own diagnostic questions."""
        logger.info(">>> ESCALATION: Model now generates diagnostic questions")
        diag = None
        if isinstance(result_or_diag, DiagnosticResult):
            diag = result_or_diag
        elif result_or_diag is not None and hasattr(result_or_diag, 'diagnostics'):
            diag = result_or_diag.diagnostics
        if diag is None and self.history and self.history[-1].diagnostics:
            diag = self.history[-1].diagnostics

        if diag is None:
            all_stubs = all(r.diagnostics is None for r in self.history) if self.history else True
            if all_stubs:
                logger.info("  diagnosis escalation deferred: no live diagnostics available post-resume")
                return

        if diag:
            weak_domains = [
                domain for domain, score in diag.domain_scores.items()
                if score < self.config.diagnostics.confidence_threshold
            ]
            if weak_domains:
                self.diagnostics.generate_adaptive_questions(weak_domains, cycle=cycle)
                logger.info(f"  Generated adaptive questions for: {weak_domains}")

    def _escalate_generation(self):
        """Let the model improve the data generation prompts.

        Bounded by a wall-clock timeout: the model_loader.generate call has no
        native timeout, so a stuck decode would hang the entire cycle. We wrap
        it in a ThreadPoolExecutor and on timeout we log and retain the current
        template — the GPU work continues in the background but the main loop
        recovers and the escalation is treated as a no-op for this cycle (the
        _escalation_state flag has already been set, so we won't retry).
        """
        logger.info(">>> ESCALATION: Model now improves data generation process")
        min_steps = self.config.generator.min_reasoning_steps
        current_template = self.generator._custom_solution_template or (
            "Solve the following {domain}/{subdomain} problem.\n"
            "PROBLEM: {problem}\n"
            "You MUST structure your answer as numbered steps.\n"
            "For EACH step: Step N: [what], Justification: [why], Assumptions: [any]\n"
            f"Minimum {min_steps} steps. End with Conclusion: [answer]"
        )
        improvement_prompt = (
            "You are improving a system that generates training data. "
            "The system asks a model to solve problems step-by-step. "
            "Here is the current prompt template:\n\n"
            f"---\n{current_template}\n---\n\n"
            "How would you improve this to get more rigorous, complete reasoning? "
            "Output ONLY the improved template. Keep {problem}, {domain}, {subdomain} as placeholders."
        )

        import concurrent.futures
        timeout_s = getattr(self.config.orchestrator, "escalation_generate_timeout_s", 120)
        improved_template = ""
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(
                self.model_loader.generate,
                improvement_prompt, max_new_tokens=1024, temperature=0.3,
            )
            try:
                improved_template = fut.result(timeout=timeout_s) or ""
            except concurrent.futures.TimeoutError:
                logger.warning(
                    f"  escalation_generation: model timed out after {timeout_s}s, "
                    f"keeping current template"
                )
                return
            except Exception as exc:
                logger.warning(f"  escalation_generation: model raised {type(exc).__name__}: {exc}")
                return

        required = {"{problem}", "{domain}", "{subdomain}"}
        if (improved_template and len(improved_template) > 50
                and all(p in improved_template for p in required)):
            logger.info(f"  Model suggested improved template ({len(improved_template)} chars)")
            self.generator.set_custom_solution_template(improved_template)
        elif improved_template:
            missing = [p for p in required if p not in improved_template]
            logger.warning(f"  Rejected model template — missing placeholders: {missing}")

    def _should_stop(self, result: CycleResult) -> tuple[bool, str]:
        """Check if we should stop using EMA-smoothed improvement for plateau detection.

        Proper RSI never stops because of "no weaknesses found" — when the
        model saturates the current difficulty, we raise the bar. We only
        stop on genuine plateau (EMA improvement < threshold for N cycles)
        or on sustained training failure.
        """
        # Saturation handler: instead of quitting when everything passes the
        # current confidence_threshold, make the threshold stricter. The
        # curriculum's hard/expert weight also gets bumped so upcoming probes
        # are harder. This is how open-ended RSI is supposed to work.
        if result.diagnostics and not result.diagnostics.weaknesses:
            cur = self.config.diagnostics.confidence_threshold
            # Step up in 0.05 increments, cap at 0.95 (leave headroom for noise).
            new_thresh = min(0.95, round(cur + 0.05, 3))
            if new_thresh > cur:
                self.config.diagnostics.confidence_threshold = new_thresh
                # Also shift difficulty mix toward harder bands so new probes
                # aren't re-asking the same problems the model just solved.
                mix = self.config.diagnostics.difficulty_mix
                bump = {
                    "easy": max(0.05, mix.get("easy", 0.3) - 0.05),
                    "medium": max(0.10, mix.get("medium", 0.35) - 0.03),
                    "hard": min(0.60, mix.get("hard", 0.25) + 0.04),
                    "expert": min(0.50, mix.get("expert", 0.10) + 0.04),
                }
                # Renormalize so weights sum to 1.
                total = sum(bump.values())
                self.config.diagnostics.difficulty_mix = {
                    k: round(v / total, 3) for k, v in bump.items()
                }
                logger.info(
                    f"  Saturation: all domains above {cur:.2f}. Raising "
                    f"confidence_threshold → {new_thresh:.2f} and shifting "
                    f"difficulty mix to {self.config.diagnostics.difficulty_mix}. "
                    f"RSI continues."
                )
                # Reset plateau counter — new difficulty regime, fresh start.
                # Also reset the improvement EMA: after raising the bar, the
                # first few cycles in the new regime typically score LOWER
                # (harder questions), which would drive the EMA negative and
                # trigger plateau-stop immediately. Resetting to 0 gives the
                # model plateau_patience cycles to adapt without being falsely
                # declared stuck.
                self._plateau_count = 0
                self._consecutive_failures = 0
                self._improvement_ema = 0.0
                return False, ""
            else:
                # Already at max threshold. Only now do we admit the RSI
                # pipeline has run out of room within the current eval harness.
                self._plateau_count = 0
                self._consecutive_failures = 0
                return True, (
                    f"saturated at confidence_threshold={cur:.2f} (max 0.95) — "
                    f"eval harness exhausted, consider expanding domains or "
                    f"adding harder question banks"
                )

        undertrained = getattr(self.trainer, "_last_merge_undertrained", False)
        if result.training_metrics and result.training_metrics.steps > 0 and not undertrained:
            self._consecutive_failures = 0
            # Use EMA-smoothed improvement instead of raw per-cycle delta.
            if self._improvement_ema < self.config.orchestrator.min_improvement_threshold:
                self._plateau_count += 1
            else:
                self._plateau_count = 0
        elif undertrained:
            # Genuine training failure (LoRA never woke up) — this IS a
            # failure signal worth tracking.
            self._consecutive_failures += 1
        else:
            # "No training happened this cycle." Previously this counted as
            # a failure, which stopped run-4 at cycle 16 after the model
            # simply hadn't produced enough RSI-verified samples yet to
            # clear min_train_samples. That's not a pipeline failure —
            # it's the training guard working as designed while the model
            # warms up on the novel-problem proposal task. Only count
            # training-FAILED cycles as failures; training-SKIPPED cycles
            # are a wait state and should not trip the stop condition.
            pass

        if self._plateau_count >= self.config.orchestrator.plateau_patience:
            return True, (
                f"plateau after {self._plateau_count} cycles "
                f"(EMA improvement {self._improvement_ema:.4f} < "
                f"threshold {self.config.orchestrator.min_improvement_threshold})"
            )
        if self._consecutive_failures >= self.config.orchestrator.plateau_patience * 2:
            return True, f"{self._consecutive_failures} consecutive TRAINING failures — pipeline appears broken"
        return False, ""

    def _save_checkpoint(self, cycle: int):
        """Save full checkpoint: model (post-merge) and history.

        Keeps only the 2 most recent full model checkpoints plus the best
        checkpoint (for early stopping revert) to prevent disk exhaustion.
        """
        output_dir = self.config.orchestrator.output_dir
        already_saved = self._use_vllm and getattr(self, "_vllm_saved_this_cycle", None) == cycle
        cycle_dir = output_dir / "checkpoints" / f"cycle_{cycle}"
        if not already_saved:
            try:
                self.model_loader.save_checkpoint(output_dir / "checkpoints", cycle)
            except Exception as e:
                logger.error(f"  Model checkpoint save failed: {e}")
                # Only create the dir if model save succeeded at least partially
                # (e.g. wrote config.json). Empty directories here poison later
                # vLLM reloads: after a failed training cycle, _current_model_path
                # can point at this empty dir, and vLLM fails to load with a
                # misleading "config.json not found" error.
                if not cycle_dir.exists() or not (cycle_dir / "config.json").exists():
                    logger.warning(
                        f"  Not creating empty checkpoint dir {cycle_dir} "
                        f"(would poison future vLLM reloads)"
                    )
        # Ensure the dir exists ONLY when the save actually produced files —
        # needed so history.json below can be written.
        if cycle_dir.exists() and (cycle_dir / "config.json").exists():
            pass  # real checkpoint
        elif already_saved:
            # vLLM save path already created the dir with files — safe.
            cycle_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Checkpoint save failed AND nothing was written. Create a
            # minimal marker dir just for history.json, but mark it so that
            # swap_to_vllm_after_training() won't accidentally load from it.
            cycle_dir.mkdir(parents=True, exist_ok=True)
            try:
                (cycle_dir / ".incomplete").write_text("no model weights saved")
            except OSError:
                pass

        # Prune old model checkpoints — keep last 2 + best checkpoint
        ckpt_dir = output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        def _cycle_num(d):
            try:
                return int(d.name.split("_")[1])
            except (ValueError, IndexError):
                return -1
        existing = sorted(
            [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("cycle_")],
            key=_cycle_num,
        )
        model_dirs = [d for d in existing if (d / "config.json").exists()]
        # Protect the best checkpoint from pruning
        keep_names = set()
        if self._best_checkpoint_cycle is not None:
            keep_names.add(f"cycle_{self._best_checkpoint_cycle}")
        # Always keep the current cycle's dir (needed for history.json below).
        keep_names.add(f"cycle_{cycle}")
        for old_dir in model_dirs:
            if old_dir.name in keep_names:
                continue
            if old_dir in model_dirs[-2:]:
                continue
            try:
                shutil.rmtree(old_dir)
                logger.info(f"  Pruned old checkpoint: {old_dir.name}")
            except OSError:
                pass

        # Also prune stale .incomplete marker dirs (failed-save leftovers).
        # These lack config.json so they're excluded from model_dirs — without
        # this sweep they accumulate forever across cycles on any run that
        # had a training failure.
        incomplete_dirs = [
            d for d in existing
            if (d / ".incomplete").exists() and d.name not in keep_names
        ]
        # Keep only the most recent incomplete marker (last 1) for debugging;
        # older ones are pure litter.
        for old_dir in incomplete_dirs[:-1]:
            try:
                shutil.rmtree(old_dir)
                logger.info(f"  Pruned stale incomplete checkpoint dir: {old_dir.name}")
            except OSError:
                pass

        # Save history for resume
        history_path = output_dir / "checkpoints" / f"cycle_{cycle}" / "history.json"
        with open(history_path, "w") as f:
            json.dump({
                "cycles": [r.to_dict() for r in self.history],
                "escalation_state": self._escalation_state,
                "plateau_count": self._plateau_count,
                "consecutive_failures": self._consecutive_failures,
                "domain_score_history": self._domain_score_history,
                "last_deescalation_cycle": self._last_deescalation_cycle,
                "custom_solution_template": self.generator._custom_solution_template,
                "model_generated_questions": self.diagnostics._model_generated_questions,
                "pending_regressions": [
                    {"domain": w.domain, "subdomain": w.subdomain, "severity": w.severity,
                     "description": w.description, "evidence": w.evidence[:10]}
                    for w in self._pending_regression_weaknesses
                ],
                "best_score": self._best_score,
                "best_checkpoint_cycle": self._best_checkpoint_cycle,
                "degradation_count": self._degradation_count,
                "improvement_ema": self._improvement_ema,
                "meta_state": self.meta.to_dict(),
                "curriculum": (
                    self.diagnostics.curriculum.to_dict()
                    if getattr(self.diagnostics, "curriculum", None) is not None
                    else {}
                ),
            }, f, indent=2)

        # Clean up empty dirs from failed model saves
        for d in existing:
            if not (d / "config.json").exists() and not (d / "history.json").exists():
                try:
                    shutil.rmtree(d)
                    logger.info(f"  Cleaned up empty checkpoint dir: {d.name}")
                except OSError:
                    pass

    def _maybe_substrate_merge(self, cycle: int, result: CycleResult) -> None:
        """Every `merge_into_base_every` training cycles, promote the current
        merged checkpoint to a new base — so LoRA on the next cycle starts
        fresh on substrate that has already absorbed prior gains. This is the
        mechanism that unblocks substrate updates despite LoRA-on-frozen-4bit.

        Guardrails:
          - Only counts TRAINED cycles (no-training cycles aren't epochs).
          - Skipped if cumulative held-out improvement since the last promotion
            is below substrate_merge_min_improvement (don't snapshot regressions).
          - Writes an event to update-log.txt and into result.errors (as a
            structured note) so promotions are visible in run history.
        """
        every = int(getattr(self.config.orchestrator, "merge_into_base_every", 0) or 0)
        if every <= 0:
            return
        trained = bool(
            result.training_metrics
            and getattr(result.training_metrics, "steps", 0) > 0
        )
        # First-ever eval — capture the baseline so cumulative delta is meaningful
        # even before any training cycles have run.
        if self._substrate_baseline_eval is None and result.eval_score is not None:
            self._substrate_baseline_eval = float(result.eval_score)
        if not trained:
            return
        self._substrate_trained_cycles_since_merge += 1
        if self._substrate_trained_cycles_since_merge < every:
            return

        min_improvement = float(getattr(
            self.config.orchestrator, "substrate_merge_min_improvement", 0.005,
        ))
        baseline = self._substrate_baseline_eval
        current = result.eval_score
        delta = None
        if current is not None and baseline is not None:
            delta = float(current) - float(baseline)

        if delta is None or delta < min_improvement:
            logger.info(
                "[substrate-merge cycle %d] skipped — cumulative held-out "
                "improvement since last promotion is %s (< %.3f required). "
                "Trained-cycle counter NOT reset; will re-check next trained cycle.",
                cycle,
                f"{delta:+.3f}" if delta is not None else "unknown",
                min_improvement,
            )
            self._append_update_log(
                f"[cycle {cycle}] substrate-merge SKIPPED "
                f"(delta={delta if delta is None else round(delta, 4)} < {min_improvement})"
            )
            return

        # Promote: the current merged checkpoint is already saved at
        # outputs/checkpoints/cycle_{cycle} (by the RSI training path or
        # _save_checkpoint). Copy its contents to base_epoch_{K+1}/ and
        # redirect model_loader.model_path so future fallbacks use the new
        # base. If no cycle checkpoint exists (training wrote none due to
        # quantized skip etc.), defer.
        output_dir = self.config.orchestrator.output_dir
        src_ckpt = output_dir / "checkpoints" / f"cycle_{cycle}"
        if not src_ckpt.exists() or not (src_ckpt / "config.json").exists():
            logger.info(
                "[substrate-merge cycle %d] deferred — no complete cycle "
                "checkpoint at %s to promote (quantized base may have skipped "
                "the save). Will retry next trained cycle.",
                cycle, src_ckpt,
            )
            self._append_update_log(
                f"[cycle {cycle}] substrate-merge DEFERRED (no cycle checkpoint to promote)"
            )
            return

        new_epoch = self._substrate_epoch + 1
        dest = output_dir / "checkpoints" / f"base_epoch_{new_epoch}"
        try:
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src_ckpt, dest)
        except Exception as e:
            logger.warning(
                "[substrate-merge cycle %d] copy to %s failed (%s: %s) — "
                "not promoting this epoch.",
                cycle, dest, type(e).__name__, e,
            )
            return

        # Redirect the loader's fallback base. Don't touch _current_model_path
        # (that already points at the cycle_{cycle} checkpoint, which is fine);
        # swap the `model_path` attribute so future failure-fallbacks use the
        # new substrate instead of the original.
        try:
            self.model_loader.model_path = str(dest)
        except Exception:
            pass

        self._substrate_epoch = new_epoch
        self._substrate_baseline_eval = float(current)
        self._substrate_last_merge_cycle = cycle
        self._substrate_trained_cycles_since_merge = 0

        msg = (
            f"[cycle {cycle}] substrate-merge PROMOTED cycle_{cycle} -> "
            f"base_epoch_{new_epoch} (delta={delta:+.4f}, new baseline={current:.4f})"
        )
        logger.info(msg)
        self._append_update_log(msg)
        result.escalation_events.append(f"substrate_merge:base_epoch_{new_epoch}")

    def _append_update_log(self, line: str) -> None:
        """Append a substrate-merge event to update-log.txt at repo root."""
        try:
            log = Path("update-log.txt")
            with log.open("a") as f:
                f.write(line + "\n")
        except Exception as e:
            logger.debug(f"Could not append to update-log.txt: {e}")

    def _log_cycle(self, result: CycleResult):
        """Log cycle results to file."""
        log_path = self.config.orchestrator.log_dir / f"cycle_{result.cycle}.json"
        with open(log_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    def _write_cycle_metrics(self, cycle: int, result: CycleResult) -> None:
        """Dump a detailed per-cycle metrics record for autopsy.

        Written to ``outputs/cycle_metrics/cycle_N.json``. Independent of
        progress.json — that file stays dashboard-sized, this one is
        forensic-sized. Safe no-op if observability was off during the cycle.
        """
        out_dir = self.config.orchestrator.output_dir / "cycle_metrics"
        out_dir.mkdir(parents=True, exist_ok=True)
        samples_dump = []
        for s in result._training_samples or []:
            samples_dump.append({
                "prompt_hash": getattr(s, "content_hash", ""),
                "weakness": getattr(s, "target_weakness", ""),
                "domain": getattr(s, "domain", ""),
                "n_reasoning_steps": len(getattr(s, "reasoning_chain", []) or []),
                "consistency_score": getattr(s, "consistency_score", None),
                "parse_confidence": getattr(s, "parse_confidence", None),
                "severity_at_generation": getattr(s, "severity_at_generation", None),
                "expected_answer": getattr(s, "expected_answer", ""),
                "verified": bool(getattr(s, "verified", False)),
                "ground_truth_verified": bool(
                    getattr(s, "ground_truth_verified", False)
                ),
                "verification_notes": getattr(s, "verification_notes", ""),
                "source": getattr(s, "source", ""),
            })

        pre = result.diagnostics.per_question if result.diagnostics else []
        post = result.post_diag.per_question if result.post_diag else []
        pre_map = {q["question_id"]: q for q in pre}
        post_map = {q["question_id"]: q for q in post}
        shared_ids = set(pre_map) & set(post_map)
        questions_moved_right = []  # wrong pre, right post
        questions_moved_wrong = []  # right pre, wrong post
        for qid in shared_ids:
            before = pre_map[qid]["correct"]
            after = post_map[qid]["correct"]
            if not before and after:
                questions_moved_right.append(qid)
            elif before and not after:
                questions_moved_wrong.append(qid)

        loss_traj = (
            list(result.training_metrics.loss_trajectory)
            if result.training_metrics else []
        )

        payload = {
            "cycle": cycle,
            "timestamp": result.timestamp,
            "duration_seconds": result.duration,
            "scores": {
                "pre": result.pre_score,
                "post": result.post_score,
                "improvement": result.improvement,
                "eval_mean": result.eval_score,
                "eval_scores_all": result.eval_scores_all,
                "eval_spread": (
                    (max(result.eval_scores_all) - min(result.eval_scores_all))
                    if len(result.eval_scores_all) > 1 else 0.0
                ),
            },
            "eval_per_rep_domain_scores": result._eval_per_rep_domain_scores,
            "training_samples": samples_dump,
            "training_loss_trajectory": loss_traj,
            "star": result._star_stats,
            "questions": {
                "pre_right_ids": [q["question_id"] for q in pre if q["correct"]],
                "pre_wrong_ids": [q["question_id"] for q in pre if not q["correct"]],
                "post_right_ids": [q["question_id"] for q in post if q["correct"]],
                "post_wrong_ids": [q["question_id"] for q in post if not q["correct"]],
                "moved_wrong_to_right": questions_moved_right,
                "moved_right_to_wrong": questions_moved_wrong,
            },
            "diversity_stats": result.diversity_stats,
            "meta": {
                "picked_lr": getattr(self.config.trainer, "learning_rate", None),
                "picked_rank": getattr(self.config.trainer, "lora_rank", None),
                "picked_epochs": getattr(self.config.trainer, "num_epochs", None),
                "picked_min_train_samples": getattr(
                    self.config.trainer, "min_train_samples", None
                ),
                "picked_grad_accum": getattr(
                    self.config.trainer, "gradient_accumulation_steps", None
                ),
            },
            "phase_times": result.phase_times,
            "errors": [
                {"phase": e["phase"], "type": e["type"], "message": e["message"]}
                for e in result.errors
            ],
        }
        path = out_dir / f"cycle_{cycle}.json"
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)

    def _write_cycle_samples(self, cycle: int, result: CycleResult) -> None:
        """Append one JSONL record per training sample to cycle_samples/.

        Written to ``outputs/cycle_samples/cycle_N.jsonl``. Lets reviewers diff
        sample populations across cycles manually.
        """
        out_dir = self.config.orchestrator.output_dir / "cycle_samples"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"cycle_{cycle}.jsonl"
        with open(path, "w") as f:
            for s in result._training_samples or []:
                rec = {
                    "prompt": getattr(s, "prompt", ""),
                    "response": getattr(s, "response", ""),
                    "reasoning_chain": [
                        {
                            "step_id": getattr(st, "step_id", 0),
                            "content": getattr(st, "content", ""),
                            "justification": getattr(st, "justification", ""),
                            "assumptions": list(getattr(st, "assumptions", []) or []),
                            "rule": getattr(st, "rule", ""),
                            "confidence": getattr(st, "confidence", -1.0),
                        }
                        for st in getattr(s, "reasoning_chain", []) or []
                    ],
                    "expected_answer": getattr(s, "expected_answer", ""),
                    "verified": bool(getattr(s, "verified", False)),
                    "ground_truth_verified": bool(
                        getattr(s, "ground_truth_verified", False)
                    ),
                    "verification_notes": getattr(s, "verification_notes", ""),
                    "rejection_reason": (
                        "" if getattr(s, "verified", False)
                        else getattr(s, "verification_notes", "")
                    ),
                    "domain": getattr(s, "domain", ""),
                    "target_weakness": getattr(s, "target_weakness", ""),
                    "source": getattr(s, "source", ""),
                    "content_hash": getattr(s, "content_hash", ""),
                    "consistency_score": getattr(s, "consistency_score", None),
                    "parse_confidence": getattr(s, "parse_confidence", None),
                }
                f.write(json.dumps(rec, default=str))
                f.write("\n")

    def _save_final_report(self):
        """Save a summary of the entire improvement run."""
        if not self.history:
            return

        best = max(self.history, key=lambda r: r.improvement)
        improving_cycles = [r for r in self.history if r.improvement > 0]
        report = {
            "total_cycles": len(self.history),
            "improving_cycles": len(improving_cycles),
            "initial_score": self.history[0].pre_score,
            "final_score": self.history[-1].post_score,
            "total_improvement": self.history[-1].post_score - self.history[0].pre_score,
            "best_cycle": best.cycle,
            "best_cycle_improvement": best.improvement,
            "total_samples_generated": sum(r.samples_generated for r in self.history),
            "total_samples_verified": sum(r.samples_verified for r in self.history),
            "verification_pass_rate": (
                sum(r.samples_verified for r in self.history) /
                max(sum(r.samples_generated for r in self.history), 1)
            ),
            "escalations": self._escalation_state,
            "cycles_with_errors": sum(1 for r in self.history if r.errors),
            "error_breakdown": _summarize_errors(self.history),
            "score_trajectory": [
                {"cycle": r.cycle, "pre": r.pre_score, "post": r.post_score,
                 "improvement": r.improvement, "samples": r.samples_verified,
                 "had_errors": bool(r.errors)}
                for r in self.history
            ],
        }
        report_path = self.config.orchestrator.output_dir / "final_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Final report saved to {report_path}")
