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
        self.diversity_stats: dict = {}
        self.phase_times: dict[str, float] = {}
        # Records non-fatal errors that occurred during the cycle.
        # Each: {"phase": str, "type": str, "message": str, "traceback": str}
        self.errors: list[dict] = []

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

        if self._use_vllm:
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
                try:
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
                if result.training_metrics and result.training_metrics.steps > 0:
                    eval_start = time.time()
                    try:
                        self._eval_phase(cycle, result)
                    except Exception as e:
                        tb = traceback.format_exc()
                        logger.warning(f"  Held-out eval failed ({type(e).__name__}): {e}")
                        result.errors.append({
                            "phase": "eval", "type": type(e).__name__,
                            "message": str(e), "traceback": tb,
                        })
                    result.phase_times["eval"] = time.time() - eval_start

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

        if not samples:
            logger.warning(f"  No samples generated — model couldn't produce valid problems")
            self._pending_regression_weaknesses.extend(injected_regressions)
            result.post_score = result.pre_score
            return result

        # 3. Verify
        logger.info(f"[Cycle {cycle}] Phase 3: VERIFY")
        phase_start = time.time()
        verified = self.verifier.verify_batch(samples)
        result.samples_verified = len(verified)
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

    # Stable seed for held-out eval questions. Using a fixed value (not cycle)
    # means the held-out set is the SAME across all cycles, so scores track
    # true generalization over time instead of random per-cycle variance.
    HELDOUT_CYCLE_SEED = 0xE7A1

    def _eval_phase(self, cycle: int, result: CycleResult):
        """Run a held-out evaluation on a stable question set."""
        logger.info(f"[Cycle {cycle}] Phase 5b: HELD-OUT EVAL")
        eval_diag = self.diagnostics.run(self.HELDOUT_CYCLE_SEED)
        result.eval_score = eval_diag.overall_score
        result.eval_domain_scores = dict(eval_diag.domain_scores)
        # Compare to previous cycle's held-out score (if any) — apples-to-apples
        # on identical questions.
        prev_eval = next(
            (r.eval_score for r in reversed(self.history) if r.eval_score is not None),
            None,
        )
        if prev_eval is not None:
            delta = result.eval_score - prev_eval
            symbol = "+" if delta > 0 else ""
            logger.info(
                f"  Held-out eval: {result.eval_score:.3f} "
                f"(prev {prev_eval:.3f}, {symbol}{delta:.3f})"
            )
        else:
            logger.info(f"  Held-out eval: {result.eval_score:.3f} (baseline)")

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
            return

        # Propose updates for next cycle.
        proposal = self.meta.propose_updates(cfg_snapshot)
        if not proposal["applied"]:
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
                 "had_errors": bool(r.errors)}
                for r in self.history
            ],
        }
        progress_path = output_dir / "progress.json"
        with open(progress_path, "w") as f:
            json.dump(progress, f, indent=2)

    def _check_early_stopping(self, cycle: int, result: CycleResult) -> tuple[bool, str]:
        """Check if model is degrading and revert to best checkpoint if needed."""
        current_score = result.post_score
        if current_score > self._best_score:
            self._best_score = current_score
            self._best_checkpoint_cycle = cycle
            self._degradation_count = 0
            return False, ""

        if len(self.history) >= 2:
            prev_score = self.history[-2].post_score
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

        has_improved = (
            len(self.history) >= 2
            and any(r.improvement > 0.01 for r in self.history[-3:])
        )
        trained_this_cycle = bool(
            result.training_metrics
            and getattr(result.training_metrics, "steps", 0) > 0
        )

        # Verification escalation
        if (cycle >= schedule.verification
                and not self._escalation_state["verification"]
                and result.post_score > 0.5
                and has_improved
                and trained_this_cycle):
            self._escalate_verification()
            result.escalation_events.append("model_assists_verification")
            self._escalation_state["verification"] = True

        # Diagnosis escalation
        if (cycle >= schedule.diagnosis
                and not self._escalation_state["diagnosis"]
                and result.post_score > 0.6
                and has_improved
                and trained_this_cycle):
            diag_for_escalation = post_diag if post_diag else result
            self._escalate_diagnosis(diag_for_escalation, cycle=cycle)
            result.escalation_events.append("model_assists_diagnosis")
            self._escalation_state["diagnosis"] = True

        # Generation escalation
        if (cycle >= schedule.generation
                and not self._escalation_state["generation"]
                and result.post_score > 0.7
                and has_improved
                and trained_this_cycle):
            self._escalate_generation()
            result.escalation_events.append("model_improves_generation")
            self._escalation_state["generation"] = True
            self._plateau_count = 0

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
        """Let the model improve the data generation prompts."""
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
        improved_template = self.model_loader.generate(
            improvement_prompt, max_new_tokens=1024, temperature=0.3,
        )
        required = {"{problem}", "{domain}", "{subdomain}"}
        if (improved_template and len(improved_template) > 50
                and all(p in improved_template for p in required)):
            logger.info(f"  Model suggested improved template ({len(improved_template)} chars)")
            self.generator.set_custom_solution_template(improved_template)
        elif improved_template:
            missing = [p for p in required if p not in improved_template]
            logger.warning(f"  Rejected model template — missing placeholders: {missing}")

    def _should_stop(self, result: CycleResult) -> tuple[bool, str]:
        """Check if we should stop using EMA-smoothed improvement for plateau detection."""
        if result.diagnostics and not result.diagnostics.weaknesses:
            self._plateau_count = 0
            self._consecutive_failures = 0
            return True, "all domains above threshold — nothing left to improve"

        undertrained = getattr(self.trainer, "_last_merge_undertrained", False)
        if result.training_metrics and result.training_metrics.steps > 0 and not undertrained:
            self._consecutive_failures = 0
            # Use EMA-smoothed improvement instead of raw per-cycle delta.
            if self._improvement_ema < self.config.orchestrator.min_improvement_threshold:
                self._plateau_count += 1
            else:
                self._plateau_count = 0
        elif undertrained:
            self._consecutive_failures += 1
        else:
            self._consecutive_failures += 1

        if self._plateau_count >= self.config.orchestrator.plateau_patience:
            return True, (
                f"plateau after {self._plateau_count} cycles "
                f"(EMA improvement {self._improvement_ema:.4f} < "
                f"threshold {self.config.orchestrator.min_improvement_threshold})"
            )
        if self._consecutive_failures >= self.config.orchestrator.plateau_patience * 2:
            return True, f"{self._consecutive_failures} consecutive failed cycles — system is unable to produce training data"
        return False, ""

    def _save_checkpoint(self, cycle: int):
        """Save full checkpoint: model (post-merge) and history.

        Keeps only the 2 most recent full model checkpoints plus the best
        checkpoint (for early stopping revert) to prevent disk exhaustion.
        """
        output_dir = self.config.orchestrator.output_dir
        already_saved = self._use_vllm and getattr(self, "_vllm_saved_this_cycle", None) == cycle
        if not already_saved:
            try:
                self.model_loader.save_checkpoint(output_dir / "checkpoints", cycle)
            except Exception as e:
                logger.error(f"  Model checkpoint save failed: {e}")
                (output_dir / "checkpoints" / f"cycle_{cycle}").mkdir(parents=True, exist_ok=True)
        else:
            (output_dir / "checkpoints" / f"cycle_{cycle}").mkdir(parents=True, exist_ok=True)

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

    def _log_cycle(self, result: CycleResult):
        """Log cycle results to file."""
        log_path = self.config.orchestrator.log_dir / f"cycle_{result.cycle}.json"
        with open(log_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

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
