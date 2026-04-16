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
import shutil
import time
import logging
from pathlib import Path

import torch

from ..utils.config import SystemConfig
from ..utils.model_loader import ModelLoader
from ..diagnostics.engine import DiagnosticsEngine, DiagnosticResult, WeaknessReport
from ..generator.data_generator import DataGenerator
from ..verifier.verifier import Verifier
from ..trainer.custom_lora import CustomLoRATrainer, TrainingMetrics

logger = logging.getLogger(__name__)


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
        self.post_diag: DiagnosticResult | None = None  # post-training diagnostics
        # Mirrors (self.diagnostics is not None) for live cycles, and is restored
        # from saved history for resumed stubs where .diagnostics is dropped.
        self.had_diagnostics: bool = False
        self.eval_score: float | None = None  # held-out eval score
        self.eval_domain_scores: dict[str, float] = {}  # per-domain held-out scores
        self.diversity_stats: dict = {}  # sample diversity metrics
        self.phase_times: dict[str, float] = {}  # timing per phase

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
            self.model_loader = ModelLoader(config.model)

        self.diagnostics = DiagnosticsEngine(config.diagnostics, self.model_loader)
        self.generator = DataGenerator(config.generator, self.model_loader)
        self.verifier = Verifier(config.verifier)
        self.trainer = CustomLoRATrainer(config.trainer, self.model_loader)
        self.history: list[CycleResult] = []
        self._plateau_count = 0
        self._escalation_state = {
            "verification": False,
            "diagnosis": False,
            "generation": False,
        }
        self._domain_score_history: dict[str, list[float]] = {}  # track regression
        self._pending_regression_weaknesses: list[WeaknessReport] = []  # injected into next cycle
        self._last_deescalation_cycle: int = -10  # cycle of last de-escalation (init far back)
        self._consecutive_failures = 0  # cycles that failed before training completed
        self._vllm_saved_this_cycle: int | None = None  # de-duplicate vLLM mid-cycle save vs. end-of-cycle save
        self._best_score: float = 0.0
        self._best_checkpoint_cycle: int | None = None
        self._degradation_count: int = 0  # consecutive cycles where score dropped

    def run(self) -> None:
        """Run the full improvement loop."""
        logger.info("=" * 60)
        logger.info("RECURSIVE SELF-IMPROVEMENT SYSTEM")
        logger.info("=" * 60)
        self._setup()

        start_cycle = 1
        if self.config.orchestrator.resume_from:
            start_cycle = self._resume()

        for cycle in range(start_cycle, self.config.orchestrator.max_cycles + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"CYCLE {cycle}")
            logger.info(f"{'='*60}")

            cycle_start = time.time()
            result = self._run_cycle(cycle)
            result.duration = time.time() - cycle_start

            # Run held-out eval after training (if training happened)
            if result.training_metrics and result.training_metrics.steps > 0:
                eval_start = time.time()
                self._eval_phase(cycle, result)
                result.phase_times["eval"] = time.time() - eval_start

            self.history.append(result)

            # Check escalation BEFORE logging/checkpointing so events are captured.
            # Pass post_diag so diagnosis escalation targets still-weak domains.
            self._check_and_escalate(cycle, result, post_diag=result.post_diag)

            self._log_cycle(result)

            if cycle % self.config.orchestrator.checkpoint_every == 0:
                self._save_checkpoint(cycle)

            # Write progress dashboard for external monitoring
            self._save_progress_dashboard(cycle, result, result.phase_times)

            # Early stopping: revert to best if degrading
            early_stop, early_reason = self._check_early_stopping(cycle, result)
            if early_stop:
                logger.info(f"Early stopping at cycle {cycle}: {early_reason}")
                break

            should_stop, stop_reason = self._should_stop(result)
            if should_stop:
                logger.info(f"Stopping at cycle {cycle}: {stop_reason}")
                break

        # Always save a final checkpoint so resume works regardless of checkpoint_every
        if self.history:
            last_cycle = self.history[-1].cycle
            # Avoid double-save if the regular checkpoint_every already covered this cycle
            if last_cycle % self.config.orchestrator.checkpoint_every != 0:
                self._save_checkpoint(last_cycle)
        self._save_final_report()
        logger.info("\n" + "=" * 60)
        logger.info("IMPROVEMENT LOOP COMPLETE")
        if self.history:
            total = self.history[-1].post_score - self.history[0].pre_score
            logger.info(f"Total improvement: {self.history[0].pre_score:.3f} -> {self.history[-1].post_score:.3f} ({total:+.3f})")
        logger.info("=" * 60)

    def _setup(self):
        """Initialize the system."""
        if not self.config.model.model_path:
            raise ValueError("model_path must be configured")
        # Catch config inconsistencies that waste cycles — generator producing
        # samples that the verifier will always reject.
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
        # Resolve relative paths against output_dir — users typically pass paths
        # like "checkpoints/cycle_5" meaning <output_dir>/checkpoints/cycle_5,
        # not relative to an arbitrary CWD.
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

            # Restore escalation state — merge with expected keys so new escalation
            # types added after the checkpoint default to False, and old removed
            # keys are dropped rather than polluting the state dict.
            if "escalation_state" in data:
                for key in self._escalation_state:
                    if key in data["escalation_state"]:
                        self._escalation_state[key] = data["escalation_state"][key]
                # Re-apply escalation side-effects that wire up components.
                # Only verification needs re-wiring (sets model_verifier reference).
                # Diagnosis/generation are restored from saved state below —
                # re-calling _escalate_diagnosis() would be a no-op (empty history),
                # and _escalate_generation() would waste an inference call only to
                # be overwritten by the saved template.
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

            # Restore generation escalation template (instead of regenerating)
            if data.get("custom_solution_template"):
                self.generator.set_custom_solution_template(data["custom_solution_template"])

            # Restore model-generated diagnostic questions (from diagnosis escalation).
            # Validate structure — corrupt checkpoints with missing keys would cause
            # KeyError in _probe_domain when accessing q["prompt"]/q["expected"].
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

            # Restore minimal history so final report and escalation have context
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
            # Verify the model checkpoint actually exists — if it doesn't (e.g.,
            # model save failed), the loaded model is the original base, not the
            # merged version from prior cycles. LoRA weights assume the merged base,
            # so applying them to the original would produce incorrect results.
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
                # vLLM was loaded with the BASE model in _setup() before resume ran.
                # Reload from the checkpoint so prior cycles' merged weights are live.
                logger.info(f"  Reloading vLLM from checkpoint: {resume_path}")
                self.model_loader.swap_to_vllm_after_training(str(resume_path))
            logger.info(f"Resumed from checkpoint: {num_cycles} cycles, escalations: {self._escalation_state}")
            return num_cycles + 1

        return 1

    def _run_cycle(self, cycle: int) -> CycleResult:
        """Execute one full improvement cycle."""
        result = CycleResult(cycle)
        # Reset so a failed cycle doesn't carry a prior cycle's saved flag.
        self._vllm_saved_this_cycle = None

        # Log peak GPU memory at phase boundaries for profiling
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
        # If diagnosis escalation is active, regenerate adaptive questions
        # each cycle so the model doesn't memorize a fixed question set.
        # Skip when no prior diagnostics exist (e.g., first cycle or stubs from resume).
        # Treat resumed stubs with a restored `had_diagnostics` flag as prior diag.
        has_prior_diag = any(
            r.diagnostics is not None or getattr(r, "had_diagnostics", False)
            for r in self.history
        )
        if self._escalation_state["diagnosis"] and has_prior_diag:
            self._escalate_diagnosis(cycle=cycle)
        # Pass current best score to diagnostics so curriculum can adapt to model performance
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
        # Keep a copy — if the cycle fails before training, re-queue them.
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
            samples = self.generator.generate_for_weaknesses(diag.weaknesses)
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
            self._pending_regression_weaknesses.extend(injected_regressions)  # re-queue
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
            self._pending_regression_weaknesses.extend(injected_regressions)  # re-queue
            result.post_score = result.pre_score
            return result

        result.phase_times["verify"] = time.time() - phase_start
        _log_peak_memory("verify")

        # 4. Train — swap to HF model if using vLLM
        # Free fragmented VRAM from inference phases before loading HF model
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"[Cycle {cycle}] Phase 4: TRAIN on {len(verified)} verified samples")
        phase_start = time.time()
        if self._use_vllm:
            self.model_loader.swap_to_hf_for_training()
        self.trainer.inject_lora(weak_layers=diag.layer_health)
        try:
            metrics = self.trainer.train(verified, cycle)
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
        # Save LoRA weights separately from full model checkpoints so they survive
        # checkpoint pruning. LoRA weights are tiny (~100MB) so keeping all is fine.
        self.trainer.save_lora_weights(
            self.config.orchestrator.output_dir / "lora_weights", cycle
        )
        self.trainer.merge_lora()  # merges and strips LoRA for clean eval

        # Save merged model checkpoint so vLLM can reload with updated weights
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
            # Swap to vLLM BEFORE eval so post-training diagnostics use vLLM
            # (5-10x faster than HF inference). The merged weights are in the
            # checkpoint that vLLM loads.
            self.model_loader.swap_to_vllm_after_training(str(tmp_ckpt))
            post_diag = self.diagnostics.run(cycle + 10000)
        else:
            # Use offset cycle number so eval generates different question variants than pre-eval
            post_diag = self.diagnostics.run(cycle + 10000)
        result.post_diag = post_diag  # used by _check_and_escalate for diagnosis targeting
        result.post_score = post_diag.overall_score
        result.improvement = result.post_score - result.pre_score

        symbol = "+" if result.improvement > 0 else ""
        logger.info(f"  Score: {result.pre_score:.3f} -> {result.post_score:.3f} ({symbol}{result.improvement:.3f})")

        # Track per-domain scores — keep last 20 per domain to bound memory/checkpoint size
        for domain, score in post_diag.domain_scores.items():
            if domain not in self._domain_score_history:
                self._domain_score_history[domain] = []
            self._domain_score_history[domain].append(score)
            if len(self._domain_score_history[domain]) > 20:
                self._domain_score_history[domain] = self._domain_score_history[domain][-20:]

        # Detect regressions: domains that got WORSE compared to pre-training.
        # Add regressed domains as weaknesses so the NEXT cycle targets them.
        regressions = []
        for domain, pre_score in diag.domain_scores.items():
            post_score = post_diag.domain_scores.get(domain, 0)
            if post_score < pre_score - 0.05:  # 5% threshold
                regressions.append(f"{domain}: {pre_score:.3f}->{post_score:.3f}")
                # Inject a weakness so next cycle's generator creates remediation data.
                # Pull evidence from post-training diagnostic. Check both weaknesses
                # (domains below threshold) AND the raw evidence from _probe_domain
                # (stored in domain_scores). For small regressions where the domain
                # is still above threshold, post_diag.weaknesses won't contain it —
                # fall back to pre-training evidence.
                regression_evidence = [
                    e for e in post_diag.weaknesses
                    if e.domain == domain
                ]
                evidence_items = []
                for w in regression_evidence:
                    evidence_items.extend(w.evidence[:5])
                # If no post-training evidence (domain still above threshold),
                # use pre-training evidence instead of leaving it empty.
                if not evidence_items:
                    pre_evidence = [
                        e for e in diag.weaknesses
                        if e.domain == domain
                    ]
                    for w in pre_evidence:
                        evidence_items.extend(w.evidence[:5])
                self._pending_regression_weaknesses.append(WeaknessReport(
                    domain=domain,
                    subdomain="regression",
                    severity=pre_score - post_score,  # uncapped — let generator scale naturally
                    evidence=evidence_items[:20],
                    description=f"Regression: {domain} dropped from {pre_score:.3f} to {post_score:.3f}",
                ))
        if regressions:
            logger.warning(f"  REGRESSION detected in: {', '.join(regressions)}")

        return result

    def _eval_phase(self, cycle: int, result: CycleResult):
        """Run a held-out evaluation after training to measure actual improvement.

        Uses a different seed offset (cycle + 50000) so eval questions are distinct
        from both the pre-training diagnostic (cycle) and the post-training diagnostic
        (cycle + 10000). This prevents the model from being tested on the same questions
        it was diagnosed with, giving a more honest measure of generalization.
        """
        logger.info(f"[Cycle {cycle}] Phase 5b: HELD-OUT EVAL")
        eval_diag = self.diagnostics.run(cycle + 50000)
        result.eval_score = eval_diag.overall_score
        result.eval_domain_scores = dict(eval_diag.domain_scores)
        delta = result.eval_score - result.pre_score
        symbol = "+" if delta > 0 else ""
        logger.info(
            f"  Held-out eval: {result.eval_score:.3f} "
            f"(vs pre-training {result.pre_score:.3f}, {symbol}{delta:.3f})"
        )

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
            "timing": phase_times,
            "escalations": dict(self._escalation_state),
            "degradation_count": self._degradation_count,
            "plateau_count": self._plateau_count,
            "history_summary": [
                {"cycle": r.cycle, "pre": r.pre_score, "post": r.post_score,
                 "improvement": r.improvement, "eval": r.eval_score}
                for r in self.history
            ],
        }
        progress_path = output_dir / "progress.json"
        with open(progress_path, "w") as f:
            json.dump(progress, f, indent=2)

    def _check_early_stopping(self, cycle: int, result: CycleResult) -> tuple[bool, str]:
        """Check if model is degrading and revert to best checkpoint if needed.

        If score drops for 2+ consecutive cycles, revert to the best checkpoint.
        """
        current_score = result.post_score
        # Track best score
        if current_score > self._best_score:
            self._best_score = current_score
            self._best_checkpoint_cycle = cycle
            self._degradation_count = 0
            return False, ""

        # Check for degradation (score strictly worse than previous cycle)
        if len(self.history) >= 2:
            prev_score = self.history[-2].post_score
            if current_score < prev_score - 0.005:  # small tolerance for noise
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
        """Check and perform escalations based on cycle and performance.

        post_diag: POST-training diagnostics. Used for diagnosis escalation to
        target domains that are STILL weak after training, not just pre-training.
        """
        schedule = self.config.orchestrator.escalation_schedule

        # Escalation gates on both absolute score AND demonstrated improvement.
        # Require a minimum absolute delta so a single stale positive blip doesn't
        # permanently satisfy the gate, and require that training actually ran
        # this cycle — otherwise post_score is just pre_score and escalation
        # would fire on the diagnostic starting score alone.
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

        # Diagnosis escalation — use post-training diagnostics so adaptive questions
        # target what's STILL weak, not what was weak before training fixed it.
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
            # Reset plateau counter — new capability may unlock further improvement
            self._plateau_count = 0

        # De-escalation: if model regresses significantly after escalation, revert
        # to safer hand-crafted pipelines. Check last 2 cycles for sustained regression.
        # Skip if an escalation just happened this call — don't immediately undo it.
        # Require 3+ cycles since last de-escalation to prevent cascading reverts
        # where one bad escalation causes all capabilities to be stripped in 3 cycles.
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
        # Enable model-assisted verification in config
        self.config.verifier.use_model_verification = True

    def _escalate_diagnosis(self, result_or_diag: 'CycleResult | DiagnosticResult | None' = None, cycle: int = 0):
        """Let the model generate its own diagnostic questions.

        Accepts either a CycleResult (extracts .diagnostics) or a DiagnosticResult directly.
        """
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
        # Show the CURRENT template (may already be model-improved from prior escalation)
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
        # Validate the template has required placeholders — without {problem}
        # the solution prompt is useless, and format() would raise KeyError
        # (caught in _build_solution_prompt, silently falling back to default
        # every cycle, wasting inference for zero benefit).
        required = {"{problem}", "{domain}", "{subdomain}"}
        if (improved_template and len(improved_template) > 50
                and all(p in improved_template for p in required)):
            logger.info(f"  Model suggested improved template ({len(improved_template)} chars)")
            self.generator.set_custom_solution_template(improved_template)
        elif improved_template:
            missing = [p for p in required if p not in improved_template]
            logger.warning(f"  Rejected model template — missing placeholders: {missing}")

    def _should_stop(self, result: CycleResult) -> tuple[bool, str]:
        """Check if we should stop. Returns (should_stop, reason)."""
        if result.diagnostics and not result.diagnostics.weaknesses:
            self._plateau_count = 0
            self._consecutive_failures = 0
            return True, "all domains above threshold — nothing left to improve"
        # Only count as plateau if training actually completed — failed cycles
        # (no training metrics) shouldn't count toward the patience limit since
        # they represent infrastructure issues, not genuine lack of improvement.
        undertrained = getattr(self.trainer, "_last_merge_undertrained", False)
        if result.training_metrics and result.training_metrics.steps > 0 and not undertrained:
            self._consecutive_failures = 0
            if result.improvement < self.config.orchestrator.min_improvement_threshold:
                self._plateau_count += 1
            else:
                self._plateau_count = 0
        elif undertrained:
            # Training ran but >50% of LoRA layers produced no gradient signal —
            # treat as failure for plateau purposes, not genuine convergence.
            self._consecutive_failures += 1
        else:
            # Track consecutive failures (generation/verification/training all failed).
            # Without this, repeated failures run forever without triggering any stop.
            self._consecutive_failures += 1

        if self._plateau_count >= self.config.orchestrator.plateau_patience:
            return True, f"plateau after {self._plateau_count} cycles with no improvement"
        if self._consecutive_failures >= self.config.orchestrator.plateau_patience * 2:
            return True, f"{self._consecutive_failures} consecutive failed cycles — system is unable to produce training data"
        return False, ""

    def _save_checkpoint(self, cycle: int):
        """Save full checkpoint: model (post-merge) and history.
        LoRA weights are saved BEFORE merge in _run_cycle.

        Keeps only the 2 most recent full model checkpoints to prevent
        disk exhaustion (70B model ≈ 140GB per checkpoint).
        LoRA weight directories are kept since they're tiny (~100MB).
        """
        output_dir = self.config.orchestrator.output_dir
        # In vLLM mode the model was already saved mid-cycle (before the swap back
        # to vLLM). HF is no longer loaded, so calling save_checkpoint now would
        # log a warning and leave an empty dir. Skip the redundant save.
        already_saved = self._use_vllm and getattr(self, "_vllm_saved_this_cycle", None) == cycle
        if not already_saved:
            try:
                self.model_loader.save_checkpoint(output_dir / "checkpoints", cycle)
            except Exception as e:
                # Model save can fail (disk full, permissions). Ensure the cycle
                # directory still exists so history.json (critical for resume) is saved.
                logger.error(f"  Model checkpoint save failed: {e}")
                (output_dir / "checkpoints" / f"cycle_{cycle}").mkdir(parents=True, exist_ok=True)
        else:
            (output_dir / "checkpoints" / f"cycle_{cycle}").mkdir(parents=True, exist_ok=True)

        # Prune old model checkpoints — keep last 2 + any LoRA-only dirs
        ckpt_dir = output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        def _cycle_num(d):
            try:
                return int(d.name.split("_")[1])
            except (ValueError, IndexError):
                return -1  # sort non-standard dirs first (won't be pruned — no config.json)
        existing = sorted(
            [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("cycle_")],
            key=_cycle_num,
        )
        # Prune full model checkpoints (keep last 2) AND empty dirs from failed
        # saves. Without cleaning up empty dirs, repeated save failures accumulate
        # directories that waste inodes and confuse manual inspection.
        model_dirs = [d for d in existing if (d / "config.json").exists()]
        for old_dir in model_dirs[:-2]:  # keep last 2
            try:
                shutil.rmtree(old_dir)
                logger.info(f"  Pruned old checkpoint: {old_dir.name}")
            except OSError:
                pass
        # Save history for resume (BEFORE empty-dir cleanup so the current cycle's
        # dir isn't deleted — if model save failed, the dir exists but is empty).
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
            }, f, indent=2)

        # Clean up empty dirs from failed model saves (no config.json, no history.json).
        # Runs AFTER history.json is written so the current cycle's dir isn't caught.
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
            # Only include per-cycle summary (score trajectory), not full to_dict().
            # Full cycle data is already saved by _log_cycle per cycle — duplicating
            # all 100+ cycles here makes final_report.json huge and redundant.
            "score_trajectory": [
                {"cycle": r.cycle, "pre": r.pre_score, "post": r.post_score,
                 "improvement": r.improvement, "samples": r.samples_verified}
                for r in self.history
            ],
        }
        report_path = self.config.orchestrator.output_dir / "final_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Final report saved to {report_path}")
