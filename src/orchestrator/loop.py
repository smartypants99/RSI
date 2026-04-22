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
from ..diagnostics.difficulty_tracker import DifficultyTracker
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
        # Task #10: which held-out eval kind this cycle ran. "full" draws
        # the complete HELD_OUT_ONLY sweep; "quick" uses the subsample
        # path to shorten wall-clock on non-anchor cycles.
        self.heldout_eval_kind: str = "full"
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
        # Curriculum escalation snapshot for this cycle (DifficultyTracker).
        self.difficulty_frontier: str = ""
        self.difficulty_floor: float = 0.0
        # Candidates admitted by §1.4 quorum this cycle — captured so the
        # orchestrator can bank them as adversarial examples if post-training
        # eval regresses. Each entry: {problem_id, candidate, domain, problem_ctx}.
        self._admitted_candidates: list[dict] = []
        # Anchor eval (Task #1, ground-truth). External-benchmark score on
        # HumanEval/MBPP/GSM8K/MATH, run after internal held-out. Decoupled
        # from the self-graded loop — if this drops while eval_score rises,
        # verifier_capture_alarm fires (see _eval_phase).
        self.anchor_score: float | None = None
        self.verifier_capture_alarm: bool = False
        # Paired-sample variance reduction (task #3). Computed in _eval_phase
        # vs. the previous cycle's per-question records. None before cycle 2.
        self.paired_delta: float | None = None
        self.paired_delta_se: float | None = None
        self.paired_delta_n: int | None = None
        self.paired_variance_reduction: float | None = None
        # Task #14: set True when solution-diversity tracker raised a
        # mode-collapse alarm this cycle.
        self.diversity_alarm: bool = False
        # Task #11 concern #2: set True when per-benchmark distinct/n falls
        # below mode_collapse_distinct_threshold on the post-train anchor
        # eval. When True the cycle is ineligible for best-promotion —
        # a "clean score with degenerate outputs" cannot become a reference.
        self.mode_collapse_detected: bool = False
        # Task #15: set True when the post-Phase-5b regression-revert guard
        # fires on this cycle (eval_score dropped > revert_threshold below the
        # reference). A reverted cycle MUST NOT advance the pending-best
        # streak even if _best_score is 0 (pre-promotion) or if no single
        # prior history cycle strictly exceeds current_score — the revert
        # itself is authoritative evidence the training step regressed.
        self.regression_reverted: bool = False

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


# Task #16: expected HELD_OUT_ONLY partition retention. Matches the
# _WEIGHTS table in src/diagnostics/eval_partition.py (HELD_OUT_ONLY=0.37).
# Kept here as a module-level constant so the quick-eval stratification
# math is a pure function of (target_n, n_domains).
_HELDOUT_PARTITION_RETENTION = 0.37


def _quick_eval_stratified_targets(
    *,
    target_n: int,
    n_domains: int,
    min_per_domain: int = 0,
    retention: float = _HELDOUT_PARTITION_RETENTION,
) -> tuple[int, int]:
    """Compute the pre-filter per-domain target for Phase-5b QUICK eval.

    Task #16: the prior implementation used ceil(target_n / N_domains)
    as the per-domain post-filter target, then divided by retention —
    which compounded with ceil rounding and HELD_OUT_ONLY variance to
    produce live-run cycle 2's observed n=207 when target=128.

    New: stratify WITHIN target_n by equal domain weight (1/N), then
    back out the pre-filter per-domain target. The expected post-filter
    total is ``round(target_n / N) * N``, bounded at ±(N_domains/2) from
    target_n due to rounding. For target=128, N=4 the expected total
    sits in [126, 130]; with min_per_domain guard the total can only
    grow, so the test tolerance of [124, 132] is comfortable.

    Returns ``(pre_filter_per_domain, expected_post_filter_total)``.
    """
    if target_n <= 0:
        return (max(0, min_per_domain), 0)
    n = max(1, int(n_domains))
    post_filter_per_domain = max(1, round(target_n / n))
    pre_filter_per_domain = max(
        int(min_per_domain),
        int(round(post_filter_per_domain / max(1e-9, retention))),
    )
    expected_total = post_filter_per_domain * n
    return (pre_filter_per_domain, expected_total)


class ImprovementLoop:
    """Orchestrates the recursive self-improvement loop."""

    def _base_is_4bit(self) -> bool:
        """True when the base model is bnb-4bit/8bit quantized.

        On bnb-4bit bases, merge_lora no-ops (packed weights can't absorb
        dense deltas) AND save_checkpoint no-ops (save_pretrained raises on
        quantized bases). Without adapter persistence, training effectively
        evaporates each cycle. This predicate gates the PEFT-adapter path.
        """
        qc = getattr(getattr(self.model_loader, "config", None),
                     "quantization_config", None)
        if not qc:
            qc = getattr(self.model_loader, "quantization_config", None)
        if not qc:
            return False
        return bool(qc.get("load_in_4bit") or qc.get("load_in_8bit"))

    def __init__(self, config: SystemConfig):
        self.config = config
        self._use_vllm = config.use_vllm
        # Measurement-corruption global halt (task #4 baseline-drift canary).
        # Set to True if the cycle_0 anchor re-eval drifts by more than
        # BASELINE_DRIFT_EPS; latches for the rest of the run.
        self._self_edit_halted: bool = False
        self._self_edit_halted_at: int | None = None
        self._cycle0_anchor_score: float | None = None
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
                max_num_seqs=int(getattr(vllm_cfg, "max_num_seqs", 0) or 0),
                # Task #19: co-resident training knobs. Default-off; user
                # flips coresident_training_enabled=True after A/B on
                # target GPU. See VLLMConfig docstring for the VRAM budget.
                enforce_eager=bool(getattr(vllm_cfg, "enforce_eager", False)),
                coresident_training_enabled=bool(
                    getattr(vllm_cfg, "coresident_training_enabled", False)
                ),
                coresident_vllm_mem_frac=float(
                    getattr(vllm_cfg, "coresident_vllm_mem_frac", 0.42)
                ),
                # Task #18 step 2: chunked prefill — default True in the
                # config, off only if the knob is explicitly flipped.
                enable_chunked_prefill=bool(
                    getattr(vllm_cfg, "enable_chunked_prefill", True)
                ),
                # Task #18 step 3: prefix-cache throughput logging. Off
                # by default; flip on for a diagnostic cycle to verify
                # the shared-prefix cache is hitting.
                log_throughput_stats=bool(
                    getattr(vllm_cfg, "log_throughput_stats", False)
                ),
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
        # Task #14: sample-quality clean-floor filter plumbed from generator
        # config through a trainer attribute. Trainer reads it at train() time.
        self.trainer._sample_quality_min_clean_floor = int(
            getattr(config.generator, "sample_quality_min_clean_floor", 0)
        )

        # Synthesis-mode components — instantiated only when the flag is on so
        # the classic path has zero overhead.
        self._synthesis_enabled = getattr(
            getattr(config, "synthesis", None), "enable_task_synthesis", False
        )
        # Fast-student manager (src/utils/fast_student.py). Constructed when
        # orchestrator.use_fast_student is True so the RSI solve_batch path
        # can harvest (prompt, completion) pairs for periodic distillation.
        # Kept here (pre-synthesis) so meta_meta records reflect the config
        # state even if synthesis itself is disabled on a given run.
        self._fast_student_mgr = None
        if getattr(config.orchestrator, "use_fast_student", False):
            try:
                from ..utils.fast_student import (
                    FastStudentConfig,
                    FastStudentManager,
                )
                fs_cfg = FastStudentConfig(
                    enabled=True,
                    model_name=getattr(
                        config.orchestrator,
                        "fast_student_model_name",
                        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                    ),
                    checkpoint_root=(
                        getattr(config.orchestrator, "output_dir", Path("outputs"))
                        / "fast_student"
                    ),
                )
                self._fast_student_mgr = FastStudentManager(fs_cfg)
                logger.info(
                    "fast_student: manager constructed (model=%s, redistill_every=%d)",
                    fs_cfg.model_name, fs_cfg.redistill_every,
                )
            except Exception as exc:
                logger.warning(
                    "fast_student: manager init failed (%s: %s) — disabled",
                    type(exc).__name__, exc,
                )
                self._fast_student_mgr = None

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
                if self._task_synthesizer is not None:
                    try:
                        self._task_synthesizer.set_registries(self._registries)
                    except Exception as e:
                        logger.warning(
                            "TaskSynthesizer.set_registries failed (%s: %s) — "
                            "library few-shot prefix disabled",
                            type(e).__name__, e,
                        )
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
        # Task #2: lagged-confirmation best-promotion. A cycle becomes the
        # confirmed best only after `best_confirm_cycles` consecutive eligible
        # cycles at-or-above a pending high-water mark. Kills outlier lock-in
        # (cycle 1's 1-sample/2-step eval=0.624 pinned the reference bank for
        # 6 hours of overnight run with zero forward progress).
        self._pending_best_score: float = 0.0
        self._pending_best_cycle: int | None = None
        self._pending_best_streak: int = 0
        # Consecutive verifier-capture alarm count (task #2, toothed alarm).
        self._capture_alarm_consecutive: int = 0
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
        # Fast-start (Task #11): pre-stash prior-session training samples so
        # cycle 1 has a non-empty pool without waiting for a full synthesis
        # round. Guarded by config flag + try/except so a cold start with
        # no prior outputs just falls through to normal cold-cycle behavior.
        if self._synthesis_enabled and getattr(config.orchestrator, "prestash_prior_samples", False):
            try:
                from ..utils.fast_start import prestash_prior_training_samples
                _sid = self._registries.sid if self._registries else ""
                prior = prestash_prior_training_samples(
                    config.orchestrator.output_dir, _sid,
                    max_samples=int(getattr(config.orchestrator, "prestash_max_samples", 30)),
                )
                self._rsi_pending_pool.extend(prior)
                if prior:
                    logger.info("fast_start: pre-stashed %d prior training samples", len(prior))
            except Exception as exc:
                logger.warning("fast_start pre-stash failed (%s): %s", type(exc).__name__, exc)
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

        # Curriculum escalation: DifficultyTracker persists proposer
        # frontier + a difficulty ratchet across restarts. rsi_tick and
        # _eval_phase feed it proposal-accept counts and held-out per-
        # question records respectively.
        self._difficulty_state_path = (
            config.orchestrator.output_dir / "difficulty_state.json"
        )
        self.difficulty_tracker = DifficultyTracker.load_or_new(
            self._difficulty_state_path
        )

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
                # Task #13: consult compute allocator at start of cycle.
                self._safe_call(
                    "compute_allocator",
                    lambda: self._consult_compute_allocator(cycle, result),
                    result,
                )
                # Task #13: optionally run component proposer every N cycles.
                self._safe_call(
                    "component_proposer",
                    lambda: self._maybe_run_component_proposer(cycle, result),
                    result,
                )
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
                # Task #23 wedge 1: also skip the full Phase-5b eval when the
                # quick regression probe already showed a regression beyond
                # the configured threshold. Full eval is ~40 min; confirming
                # a regression we already detected in the quick probe is
                # wasted compute. The quick-probe revert path above has
                # already banked the adversarial samples and rolled vLLM
                # back to last-good, so skipping the full eval here does not
                # change best-promotion eligibility (the cycle was not going
                # to be promoted anyway). Carry forward the prior eval_score
                # so downstream meta / early-stop have something to read.
                ocfg_es = self.config.orchestrator
                _skip_full_on_quick_regression = bool(getattr(
                    ocfg_es, "skip_full_heldout_on_quick_regression", False,
                ))
                _quick_skip_threshold = float(getattr(
                    ocfg_es, "quick_regression_skip_threshold", 0.10,
                ))
                _quick_regressed = (
                    _skip_full_on_quick_regression
                    and trained
                    and result.improvement < -_quick_skip_threshold
                )
                if _quick_regressed and not skip_eval:
                    skip_eval = True
                    logger.info(
                        "[Cycle %d] Phase 5b: HELD-OUT EVAL skipped — quick "
                        "probe already regressed (%+.3f < -%.2f); saves "
                        "~%d min by not confirming what we already know.",
                        cycle, result.improvement, _quick_skip_threshold, 40,
                    )
                    result.heldout_eval_kind = "skipped_quick_regression"
                eval_start = time.time()
                try:
                    if skip_eval:
                        if not _quick_regressed:
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
                    # Robust reference (task #2): prefer the CONFIRMED best, but
                    # clamp it to the trimmed mean of the last K held-out scores
                    # so a single stale high doesn't trap us into perpetual
                    # reverts. Pre-promotion (no confirmed best yet) falls back
                    # to pre_score so the first trained cycle still has a sane
                    # comparison point without deferring to an unconfirmed
                    # outlier.
                    reference = self._revert_reference(result)
                    full_eval_drop = reference - result.eval_score
                    if full_eval_drop > revert_threshold:
                        # Task #15: mark result BEFORE _check_early_stopping
                        # runs so its streak-advance gate sees it. Without
                        # this the live run showed "streak=1/2 — awaiting
                        # confirmation" on the very cycle that just got
                        # reverted (cycle 2: 0.478 vs ref 0.633).
                        result.regression_reverted = True
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
                        # Task #15: regression-revert clears any pending-best
                        # tracking from prior cycles. A candidate held-over
                        # from cycle N-1 cannot ride through a cycle N that
                        # just got reverted — subsequent confirmation must
                        # restart from scratch on a clean cycle.
                        if self._pending_best_cycle is not None:
                            logger.info(
                                "  regression-revert: clearing pending-best "
                                "state (was cycle=%s score=%.4f streak=%d)",
                                self._pending_best_cycle,
                                self._pending_best_score,
                                self._pending_best_streak,
                            )
                        self._pending_best_streak = 0
                        self._pending_best_cycle = None
                        self._pending_best_score = 0.0

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
                # Task #10: push per-phase wall-time into meta_meta sidecar and
                # emit a 10-cycle-window trend line.
                self._safe_call("wall_time",
                                lambda: self._record_wall_time(cycle, result),
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
            # Task #2: lagged-confirmation pending-best + capture-alarm streak.
            if "pending_best_score" in data:
                self._pending_best_score = data["pending_best_score"]
            if "pending_best_cycle" in data:
                self._pending_best_cycle = data["pending_best_cycle"]
            if "pending_best_streak" in data:
                self._pending_best_streak = data["pending_best_streak"]
            if "capture_alarm_consecutive" in data:
                self._capture_alarm_consecutive = data["capture_alarm_consecutive"]

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
                                accept_policy=getattr(
                                    self.config.verifier,
                                    "verifier_accept_policy",
                                    "any_fail_veto",
                                ),
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

        # 5. Save LoRA weights BEFORE merge (merge destroys them), then evaluate.
        # save_lora_weights ALSO emits a PEFT-format adapter dir — when the
        # base is bnb-4bit, merge_lora no-ops (packed weights ≠ dense delta),
        # so this adapter is the ONLY thing that persists training across
        # cycles. vLLM loads it at inference via LoRARequest.
        logger.info(f"[Cycle {cycle}] Phase 5: EVALUATE")
        adapter_path = self.trainer.save_lora_weights(
            self.config.orchestrator.output_dir / "lora_weights", cycle
        )
        self.trainer.merge_lora()
        use_adapter = bool(
            adapter_path is not None
            and getattr(self.config.orchestrator, "use_lora_adapter_persistence", True)
            and self._base_is_4bit()
        )

        if self._use_vllm:
            ckpt_root = self.config.orchestrator.output_dir / "checkpoints"
            self.model_loader.save_checkpoint(ckpt_root, cycle)
            tmp_ckpt = ckpt_root / f"cycle_{cycle}"
            ckpt_ok = (
                (tmp_ckpt / "config.json").exists()
                and any(tmp_ckpt.glob("*.safetensors"))
            )
            # On bnb-4bit, save_checkpoint correctly skips (marker .incomplete
            # is dropped) because the base weights can't round-trip. Don't fail
            # the swap — adapter persistence covers us. Fall back to the base
            # model path and attach the adapter.
            if not ckpt_ok and not use_adapter:
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
                swap_kwargs = {}
                if use_adapter:
                    swap_kwargs["adapter_path"] = str(adapter_path)
                swap_ckpt = str(tmp_ckpt) if ckpt_ok else None
                self.model_loader.swap_to_vllm_after_training(
                    swap_ckpt, **swap_kwargs
                )
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
        # Fast-start (Task #11): cycle 1 uses a smaller propose budget.
        from ..utils.fast_start import bootstrap_tasks_per_cycle
        ts = bootstrap_tasks_per_cycle(self.config.synthesis, cycle)

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
            elif cycle == 1 and getattr(self.config.orchestrator, "skip_first_diagnostics", False):
                # Fast-start (Task #11): cycle 1 uses a uniform-default
                # WeaknessReport to save ~6 min on a cold start. Intentionally
                # do NOT populate the cache, so cycle 2 triggers a real probe.
                from ..utils.fast_start import default_weakness_diag
                diag_result = default_weakness_diag(self.config.diagnostics.domains, cycle=cycle)
                logger.info("[RSI tick %d] skip_first_diagnostics=True — using uniform default diag", cycle)
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
        # Curriculum escalation: push the current frontier skill-pair +
        # ratcheted difficulty floor into the synthesizer so propose_batch_code
        # biases frontier_fraction of prompts toward the failing zone and
        # rejects proposals below the ratchet floor.
        try:
            # Domain-scoped frontier: propose_batch_code synthesizes code-only
            # problems, so the hint must come from a code subdomain. Without
            # this scope the tracker's global argmin kept landing on
            # "math/percentage" (lowest accuracy overall) and getting spliced
            # into the code-proposal prompt, biasing the model toward a
            # domain the proposal format cannot express.
            frontier_skill = self.difficulty_tracker.frontier(domain="code")
            if hasattr(synth, "set_frontier_hint"):
                synth.set_frontier_hint(frontier_skill)
            if hasattr(synth, "set_difficulty_floor"):
                synth.set_difficulty_floor(self.difficulty_tracker.difficulty_floor)
            logger.info(
                "[RSI tick %d] curriculum: frontier=%r floor=%.2f",
                cycle, frontier_skill or "(none)",
                self.difficulty_tracker.difficulty_floor,
            )
        except Exception as exc:
            logger.debug("difficulty_tracker wiring failed: %s", exc)
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
        try:
            accepted_n = int(result.novel_problems_proposed)
            requested_n = int(ts)
            self.difficulty_tracker.record_proposals(
                accepted=accepted_n,
                rejected=max(0, requested_n - accepted_n),
            )
        except Exception as exc:
            logger.debug("difficulty_tracker.record_proposals failed: %s", exc)

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

        # ── Fast-student harvest (Task #3 activation) ─────────────────────
        # Record (prompt, completion) pairs from the teacher's solve_batch
        # so FastStudentManager can distill periodically. Harvest is bounded
        # by the manager's rolling buffer; the call is a no-op when
        # fast_student is disabled.
        if self._fast_student_mgr is not None:
            try:
                harvest_prompts: list[str] = []
                harvest_completions: list[str] = []
                for _p in proposed_problems:
                    pid = getattr(_p, "problem_id", None)
                    if not pid:
                        continue
                    entry = (getattr(_p, "problem_ctx", None) or {}).get(
                        "entry_point"
                    ) or "solve"
                    # Mirror solve_batch's prompt structure without the
                    # strategy prefix (prefix is a session-local decoration).
                    _prompt = (
                        "Write a Python function to solve this problem.\n\n"
                        f"PROBLEM: {getattr(_p, 'problem_text', '')}\n\n"
                        f"Requirements: the function MUST be named "
                        f"`{entry}` exactly, and be a single self-contained "
                        "definition."
                    )
                    for comp in candidates_by_problem.get(pid, []) or []:
                        if not comp:
                            continue
                        harvest_prompts.append(_prompt)
                        harvest_completions.append(comp)
                if harvest_prompts:
                    self._fast_student_mgr.record_teacher_generation(
                        harvest_prompts, harvest_completions, cycle=cycle,
                    )
                    logger.debug(
                        "fast_student: recorded %d pairs (buffer=%d)",
                        len(harvest_prompts),
                        self._fast_student_mgr.buffer_size(),
                    )
            except Exception as _fs_exc:
                logger.debug(
                    "fast_student: record_teacher_generation failed (%s): %s",
                    type(_fs_exc).__name__, _fs_exc,
                )

        # ── STEP 3b: solution-diversity tracker (task #4) ───────────────────
        # Leading indicator for mode collapse: if the k-candidate set per
        # problem collapses onto one canonical answer, the held-out benchmark
        # will degrade 10-20 cycles later. Compute once per cycle, stash on
        # CycleResult.diversity_stats, and append a jsonl row.
        try:
            from ..diagnostics.solution_diversity import compute_diversity
            div_report = compute_diversity(cycle, candidates_by_problem)
            result.diversity_stats = {
                "mean": div_report.diversity_mean,
                "variance": div_report.diversity_variance,
                "n_problems": div_report.n_problems,
                "n_collapsed": div_report.n_problems_above_threshold,
                "fraction_collapsed": div_report.fraction_above_threshold,
                "mode_collapse_alarm": div_report.mode_collapse_alarm,
                "embedding_backend": div_report.embedding_backend,
            }
            # Task #14: surface the alarm on CycleResult so downstream logging
            # / dashboards can treat this as a first-class signal.
            if div_report.mode_collapse_alarm:
                result.diversity_alarm = True
                logger.warning(
                    "[RSI tick %d] solution diversity alarm: %d/%d problems collapsed (frac=%.2f)",
                    cycle, div_report.n_problems_above_threshold,
                    div_report.n_problems, div_report.fraction_above_threshold,
                )
            try:
                out_dir = self.config.orchestrator.output_dir / "cycle_metrics"
                out_dir.mkdir(parents=True, exist_ok=True)
                with open(out_dir / "diversity.jsonl", "a") as _f:
                    _f.write(json.dumps({
                        "cycle": cycle,
                        **result.diversity_stats,
                        "timestamp": time.time(),
                    }) + "\n")
            except Exception as _exc:
                logger.debug("diversity.jsonl write failed: %s", _exc)
        except Exception as exc:
            logger.warning("[RSI tick %d] diversity tracker failed (%s): %s",
                           cycle, type(exc).__name__, exc)

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

        # Task #12 (warm-speed): cross-candidate parallel verify. Each
        # _pe_verify call is subprocess-bound (SandboxedExecutor spawns a
        # child per property; 8ea90e4 already parallelizes WITHIN a candidate).
        # Stacking across-candidate parallelism on top gives real wall-clock
        # speedup on the verify phase, which is dominant at steady state.
        # Registry / training-pool / _admitted_candidates writes stay serial
        # on the main thread in the bookkeeping pass below — no locking
        # needed, and the original (pid, idx) order is preserved.
        _verify_work: list = []
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
                _verify_work.append((problem, pid, idx, cand_id, candidate_str, props))

        _accept_policy = getattr(
            self.config.verifier, "verifier_accept_policy", "any_fail_veto",
        )

        def _verify_one(item):
            _problem, _pid, _idx, _cand_id, _cand_str, _props = item
            try:
                rec = _pe_verify(
                    problem_id=_pid,
                    candidate=_cand_str,
                    admitted_properties=_props,
                    executor=_executor,
                    calibration=reg.calibration_ledger,
                    quorum_distinct_classes_required=2,
                    min_properties=2,
                    accept_policy=_accept_policy,
                )
                return (item, rec, None)
            except Exception as exc:
                return (item, None, exc)

        _verify_results: list = []
        if _verify_work:
            from concurrent.futures import ThreadPoolExecutor as _VerifyPool
            # max_workers=8 chosen to saturate on typical 8-16 core machines
            # while avoiding subprocess-spawn storms. property_engine's
            # inner pool is max_workers=min(8, len(props)) after task #12.
            with _VerifyPool(max_workers=8) as _vex:
                _verify_results = list(_vex.map(_verify_one, _verify_work))

        for _item, pe_record, _exc in _verify_results:
            problem, pid, idx, cand_id, candidate_str, props = _item
            if _exc is not None:
                logger.debug("  verify() raised for candidate %s: %s", cand_id, _exc)
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
                for prop_rec in staged_prop_records.get(pid, []):
                    try:
                        reg.property_registry.append_property(prop_rec, bundle_passed_vov=True)
                        result.properties_admitted += 1
                    except Exception as exc:
                        logger.debug("PropertyRecord flush failed: %s", exc)
                try:
                    from ..generator.data_generator import TrainingSample
                    ts_obj = TrainingSample(
                        prompt=getattr(problem, "problem_text", ""),
                        response=candidate_str,
                        domain=getattr(problem, "domain", "unknown"),
                        verified=True,
                        source="rsi_property",
                        verdict_warnings=tuple(
                            getattr(pe_record, "verdict_warnings", ()) or ()
                        ),
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
                    rsi_adapter_path = self.trainer.save_lora_weights(
                        self.config.orchestrator.output_dir / "lora_weights", cycle
                    )
                    self.trainer.merge_lora()
                    rsi_use_adapter = bool(
                        rsi_adapter_path is not None
                        and getattr(
                            self.config.orchestrator,
                            "use_lora_adapter_persistence", True,
                        )
                        and self._base_is_4bit()
                    )

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
                    # On vLLM, skip the quick HF-path probe entirely — it was
                    # OOM'ing on 32B-4bit post-training (48 prompts through
                    # HF with no VRAM headroom). Swap to vLLM first with the
                    # merged weights, then let the full Phase 5b eval (which
                    # runs via vLLM and uses the stronger eval-isolation
                    # partition anyway) determine regression. If it regresses,
                    # the post-Phase-5b revert guard at the eval-phase site
                    # swaps back. We set a neutral post_score here so we don't
                    # block the swap path; the real decision happens in
                    # _eval_phase.
                    skip_quick_probe = bool(getattr(self, "_use_vllm", False))
                    if skip_quick_probe:
                        post_score = result.pre_score  # neutral — defer to Phase 5b
                    else:
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
                            swap_kwargs = {}
                            if rsi_use_adapter:
                                swap_kwargs["adapter_path"] = str(rsi_adapter_path)
                            tmp_ok = (
                                (tmp_ckpt / "config.json").exists()
                                and any(tmp_ckpt.glob("*.safetensors"))
                            )
                            swap_ckpt = str(tmp_ckpt) if tmp_ok else None
                            self.model_loader.swap_to_vllm_after_training(
                                swap_ckpt, **swap_kwargs
                            )
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

                        # Fast-student: count this cycle as "trained". The
                        # inline distill path is gated behind
                        # fast_student_distill_inline (default False) because
                        # at this hook point the HF teacher is still resident
                        # (vLLM swap-back happens later) and co-resident
                        # teacher + 1.5B student + AdamW has no GPU-headroom
                        # guarantee on A6000 48GB. When disabled we still
                        # HARVEST pairs above (buffer accumulates) so a
                        # future offline distill pass can consume them; only
                        # the in-line on_trained_cycle call is suppressed.
                        # Flip True after a live-GPU headroom check with the
                        # teacher unloaded — cross-review issue A.
                        if self._fast_student_mgr is not None and bool(getattr(
                            self.config.orchestrator,
                            "fast_student_distill_inline",
                            False,
                        )):
                            try:
                                self._fast_student_mgr.on_trained_cycle(cycle)
                            except Exception as _fs_exc:
                                logger.debug(
                                    "fast_student: on_trained_cycle failed (%s): %s",
                                    type(_fs_exc).__name__, _fs_exc,
                                )

                        # ── FOOM consolidation hooks (Tasks #7, #8) ──────────
                        # Gated by config; default 0 = no-op. Wrapped in broad
                        # try/except so a failure in grow/self-edit never kills
                        # the RSI cycle — we log and continue.
                        #
                        # Snapshot the just-trained pool BEFORE the clear above
                        # would have wiped it — used as distill batches for
                        # grow_and_distill. (Pool was cleared on success path;
                        # we keep a local copy here only for the hook.)
                        _pool_snapshot = list(training_samples)
                        _ge = int(getattr(self.config.orchestrator, "grow_every", 0))
                        if _ge > 0 and cycle % _ge == 0:
                            try:
                                from ..trainer.growth import GrowthConfig, grow_and_distill
                                logger.info(
                                    "[RSI tick %d] grow_every=%d triggered — attempting N→1.5N distill",
                                    cycle, _ge,
                                )

                                # Build pool_batches: tokenize pool items into
                                # {input_ids, labels} dicts using the same
                                # tokenizer the trainer uses.
                                _tok = getattr(self.model_loader, "tokenizer", None)
                                _teacher = getattr(self.model_loader, "model", None)
                                if _tok is None or _teacher is None:
                                    logger.warning(
                                        "[RSI tick %d] grow hook: tokenizer/model unavailable — skipping",
                                        cycle,
                                    )
                                else:
                                    def _mk_batches(samples, tok, max_len=1024):
                                        import torch as _torch
                                        out = []
                                        for s in samples:
                                            prompt = getattr(s, "prompt", None) or (
                                                s.get("prompt", "") if isinstance(s, dict) else ""
                                            )
                                            completion = getattr(s, "completion", None) or (
                                                s.get("completion", "") if isinstance(s, dict) else ""
                                            )
                                            text = f"{prompt}\n{completion}".strip()
                                            if not text:
                                                continue
                                            ids = tok(
                                                text, add_special_tokens=True,
                                                truncation=True, max_length=max_len,
                                                return_tensors="pt",
                                            )["input_ids"]
                                            out.append({"input_ids": ids, "labels": ids.clone()})
                                        return out

                                    pool_batches = _mk_batches(_pool_snapshot, _tok)

                                    # heldout_fn: reuse the quick regression
                                    # probe. It evaluates the CURRENT resident
                                    # model via the diagnostics engine and
                                    # ignores the passed-in module argument
                                    # (the probe reads self.model_loader state).
                                    # This is intentional: growth swaps weights
                                    # in place, so the next probe call reflects
                                    # the student's score once it's resident.
                                    def _heldout_fn(_model_unused):
                                        return self._quick_regression_probe(cycle)

                                    ocfg = self.config.orchestrator
                                    gcfg = GrowthConfig(
                                        grow_every=_ge,
                                        arch_search_enabled=bool(getattr(
                                            ocfg, "arch_search_enabled", False)),
                                        arch_search_every=int(getattr(
                                            ocfg, "arch_search_every", 30)),
                                        arch_search_min_delta=float(getattr(
                                            ocfg, "arch_search_min_delta", 0.005)),
                                    )
                                    _motif = None
                                    if (
                                        gcfg.arch_search_enabled
                                        and cycle % max(1, gcfg.arch_search_every) == 0
                                    ):
                                        try:
                                            from ..trainer.arch_search import run_arch_search
                                            _ares = run_arch_search(
                                                _teacher, gcfg,
                                                train_fn=lambda m: m,
                                                eval_fn=lambda m: 0.0,
                                            )
                                            _motif = _ares.accepted_motif
                                            logger.info(
                                                "[RSI tick %d] arch_search motif=%s delta=%.4f",
                                                cycle, _motif, _ares.best_delta,
                                            )
                                        except Exception as _exc:
                                            logger.warning(
                                                "[RSI tick %d] arch_search failed (%s): %s",
                                                cycle, type(_exc).__name__, _exc,
                                            )
                                    _moe_enabled = bool(getattr(
                                        ocfg, "moe_conversion_enabled", False))
                                    _moe_n = int(getattr(ocfg, "moe_num_experts", 0))
                                    if _moe_enabled and _moe_n > 0:
                                        try:
                                            from ..trainer.moe_conversion import (
                                                MoEConversionConfig,
                                                convert_model_ffn_to_moe,
                                            )
                                            _mcfg = MoEConversionConfig(
                                                num_experts=_moe_n,
                                                top_k=int(getattr(ocfg, "moe_top_k", 2)),
                                                shared_experts=int(getattr(
                                                    ocfg, "moe_shared_experts", 1)),
                                                init_method=str(getattr(
                                                    ocfg, "moe_init_method", "clustering")),
                                                router_noise_std=float(getattr(
                                                    ocfg, "moe_router_noise_std", 0.02)),
                                            )
                                            convert_model_ffn_to_moe(_teacher, _mcfg)
                                            logger.info(
                                                "[RSI tick %d] MoE conversion path taken "
                                                "(experts=%d, top_k=%d) — layer expansion "
                                                "skipped this cycle",
                                                cycle, _mcfg.num_experts, _mcfg.top_k,
                                            )
                                        except Exception as _exc:
                                            logger.warning(
                                                "[RSI tick %d] MoE conversion failed (%s): %s",
                                                cycle, type(_exc).__name__, _exc,
                                            )
                                    else:
                                        _student, _gres = grow_and_distill(
                                            _teacher, pool_batches, gcfg,
                                            heldout_eval=_heldout_fn,
                                        )
                                        logger.info(
                                            "[RSI tick %d] grow_and_distill: grew=%s motif=%s",
                                            cycle, getattr(_gres, "grew", False), _motif,
                                        )
                            except Exception as exc:
                                logger.warning(
                                    "[RSI tick %d] grow_and_distill hook failed (%s): %s",
                                    cycle, type(exc).__name__, exc,
                                )

                        _sse = int(getattr(self.config.orchestrator, "self_edit_every", 0))
                        if getattr(self, "_self_edit_halted", False):
                            logger.warning(
                                "[RSI tick %d] self-edit HALTED globally "
                                "(measurement-corruption alarm raised at cycle %s) — "
                                "skipping meta-cycle.",
                                cycle, getattr(self, "_self_edit_halted_at", "?"),
                            )
                        elif _sse > 0 and cycle % _sse == 0:
                            try:
                                from pathlib import Path as _Path
                                from .self_edit import (
                                    should_run_meta_cycle,
                                    run_self_edit_meta_cycle,
                                    subprocess_smoke_eval,
                                    SelfEditConfig,
                                )
                                if should_run_meta_cycle(cycle, _sse):
                                    logger.info(
                                        "[RSI tick %d] self_edit_every=%d triggered — "
                                        "running meta-cycle",
                                        cycle, _sse,
                                    )

                                    # model_propose: wrap model_loader.generate
                                    # (task_synthesizer's generate internally
                                    # drives the same loader for freeform text).
                                    def _model_propose(prompt: str) -> str:
                                        return self.model_loader.generate(
                                            prompt, max_new_tokens=2048, temperature=0.2
                                        )

                                    # smoke_eval: run a real subprocess harness
                                    # that imports the patched file FROM the
                                    # worktree (baseline path = repo_root ->
                                    # unpatched code; patched path = wt_path ->
                                    # patched code). This gives a genuine
                                    # before/after signal on import-health +
                                    # structural properties, replacing the
                                    # prior in-process evaluator which
                                    # returned identical scores for both
                                    # paths (always-reject pathology).
                                    def _smoke_eval(wt_path) -> float:
                                        return subprocess_smoke_eval(wt_path)

                                    se_cfg = SelfEditConfig(self_edit_every=_sse)
                                    _outcome = run_self_edit_meta_cycle(
                                        cycle=cycle,
                                        repo_root=_Path.cwd(),
                                        candidate_path="src/generator/task_synthesizer.py",
                                        delta_history=list(getattr(self, "_delta_history", []) or []),
                                        model_propose=_model_propose,
                                        smoke_eval=_smoke_eval,
                                        config=se_cfg,
                                    )
                                    logger.info(
                                        "[RSI tick %d] self_edit outcome: %s (%s)",
                                        cycle, _outcome.decision,
                                        "; ".join(_outcome.reasons or []),
                                    )
                            except Exception as exc:
                                logger.warning(
                                    "[RSI tick %d] self_edit hook failed (%s): %s",
                                    cycle, type(exc).__name__, exc,
                                )
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

    def _bank_admitted_as_adversarial(
        self, cycle: int, result: "CycleResult", *, reason: str,
    ) -> None:
        """Append this cycle's admitted candidates to the VoV adversarial bank.

        Called from regression-revert paths (quick-probe and full-eval).
        Each admitted candidate cleared §1.4 quorum yet was implicated in a
        post-training score drop, so it's evidence the property set was
        toothless on this failure mode. Future VoV audits test admitted
        properties against every bank entry; any property that accepts an
        entry is rejected.
        """
        candidates = getattr(result, "_admitted_candidates", None) or []
        if not candidates:
            return
        try:
            from ..verifier.vov_adversarial_bank import get_default_bank
            bank = get_default_bank()
        except Exception as exc:
            logger.debug("VoV bank unavailable: %s", exc)
            return
        added = 0
        for c in candidates:
            try:
                bank.append(
                    problem_id=str(c.get("problem_id", "")),
                    candidate=c.get("candidate"),
                    domain=str(c.get("domain", "code")),
                    problem_ctx=c.get("problem_ctx") or {},
                    cycle=cycle,
                    reason=reason,
                )
                added += 1
            except Exception as exc:
                logger.debug("VoV bank append failed: %s", exc)
        logger.warning(
            "[RSI tick %d] VoV adversarial bank: appended %d candidates (reason=%s; bank size=%d)",
            cycle, added, reason, len(bank),
        )

    def _revert_reference(self, result: "CycleResult") -> float:
        """Compute the held-out reference for the full-eval regression guard.

        Design (task #2, orchestrator-auditor): the 7-cycle overnight run
        showed cycle-1's 0.624 becoming a permanent trap — its raw value
        pinned the reference, so every subsequent cycle that scored below
        0.52 triggered a revert, preventing any forward progress.

        We return the MINIMUM of:
          (a) the confirmed best (``self._best_score``); promotion is
              already gated by best_confirm_cycles + samples_verified +
              capture-alarm ineligibility in ``_check_early_stopping``.
          (b) the trimmed mean of the last K held-out eval scores (drop
              the single highest + single lowest). This absorbs a stale
              historical high once the local level has genuinely drifted
              downward — the revert still fires against a local
              reference, not a fossil.

        When no confirmed best and no history exist, fall back to the
        cycle's own pre-training pre_score so the first trained cycle
        still has a sane comparison point.
        """
        prior_evals = [
            r.eval_score for r in self.history
            if getattr(r, "eval_score", None) is not None
            and not getattr(r, "verifier_capture_alarm", False)
        ]
        K = 5
        window = prior_evals[-K:]
        trimmed_mean: float | None = None
        if len(window) >= 3:
            # Drop min and max before averaging — robust to one-cycle outliers.
            ordered = sorted(window)
            core = ordered[1:-1]
            trimmed_mean = sum(core) / len(core) if core else None
        elif window:
            trimmed_mean = sum(window) / len(window)

        candidates: list[float] = []
        if self._best_score and self._best_score > 0:
            candidates.append(float(self._best_score))
        if trimmed_mean is not None:
            candidates.append(float(trimmed_mean))

        if candidates:
            return min(candidates)
        return float(result.pre_score or 0.0)

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

        # Ensure vLLM is resident before Phase 5b. After training (Step 8),
        # HF is loaded and vLLM is unloaded; if the post-train swap path
        # didn't run (e.g. trainer guard skipped because 0 steps, or
        # save_checkpoint raised silently), we'd end up running 1200-question
        # eval through HF — which on 32B-4bit OOMs and enters endless retry.
        # Guarantee vLLM is up here no matter what happened in Step 8.
        if getattr(self, "_use_vllm", False) and getattr(self.model_loader, "_llm", None) is None:
            logger.info(
                "[Cycle %d] Phase 5b: vLLM not resident (training path skipped "
                "swap-back); reloading from current model_path before eval.",
                cycle,
            )
            try:
                # Prefer the latest saved checkpoint if present; else base.
                ckpt_root = self.config.orchestrator.output_dir / "checkpoints"
                candidate = ckpt_root / f"cycle_{cycle}"
                if candidate.exists() and (candidate / "config.json").exists():
                    self.model_loader.swap_to_vllm_after_training(str(candidate))
                else:
                    self.model_loader.swap_to_vllm_after_training(None)
            except Exception as exc:
                logger.warning(
                    "[Cycle %d] vLLM pre-eval reload failed (%s) — eval may "
                    "fall back to HF path. Error: %s",
                    cycle, type(exc).__name__, exc,
                )
        eval_diag = None
        scores: list[float] = []
        per_rep_domain_scores: list[dict] = []
        per_rep_per_question: list[list[dict]] = []
        # Freeze the diagnostics engine so eval samples ONLY from the
        # deterministic ground-truth bank — curriculum state and templates
        # would otherwise drift question composition between runs and
        # inject ~25% of the noise that masked cycle 2's real signal.
        #
        # Oversample during frozen eval so the HELD_OUT_ONLY partition
        # (~37% of items, task #3) yields ~200 per-domain → ~1200 total.
        # At p=0.5 this puts SE ≈ 0.014, enough to resolve |delta| > 0.02
        # at high confidence. Without the oversample we'd only land ~30
        # held-out items per domain after partition filtering.
        cfg = self.config.diagnostics
        orig_n = cfg.questions_per_domain
        orig_max = cfg.max_questions_per_domain
        heldout_target_per_domain = int(
            getattr(self.config.orchestrator, "heldout_questions_per_domain", 540)
        )
        # Task #10: quick-eval gating. Every heldout_full_every cycles we
        # run the full sweep; on other cycles we shrink the per-domain
        # target so the total lands near heldout_quick_subsample_n after
        # the HELD_OUT_ONLY partition filter (~37% retain). Cycle 1 always
        # runs full so the base-model reference is cached from a full draw.
        _quick_n = int(getattr(
            self.config.orchestrator, "heldout_quick_subsample_n", 0
        ))
        _full_every = int(getattr(
            self.config.orchestrator, "heldout_full_every", 5
        ))
        is_full_cycle = (
            cycle == 1
            or _full_every <= 1
            or (cycle % max(1, _full_every) == 0)
        )
        try:
            result.heldout_eval_kind = "full" if is_full_cycle else "quick"
        except AttributeError:
            pass
        if _quick_n > 0 and not is_full_cycle:
            # Task #16: stratify WITHIN target_n by domain proportion, not
            # by ceil-per-domain × N_domains. See
            # _quick_eval_stratified_targets for math + rationale; unit-
            # tested in test_quick_eval_stratification.
            _pre_per_domain, _expected_total = _quick_eval_stratified_targets(
                target_n=_quick_n,
                n_domains=max(1, len(cfg.domains)),
                min_per_domain=int(cfg.min_questions_per_domain),
            )
            heldout_target_per_domain = min(
                heldout_target_per_domain, _pre_per_domain
            )
            logger.info(
                "[Cycle %d] Phase 5b: QUICK eval — target_n=%d "
                "per-domain=%d (expected_post_filter_total≈%d, "
                "full every %d cycles)",
                cycle, _quick_n, heldout_target_per_domain,
                _expected_total, _full_every,
            )
        prev_mode = getattr(self.diagnostics, "_frozen_eval_mode", False)
        self.diagnostics._frozen_eval_mode = True
        try:
            cfg.max_questions_per_domain = max(orig_max, heldout_target_per_domain)
            cfg.questions_per_domain = heldout_target_per_domain
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
            cfg.questions_per_domain = orig_n
            cfg.max_questions_per_domain = orig_max
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
        delta: float | None = None
        if prev_eval is not None:
            delta = mean_score - prev_eval
            symbol = "+" if delta > 0 else ""
            logger.info(f"    (prev {prev_eval:.3f}, {symbol}{delta:.3f})")

        # Paired-sample variance reduction (task #3). When the previous
        # cycle cached its per-question records, compute a paired delta +
        # SE. Frozen eval holds the question set constant, so pairing by
        # (prompt, expected) matches every question — typical VR is 3-5×.
        # Gated on paired_eval_enabled (default True, consolidation flip).
        if getattr(self.config.orchestrator, "paired_eval_enabled", True):
            try:
                prev_per_q = next(
                    (
                        getattr(r, "_eval_per_rep_per_question", None)
                        for r in reversed(self.history)
                        if getattr(r, "_eval_per_rep_per_question", None)
                    ),
                    None,
                )
                cur_per_q = per_rep_per_question[-1] if per_rep_per_question else None
                if prev_per_q and cur_per_q:
                    from ..diagnostics.paired_eval import paired_delta
                    pd = paired_delta(prev_per_q[-1], cur_per_q)
                    if pd is not None:
                        result.paired_delta = pd.delta
                        result.paired_delta_se = pd.delta_se
                        result.paired_delta_n = pd.n
                        result.paired_variance_reduction = pd.variance_reduction
                        logger.info(
                            "    paired delta: %+.4f ± %.4f (n=%d, z=%.2f, VR=%.1fx)",
                            pd.delta, pd.delta_se, pd.n, pd.z, pd.variance_reduction,
                        )
            except Exception as exc:
                logger.warning("paired delta computation failed (non-fatal): %s", exc)

        # Anchor eval (Task #1, ground-truth). Ground-truth external
        # benchmarks — detects verifier capture when internal loop improves
        # while real-world tasks regress.
        anchor_delta: float | None = None
        if self.config.orchestrator.anchor_eval_enabled:
            try:
                from ..utils.external_benchmarks import (
                    run_anchor_eval,
                    detect_verifier_capture,
                    fire_capture_alarm,
                )
                ocfg = self.config.orchestrator
                benchmarks = list(ocfg.anchor_eval_benchmarks)
                per_bench = max(1, ocfg.anchor_eval_size // max(1, len(benchmarks)))

                def _anchor_model_fn(prompt: str) -> str:
                    return self.model_loader.generate(
                        prompt, max_new_tokens=256, temperature=0.0, top_p=1.0
                    )

                def _anchor_batch_fn(prompts: list[str]) -> list[str]:
                    # One vLLM batched call across the full anchor set —
                    # ~15x faster than serial generate() on a 100-item set.
                    return list(self.model_loader.generate_batch(
                        prompts, max_new_tokens=256, temperature=0.0, top_p=1.0,
                    ))

                summary = run_anchor_eval(
                    _anchor_model_fn,
                    benchmarks=benchmarks,
                    per_benchmark=per_bench,
                    cache_dir=ocfg.anchor_eval_cache_dir,
                    batch_model_fn=_anchor_batch_fn,
                )
                result.anchor_score = summary["anchor_score"]
                logger.info(
                    "  anchor eval: %.3f (n=%d) per_bench=%s per_bench_n=%s distinct=%s offline=%s",
                    result.anchor_score, summary["n"], summary["per_benchmark"],
                    summary.get("per_benchmark_n", {}),
                    summary.get("per_benchmark_distinct", {}),
                    summary.get("per_benchmark_offline", {}),
                )
                # Task #9: per-benchmark suspicious-clean alarms (degenerate
                # predictions or offline-fixture-only score). Logged as WARN
                # inside run_anchor_eval; surface here so cycle_metrics JSON
                # carries the list for cross-reviewer inspection.
                _suspect = summary.get("per_benchmark_suspect") or []
                if _suspect:
                    logger.warning(
                        "  anchor eval SUSPECT per-benchmark alarms: %s",
                        _suspect,
                    )
                # Task #11 concern #2: mode-collapse automatic response.
                # If distinct/n < threshold on any benchmark that ran this
                # cycle, mark cycle ineligible for best-promotion. This
                # catches "score looks clean but outputs collapsed" before
                # the cycle gets confirmed as the new best reference.
                _mc_threshold = float(getattr(
                    ocfg, "mode_collapse_distinct_threshold", 0.0,
                ))
                if _mc_threshold > 0:
                    _pb_n = summary.get("per_benchmark_n", {}) or {}
                    _pb_distinct = summary.get("per_benchmark_distinct", {}) or {}
                    _collapsed: list[dict] = []
                    for _b, _n in _pb_n.items():
                        if _n <= 0:
                            continue
                        _d = _pb_distinct.get(_b, 0)
                        _ratio = _d / _n
                        if _ratio < _mc_threshold:
                            _collapsed.append({
                                "benchmark": _b,
                                "distinct": _d,
                                "n": _n,
                                "ratio": round(_ratio, 3),
                            })
                    if _collapsed:
                        result.mode_collapse_detected = True
                        logger.warning(
                            "  mode_collapse_detected=True benchmarks=%s "
                            "(distinct/n < %.2f) — cycle ineligible for best-promotion",
                            _collapsed, _mc_threshold,
                        )
                # Baseline-drift canary (task #4). The cycle_0 anchor score
                # is frozen here; every 10 cycles we compare the current
                # anchor to that baseline. If drift > ±0.01 we HALT self-edit
                # globally: either the anchor bank changed under our feet,
                # or the model is gaming the measurement surface — in either
                # case, self-edit must stop until a human audits.
                #
                # The canary assumes the anchor bank + scorer are
                # deterministic w.r.t. model-id; if the model did not change
                # since cycle_0 (e.g. cycle_0 itself re-evaluated), drift
                # would be exactly 0. We use ±0.01 to absorb generator
                # noise for anchor benchmarks with small n.
                BASELINE_DRIFT_EPS = 0.01
                if cycle == 0 and self._cycle0_anchor_score is None:
                    self._cycle0_anchor_score = result.anchor_score
                    logger.info(
                        "  baseline-drift canary: cycle_0 anchor=%.4f frozen",
                        self._cycle0_anchor_score,
                    )
                elif (
                    self._cycle0_anchor_score is not None
                    and cycle > 0
                    and cycle % 10 == 0
                    and not self._self_edit_halted
                ):
                    drift = result.anchor_score - self._cycle0_anchor_score
                    logger.info(
                        "  baseline-drift canary: cycle_%d anchor=%.4f vs cycle_0=%.4f drift=%+.4f",
                        cycle, result.anchor_score, self._cycle0_anchor_score, drift,
                    )
                    if abs(drift) > BASELINE_DRIFT_EPS:
                        logger.error(
                            "  MEASUREMENT-CORRUPTION ALARM: cycle_0 anchor "
                            "drifted by %+.4f (> ±%.4f). Halting self-edit "
                            "globally. Human audit required.",
                            drift, BASELINE_DRIFT_EPS,
                        )
                        self._self_edit_halted = True
                        self._self_edit_halted_at = cycle
                        result.verifier_capture_alarm = True
                        try:
                            alarm_dir = self.config.orchestrator.output_dir / "cycle_metrics"
                            alarm_dir.mkdir(parents=True, exist_ok=True)
                            with open(alarm_dir / "measurement_corruption.jsonl", "a") as _af:
                                _af.write(json.dumps({
                                    "cycle": cycle,
                                    "cycle0_anchor": self._cycle0_anchor_score,
                                    "current_anchor": result.anchor_score,
                                    "drift": drift,
                                    "threshold": BASELINE_DRIFT_EPS,
                                    "timestamp": time.time(),
                                }) + "\n")
                        except Exception as _aexc:
                            logger.debug("measurement_corruption.jsonl write failed: %s", _aexc)
                prev_anchor = next(
                    (r.anchor_score for r in reversed(self.history)
                     if getattr(r, "anchor_score", None) is not None),
                    None,
                )
                if prev_anchor is not None:
                    anchor_delta = result.anchor_score - prev_anchor
                    logger.info(
                        "    (anchor prev %.3f, %s%.3f)",
                        prev_anchor,
                        "+" if anchor_delta > 0 else "",
                        anchor_delta,
                    )
                if delta is not None and anchor_delta is not None:
                    if detect_verifier_capture(
                        internal_delta=delta,
                        anchor_delta=anchor_delta,
                        threshold=ocfg.verifier_capture_alarm_threshold,
                    ):
                        fire_capture_alarm(
                            cycle=cycle,
                            internal_delta=delta,
                            anchor_delta=anchor_delta,
                        )
                        result.verifier_capture_alarm = True
                        # Task #2: toothed alarm. Before this block the alarm
                        # only logged — pipeline continued, a captured-verifier
                        # cycle could still become best. Now:
                        #  (a) bump consecutive-alarm counter; halt self-edit
                        #      globally if it exceeds configured threshold,
                        #  (b) revert vLLM to last confirmed-best checkpoint
                        #      so next cycle starts from weights that passed
                        #      the ground-truth bar, not the captured ones,
                        #  (c) bank this cycle's admitted candidates as VoV
                        #      adversarial — they cleared §1.4 yet correlate
                        #      with an anchor drop, so they're exactly the
                        #      toothless-verifier pattern VoV exists to catch.
                        # _check_early_stopping sees verifier_capture_alarm
                        # and excludes this cycle from best-promotion.
                        self._capture_alarm_consecutive += 1
                        halt_n = int(getattr(
                            ocfg, "verifier_capture_halt_consecutive", 2,
                        ))
                        if (
                            self._capture_alarm_consecutive >= halt_n
                            and not self._self_edit_halted
                        ):
                            logger.error(
                                "  VERIFIER-CAPTURE HALT: %d consecutive "
                                "alarm cycles (>=%d). Halting self-edit "
                                "globally. Human audit required.",
                                self._capture_alarm_consecutive, halt_n,
                            )
                            self._self_edit_halted = True
                            self._self_edit_halted_at = cycle
                        try:
                            best_cycle = self._best_checkpoint_cycle
                            if best_cycle is not None:
                                best_ckpt = (
                                    ocfg.output_dir / "checkpoints"
                                    / f"cycle_{best_cycle}"
                                )
                                if (
                                    best_ckpt.exists()
                                    and (best_ckpt / "config.json").exists()
                                ):
                                    logger.warning(
                                        "  capture-alarm revert: reloading "
                                        "vLLM from cycle %d",
                                        best_cycle,
                                    )
                                    self.model_loader.swap_to_vllm_after_training(
                                        str(best_ckpt)
                                    )
                                else:
                                    logger.warning(
                                        "  capture-alarm revert: best "
                                        "checkpoint cycle %d missing; "
                                        "falling back to base.",
                                        best_cycle,
                                    )
                                    self.model_loader.swap_to_vllm_after_training(
                                        str(getattr(
                                            self.model_loader, "model_path", None,
                                        ))
                                    )
                            else:
                                logger.warning(
                                    "  capture-alarm revert: no confirmed-"
                                    "best checkpoint yet; reverting to base.",
                                )
                                self.model_loader.swap_to_vllm_after_training(
                                    str(getattr(
                                        self.model_loader, "model_path", None,
                                    ))
                                )
                        except Exception as _rexc:
                            logger.warning(
                                "  capture-alarm revert failed (%s): %s",
                                type(_rexc).__name__, _rexc,
                            )
                        self._bank_admitted_as_adversarial(
                            cycle, result,
                            reason=(
                                f"verifier_capture internal=+{delta:.3f} "
                                f"anchor={anchor_delta:+.3f}"
                            ),
                        )
                    else:
                        # Clean anchor cycle: reset the consecutive-alarm
                        # counter so isolated alarms don't accumulate forever.
                        self._capture_alarm_consecutive = 0
            except Exception as exc:
                logger.warning("anchor_eval failed (non-fatal): %s", exc)

        # Curriculum escalation: feed held-out per-question records + the
        # delta into the DifficultyTracker, ratchet the proposal difficulty
        # floor, and persist state so restarts pick up where we left off.
        try:
            last_per_q = (
                per_rep_per_question[-1] if per_rep_per_question else []
            )
            self.difficulty_tracker.record_heldout(last_per_q)
            if delta is not None:
                self.difficulty_tracker.update_ratchet(delta, cycle=cycle)
            result.difficulty_frontier = self.difficulty_tracker.frontier()
            result.difficulty_floor = self.difficulty_tracker.difficulty_floor
            self.difficulty_tracker.save()
            # cycle_metrics jsonl: one line per held-out eval with the
            # ratchet state + frontier. Independent of the full per-cycle
            # JSON dump so operators can tail it cheaply across runs.
            try:
                out_dir = self.config.orchestrator.output_dir / "cycle_metrics"
                out_dir.mkdir(parents=True, exist_ok=True)
                jsonl_path = out_dir / "curriculum.jsonl"
                with open(jsonl_path, "a") as _f:
                    _f.write(json.dumps({
                        "cycle": cycle,
                        "eval_score": mean_score,
                        "heldout_delta": delta,
                        "anchor_score": result.anchor_score,
                        "anchor_delta": anchor_delta,
                        "verifier_capture_alarm": result.verifier_capture_alarm,
                        "frontier": result.difficulty_frontier,
                        "difficulty_floor": result.difficulty_floor,
                        "proposals_last_accepted": self.difficulty_tracker.last_accepted,
                        "proposals_last_rejected": self.difficulty_tracker.last_rejected,
                        "timestamp": time.time(),
                    }) + "\n")
            except Exception as exc:
                logger.debug("curriculum.jsonl write failed: %s", exc)
            logger.info(
                "  curriculum: frontier=%r floor=%.2f (delta=%s)",
                result.difficulty_frontier or "(none)",
                result.difficulty_floor,
                f"{delta:+.3f}" if delta is not None else "n/a",
            )
        except Exception as exc:
            logger.debug("difficulty_tracker post-eval update failed: %s", exc)

        # meta_meta cycle record (Task #4). Appends one row per cycle to the
        # JSONL history with the components-active snapshot + held-out delta,
        # so component_contributions / effective_allow_list can attribute
        # future improvements. Gated on meta_meta_enabled (default True).
        if getattr(self.config.orchestrator, "meta_meta_enabled", True) and delta is not None:
            try:
                from pathlib import Path as _P
                from .meta_meta import record_cycle as _mm_record
                ocfg = self.config.orchestrator
                scfg = self.config.synthesis
                # fast_student is "active" when the manager was constructed
                # and its config.enabled is True — that means the solve_batch
                # harvest path is live and on_trained_cycle is being called.
                _fs_mgr = getattr(self, "_fast_student_mgr", None)
                _fs_active = bool(
                    _fs_mgr is not None
                    and getattr(getattr(_fs_mgr, "cfg", None), "enabled", False)
                )
                components_active = {
                    "fast_student": _fs_active,
                    "ood": bool(getattr(scfg, "ood_enabled", False)),
                    "curriculum_ratchet": bool(
                        getattr(self.config.diagnostics, "difficulty_curriculum", False)
                    ),
                    "growth": int(getattr(ocfg, "grow_every", 0)) > 0,
                    "self_edit": int(getattr(ocfg, "self_edit_every", 0)) > 0,
                    "grpo": self.config.trainer.training_mode == "grpo",
                }
                _grad_health = None
                _gs = getattr(self.trainer, "_last_grad_summary", None)
                if _gs is not None:
                    try:
                        _grad_health = _gs.to_dict()
                    except Exception:
                        _grad_health = None
                _mm_record(
                    _P(ocfg.meta_meta_history_path),
                    cycle_id=cycle,
                    components_active=components_active,
                    held_out_delta=float(delta),
                    self_edit_tier=None,
                    gradient_health=_grad_health,
                )
            except Exception as exc:
                logger.debug("meta_meta record_cycle failed (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # Task #10: per-phase wall-time observability
    # ------------------------------------------------------------------
    def _record_wall_time(self, cycle: int, result: CycleResult) -> None:
        """Append each phase's wall time (ms) to meta_meta's wall-time sidecar
        and, at the end of every 10-cycle window, log the cycle-time trend.
        """
        ocfg = self.config.orchestrator
        wall_path_s = getattr(
            ocfg, "meta_meta_wall_time_path", "outputs/meta_meta_wall_time.jsonl",
        )
        try:
            from pathlib import Path as _P
            from .meta_meta import (
                record_wall_time as _rwt,
                load_wall_time as _lwt,
                wall_time_trend as _wtt,
            )
            wall_path = _P(wall_path_s)
            for phase, seconds in (result.phase_times or {}).items():
                if seconds is None or seconds <= 0:
                    continue
                _rwt(wall_path, cycle, phase, float(seconds) * 1000.0)
            if cycle > 0 and cycle % 10 == 0:
                trend = _wtt(_lwt(wall_path), window=10)
                if trend is not None:
                    direction = "down" if trend["pct_change_down"] >= 0 else "up"
                    logger.info(
                        "[meta_meta] cycle time trending %s by %.1f%%/10 cycles "
                        "(older=%.0fms newer=%.0fms)",
                        direction, abs(trend["pct_change_down"]),
                        trend["mean_ms_older"], trend["mean_ms_newer"],
                    )
        except Exception as exc:
            logger.debug("wall_time record failed (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # Task #13: compute allocator + component proposer hooks (default off)
    # ------------------------------------------------------------------
    def _consult_compute_allocator(self, cycle: int, result: CycleResult) -> None:
        """If enabled, pick a strategy via UCB1 and apply it for THIS cycle."""
        ocfg = self.config.orchestrator
        if not getattr(ocfg, "compute_allocator_enabled", False):
            return
        try:
            from pathlib import Path as _P
            from .compute_allocator import allocator_from_history
            hist_path = _P(getattr(
                ocfg, "compute_allocator_history_path",
                "outputs/compute_allocator_history.jsonl",
            ))
            budget = float(getattr(ocfg, "compute_allocator_budget_tokens", 1e9))
            alloc = allocator_from_history(hist_path)
            strat = alloc.select(remaining_budget=budget)
            try:
                self.config.synthesis.candidates_per_problem = int(strat.k_candidates)
            except Exception:
                pass
            try:
                self.config.generator.max_new_tokens = int(strat.token_budget)
            except Exception:
                pass
            logger.info(
                "[cycle %d] compute_allocator → strategy=%s k=%d tokens=%d mode=%s",
                cycle, strat.name, strat.k_candidates, strat.token_budget,
                strat.train_mode,
            )
        except Exception as exc:
            logger.debug("compute_allocator consult failed (non-fatal): %s", exc)

    def _maybe_run_component_proposer(self, cycle: int, result: CycleResult) -> None:
        """Run ComponentProposer every N cycles. 0 = off (default)."""
        ocfg = self.config.orchestrator
        every = int(getattr(ocfg, "component_proposer_every", 0) or 0)
        if every <= 0 or cycle <= 0 or cycle % every != 0:
            return
        proposer = getattr(self, "component_proposer", None)
        if proposer is None:
            logger.debug(
                "[cycle %d] component_proposer_every=%d but no proposer wired "
                "on self.component_proposer — skipping", cycle, every,
            )
            return
        try:
            if hasattr(proposer, "propose_and_test"):
                proposer.propose_and_test(cycle=cycle)
            logger.info("[cycle %d] component_proposer executed", cycle)
        except Exception as exc:
            logger.warning(
                "[cycle %d] component_proposer failed (%s): %s",
                cycle, type(exc).__name__, exc,
            )

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

        # GRPO auto-switch: escalate SFT→GRPO on paired-held-out plateau.
        # Uses the new should_switch_to_grpo signature: paired_delta +
        # z-score gate over a sliding window of training-cycle records.
        try:
            if self.config.trainer.training_mode == "sft":
                from ..trainer.grpo import should_switch_to_grpo
                hist = []
                for r in self.history:
                    if r.paired_delta is None:
                        continue
                    hist.append({
                        "paired_delta": float(r.paired_delta),
                        "paired_se": float(r.paired_delta_se or 0.0),
                        "n": int(r.paired_delta_n or 0),
                    })
                if should_switch_to_grpo(hist):
                    logger.info(
                        "[meta] SFT plateau detected (paired-delta z-gate) — "
                        "switching training_mode to grpo at cycle %d", cycle,
                    )
                    self.config.trainer.training_mode = "grpo"
        except Exception as exc:
            logger.debug("grpo auto-switch check failed (non-fatal): %s", exc)

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

        ocfg = self.config.orchestrator
        min_samples = int(getattr(ocfg, "best_min_samples_verified", 8))
        confirm_n = max(1, int(getattr(ocfg, "best_confirm_cycles", 2)))

        eligible = (
            result.samples_verified >= min_samples
            and not getattr(result, "verifier_capture_alarm", False)
            # Task #11 concern #2: mode-collapse cycles cannot be promoted.
            # A clean score with degenerate outputs is the exact failure
            # mode that would lock in a reference the verifier "approves"
            # of but that doesn't generalize.
            and not getattr(result, "mode_collapse_detected", False)
            # Task #15: a cycle that just tripped the post-Phase-5b
            # regression-revert guard is hard-ineligible for promotion
            # AND for streak advancement. The revert itself is
            # authoritative: training regressed the model, so no streak
            # from prior cycles may ride through this cycle. Live bug:
            # cycle 2 held-out=0.478 vs reference=0.633 was reverted,
            # but the very next log line was "streak=1/2 — awaiting
            # confirmation". Setting this flag into `eligible` routes
            # the cycle through the reset branch below.
            and not getattr(result, "regression_reverted", False)
        )

        # Task #11 concern #3: regression guard. Before advancing the
        # streak, check whether the current cycle REGRESSED relative to any
        # prior cycle's eval/post score. The overnight run showed
        # "streak=1/2 — awaiting confirmation" tick for cycle 2 even though
        # cycle 2 was reverted vs. reference. Root cause: _best_score=0 on
        # first-ever confirmed-best means current_score > _best_score is
        # trivially true, and the old "else" branch restarted the streak
        # at 1 on a LOWER score. Fix: if there's a prior history cycle
        # with score > current_score + tolerance, treat as regression and
        # reset pending without incrementing. Tolerance 0.005 matches the
        # _degradation_count threshold used below.
        REGRESSION_TOL = 0.005

        def _eligible_for_max(r: CycleResult) -> bool:
            if r.samples_verified < min_samples:
                return False
            if getattr(r, "verifier_capture_alarm", False):
                return False
            if getattr(r, "mode_collapse_detected", False):
                return False
            return True

        prior_scores = [
            (r.eval_score if r.eval_score is not None else r.post_score)
            for r in self.history[:-1]  # exclude the current cycle's own record
            if _eligible_for_max(r)
        ]
        prior_scores = [s for s in prior_scores if s is not None]
        regressed_vs_prior = bool(prior_scores) and (
            max(prior_scores) > current_score + REGRESSION_TOL
        )

        # Task #15: explicit no-regression-vs-best gate. When a confirmed best
        # exists, require current_score >= best_score - 0.005 before the streak
        # may advance. Before task #15, `current_score > self._best_score` was
        # the only gate, and with _best_score=0 (pre-promotion) any positive
        # score sailed through — that is how cycle 2's 0.478 advanced the
        # streak despite triggering a regression-revert against reference
        # 0.633. The `regression_reverted` flag is already folded into
        # `eligible` above; this check adds the symmetric protection for the
        # post-promotion regime.
        not_regressed_vs_best = (
            self._best_score <= 0.0
            or current_score >= self._best_score - 0.005
        )

        if (
            current_score > self._best_score
            and eligible
            and not regressed_vs_prior
            and not_regressed_vs_best
        ):
            # Candidate new best — lagged N-cycle confirmation gate (task #2).
            if (
                self._pending_best_cycle is not None
                and current_score >= self._pending_best_score - 1e-9
            ):
                self._pending_best_streak += 1
            else:
                self._pending_best_score = current_score
                self._pending_best_cycle = cycle
                self._pending_best_streak = 1

            if self._pending_best_streak >= confirm_n:
                # Adopt the EARLIER cycle that first hit this high-water mark:
                # its weights produced the reproduced score. Revert path should
                # point at the weights that actually did it.
                self._best_score = self._pending_best_score
                self._best_checkpoint_cycle = self._pending_best_cycle
                self._pending_best_streak = 0
                self._degradation_count = 0
                logger.info(
                    "  PROMOTE: new confirmed best held-out=%.4f (cycle %d, "
                    "confirmed after %d consecutive eligible cycles)",
                    self._best_score, self._best_checkpoint_cycle, confirm_n,
                )
                return False, ""
            logger.info(
                "  best-candidate: held-out=%.4f (cycle %d) streak=%d/%d — "
                "awaiting confirmation",
                self._pending_best_score, self._pending_best_cycle,
                self._pending_best_streak, confirm_n,
            )
            return False, ""
        elif current_score > self._best_score and regressed_vs_prior:
            # Task #11 concern #3: current_score beat the confirmed best
            # (which is 0.0 or stale), but is strictly below a prior cycle's
            # score. Do NOT advance the streak, do NOT log "streak=1/2 —
            # awaiting confirmation" — that was the overnight-run bug where
            # cycle 2 ticked streak=1 despite regressing vs. cycle 1.
            prior_max = max(prior_scores)
            logger.warning(
                "  best-candidate REGRESSION: held-out=%.4f cycle=%d < "
                "prior_max=%.4f (tol=%.3f) — streak NOT advanced.",
                current_score, cycle, prior_max, REGRESSION_TOL,
            )
            self._pending_best_streak = 0
            self._pending_best_cycle = None
            self._pending_best_score = 0.0
        elif current_score > self._best_score and not eligible:
            # Outlier guard: score beat the bar but cycle is ineligible (tiny
            # sample count or capture-alarm). Reset pending streak outright —
            # these are exactly the cycles that gave us the 1-sample/2-step
            # reference-bank lock-in.
            logger.warning(
                "  best-candidate IGNORED: held-out=%.4f cycle=%d but "
                "samples_verified=%d (<%d) or capture_alarm=%s or "
                "mode_collapse=%s — ineligible for best-promotion.",
                current_score, cycle, result.samples_verified, min_samples,
                getattr(result, "verifier_capture_alarm", False),
                getattr(result, "mode_collapse_detected", False),
            )
            self._pending_best_streak = 0
            self._pending_best_cycle = None
            self._pending_best_score = 0.0
            return False, ""
        else:
            # Score didn't beat best — reset pending streak. Only CONSECUTIVE
            # at-or-above cycles count toward confirmation.
            self._pending_best_streak = 0
            self._pending_best_cycle = None
            self._pending_best_score = 0.0

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
        # Task #2: also protect the pending-best candidate so that if it gets
        # confirmed next cycle, _check_early_stopping can still revert to its
        # weights. Without this, a pending cycle 3 high-water could be pruned
        # between cycles 3 and 4 right before its confirmation lands.
        if self._pending_best_cycle is not None:
            keep_names.add(f"cycle_{self._pending_best_cycle}")
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
                "pending_best_score": self._pending_best_score,
                "pending_best_cycle": self._pending_best_cycle,
                "pending_best_streak": self._pending_best_streak,
                "capture_alarm_consecutive": self._capture_alarm_consecutive,
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
