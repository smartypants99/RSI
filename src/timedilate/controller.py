import logging
import time
from dataclasses import dataclass, field
from timedilate.config import TimeDilateConfig
from timedilate.improver import ImprovementEngine
from timedilate.scorer import Scorer
from timedilate.directives import DirectiveGenerator
from timedilate.checkpoint import CheckpointManager
from timedilate.metrics import RunMetrics
from timedilate.logging_config import log_cycle_summary
from timedilate.meta import MetaLearner


@dataclass
class DilationResult:
    output: str
    score: int
    cycles_completed: int
    elapsed_seconds: float
    convergence_detected: bool
    interrupted: bool = False
    resumed_from_cycle: int = 0
    metrics: RunMetrics | None = None

    def to_report(self, config: "TimeDilateConfig | None" = None) -> dict:
        """Export a complete run report as a dict for JSON serialization."""
        from timedilate import __version__
        import time as _time
        report = {
            "version": __version__,
            "timestamp": _time.time(),
            "score": self.score,
            "cycles_completed": self.cycles_completed,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "convergence_detected": self.convergence_detected,
            "interrupted": self.interrupted,
            "output_length": len(self.output),
        }
        if config:
            report["config"] = {
                "model": config.model,
                "dilation_factor": config.dilation_factor,
                "branch_factor": config.branch_factor,
                "budget_seconds": config.budget_seconds,
                "temperature": config.temperature,
                "use_reflection": config.use_reflection,
                "use_meta_learning": config.use_meta_learning,
                "score_weights": config.score_weights,
                "task_type_override": config.task_type_override,
            }
        if self.metrics:
            report["metrics"] = self.metrics.to_dict()
        return report


logger = logging.getLogger(__name__)


class DilationController:
    def __init__(self, config: TimeDilateConfig, engine):
        config.validate()
        self.config = config
        self.engine = engine
        self.improver = ImprovementEngine(engine, config)
        self.scorer = Scorer()
        self.directives = DirectiveGenerator()
        self.checkpoint = CheckpointManager(config.checkpoint_dir)
        self.meta = MetaLearner(path=config.checkpoint_dir + "/.meta.json") if config.use_meta_learning else None

    def _build_history_summary(self, metrics: RunMetrics, max_entries: int = 5) -> str:
        """Build a structured summary that highlights what worked and what to avoid."""
        if not metrics.cycles:
            return ""
        recent = metrics.cycles[-max_entries:]

        worked = []
        failed = []
        for c in recent:
            if c.score > c.previous_score:
                worked.append(f'"{c.directive}" (+{c.score - c.previous_score} pts)')
            else:
                failed.append(f'"{c.directive}"')

        lines = []
        if worked:
            lines.append(f"Approaches that improved the score: {'; '.join(worked)}")
        if failed:
            lines.append(f"Approaches that did NOT help (avoid these): {'; '.join(failed)}")
        lines.append(f"Current trajectory: {' -> '.join(str(c.score) for c in recent)}")
        return "\n".join(lines)

    def _adaptive_branch_factor(self, cycle: int, metrics: RunMetrics) -> int:
        """Adapt branch factor based on timing, stagnation, and cycle position."""
        base = self.config.branch_factor
        refinement_cycles = self.config.dilation_factor - 1
        if not metrics.cycles:
            return base

        # If last cycle took more than budget_seconds / dilation_factor,
        # we're running slow — reduce branching
        last_elapsed = metrics.cycles[-1].elapsed_seconds
        target_per_cycle = self.config.budget_seconds / max(self.config.dilation_factor, 1)
        if target_per_cycle > 0 and last_elapsed > target_per_cycle * 2:
            reduced = max(1, base // 2)
            if reduced < base:
                logger.info("Cycle %d slow (%.1fs), reducing branches %d -> %d",
                            cycle, last_elapsed, base, reduced)
            return reduced

        # If stagnating, increase branches to explore more
        if metrics.stagnant_streak >= 3 and base < 5:
            return min(base + 1, 5)

        # Late-cycle reduction: in the last 20% of cycles with high scores,
        # reduce branches since gains are marginal and we want speed
        if refinement_cycles >= 5 and cycle > refinement_cycles * 0.8:
            current_score = metrics.cycles[-1].score if metrics.cycles else 0
            if current_score >= 80 and base > 1:
                return max(1, base - 1)

        return base

    def _effective_convergence_threshold(self, current_score: int) -> int:
        """Adaptive convergence threshold: more patient at high scores
        where improvement is naturally harder."""
        base = self.config.convergence_threshold
        if current_score >= 85:
            return base + 3  # much harder to improve, allow more attempts
        if current_score >= 70:
            return base + 1
        return base

    def _should_prefer_generated(self, metrics: RunMetrics) -> bool:
        """After enough data, prefer generated directives if they outperform builtins."""
        eff = metrics.directive_effectiveness
        if "builtin" not in eff or "generated" not in eff:
            return False
        # Need at least 3 builtin and 2 generated samples
        builtin_count = sum(1 for c in metrics.cycles if c.directive_source == "builtin")
        generated_count = sum(1 for c in metrics.cycles if c.directive_source == "generated")
        if builtin_count < 3 or generated_count < 2:
            return False
        return eff["generated"] > eff["builtin"] + 0.2  # need meaningful advantage

    def run(self, prompt: str, on_cycle=None, resume: bool = False) -> DilationResult:
        start = time.time()
        task_type = self.config.task_type_override or self.directives.classify_task(prompt)
        self.improver.task_type = task_type
        refinement_cycles = self.config.dilation_factor - 1

        metrics = RunMetrics(
            prompt=prompt,
            task_type=task_type,
            dilation_factor=self.config.dilation_factor,
            branch_factor=self.config.branch_factor,
            start_time=start,
        )

        # Resume from checkpoint if requested
        resumed_from = 0
        if resume:
            checkpoint = self.checkpoint.load_latest()
            if checkpoint:
                # Validate prompt matches — don't resume a different task
                saved_prompt = checkpoint.get("prompt", "")
                if saved_prompt and saved_prompt != prompt:
                    logger.warning(
                        "Checkpoint prompt mismatch — saved: '%.50s...', current: '%.50s...'. "
                        "Starting fresh instead of resuming.",
                        saved_prompt, prompt
                    )
                    resume = False
                else:
                    current_best = checkpoint["output"]
                    current_score = checkpoint["score"]
                    resumed_from = checkpoint["cycle"]
                    logger.info("Resumed from checkpoint at cycle %d (score %d)",
                                resumed_from, current_score)
            else:
                resume = False

        if not resume:
            # Initial generation
            current_best = self.engine.generate(prompt)

            if refinement_cycles <= 0:
                return DilationResult(
                    output=current_best,
                    score=0,
                    cycles_completed=0,
                    elapsed_seconds=time.time() - start,
                    convergence_detected=False,
                    metrics=metrics,
                )

            # Score the initial output (task-aware when applicable)
            if task_type != "general":
                score_prompt = self.scorer.build_task_aware_scoring_prompt(prompt, current_best, task_type)
            else:
                score_prompt = self.scorer.build_scoring_prompt(prompt, current_best)
            raw_score = self.engine.generate(
                score_prompt, temperature=self.config.scoring_temperature
            )
            current_score = self.scorer.parse_score(raw_score)

        self.improver.initial_output_length = len(current_best)

        # Scoring consistency check: re-score initial output to detect unreliable scoring
        if not resume and refinement_cycles >= 3:
            try:
                if task_type != "general":
                    check_prompt = self.scorer.build_task_aware_scoring_prompt(prompt, current_best, task_type)
                else:
                    check_prompt = self.scorer.build_scoring_prompt(prompt, current_best)
                raw_check = self.engine.generate(check_prompt, temperature=self.config.scoring_temperature)
                check_score = self.scorer.parse_score(raw_check)
                score_delta = abs(current_score - check_score)
                if score_delta > 15:
                    logger.warning(
                        "Scoring inconsistency detected: %d vs %d (delta=%d). "
                        "Enabling ensemble scoring for this run.",
                        current_score, check_score, score_delta
                    )
                    self.improver.force_ensemble = True
                    # Use average as the baseline
                    current_score = (current_score + check_score) // 2
                elif score_delta > 0:
                    logger.info("Scoring consistency check: %d vs %d (delta=%d, acceptable)",
                                current_score, check_score, score_delta)
            except Exception:
                logger.debug("Scoring consistency check failed, continuing normally")

        no_improvement_count = 0
        convergence_detected = False
        built_in_exhausted = False
        directive_offset = 0
        built_in_count = len(self.directives.get_directives(task_type))
        meta_directives = self.meta.best_directives(task_type, top_n=3) if self.meta else []
        completed_cycles = resumed_from
        start_cycle = resumed_from
        # Track best output seen across the entire run (may differ from current_best
        # if comparative validation or sanity checks rejected a better variant)
        best_ever_output = current_best
        best_ever_score = current_score
        consecutive_errors = 0
        failed_directives: set[str] = set()

        try:
            for cycle in range(start_cycle, refinement_cycles):
                cycle_start = time.time()

                # Adaptive branch factor
                branch_factor = self._adaptive_branch_factor(cycle, metrics)
                self.improver.config = TimeDilateConfig(
                    **{**self.config.__dict__, "branch_factor": branch_factor}
                )

                # Use ensemble scoring when scores are oscillating, biased,
                # or comparative validation is frequently overruling selections
                self.improver.force_ensemble = (
                    metrics.score_oscillating
                    or metrics.scoring_bias != "normal"
                    or (len(metrics.cycles) >= 3 and metrics.comparative_overrule_rate > 0.3)
                )
                # Boost temperature diversity when stagnating
                self.improver.stagnation_boost = metrics.stagnant_streak >= 2
                self.improver.cycles_remaining = refinement_cycles - cycle - 1

                # Every 5 cycles, do a detailed score to target weaknesses
                # Every 3 cycles (not overlapping with 5), do feedback scoring for strengths/weaknesses
                score_feedback_text = ""
                if cycle > 0 and cycle % 3 == 0 and cycle % 5 != 0:
                    try:
                        fb_prompt = self.scorer.build_feedback_scoring_prompt(prompt, current_best)
                        fb_raw = self.engine.generate(fb_prompt, temperature=self.config.scoring_temperature)
                        strengths, weaknesses = self.scorer.parse_strengths_weaknesses(fb_raw)
                        if strengths:
                            score_feedback_text = f"PRESERVE these strengths:\n{strengths}\n"
                        if weaknesses:
                            score_feedback_text += f"FIX these weaknesses:\n{weaknesses}"
                        logger.info("Cycle %d: feedback scoring — strengths found: %s",
                                    cycle + 1, bool(strengths))
                    except Exception:
                        logger.debug("Feedback scoring failed, continuing without")
                if cycle > 0 and cycle % 5 == 0:
                    try:
                        detail_prompt = self.scorer.build_detailed_scoring_prompt(prompt, current_best)
                        detail_raw = self.engine.generate(detail_prompt, temperature=self.config.scoring_temperature)
                        detailed = self.scorer.parse_detailed_score(detail_raw)
                        weakness = detailed.weakest_aspect
                        directive = self.directives.directive_for_weakness(weakness)
                        directive_source = "targeted"
                        score_feedback_text = (
                            f"Aspect scores — Correctness: {detailed.correctness}/25, "
                            f"Completeness: {detailed.completeness}/25, "
                            f"Quality: {detailed.quality}/25, "
                            f"Elegance: {detailed.elegance}/25. "
                            f"Weakest: {weakness}."
                        )
                        logger.info("Cycle %d: targeting weakest aspect '%s' (scores: %s)",
                                    cycle + 1, weakness, detailed.to_dict())
                    except Exception:
                        logger.warning("Detailed scoring failed, falling back to normal directive")
                        directive = self.directives.next_directive(task_type, cycle + directive_offset, current_score=current_score)
                        directive_source = "builtin"
                elif meta_directives and cycle < len(meta_directives):
                    directive = meta_directives[cycle]
                    directive_source = "meta"
                    logger.info("Cycle %d: using meta-learned directive", cycle + 1)
                elif built_in_exhausted or self._should_prefer_generated(metrics) or metrics.diminishing_returns:
                    custom_prompt = self.directives.generate_custom_directive_prompt(
                        task_type, prompt, current_best
                    )
                    directive = self.engine.generate(custom_prompt)
                    directive_source = "generated"
                else:
                    # Try trajectory-aware directives, skipping ones that already failed
                    directive = self.directives.trajectory_aware_directive(
                        task_type, cycle + directive_offset,
                        current_score=current_score,
                        score_history=metrics.score_history,
                    )
                    attempts = 0
                    while directive in failed_directives and attempts < 5:
                        directive_offset += 1
                        directive = self.directives.next_directive(
                            task_type, cycle + directive_offset, current_score=current_score
                        )
                        attempts += 1
                    directive_source = "builtin"

                previous_score = current_score
                previous_best = current_best
                history_summary = self._build_history_summary(metrics)

                try:
                    new_best, new_score, best_idx = self.improver.run_cycle(
                        original_prompt=prompt,
                        current_best=current_best,
                        current_score=current_score,
                        directive=directive,
                        history_summary=history_summary,
                        score_feedback=score_feedback_text,
                    )
                    consecutive_errors = 0
                except Exception as e:
                    consecutive_errors += 1
                    logger.warning("Cycle %d failed: %s (consecutive errors: %d)",
                                   cycle + 1, e, consecutive_errors)
                    if consecutive_errors >= 3:
                        logger.error("Too many consecutive errors, stopping early")
                        break
                    # Skip this cycle but continue
                    new_best, new_score, best_idx = current_best, current_score, -1

                # Compute output change magnitude before updating current_best
                output_delta = 1.0 - self.improver._similarity_ratio(new_best, previous_best)

                if new_score > current_score:
                    current_best = new_best
                    current_score = new_score
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    failed_directives.add(directive)

                # Track best output ever seen
                if current_score > best_ever_score:
                    best_ever_output = current_best
                    best_ever_score = current_score

                # Estimate inference calls: generation + scoring per branch
                # + potential comparative/crossover/ensemble calls
                est_calls = branch_factor  # generation
                if self.improver.force_ensemble:
                    est_calls += branch_factor * 2  # 2 scores per variant
                else:
                    est_calls += branch_factor  # 1 score per variant
                if best_idx == -2:
                    est_calls += 2  # crossover generation + scoring
                if score_feedback_text:
                    est_calls += 1  # feedback/detailed scoring call

                metrics.record_cycle(
                    cycle=cycle + 1,
                    score=current_score,
                    previous_score=previous_score,
                    directive=directive,
                    directive_source=directive_source,
                    branch_count=branch_factor,
                    best_variant_index=best_idx,
                    elapsed_seconds=time.time() - cycle_start,
                    output_delta=output_delta,
                    output_length=len(current_best),
                    crossover_used=(best_idx == -2),
                    inference_calls=est_calls,
                )

                total_elapsed = time.time() - start
                budget_pct = (total_elapsed / self.config.budget_seconds * 100) if self.config.budget_seconds > 0 else 0
                log_cycle_summary(
                    logger, cycle + 1, current_score, previous_score,
                    directive_source, time.time() - cycle_start, branch_factor,
                    output_delta=output_delta,
                    peak_score=metrics.peak_score,
                    improvement_rate=metrics.improvement_rate,
                    budget_used_pct=budget_pct,
                )

                # Record directive outcome for meta-learning
                if self.meta:
                    self.meta.record_directive(
                        task_type, directive, current_score > previous_score
                    )

                if no_improvement_count >= self._effective_convergence_threshold(current_score):
                    convergence_detected = True
                    # Try a fresh attempt to break out of plateau
                    fresh_output, fresh_score = self.improver.fresh_attempt(
                        prompt, directive,
                        best_directive=metrics.best_directive,
                        score_history=metrics.score_history,
                    )
                    if fresh_output and fresh_score > current_score:
                        logger.info("Fresh attempt broke plateau (score %d -> %d)",
                                    current_score, fresh_score)
                        current_best = fresh_output
                        current_score = fresh_score
                    if not built_in_exhausted:
                        directive_offset += built_in_count
                        if directive_offset >= built_in_count * 2:
                            built_in_exhausted = True
                    no_improvement_count = 0

                if (cycle + 1) % self.config.checkpoint_interval == 0:
                    self.checkpoint.save(cycle + 1, current_best, current_score,
                                        prompt=prompt, task_type=task_type,
                                        no_improvement_count=no_improvement_count)

                completed_cycles = cycle + 1

                # Early exit if perfect score
                if current_score >= 100:
                    logger.info("Perfect score reached at cycle %d", cycle + 1)
                    break

                # Early termination if projected gains are negligible
                if metrics.should_early_terminate:
                    logger.info("Early termination: projected gains negligible at cycle %d (score %d)",
                                cycle + 1, current_score)
                    break

                if on_cycle:
                    elapsed = time.time() - start
                    on_cycle(cycle + 1, refinement_cycles, current_score, elapsed)

        except KeyboardInterrupt:
            # Save checkpoint on interrupt so we can resume
            self.checkpoint.save(completed_cycles, current_best, current_score,
                                prompt=prompt, task_type=task_type,
                                no_improvement_count=no_improvement_count)
            return DilationResult(
                output=current_best,
                score=current_score,
                cycles_completed=completed_cycles,
                elapsed_seconds=time.time() - start,
                convergence_detected=convergence_detected,
                interrupted=True,
                resumed_from_cycle=resumed_from,
                metrics=metrics,
            )

        self.checkpoint.cleanup()
        if self.meta:
            try:
                self.meta.save()
            except Exception:
                logger.warning("Failed to save meta-learning data")

        # Use best-ever output if it's better than current
        final_output = best_ever_output if best_ever_score > current_score else current_best
        final_score = max(best_ever_score, current_score)

        return DilationResult(
            output=final_output,
            score=final_score,
            cycles_completed=completed_cycles,
            elapsed_seconds=time.time() - start,
            convergence_detected=convergence_detected,
            resumed_from_cycle=resumed_from,
            metrics=metrics,
        )
