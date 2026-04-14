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
        self.meta = MetaLearner(path=config.checkpoint_dir + "/.meta.json")

    def _build_history_summary(self, metrics: RunMetrics, max_entries: int = 5) -> str:
        """Build a concise summary of recent cycles for the improvement prompt."""
        if not metrics.cycles:
            return ""
        recent = metrics.cycles[-max_entries:]
        lines = []
        for c in recent:
            improved = "improved" if c.score > c.previous_score else "no improvement"
            lines.append(f"- Cycle {c.cycle}: \"{c.directive}\" -> {improved} (score {c.previous_score}->{c.score})")
        return "\n".join(lines)

    def _adaptive_branch_factor(self, cycle: int, metrics: RunMetrics) -> int:
        """Reduce branch factor if cycles are slow or stagnating."""
        base = self.config.branch_factor
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
        task_type = self.directives.classify_task(prompt)
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

            # Score the initial output
            score_prompt = self.scorer.build_scoring_prompt(prompt, current_best)
            raw_score = self.engine.generate(
                score_prompt, temperature=self.config.scoring_temperature
            )
            current_score = self.scorer.parse_score(raw_score)

        no_improvement_count = 0
        convergence_detected = False
        built_in_exhausted = False
        directive_offset = 0
        built_in_count = len(self.directives.get_directives(task_type))
        meta_directives = self.meta.best_directives(task_type, top_n=3)
        completed_cycles = resumed_from
        start_cycle = resumed_from

        try:
            for cycle in range(start_cycle, refinement_cycles):
                cycle_start = time.time()

                # Adaptive branch factor
                branch_factor = self._adaptive_branch_factor(cycle, metrics)
                self.improver.config = TimeDilateConfig(
                    **{**self.config.__dict__, "branch_factor": branch_factor}
                )

                # Every 5 cycles, do a detailed score to target weaknesses
                if cycle > 0 and cycle % 5 == 0:
                    try:
                        detail_prompt = self.scorer.build_detailed_scoring_prompt(prompt, current_best)
                        detail_raw = self.engine.generate(detail_prompt, temperature=self.config.scoring_temperature)
                        detailed = self.scorer.parse_detailed_score(detail_raw)
                        weakness = detailed.weakest_aspect
                        directive = self.directives.directive_for_weakness(weakness)
                        directive_source = "targeted"
                        logger.info("Cycle %d: targeting weakest aspect '%s' (scores: %s)",
                                    cycle + 1, weakness, detailed.to_dict())
                    except Exception:
                        logger.warning("Detailed scoring failed, falling back to normal directive")
                        directive = self.directives.next_directive(task_type, cycle + directive_offset)
                        directive_source = "builtin"
                elif meta_directives and cycle < len(meta_directives):
                    directive = meta_directives[cycle]
                    directive_source = "meta"
                    logger.info("Cycle %d: using meta-learned directive", cycle + 1)
                elif built_in_exhausted or self._should_prefer_generated(metrics):
                    custom_prompt = self.directives.generate_custom_directive_prompt(
                        task_type, prompt, current_best
                    )
                    directive = self.engine.generate(custom_prompt)
                    directive_source = "generated"
                else:
                    directive = self.directives.next_directive(
                        task_type, cycle + directive_offset
                    )
                    directive_source = "builtin"

                previous_score = current_score
                history_summary = self._build_history_summary(metrics)
                new_best, new_score, best_idx = self.improver.run_cycle(
                    original_prompt=prompt,
                    current_best=current_best,
                    current_score=current_score,
                    directive=directive,
                    history_summary=history_summary,
                    score_feedback="",
                )

                if new_score > current_score:
                    current_best = new_best
                    current_score = new_score
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                metrics.record_cycle(
                    cycle=cycle + 1,
                    score=current_score,
                    previous_score=previous_score,
                    directive=directive,
                    directive_source=directive_source,
                    branch_count=branch_factor,
                    best_variant_index=best_idx,
                    elapsed_seconds=time.time() - cycle_start,
                )

                log_cycle_summary(
                    logger, cycle + 1, current_score, previous_score,
                    directive_source, time.time() - cycle_start, branch_factor,
                )

                # Record directive outcome for meta-learning
                self.meta.record_directive(
                    task_type, directive, current_score > previous_score
                )

                if no_improvement_count >= self.config.convergence_threshold:
                    convergence_detected = True
                    # Try a fresh attempt to break out of plateau
                    fresh_output, fresh_score = self.improver.fresh_attempt(
                        prompt, directive
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
        try:
            self.meta.save()
        except Exception:
            logger.warning("Failed to save meta-learning data")

        return DilationResult(
            output=current_best,
            score=current_score,
            cycles_completed=completed_cycles,
            elapsed_seconds=time.time() - start,
            convergence_detected=convergence_detected,
            resumed_from_cycle=resumed_from,
            metrics=metrics,
        )
