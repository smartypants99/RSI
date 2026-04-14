import logging
import time
from dataclasses import dataclass, field
from timedilate.config import TimeDilateConfig
from timedilate.improver import ImprovementEngine
from timedilate.scorer import Scorer
from timedilate.directives import DirectiveGenerator
from timedilate.checkpoint import CheckpointManager
from timedilate.metrics import RunMetrics


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
        self.config = config
        self.engine = engine
        self.improver = ImprovementEngine(engine, config)
        self.scorer = Scorer()
        self.directives = DirectiveGenerator()
        self.checkpoint = CheckpointManager(config.checkpoint_dir)

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

                # Choose directive: built-in or self-generated
                if built_in_exhausted:
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
                new_best, new_score, best_idx = self.improver.run_cycle(
                    original_prompt=prompt,
                    current_best=current_best,
                    current_score=current_score,
                    directive=directive,
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

                if no_improvement_count >= self.config.convergence_threshold:
                    convergence_detected = True
                    if not built_in_exhausted:
                        directive_offset += built_in_count
                        if directive_offset >= built_in_count * 2:
                            built_in_exhausted = True
                    no_improvement_count = 0

                if (cycle + 1) % self.config.checkpoint_interval == 0:
                    self.checkpoint.save(cycle + 1, current_best, current_score)

                completed_cycles = cycle + 1

                # Early exit if perfect score
                if current_score >= 100:
                    logger.info("Perfect score reached at cycle %d", cycle + 1)
                    break

                if on_cycle:
                    elapsed = time.time() - start
                    on_cycle(cycle + 1, refinement_cycles, current_score, elapsed)

        except KeyboardInterrupt:
            # Save checkpoint on interrupt so we can resume
            self.checkpoint.save(completed_cycles, current_best, current_score)
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

        return DilationResult(
            output=current_best,
            score=current_score,
            cycles_completed=completed_cycles,
            elapsed_seconds=time.time() - start,
            convergence_detected=convergence_detected,
            resumed_from_cycle=resumed_from,
            metrics=metrics,
        )
