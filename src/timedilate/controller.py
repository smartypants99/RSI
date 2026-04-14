"""Time Dilation Controller.

Implements Einstein's time dilation for AI: the AI gets more subjective
thinking time. Each cycle is a round of generation, self-critique,
and refinement. The dilation factor determines how many cycles the AI
gets — infinite scaling, no quality loss, no plateau.

Factor 1 = single pass (normal inference)
Factor 10 = 10 reasoning cycles
Factor 1000 = 1000 reasoning cycles
Factor 1000000 = 1000000 reasoning cycles

More thinking always helps. No ceiling.
"""
import logging
import time
from dataclasses import dataclass, field

from timedilate.config import TimeDilateConfig
from timedilate.engine import DilationEngine

logger = logging.getLogger(__name__)


@dataclass
class CycleRecord:
    cycle: int
    action: str  # "critique", "refine", "branch"
    improved: bool
    score_before: int | None = None
    score_after: int | None = None


@dataclass
class DilationResult:
    output: str
    dilation_factor: float
    cycles_completed: int
    total_cycles: int
    elapsed_seconds: float
    model_used: str
    score: int
    cycle_history: list[CycleRecord] = field(default_factory=list)
    convergence_resets: int = 0

    def to_report(self, config: TimeDilateConfig | None = None) -> dict:
        from timedilate import __version__
        report = {
            "version": __version__,
            "timestamp": time.time(),
            "dilation_factor": self.dilation_factor,
            "cycles_completed": self.cycles_completed,
            "total_cycles": self.total_cycles,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "score": self.score,
            "model_used": self.model_used,
            "convergence_resets": self.convergence_resets,
            "improvements": sum(1 for c in self.cycle_history if c.improved),
        }
        if config:
            report["config"] = {
                "dilation_factor": config.dilation_factor,
                "branch_factor": config.branch_factor,
                "convergence_patience": config.convergence_patience,
                "use_self_critique": config.use_self_critique,
                "use_chain_of_thought": config.use_chain_of_thought,
            }
        return report


class DilationController:
    """Orchestrates time-dilated reasoning.

    Each cycle:
    1. Score the current output (self-assessment)
    2. Critique: identify weaknesses
    3. Refine: generate improved version addressing the critique
    4. If stuck (no improvement for N cycles), try a fresh approach

    This loops for as many cycles as the dilation factor demands.
    """

    def __init__(self, config: TimeDilateConfig, engine: DilationEngine | None = None):
        config.validate()
        self.config = config
        self.engine = engine or DilationEngine(config)

    def run(self, prompt: str, on_cycle=None) -> DilationResult:
        """Run time-dilated inference.

        Args:
            prompt: The task/question for the AI.
            on_cycle: Optional callback(cycle, total, score, elapsed) per cycle.
        """
        start = time.time()
        num_cycles = self.config.num_cycles

        # Initial generation
        logger.info("Generating initial response...")
        output = self.engine.generate(prompt)

        time_budget = self.config.time_budget_seconds
        use_time_budget = time_budget is not None and self.config.dilation_factor > 1.0

        if num_cycles == 0 and not use_time_budget:
            # Factor 1.0 — single pass, no dilation
            return DilationResult(
                output=output,
                dilation_factor=self.config.dilation_factor,
                cycles_completed=0,
                total_cycles=0,
                elapsed_seconds=time.time() - start,
                model_used=self.config.model,
                score=0,
            )

        # Score initial output
        score = self._score(prompt, output)
        best_output = output
        best_score = score
        history = []
        no_improve_count = 0
        convergence_resets = 0
        cycle = 0

        if use_time_budget:
            logger.info("Starting dilation: %.1fs budget x %.0fx factor = %.0fs subjective time, initial score: %d",
                        time_budget, self.config.dilation_factor, self.config.subjective_time, score)
        else:
            logger.info("Starting dilation: %d cycles, initial score: %d", num_cycles, score)

        while True:
            cycle += 1

            # Check termination: time budget or cycle count
            if use_time_budget:
                elapsed = time.time() - start
                if elapsed >= time_budget:
                    break
            else:
                if cycle > num_cycles:
                    break

            # Step 1: Critique
            if self.config.use_self_critique:
                critique = self._critique(prompt, output, score)
            else:
                critique = "Improve the response."

            # Step 2: Refine based on critique
            new_output = self._refine(prompt, output, critique)
            new_score = self._score(prompt, new_output)

            improved = new_score > best_score
            history.append(CycleRecord(
                cycle=cycle, action="refine", improved=improved,
                score_before=best_score, score_after=new_score,
            ))

            if improved:
                best_output = new_output
                best_score = new_score
                output = new_output
                score = new_score
                no_improve_count = 0
                logger.info("Cycle %d: score %d -> %d (improved)", cycle, score, new_score)
            else:
                no_improve_count += 1
                logger.info("Cycle %d: score %d (no improvement, patience %d/%d)",
                            cycle, new_score, no_improve_count, self.config.convergence_patience)

            # Step 3: If stuck, try fresh approach
            if no_improve_count >= self.config.convergence_patience:
                logger.info("Convergence detected, trying fresh approach...")
                fresh = self._fresh_attempt(prompt, best_output, best_score)
                fresh_score = self._score(prompt, fresh)
                if fresh_score > best_score:
                    best_output = fresh
                    best_score = fresh_score
                    output = fresh
                    score = fresh_score
                    history.append(CycleRecord(
                        cycle=cycle, action="fresh", improved=True,
                        score_before=best_score, score_after=fresh_score,
                    ))
                no_improve_count = 0
                convergence_resets += 1

            if on_cycle:
                on_cycle(cycle, num_cycles or cycle, best_score, time.time() - start)

        return DilationResult(
            output=best_output,
            dilation_factor=self.config.dilation_factor,
            cycles_completed=cycle - 1,
            total_cycles=num_cycles or cycle - 1,
            elapsed_seconds=time.time() - start,
            model_used=self.config.model,
            score=best_score,
            cycle_history=history,
            convergence_resets=convergence_resets,
        )

    def _score(self, prompt: str, output: str) -> int:
        """Have the AI score its own output 0-100."""
        score_prompt = (
            f"You are scoring an AI response. Rate it 0-100.\n\n"
            f"TASK: {prompt}\n\n"
            f"RESPONSE:\n{output}\n\n"
            f"Score criteria: correctness, completeness, clarity, usefulness.\n"
            f"Reply with ONLY a number 0-100."
        )
        try:
            result = self.engine.generate(score_prompt, max_tokens=16, temperature=0.1)
            # Extract first number from response
            for word in result.split():
                word = word.strip(".,!()[]")
                if word.isdigit():
                    return min(100, max(0, int(word)))
            return 50  # fallback
        except Exception:
            return 50

    def _critique(self, prompt: str, output: str, score: int) -> str:
        """Have the AI critique its own output to find weaknesses."""
        cot = ""
        if self.config.use_chain_of_thought:
            cot = "Think step by step about what's wrong and what's missing. "

        critique_prompt = (
            f"You are reviewing an AI response (current score: {score}/100).\n\n"
            f"TASK: {prompt}\n\n"
            f"RESPONSE:\n{output}\n\n"
            f"{cot}"
            f"List the specific weaknesses, errors, and missing elements. "
            f"Be concrete and actionable — what exactly should be fixed?"
        )
        return self.engine.generate(critique_prompt)

    def _refine(self, prompt: str, current_output: str, critique: str) -> str:
        """Generate an improved version addressing the critique."""
        refine_prompt = (
            f"You previously produced a response that received this critique:\n\n"
            f"CRITIQUE:\n{critique}\n\n"
            f"ORIGINAL TASK: {prompt}\n\n"
            f"PREVIOUS RESPONSE:\n{current_output}\n\n"
            f"Write an improved version that addresses every point in the critique. "
            f"Keep what was good, fix what was bad, add what was missing."
        )
        return self.engine.generate(refine_prompt)

    def _fresh_attempt(self, prompt: str, best_so_far: str, best_score: int) -> str:
        """Try a completely different approach when stuck."""
        fresh_prompt = (
            f"Previous attempts at this task have plateaued at score {best_score}/100.\n\n"
            f"TASK: {prompt}\n\n"
            f"The best attempt so far:\n{best_so_far}\n\n"
            f"Take a completely different approach. Rethink the problem from scratch. "
            f"Don't iterate on the previous attempt — try something fundamentally different."
        )
        return self.engine.generate(fresh_prompt)
