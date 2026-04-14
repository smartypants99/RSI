import logging

from timedilate.config import TimeDilateConfig
from timedilate.scorer import Scorer

logger = logging.getLogger(__name__)


class ImprovementEngine:
    def __init__(self, engine, config: TimeDilateConfig):
        self.engine = engine
        self.config = config
        self.scorer = Scorer()

    def _build_improvement_prompt(
        self, original_prompt: str, current_best: str, directive: str, current_score: int = 0
    ) -> str:
        return (
            f"Original task: {original_prompt}\n\n"
            f"Current solution (scored {current_score}/100):\n{current_best}\n\n"
            f"Improvement directive: {directive}\n\n"
            f"Produce a meaningfully improved version. Think carefully about what "
            f"specific changes will increase the quality score. "
            f"Output ONLY the improved solution, nothing else."
        )

    def _maybe_summarize(self, text: str, original_prompt: str) -> str:
        """If text exceeds 75% of context window, ask model to summarize."""
        token_est = self.engine.estimate_tokens(text)
        limit = int(self.config.context_window * 0.75)
        if token_est <= limit:
            return text
        logger.info("Output exceeds context limit (%d > %d tokens), summarizing", token_est, limit)
        summary_prompt = (
            f"The following is a solution to this task: {original_prompt}\n\n"
            f"Solution:\n{text}\n\n"
            f"The solution is very long. Produce a condensed version that preserves "
            f"all key functionality and correctness. Output ONLY the condensed solution."
        )
        return self.engine.generate(summary_prompt)

    def _generate_variant(self, original_prompt: str, current_best: str, directive: str, current_score: int) -> str | None:
        """Generate a single variant, returning None on failure."""
        try:
            prompt = self._build_improvement_prompt(
                original_prompt, current_best, directive, current_score
            )
            variant = self.engine.generate(prompt)
            if not variant or not variant.strip():
                logger.warning("Empty variant generated, skipping")
                return None
            return variant
        except Exception as e:
            logger.warning("Variant generation failed: %s", e)
            return None

    def _score_variant(self, original_prompt: str, variant: str) -> int:
        """Score a variant, returning 0 on failure."""
        try:
            score_prompt = self.scorer.build_scoring_prompt(original_prompt, variant)
            raw_score = self.engine.generate(
                score_prompt, temperature=self.config.scoring_temperature
            )
            return self.scorer.parse_score(raw_score)
        except Exception as e:
            logger.warning("Scoring failed: %s", e)
            return 0

    def run_cycle(
        self,
        original_prompt: str,
        current_best: str,
        current_score: int,
        directive: str,
    ) -> tuple[str, int, int]:
        """Returns (best_output, best_score, best_variant_index).
        best_variant_index is -1 if no variant beat the current best."""
        current_best = self._maybe_summarize(current_best, original_prompt)

        variants = []
        for _ in range(self.config.branch_factor):
            variant = self._generate_variant(
                original_prompt, current_best, directive, current_score
            )
            if variant is not None:
                variants.append(variant)

        if not variants:
            logger.warning("All variant generations failed, keeping current best")
            return current_best, current_score, -1

        best_variant = current_best
        best_score = current_score
        best_index = -1

        for i, variant in enumerate(variants):
            score = self._score_variant(original_prompt, variant)
            if score > best_score:
                best_variant = variant
                best_score = score
                best_index = i

        return best_variant, best_score, best_index
