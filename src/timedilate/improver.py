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
        self,
        original_prompt: str,
        current_best: str,
        directive: str,
        current_score: int = 0,
        history_summary: str = "",
    ) -> str:
        history_block = ""
        if history_summary:
            history_block = f"Previous improvement attempts:\n{history_summary}\n\n"
        return (
            f"Original task: {original_prompt}\n\n"
            f"Current solution (scored {current_score}/100):\n{current_best}\n\n"
            f"{history_block}"
            f"Improvement directive: {directive}\n\n"
            f"Produce a meaningfully improved version. Think carefully about what "
            f"specific changes will increase the quality score. "
            f"Do NOT repeat changes that already failed to improve the score. "
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

    def _branch_temperature(self, branch_index: int) -> float:
        """Vary temperature across branches for diverse exploration.
        Branch 0 uses base temp, others spread from 0.3 to 1.0."""
        if self.config.branch_factor <= 1:
            return self.config.temperature
        if branch_index == 0:
            return self.config.temperature
        # Spread remaining branches across 0.3 - 1.0
        t = 0.3 + (branch_index / (self.config.branch_factor - 1)) * 0.7
        return round(min(t, 1.0), 2)

    def _generate_variant(self, original_prompt: str, current_best: str, directive: str, current_score: int, history_summary: str = "", temperature: float | None = None) -> str | None:
        """Generate a single variant, returning None on failure."""
        try:
            prompt = self._build_improvement_prompt(
                original_prompt, current_best, directive, current_score, history_summary
            )
            variant = self.engine.generate(prompt, temperature=temperature)
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
        history_summary: str = "",
    ) -> tuple[str, int, int]:
        """Returns (best_output, best_score, best_variant_index).
        best_variant_index is -1 if no variant beat the current best."""
        current_best = self._maybe_summarize(current_best, original_prompt)

        variants = []
        for i in range(self.config.branch_factor):
            temp = self._branch_temperature(i)
            variant = self._generate_variant(
                original_prompt, current_best, directive, current_score, history_summary, temperature=temp
            )
            if variant is not None:
                variants.append(variant)

        if not variants:
            logger.warning("All variant generations failed, keeping current best")
            return current_best, current_score, -1

        # Choose selection strategy
        if len(variants) >= 4:
            winner_variant, winner_index = self._tournament_select(original_prompt, variants)
            # Tournament uses comparisons, need to score the winner
            winner_score = self._score_variant(original_prompt, winner_variant)
        else:
            winner_variant, winner_index, winner_score = self._score_select(original_prompt, variants)

        if winner_score <= current_score:
            return current_best, current_score, -1

        # Comparative validation for close scores
        if (winner_score - current_score) <= 5:
            result = self._compare_outputs(original_prompt, current_best, winner_variant)
            if result == "A":
                logger.info("Comparative check overruled selection (delta=%d)",
                            winner_score - current_score)
                return current_best, current_score, -1

        return winner_variant, winner_score, winner_index

    def _score_select(self, original_prompt: str, variants: list[str]) -> tuple[str, int, int]:
        """Score each variant, return (best_variant, best_index, best_score)."""
        best_variant = variants[0]
        best_score = -1
        best_index = 0

        for i, variant in enumerate(variants):
            score = self._score_variant(original_prompt, variant)
            if score > best_score:
                best_variant = variant
                best_score = score
                best_index = i

        return best_variant, best_index, best_score

    def _tournament_select(self, original_prompt: str, variants: list[str]) -> tuple[str, int]:
        """Pairwise tournament: compare adjacent pairs, winners advance.
        Uses fewer inference calls than scoring every variant."""
        indexed = list(enumerate(variants))

        while len(indexed) > 1:
            next_round = []
            for j in range(0, len(indexed), 2):
                if j + 1 >= len(indexed):
                    next_round.append(indexed[j])
                    continue
                idx_a, a = indexed[j]
                idx_b, b = indexed[j + 1]
                winner = self._compare_outputs(original_prompt, a, b)
                if winner == "B":
                    next_round.append((idx_b, b))
                else:
                    next_round.append((idx_a, a))  # A or TIE -> keep A
            indexed = next_round

        return indexed[0][1], indexed[0][0]

    def _compare_outputs(self, original_prompt: str, output_a: str, output_b: str) -> str:
        """A/B compare two outputs, returns 'A', 'B', or 'TIE'."""
        try:
            prompt = self.scorer.build_comparative_prompt(original_prompt, output_a, output_b)
            raw = self.engine.generate(prompt, temperature=self.config.scoring_temperature)
            return self.scorer.parse_comparison(raw)
        except Exception as e:
            logger.warning("Comparison failed: %s", e)
            return "TIE"
