import logging
import time

from timedilate.config import TimeDilateConfig
from timedilate.scorer import Scorer

logger = logging.getLogger(__name__)


class ImprovementEngine:
    def __init__(self, engine, config: TimeDilateConfig):
        self.engine = engine
        self.config = config
        self.scorer = Scorer()
        self.task_type: str = "general"
        self.force_ensemble: bool = False
        self.stagnation_boost: bool = False
        self.initial_output_length: int = 0
        self.cycles_remaining: int = 0

    def _estimate_prompt_tokens(self, *parts: str) -> int:
        return sum(self.engine.estimate_tokens(p) for p in parts if p)

    def _build_improvement_prompt(
        self,
        original_prompt: str,
        current_best: str,
        directive: str,
        current_score: int = 0,
        history_summary: str = "",
        score_feedback: str = "",
    ) -> str:
        history_block = ""
        if history_summary:
            history_block = f"Previous improvement attempts:\n{history_summary}\n\n"
        feedback_block = ""
        if score_feedback:
            feedback_block = f"Evaluator feedback on current solution:\n{score_feedback}\n\n"

        # Check if prompt would exceed context budget (60% of window, leaving room for generation)
        prompt_budget = int(self.config.context_window * 0.6)
        total_est = self._estimate_prompt_tokens(
            original_prompt, current_best, history_block, feedback_block, directive
        )
        if total_est > prompt_budget:
            # Trim history first, then feedback
            logger.info("Prompt too long (%d tokens), trimming to fit %d budget",
                        total_est, prompt_budget)
            history_block = ""
            total_est = self._estimate_prompt_tokens(
                original_prompt, current_best, feedback_block, directive
            )
            if total_est > prompt_budget:
                feedback_block = ""

        # Score-aware generation guidance
        if current_score >= 85:
            guidance = (
                "This solution is already strong. Make surgical, precise changes only. "
                "Do NOT restructure what's working. Focus on the specific weakness identified. "
                "Small, targeted fixes are better than rewrites at this stage."
            )
        elif current_score >= 65:
            guidance = (
                "Produce a meaningfully improved version. Think carefully about what "
                "specific changes will increase the quality score. "
                "Address the evaluator's feedback directly."
            )
        else:
            guidance = (
                "This solution needs substantial improvement. "
                "Consider a different approach if the current one is fundamentally flawed. "
                "Make bold changes — incremental tweaks won't be enough."
            )

        urgency = ""
        if 0 < self.cycles_remaining <= 3:
            urgency = (
                f"IMPORTANT: Only {self.cycles_remaining} improvement cycle(s) remaining. "
                f"Make this change count — prioritize the highest-impact fix.\n\n"
            )

        return (
            f"Original task: {original_prompt}\n\n"
            f"Current solution (scored {current_score}/100):\n{current_best}\n\n"
            f"{feedback_block}"
            f"{history_block}"
            f"{urgency}"
            f"Improvement directive: {directive}\n\n"
            f"{guidance} "
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
        Branch 0 uses base temp, others spread from 0.3 to 1.0.
        When stagnation_boost is active, widen range to 0.5-1.3 for more exploration."""
        if self.config.branch_factor <= 1:
            base = self.config.temperature
            return min(base + 0.2, 1.3) if self.stagnation_boost else base
        if branch_index == 0:
            return self.config.temperature
        if self.stagnation_boost:
            # Wider spread for exploration: 0.5 - 1.3
            t = 0.5 + (branch_index / (self.config.branch_factor - 1)) * 0.8
            return round(min(t, 1.3), 2)
        # Normal spread: 0.3 - 1.0
        t = 0.3 + (branch_index / (self.config.branch_factor - 1)) * 0.7
        return round(min(t, 1.0), 2)

    def _build_reflection_prompt(
        self,
        original_prompt: str,
        current_best: str,
        directive: str,
        current_score: int,
        score_feedback: str = "",
    ) -> str:
        """Ask the model to reflect on what to change before generating."""
        feedback_block = f"\nEvaluator feedback: {score_feedback}\n" if score_feedback else ""
        return (
            f"Original task: {original_prompt}\n\n"
            f"Current solution (scored {current_score}/100):\n{current_best}\n"
            f"{feedback_block}\n"
            f"Improvement directive: {directive}\n\n"
            f"Before making changes, analyze:\n"
            f"1. What are the 2-3 most impactful changes that would raise the score?\n"
            f"2. What should NOT be changed (things that are already good)?\n"
            f"3. What is the biggest risk of regression?\n\n"
            f"Be specific and concise. 3-5 sentences total."
        )

    def _generate_with_reflection(self, original_prompt: str, current_best: str, directive: str, current_score: int, history_summary: str = "", temperature: float | None = None, score_feedback: str = "") -> str | None:
        """Generate a variant using a reflect-then-act pattern."""
        try:
            # Step 1: Reflect
            reflection_prompt = self._build_reflection_prompt(
                original_prompt, current_best, directive, current_score, score_feedback
            )
            reflection = self.engine.generate(reflection_prompt, temperature=0.3)
            if not reflection:
                return self._generate_variant(
                    original_prompt, current_best, directive, current_score,
                    history_summary, temperature, score_feedback
                )

            # Step 2: Generate guided by reflection
            prompt = self._build_improvement_prompt(
                original_prompt, current_best, directive, current_score,
                history_summary, score_feedback
            )
            prompt += f"\n\nYour analysis of what to change:\n{reflection}\n\nNow produce the improved version:"
            variant = self.engine.generate(prompt, temperature=temperature)
            if not variant or not variant.strip():
                return None
            return variant
        except Exception as e:
            logger.warning("Reflection-based generation failed: %s", e)
            return None

    def _similarity_ratio(self, a: str, b: str) -> float:
        """Quick similarity check using character overlap ratio."""
        if not a or not b:
            return 0.0
        a_stripped = a.strip()
        b_stripped = b.strip()
        if a_stripped == b_stripped:
            return 1.0
        shorter = min(len(a_stripped), len(b_stripped))
        longer = max(len(a_stripped), len(b_stripped))
        if longer == 0:
            return 1.0
        # Count matching prefix + suffix as a fast approximation
        prefix = 0
        for i in range(min(shorter, longer)):
            if a_stripped[i] == b_stripped[i]:
                prefix += 1
            else:
                break
        return prefix / longer

    def _validate_variant(self, variant: str, current_best: str, original_prompt: str) -> bool:
        """Quick validation to reject obviously bad variants without scoring."""
        if not variant or not variant.strip():
            return False
        # Reject if variant is just the original prompt echoed back
        if variant.strip() == original_prompt.strip():
            logger.info("Variant rejected: echoes original prompt")
            return False
        # Reject if variant is drastically shorter (< 30% of current best length)
        if len(current_best) > 50 and len(variant) < len(current_best) * 0.3:
            logger.info("Variant rejected: too short (%d vs %d chars)", len(variant), len(current_best))
            return False
        # Reject if nearly identical to current best (>95% similar)
        if self._similarity_ratio(variant, current_best) > 0.95:
            logger.info("Variant rejected: too similar to current best")
            return False
        return True

    def _generate_variant(self, original_prompt: str, current_best: str, directive: str, current_score: int, history_summary: str = "", temperature: float | None = None, score_feedback: str = "") -> str | None:
        """Generate a single variant, returning None on failure. Retries once with backoff."""
        for attempt in range(2):
            try:
                prompt = self._build_improvement_prompt(
                    original_prompt, current_best, directive, current_score, history_summary, score_feedback
                )
                variant = self.engine.generate(prompt, temperature=temperature)
                if not variant or not variant.strip():
                    logger.warning("Empty variant generated, skipping")
                    return None
                return variant
            except Exception as e:
                if attempt == 0:
                    logger.info("Variant generation failed: %s, retrying in 0.5s", e)
                    time.sleep(0.5)
                    continue
                logger.warning("Variant generation failed after retry: %s", e)
                return None
        return None

    def _score_variant(self, original_prompt: str, variant: str, use_cot: bool = False, ensemble: bool = False, retries: int = 1, current_score: int = 0) -> int:
        """Score a variant, returning 0 on failure.
        If use_cot=True, uses chain-of-thought scoring for higher accuracy.
        If ensemble=True, scores twice (normal + CoT) and averages.
        Uses task-aware rubric when task_type is set, and progressive
        harshness when current_score >= 75.
        Retries on failure up to `retries` times."""
        for attempt in range(retries + 1):
            try:
                if ensemble:
                    s1 = self._score_variant(original_prompt, variant, use_cot=False, current_score=current_score)
                    s2 = self._score_variant(original_prompt, variant, use_cot=True, current_score=current_score)
                    return (s1 + s2) // 2
                if use_cot:
                    score_prompt = self.scorer.build_cot_scoring_prompt(
                        original_prompt, variant, task_type=self.task_type
                    )
                    if current_score >= 75:
                        score_prompt += self.scorer.HARSH_ADDENDUM.format(score=current_score)
                    if self.initial_output_length > 0 and len(variant) > self.initial_output_length * 2.5:
                        score_prompt += (
                            "\nNote: This output is significantly longer than the initial version. "
                            "Penalize unnecessary verbosity, padding, or repetition. "
                            "Conciseness is a quality signal."
                        )
                    raw_score = self.engine.generate(
                        score_prompt, temperature=self.config.scoring_temperature
                    )
                    score = self.scorer.parse_cot_score(raw_score)
                elif self.config.score_weights:
                    # Use detailed scoring with custom weights
                    score_prompt = self.scorer.build_detailed_scoring_prompt(original_prompt, variant)
                    if current_score >= 75:
                        score_prompt += self.scorer.HARSH_ADDENDUM.format(score=current_score)
                    raw_score = self.engine.generate(
                        score_prompt, temperature=self.config.scoring_temperature
                    )
                    detailed = self.scorer.parse_detailed_score(raw_score)
                    score = detailed.weighted_total(self.config.score_weights)
                else:
                    # Use task-aware scoring with progressive harshness
                    if self.task_type != "general":
                        score_prompt = self.scorer.build_task_aware_scoring_prompt(
                            original_prompt, variant, self.task_type
                        )
                    else:
                        score_prompt = self.scorer.build_scoring_prompt(original_prompt, variant)
                    if current_score >= 75:
                        score_prompt += self.scorer.HARSH_ADDENDUM.format(score=current_score)
                    if self.initial_output_length > 0 and len(variant) > self.initial_output_length * 2.5:
                        score_prompt += (
                            "\nNote: This output is significantly longer than the initial version. "
                            "Penalize unnecessary verbosity, padding, or repetition. "
                            "Conciseness is a quality signal."
                        )
                    raw_score = self.engine.generate(
                        score_prompt, temperature=self.config.scoring_temperature
                    )
                    score = self.scorer.parse_score(raw_score)
                if score == 0 and attempt < retries:
                    logger.info("Score was 0, retrying (%d/%d)", attempt + 1, retries)
                    time.sleep(min(0.5 * (2 ** attempt), 5.0))
                    continue
                return score
            except Exception as e:
                if attempt < retries:
                    backoff = min(0.5 * (2 ** attempt), 5.0)
                    logger.info("Scoring attempt %d failed: %s, retrying in %.1fs",
                                attempt + 1, e, backoff)
                    time.sleep(backoff)
                    continue
                logger.warning("Scoring failed after %d attempts: %s", attempt + 1, e)
                return 0
        return 0

    def fresh_attempt(self, original_prompt: str, directive: str,
                      best_directive: str | None = None,
                      score_history: list[int] | None = None) -> tuple[str | None, int]:
        """Generate a completely fresh attempt (not based on current best).
        Returns (output, score). Used when refinement has plateaued.
        Optionally uses insights from the run (best directive, score trajectory)."""
        try:
            insights = ""
            if best_directive:
                insights += f"The most effective improvement approach so far was: \"{best_directive}\"\n"
            if score_history and len(score_history) >= 2:
                insights += f"Previous scores plateaued at: {' -> '.join(str(s) for s in score_history[-5:])}\n"
                insights += "Try a fundamentally different approach.\n"

            prompt = (
                f"{original_prompt}\n\n"
                f"Additional guidance: {directive}\n\n"
                f"{insights}"
                f"Produce an excellent, complete solution. Output ONLY the solution."
            )
            output = self.engine.generate(prompt)
            if not output or not output.strip():
                return None, 0
            score = self._score_variant(original_prompt, output)
            return output, score
        except Exception as e:
            logger.warning("Fresh attempt failed: %s", e)
            return None, 0

    def run_cycle(
        self,
        original_prompt: str,
        current_best: str,
        current_score: int,
        directive: str,
        history_summary: str = "",
        score_feedback: str = "",
    ) -> tuple[str, int, int]:
        """Returns (best_output, best_score, best_variant_index).
        best_variant_index is -1 if no variant beat the current best."""
        current_best = self._maybe_summarize(current_best, original_prompt)

        variants = []
        use_reflection = self.config.use_reflection and current_score >= 60
        cycle_start = time.time()
        budget_per_branch = self.config.budget_seconds / max(self.config.dilation_factor, 1) / max(self.config.branch_factor, 1)

        for i in range(self.config.branch_factor):
            # Skip remaining branches if we've used >80% of per-cycle budget
            if i > 0 and budget_per_branch > 0:
                elapsed = time.time() - cycle_start
                if elapsed > budget_per_branch * self.config.branch_factor * 0.8:
                    logger.info("Skipping branch %d/%d — time budget exceeded (%.1fs)",
                                i + 1, self.config.branch_factor, elapsed)
                    break

            temp = self._branch_temperature(i)
            if i == 0 and use_reflection:
                variant = self._generate_with_reflection(
                    original_prompt, current_best, directive, current_score, history_summary, temperature=temp, score_feedback=score_feedback
                )
            else:
                variant = self._generate_variant(
                    original_prompt, current_best, directive, current_score, history_summary, temperature=temp, score_feedback=score_feedback
                )
            if variant is not None and self._validate_variant(variant, current_best, original_prompt):
                variants.append(variant)

        gen_time = time.time() - cycle_start

        if not variants:
            logger.warning("All variant generations failed, keeping current best")
            return current_best, current_score, -1

        # Deduplicate variants before scoring to save inference
        if len(variants) > 1:
            diversity = self._variant_diversity(variants)
            logger.info("Variant diversity: %.0f%% across %d variants", diversity * 100, len(variants))
            deduped = self._deduplicate_variants(variants)
            variants = [v for _, v in deduped]

        # Choose selection strategy
        score_start = time.time()
        if len(variants) >= 4:
            winner_variant, winner_index = self._tournament_select(original_prompt, variants)
            # Tournament uses comparisons, need to score the winner
            winner_score = self._score_variant(original_prompt, winner_variant, current_score=current_score)
        else:
            winner_variant, winner_index, winner_score = self._score_select(original_prompt, variants, current_score=current_score)
        score_time = time.time() - score_start
        logger.info("Cycle timing: gen=%.2fs score=%.2fs total=%.2fs",
                     gen_time, score_time, gen_time + score_time)

        if winner_score <= current_score:
            return current_best, current_score, -1

        # Sanity check: flag suspicious score jumps and force re-score
        if not self.scorer.sanity_check_score(winner_score, current_score, winner_variant, current_best):
            logger.warning("Score sanity check failed (delta=%d), re-scoring",
                           winner_score - current_score)
            rescore = self._score_variant(original_prompt, winner_variant, use_cot=True, current_score=current_score)
            if rescore <= current_score:
                return current_best, current_score, -1
            winner_score = rescore

        # Comparative validation — more aggressive at high scores where
        # self-scoring is less reliable
        comparative_threshold = 5 if current_score < 80 else 10
        if (winner_score - current_score) <= comparative_threshold:
            result = self._compare_outputs(original_prompt, current_best, winner_variant)
            if result == "A":
                logger.info("Comparative check overruled selection (delta=%d, threshold=%d)",
                            winner_score - current_score, comparative_threshold)
                return current_best, current_score, -1

        return winner_variant, winner_score, winner_index

    def _variant_diversity(self, variants: list[str]) -> float:
        """Average pairwise dissimilarity (0=identical, 1=completely different)."""
        if len(variants) < 2:
            return 1.0
        total = 0.0
        pairs = 0
        for i in range(len(variants)):
            for j in range(i + 1, len(variants)):
                total += 1.0 - self._similarity_ratio(variants[i], variants[j])
                pairs += 1
        return total / pairs if pairs > 0 else 1.0

    def _deduplicate_variants(self, variants: list[str], threshold: float = 0.85) -> list[tuple[int, str]]:
        """Remove near-duplicate variants, keeping the first of each cluster.
        Returns list of (original_index, variant)."""
        kept: list[tuple[int, str]] = []
        for i, v in enumerate(variants):
            is_dup = False
            for _, existing in kept:
                if self._similarity_ratio(v, existing) > threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append((i, v))
        if len(kept) < len(variants):
            logger.info("Deduplicated %d -> %d unique variants", len(variants), len(kept))
        return kept

    def _score_select(self, original_prompt: str, variants: list[str], current_score: int = 0) -> tuple[str, int, int]:
        """Score each variant, return (best_variant, best_index, best_score).
        Uses CoT scoring when there are multiple variants for better discrimination.
        Attempts crossover when top 2 variants have close scores.
        Skips scoring remaining variants if one already beats current by a wide margin."""
        use_cot = len(variants) > 1
        scored = []
        for i, variant in enumerate(variants):
            if self.force_ensemble:
                score = self._score_variant(original_prompt, variant, ensemble=True, current_score=current_score)
            else:
                score = self._score_variant(original_prompt, variant, use_cot=use_cot, current_score=current_score)
            scored.append((score, i, variant))
            # Early exit: if we found a variant that beats current by >20 points
            # and we've scored at least 2, skip remaining to save inference
            if len(variants) > 2 and i >= 1 and score > current_score + 20:
                remaining = len(variants) - i - 1
                if remaining > 0:
                    logger.info("Early exit scoring: variant %d scored %d (>%d+20), skipping %d remaining",
                                i, score, current_score, remaining)
                    break

        scored.sort(reverse=True, key=lambda x: x[0])
        best_score, best_index, best_variant = scored[0]

        # Try crossover if top 2 are close (within 10 points) and we have 2+ variants
        if len(scored) >= 2 and (scored[0][0] - scored[1][0]) <= 10:
            crossover = self._crossover(original_prompt, scored[0][2], scored[1][2],
                                         score_a=scored[0][0], score_b=scored[1][0])
            if crossover:
                cross_score = self._score_variant(original_prompt, crossover, use_cot=use_cot, current_score=current_score)
                if cross_score > best_score:
                    logger.info("Crossover produced better variant (%d > %d)", cross_score, best_score)
                    return crossover, -2, cross_score  # -2 indicates crossover origin

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

    def _crossover(self, original_prompt: str, variant_a: str, variant_b: str,
                   score_a: int = 0, score_b: int = 0) -> str | None:
        """Combine the best parts of two variants into a new output."""
        try:
            score_context = ""
            if score_a > 0 and score_b > 0:
                score_context = (
                    f"Solution A scored {score_a}/100, Solution B scored {score_b}/100. "
                    f"Take more from the higher-scoring solution but incorporate "
                    f"any unique strengths from the other.\n\n"
                )
            prompt = (
                f"Original task: {original_prompt}\n\n"
                f"Two candidate solutions were generated. Combine the best aspects of each "
                f"into a single superior solution.\n\n"
                f"{score_context}"
                f"Solution A:\n{variant_a}\n\n"
                f"Solution B:\n{variant_b}\n\n"
                f"Output ONLY the combined, improved solution."
            )
            result = self.engine.generate(prompt)
            if result and result.strip():
                return result
            return None
        except Exception as e:
            logger.warning("Crossover failed: %s", e)
            return None

    def _compare_outputs(self, original_prompt: str, output_a: str, output_b: str) -> str:
        """A/B compare two outputs, returns 'A', 'B', or 'TIE'."""
        try:
            prompt = self.scorer.build_comparative_prompt(original_prompt, output_a, output_b)
            raw = self.engine.generate(prompt, temperature=self.config.scoring_temperature)
            return self.scorer.parse_comparison(raw)
        except Exception as e:
            logger.warning("Comparison failed: %s", e)
            return "TIE"
