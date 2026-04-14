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
        self._score_cache: dict[int, int] = {}  # hash(variant_text) -> score

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
            # Graduated trimming: shorten history first, then drop it, then drop feedback
            if history_block:
                # Keep only the last 2 lines of history
                lines = history_block.strip().split("\n")
                if len(lines) > 2:
                    history_block = "\n".join(lines[-2:]) + "\n"
                    total_est = self._estimate_prompt_tokens(
                        original_prompt, current_best, history_block, feedback_block, directive
                    )
            if total_est > prompt_budget:
                logger.info("Prompt too long (%d tokens), dropping history to fit %d budget",
                            total_est, prompt_budget)
                history_block = ""
                total_est = self._estimate_prompt_tokens(
                    original_prompt, current_best, feedback_block, directive
                )
            if total_est > prompt_budget:
                logger.info("Still too long, dropping feedback too")
                feedback_block = ""

        # Score context: tell the model what the score means on the rubric
        score_context = ""
        if current_score > 0:
            if current_score >= 91:
                score_context = f"(Rating: exceptional — only polish-level fixes remain)\n"
            elif current_score >= 81:
                score_context = f"(Rating: very good — only nitpicks remain)\n"
            elif current_score >= 61:
                score_context = f"(Rating: good with minor issues)\n"
            elif current_score >= 41:
                score_context = f"(Rating: acceptable but significant room for improvement)\n"
            else:
                score_context = f"(Rating: major issues present)\n"

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

        # Exemplar: if history shows what worked, highlight it
        exemplar_block = ""
        if history_summary and "improved the score" in history_summary:
            exemplar_block = (
                "Learn from what worked before — apply a similar level of change "
                "and specificity to this directive.\n\n"
            )

        urgency = ""
        if 0 < self.cycles_remaining <= 3:
            urgency = (
                f"IMPORTANT: Only {self.cycles_remaining} improvement cycle(s) remaining. "
                f"Make this change count — prioritize the highest-impact fix.\n\n"
            )

        return (
            f"Original task: {original_prompt}\n\n"
            f"Current solution (scored {current_score}/100):\n{score_context}{current_best}\n\n"
            f"{feedback_block}"
            f"{history_block}"
            f"{exemplar_block}"
            f"{urgency}"
            f"Improvement directive: {directive}\n\n"
            f"{guidance} "
            f"Do NOT repeat changes that already failed to improve the score. "
            f"Output ONLY the improved solution, nothing else."
            f"{self._format_hint()}"
        )

    def _strip_wrapper(self, text: str) -> str:
        """Strip markdown code fences and common wrapper patterns models add."""
        stripped = text.strip()
        if self.task_type == "code":
            # Strip ```python ... ``` or ``` ... ```
            import re
            match = re.match(r'^```\w*\n(.*?)```\s*$', stripped, re.DOTALL)
            if match:
                return match.group(1).strip()
        return stripped

    def _format_hint(self) -> str:
        """Return a format hint to prevent models from wrapping output."""
        if self.task_type == "code":
            return (
                "\nDo NOT wrap in markdown code fences (```). "
                "Do NOT add explanations before or after the code."
            )
        if self.task_type == "prose":
            return "\nDo NOT add meta-commentary about changes made."
        return ""

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

        if self.task_type == "code":
            analysis_points = (
                "1. What are the 2-3 most impactful changes that would raise the score?\n"
                "2. Which functions/sections should NOT be touched (they're correct)?\n"
                "3. Are there any edge cases or error paths not handled?\n"
                "4. What is the biggest risk of introducing a bug?"
            )
        elif self.task_type == "prose":
            analysis_points = (
                "1. Which paragraph or section is weakest and how should it change?\n"
                "2. What parts of the argument are strong and should be preserved?\n"
                "3. Is the structure logical? Any flow issues?\n"
                "4. What is the biggest risk of weakening the writing?"
            )
        else:
            analysis_points = (
                "1. What are the 2-3 most impactful changes that would raise the score?\n"
                "2. What should NOT be changed (things that are already good)?\n"
                "3. What is the biggest risk of regression?"
            )

        return (
            f"Original task: {original_prompt}\n\n"
            f"Current solution (scored {current_score}/100):\n{current_best}\n"
            f"{feedback_block}\n"
            f"Improvement directive: {directive}\n\n"
            f"Before making changes, analyze:\n"
            f"{analysis_points}\n\n"
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
            return self._strip_wrapper(variant)
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
        suffix = 0
        for i in range(1, min(shorter - prefix, longer - prefix) + 1):
            if a_stripped[-i] == b_stripped[-i]:
                suffix += 1
            else:
                break
        return (prefix + suffix) / longer

    def _validate_variant(self, variant: str, current_best: str, original_prompt: str) -> bool:
        """Quick validation to reject obviously bad variants without scoring."""
        reason = self._validation_failure_reason(variant, current_best, original_prompt)
        if reason:
            logger.info("Variant rejected: %s", reason)
            return False
        return True

    def _validation_failure_reason(self, variant: str, current_best: str, original_prompt: str) -> str | None:
        """Return rejection reason or None if variant is valid."""
        if not variant or not variant.strip():
            return "empty"
        if variant.strip() == original_prompt.strip():
            return "echoes original prompt"
        if len(current_best) > 50 and len(variant) < len(current_best) * 0.3:
            return f"too short ({len(variant)} vs {len(current_best)} chars)"
        if self._similarity_ratio(variant, current_best) > 0.95:
            return "too similar to current best"
        if len(variant) > len(current_best) * 1.5 and len(current_best) > 100:
            if current_best.strip() in variant:
                return "original embedded with padding"
        variant_lower = variant[:200].lower()
        meta_markers = ["here is the improved", "i've made the following", "changes made:",
                        "here's the updated", "i have improved", "the improved version"]
        if any(m in variant_lower for m in meta_markers):
            return "contains meta-commentary instead of output"
        if len(original_prompt) > 20 and variant.strip().startswith(original_prompt[:50]):
            return "starts with original prompt"
        return None

    def _generate_variant(self, original_prompt: str, current_best: str, directive: str, current_score: int, history_summary: str = "", temperature: float | None = None, score_feedback: str = "") -> str | None:
        """Generate a single variant, returning None on failure. Retries once with backoff."""
        for attempt in range(2):
            try:
                prompt = self._build_improvement_prompt(
                    original_prompt, current_best, directive, current_score, history_summary, score_feedback
                )
                max_tokens = self.config.max_output_tokens or None
                variant = self.engine.generate(prompt, temperature=temperature, max_tokens=max_tokens)
                if not variant or not variant.strip():
                    logger.warning("Empty variant generated, skipping")
                    return None
                return self._strip_wrapper(variant)
            except Exception as e:
                if attempt == 0:
                    logger.info("Variant generation failed: %s, retrying in 0.5s", e)
                    time.sleep(0.5)
                    continue
                logger.warning("Variant generation failed after retry: %s", e)
                return None
        return None

    def _scoring_addenda(self, variant: str, current_score: int) -> str:
        """Build scoring addenda for harshness and anti-bloat."""
        addenda = ""
        if current_score >= 75:
            addenda += self.scorer.HARSH_ADDENDUM.format(score=current_score)
        if self.initial_output_length > 0 and len(variant) > self.initial_output_length * 2.5:
            addenda += (
                "\nNote: This output is significantly longer than the initial version. "
                "Penalize unnecessary verbosity, padding, or repetition. "
                "Conciseness is a quality signal."
            )
        return addenda

    def _score_variant(self, original_prompt: str, variant: str, use_cot: bool = False, ensemble: bool = False, retries: int = 1, current_score: int = 0) -> int:
        """Score a variant, returning 0 on failure.
        If use_cot=True, uses chain-of-thought scoring for higher accuracy.
        If ensemble=True, scores twice (normal + CoT) and averages.
        Uses task-aware rubric when task_type is set, and progressive
        harshness when current_score >= 75.
        Retries on failure up to `retries` times.
        Results are cached by variant text hash to avoid redundant scoring."""
        # Check cache (only for non-ensemble, non-CoT — those vary by mode)
        if not ensemble and not use_cot:
            cache_key = hash(variant)
            if cache_key in self._score_cache:
                logger.debug("Score cache hit")
                return self._score_cache[cache_key]
        else:
            cache_key = None
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
                    score_prompt += self._scoring_addenda(variant, current_score)
                    raw_score = self.engine.generate(
                        score_prompt, temperature=self.config.scoring_temperature
                    )
                    score = self.scorer.parse_cot_score(raw_score)
                    subscores = self.scorer.parse_cot_subscores(raw_score)
                    if subscores:
                        logger.debug("CoT subscores: %s -> total %d", subscores, score)
                elif self.config.score_weights:
                    # Use detailed scoring with custom weights
                    score_prompt = self.scorer.build_detailed_scoring_prompt(original_prompt, variant, self.task_type)
                    score_prompt += self._scoring_addenda(variant, current_score)
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
                    score_prompt += self._scoring_addenda(variant, current_score)
                    raw_score = self.engine.generate(
                        score_prompt, temperature=self.config.scoring_temperature
                    )
                    score = self.scorer.parse_score(raw_score)
                if score == 0 and attempt < retries:
                    logger.info("Score was 0, retrying (%d/%d)", attempt + 1, retries)
                    time.sleep(min(0.5 * (2 ** attempt), 5.0))
                    continue
                if score == 0 and attempt == retries:
                    # All retries returned 0 — try fallback scoring
                    return self._fallback_score(original_prompt, variant)
                if cache_key is not None:
                    self._score_cache[cache_key] = score
                return score
            except Exception as e:
                if attempt < retries:
                    backoff = min(0.5 * (2 ** attempt), 5.0)
                    logger.info("Scoring attempt %d failed: %s, retrying in %.1fs",
                                attempt + 1, e, backoff)
                    time.sleep(backoff)
                    continue
                logger.warning("Scoring failed after %d attempts: %s, trying fallback",
                               attempt + 1, e)
                return self._fallback_score(original_prompt, variant)
        return 0

    def _fallback_score(self, original_prompt: str, variant: str) -> int:
        """Last-resort scoring with a minimal prompt that's easy to parse."""
        try:
            prompt = self.scorer.build_fallback_scoring_prompt(original_prompt, variant)
            raw = self.engine.generate(prompt, temperature=0.0)
            score = self.scorer.parse_score(raw)
            if score > 0:
                logger.info("Fallback scoring succeeded: %d", score)
            return score
        except Exception:
            logger.warning("Fallback scoring also failed")
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
            output = self._strip_wrapper(output)
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
        self._score_cache.clear()  # fresh cache per cycle
        current_best = self._maybe_summarize(current_best, original_prompt)

        variants = []
        rejected_count = 0
        rejection_reasons: set[str] = set()
        # Reflection is most valuable for code (catches bugs before they happen)
        # so activate it earlier for code tasks
        reflection_threshold = 50 if self.task_type == "code" else 60
        use_reflection = self.config.use_reflection and current_score >= reflection_threshold
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
            elif variant is not None:
                rejected_count += 1
                reason = self._validation_failure_reason(variant, current_best, original_prompt)
                if reason:
                    rejection_reasons.add(reason)

        if rejected_count > 0:
            logger.info("Rejected %d/%d variants (validation failures: %s)",
                        rejected_count, rejected_count + len(variants),
                        ", ".join(rejection_reasons))

        gen_time = time.time() - cycle_start

        # If all variants were rejected, retry once with anti-pattern guidance
        if not variants and rejected_count > 0 and rejection_reasons:
            anti_hint = "CRITICAL: " + "; ".join(
                f"Do NOT {r}" for r in rejection_reasons
            )
            logger.info("All variants rejected, retrying with anti-pattern hint")
            retry_variant = self._generate_variant(
                original_prompt, current_best, directive, current_score,
                history_summary, score_feedback=score_feedback + "\n" + anti_hint,
            )
            if retry_variant and self._validate_variant(retry_variant, current_best, original_prompt):
                variants.append(retry_variant)

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
            winner_variant, winner_index, winner_score = self._score_select(original_prompt, variants, current_score=current_score, current_best=current_best, task_type=self.task_type)
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

    def _score_select(self, original_prompt: str, variants: list[str], current_score: int = 0, current_best: str = "", task_type: str = "general") -> tuple[str, int, int]:
        """Score each variant, return (best_variant, best_index, best_score).
        Uses CoT scoring when there are multiple variants for better discrimination.
        Attempts crossover when top 2 variants have close scores.
        When top variants are within 3 points, prefers the more different one.
        Skips scoring remaining variants if one already beats current by a wide margin."""
        use_cot = len(variants) > 1
        scored = []
        # Adaptive early-exit threshold: lower when many variants, higher when few
        early_exit_margin = max(10, 30 - len(variants) * 5)

        # Score most-diverse-first: variants most different from current_best
        # are more likely genuine improvements, so score them first for early exit
        if current_best and len(variants) >= 3:
            indexed = list(enumerate(variants))
            indexed.sort(key=lambda iv: self._similarity_ratio(iv[1], current_best))
            scoring_order = [idx for idx, _ in indexed]
        else:
            scoring_order = list(range(len(variants)))

        for order_pos, i in enumerate(scoring_order):
            variant = variants[i]
            if self.force_ensemble:
                score = self._score_variant(original_prompt, variant, ensemble=True, current_score=current_score)
            else:
                score = self._score_variant(original_prompt, variant, use_cot=use_cot, current_score=current_score)
            scored.append((score, i, variant))
            # Early exit: skip remaining once we find a clearly better variant
            if len(variants) > 2 and order_pos >= 1 and score > current_score + early_exit_margin:
                remaining = len(variants) - i - 1
                if remaining > 0:
                    logger.info("Early exit scoring: variant %d scored %d (>%d+%d), skipping %d remaining",
                                i, score, current_score, early_exit_margin, remaining)
                    break

        scored.sort(reverse=True, key=lambda x: x[0])
        best_score, best_index, best_variant = scored[0]

        # Diversity tiebreaker: when top variants are within 3 points,
        # prefer the one most different from current_best (more likely genuine improvement)
        if current_best and len(scored) >= 2 and (scored[0][0] - scored[1][0]) <= 3:
            candidates = [s for s in scored if scored[0][0] - s[0] <= 3]
            most_diverse = max(candidates,
                               key=lambda s: 1.0 - self._similarity_ratio(s[2], current_best))
            if most_diverse != scored[0]:
                logger.info("Diversity tiebreaker: chose variant %d (score %d) over %d (score %d)",
                            most_diverse[1], most_diverse[0], scored[0][1], scored[0][0])
                best_score, best_index, best_variant = most_diverse

        # Try crossover if top 2 are close (within 10 points), both beat current,
        # and the top score is worth combining (above 40 — don't crossover garbage)
        if (len(scored) >= 2
                and (scored[0][0] - scored[1][0]) <= 10
                and scored[1][0] > current_score
                and scored[0][0] >= 40):
            crossover = self._crossover(original_prompt, scored[0][2], scored[1][2],
                                         score_a=scored[0][0], score_b=scored[1][0],
                                         task_type=task_type)
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
                   score_a: int = 0, score_b: int = 0,
                   task_type: str = "general") -> str | None:
        """Combine the best parts of two variants into a new output."""
        try:
            score_context = ""
            if score_a > 0 and score_b > 0:
                score_context = (
                    f"Solution A scored {score_a}/100, Solution B scored {score_b}/100. "
                    f"Take more from the higher-scoring solution but incorporate "
                    f"any unique strengths from the other.\n\n"
                )
            task_guidance = ""
            if task_type == "code":
                task_guidance = (
                    "This is code. Ensure the combined solution compiles/runs correctly. "
                    "Merge the best logic, error handling, and structure from each.\n\n"
                )
            elif task_type == "prose":
                task_guidance = (
                    "This is prose. Merge the strongest arguments, examples, and phrasing "
                    "from each. Ensure consistent tone and smooth transitions.\n\n"
                )
            prompt = (
                f"Original task: {original_prompt}\n\n"
                f"Two candidate solutions were generated. Combine the best aspects of each "
                f"into a single superior solution.\n\n"
                f"{task_guidance}{score_context}"
                f"Solution A:\n{variant_a}\n\n"
                f"Solution B:\n{variant_b}\n\n"
                f"Output ONLY the combined, improved solution."
            )
            result = self.engine.generate(prompt)
            if result and result.strip():
                return self._strip_wrapper(result)
            return None
        except Exception as e:
            logger.warning("Crossover failed: %s", e)
            return None

    def _compare_outputs(self, original_prompt: str, output_a: str, output_b: str) -> str:
        """A/B compare two outputs, returns 'A', 'B', or 'TIE'."""
        try:
            prompt = self.scorer.build_comparative_prompt(original_prompt, output_a, output_b, task_type=self.task_type)
            raw = self.engine.generate(prompt, temperature=self.config.scoring_temperature)
            return self.scorer.parse_comparison(raw)
        except Exception as e:
            logger.warning("Comparison failed: %s", e)
            return "TIE"
