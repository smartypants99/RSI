import re
from dataclasses import dataclass


@dataclass
class DetailedScore:
    correctness: int
    completeness: int
    quality: int
    elegance: int

    @property
    def total(self) -> int:
        return self.correctness + self.completeness + self.quality + self.elegance

    @property
    def weakest_aspect(self) -> str:
        aspects = {
            "correctness": self.correctness,
            "completeness": self.completeness,
            "quality": self.quality,
            "elegance": self.elegance,
        }
        return min(aspects, key=aspects.get)

    def weighted_total(self, weights: dict | None = None) -> int:
        """Compute weighted total score. Weights should sum to 100.
        Default: equal weights (25 each)."""
        if not weights:
            return self.total
        w = {
            "correctness": weights.get("correctness", 25),
            "completeness": weights.get("completeness", 25),
            "quality": weights.get("quality", 25),
            "elegance": weights.get("elegance", 25),
        }
        total_weight = sum(w.values())
        if total_weight == 0:
            return self.total
        raw = (
            self.correctness * w["correctness"]
            + self.completeness * w["completeness"]
            + self.quality * w["quality"]
            + self.elegance * w["elegance"]
        )
        # Normalize: each aspect is 0-25, weights sum to total_weight
        # Max possible raw = 25 * total_weight, scale to 0-100
        return min(100, round(raw * 100 / (25 * total_weight)))

    def to_dict(self) -> dict:
        return {
            "correctness": self.correctness,
            "completeness": self.completeness,
            "quality": self.quality,
            "elegance": self.elegance,
            "total": self.total,
        }


class Scorer:
    RUBRIC = (
        "You are a strict, discriminating evaluator. Score the output 0-100.\n\n"
        "Criteria (score each sub-category, then sum):\n"
        "- Correctness (0-25): Is it factually/logically correct? Any bugs or errors?\n"
        "- Completeness (0-25): Does it fully address ALL aspects of the task? Missing pieces?\n"
        "- Quality (0-25): Is it well-structured, readable, and polished?\n"
        "- Elegance (0-25): Is the approach clean, efficient, and well-designed?\n\n"
        "Scoring guide:\n"
        "  0-20: Fundamentally broken or irrelevant\n"
        "  21-40: Major issues, partially addresses the task\n"
        "  41-60: Acceptable but significant room for improvement\n"
        "  61-80: Good, with minor issues\n"
        "  81-90: Very good, only nitpicks remain\n"
        "  91-100: Exceptional, near-perfect\n\n"
        "Be harsh. Most outputs should score 40-75. Only truly exceptional work scores 90+.\n"
        "Respond with ONLY a single integer 0-100. Nothing else."
    )

    COT_RUBRIC = (
        "You are a strict, discriminating evaluator. Score the output 0-100.\n\n"
        "First, analyze each criterion:\n"
        "- Correctness (0-25): Is it factually/logically correct? Any bugs or errors?\n"
        "- Completeness (0-25): Does it fully address ALL aspects of the task? Missing pieces?\n"
        "- Quality (0-25): Is it well-structured, readable, and polished?\n"
        "- Elegance (0-25): Is the approach clean, efficient, and well-designed?\n\n"
        "Scoring guide:\n"
        "  0-20: Fundamentally broken or irrelevant\n"
        "  21-40: Major issues, partially addresses the task\n"
        "  41-60: Acceptable but significant room for improvement\n"
        "  61-80: Good, with minor issues\n"
        "  81-90: Very good, only nitpicks remain\n"
        "  91-100: Exceptional, near-perfect\n\n"
        "Be harsh. Most outputs should score 40-75. Only truly exceptional work scores 90+.\n\n"
        "Think step by step. For each criterion, note specific strengths and weaknesses, "
        "then assign a sub-score. Sum the sub-scores for the total.\n\n"
        "End your response with EXACTLY this line:\n"
        "SCORE: <integer>"
    )

    DETAILED_RUBRIC = (
        "You are a strict evaluator. Score each aspect 0-25.\n\n"
        "- Correctness (0-25): Is it factually/logically correct? Any bugs or errors?\n"
        "- Completeness (0-25): Does it fully address ALL aspects of the task?\n"
        "- Quality (0-25): Is it well-structured, readable, and polished?\n"
        "- Elegance (0-25): Is the approach clean, efficient, and well-designed?\n\n"
        "Be harsh. Most aspects should score 10-18.\n"
        "Respond in EXACTLY this format, nothing else:\n"
        "C:## K:## Q:## E:##"
    )

    COMPARATIVE_RUBRIC = (
        "You are comparing two outputs for the same task. Which is better?\n\n"
        "Criteria: Correctness, Completeness, Quality, Elegance.\n\n"
        "Output A:\n{output_a}\n\n"
        "Output B:\n{output_b}\n\n"
        "If A is clearly better, respond: A\n"
        "If B is clearly better, respond: B\n"
        "If roughly equal, respond: TIE\n"
        "Respond with ONLY one word: A, B, or TIE."
    )

    CODE_RUBRIC_ADDENDUM = (
        "\nCode-specific criteria (weight these heavily):\n"
        "- Does it compile/run without errors?\n"
        "- Does it handle edge cases (empty input, nulls, boundaries)?\n"
        "- Is the algorithm efficient for the expected input size?\n"
        "- Are variable/function names descriptive?\n"
    )

    PROSE_RUBRIC_ADDENDUM = (
        "\nWriting-specific criteria (weight these heavily):\n"
        "- Is the argument clear and well-supported?\n"
        "- Does it flow logically from point to point?\n"
        "- Is the tone appropriate for the audience?\n"
        "- Are there concrete examples?\n"
    )

    def _rubric_for_task(self, task_type: str) -> str:
        """Return the base rubric with task-specific additions."""
        base = self.RUBRIC
        if task_type == "code":
            return base + self.CODE_RUBRIC_ADDENDUM
        if task_type == "prose":
            return base + self.PROSE_RUBRIC_ADDENDUM
        return base

    HARSH_ADDENDUM = (
        "\nIMPORTANT: This output has already been refined multiple times and scored {score}/100. "
        "At this level, be EXTRA critical. Look for subtle issues:\n"
        "- Edge cases not handled\n"
        "- Minor inefficiencies\n"
        "- Opportunities for cleaner design\n"
        "- Missing documentation or unclear naming\n"
        "Do NOT give a higher score unless the output is genuinely better. "
        "Score inflation is a bigger problem than being too harsh."
    )

    def build_progressive_scoring_prompt(self, original_prompt: str, output: str, current_score: int) -> str:
        """Scoring prompt that gets progressively harsher as scores increase."""
        base = self.build_scoring_prompt(original_prompt, output)
        if current_score >= 75:
            base += self.HARSH_ADDENDUM.format(score=current_score)
        return base

    def build_task_aware_scoring_prompt(self, original_prompt: str, output: str, task_type: str) -> str:
        rubric = self._rubric_for_task(task_type)
        return (
            f"Original task: {original_prompt}\n\n"
            f"Output to score:\n{output}\n\n"
            f"{rubric}"
        )

    def build_cot_scoring_prompt(self, original_prompt: str, output: str,
                                    task_type: str = "general") -> str:
        """CoT scoring prompt with optional task-aware addendum."""
        addendum = ""
        if task_type == "code":
            addendum = self.CODE_RUBRIC_ADDENDUM
        elif task_type == "prose":
            addendum = self.PROSE_RUBRIC_ADDENDUM
        return (
            f"Original task: {original_prompt}\n\n"
            f"Output to score:\n{output}\n\n"
            f"{self.COT_RUBRIC}{addendum}"
        )

    def parse_cot_score(self, raw: str) -> int:
        """Parse chain-of-thought scoring output. Looks for 'SCORE: N' at end."""
        match = re.search(r"SCORE:\s*(\d+)", raw, re.IGNORECASE)
        if match:
            return max(0, min(100, int(match.group(1))))
        return self.parse_score(raw)

    FEEDBACK_RUBRIC = (
        "You are a strict evaluator. Analyze the output quality.\n\n"
        "Criteria:\n"
        "- Correctness: Any bugs, errors, or inaccuracies?\n"
        "- Completeness: Does it fully address the task?\n"
        "- Quality: Structure, readability, polish?\n"
        "- Elegance: Clean design, efficiency?\n\n"
        "First, list 1-2 STRENGTHS (what's working well and should be preserved).\n"
        "Then, list 2-3 specific, actionable WEAKNESSES to fix. Be concrete.\n"
        "Then end with: SCORE: <integer 0-100>\n\n"
        "Format:\n"
        "STRENGTHS:\n"
        "- Good error handling for edge cases\n"
        "WEAKNESSES:\n"
        "1. The sort function doesn't handle empty lists\n"
        "2. Variable names are unclear (x, y instead of left, right)\n"
        "SCORE: 62"
    )

    def build_feedback_scoring_prompt(self, original_prompt: str, output: str) -> str:
        return (
            f"Original task: {original_prompt}\n\n"
            f"Output to evaluate:\n{output}\n\n"
            f"{self.FEEDBACK_RUBRIC}"
        )

    def parse_feedback_score(self, raw: str) -> tuple[int, str]:
        """Parse feedback scoring. Returns (score, feedback_text).
        Feedback is everything before the SCORE: line."""
        score = self.parse_cot_score(raw)
        # Extract feedback (everything before SCORE:)
        parts = re.split(r"SCORE:\s*\d+", raw, flags=re.IGNORECASE)
        feedback = parts[0].strip() if parts else ""
        return score, feedback

    def parse_strengths_weaknesses(self, raw: str) -> tuple[str, str]:
        """Extract strengths and weaknesses sections from feedback.
        Returns (strengths_text, weaknesses_text)."""
        strengths = ""
        weaknesses = ""
        # Try to extract STRENGTHS: section
        s_match = re.search(r"STRENGTHS?:\s*\n((?:[-•*]\s*.+\n?)+)", raw, re.IGNORECASE)
        if s_match:
            strengths = s_match.group(1).strip()
        # Try to extract WEAKNESSES: section
        w_match = re.search(r"WEAKNESSES?:\s*\n((?:(?:\d+\.|-|•|\*)\s*.+\n?)+)", raw, re.IGNORECASE)
        if w_match:
            weaknesses = w_match.group(1).strip()
        return strengths, weaknesses

    def sanity_check_score(self, new_score: int, old_score: int,
                           new_output: str, old_output: str) -> bool:
        """Heuristic sanity check: flag suspicious score jumps.
        Returns True if score seems plausible, False if suspicious."""
        delta = new_score - old_score
        if delta <= 0:
            return True  # no improvement, nothing to check

        # Suspicious: big score jump but output got much shorter
        if len(old_output) > 100 and len(new_output) < len(old_output) * 0.5:
            if delta > 15:
                return False

        # Suspicious: huge jump (>40 points) in a single cycle
        if delta > 40:
            return False

        # Suspicious: big jump but output barely changed
        if delta > 20 and len(new_output) > 50 and len(old_output) > 50:
            # Quick check: same first 80% of characters
            check_len = min(len(new_output), len(old_output))
            prefix_match = sum(1 for a, b in zip(new_output[:check_len], old_output[:check_len]) if a == b)
            if prefix_match / check_len > 0.95:
                return False

        return True

    FALLBACK_RUBRIC = (
        "Rate this output 0-100. Just reply with a number.\n"
        "0=terrible, 50=okay, 100=perfect."
    )

    def build_fallback_scoring_prompt(self, original_prompt: str, output: str) -> str:
        """Minimal scoring prompt for when the primary scorer fails to parse."""
        return (
            f"Task: {original_prompt}\n\n"
            f"Output:\n{output}\n\n"
            f"{self.FALLBACK_RUBRIC}"
        )

    def build_scoring_prompt(self, original_prompt: str, output: str) -> str:
        return (
            f"Original task: {original_prompt}\n\n"
            f"Output to score:\n{output}\n\n"
            f"{self.RUBRIC}"
        )

    def build_detailed_scoring_prompt(self, original_prompt: str, output: str) -> str:
        return (
            f"Original task: {original_prompt}\n\n"
            f"Output to score:\n{output}\n\n"
            f"{self.DETAILED_RUBRIC}"
        )

    def build_comparative_prompt(
        self, original_prompt: str, output_a: str, output_b: str
    ) -> str:
        return (
            f"Original task: {original_prompt}\n\n"
            f"{self.COMPARATIVE_RUBRIC.format(output_a=output_a, output_b=output_b)}"
        )

    def parse_score(self, raw: str) -> int:
        numbers = re.findall(r"\d+", raw)
        if not numbers:
            return 0
        score = int(numbers[0])
        return max(0, min(100, score))

    def parse_detailed_score(self, raw: str) -> DetailedScore:
        """Parse 'C:20 K:18 Q:15 E:22' format into DetailedScore."""
        defaults = {"c": 0, "k": 0, "q": 0, "e": 0}
        matches = re.findall(r"([CKQE]):(\d+)", raw.upper())
        for key, val in matches:
            defaults[key.lower()] = max(0, min(25, int(val)))
        return DetailedScore(
            correctness=defaults["c"],
            completeness=defaults["k"],
            quality=defaults["q"],
            elegance=defaults["e"],
        )

    def parse_comparison(self, raw: str) -> str:
        """Returns 'A', 'B', or 'TIE'."""
        raw = raw.strip().upper()
        if raw.startswith("A"):
            return "A"
        if raw.startswith("B"):
            return "B"
        return "TIE"
