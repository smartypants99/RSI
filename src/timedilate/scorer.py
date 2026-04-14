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
