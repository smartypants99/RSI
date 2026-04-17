"""Diagnostics engine — finds what the model is bad at."""

import hashlib
import logging
import operator
import random
import re
import sys
import textwrap
import time
from dataclasses import dataclass, field
from fractions import Fraction
from math import comb, gcd
from typing import Optional

import torch

from ..utils.config import DiagnosticsConfig
from ..utils.model_loader import ModelLoader

logger = logging.getLogger(__name__)


@dataclass
class WeaknessReport:
    """A diagnosed weakness in the model."""
    domain: str
    subdomain: str
    severity: float  # 0-1, higher = worse
    evidence: list[dict] = field(default_factory=list)
    weak_layers: list[str] = field(default_factory=list)
    description: str = ""
    difficulty: str = "mixed"
    n_questions: int = 0
    n_failures: int = 0
    confidence_lower: float = 0.0
    confidence_upper: float = 1.0
    significance: float = 1.0
    calibrated_confidence: float = 0.0
    expected_answer_type: str = "freeform"
    check_type_distribution: dict = field(default_factory=dict)
    dominant_check_type: str = "contains"
    invariants: list[str] = field(default_factory=list)


@dataclass
class DiagnosticResult:
    """Full diagnostic result for one cycle."""
    cycle: int
    timestamp: float
    weaknesses: list[WeaknessReport] = field(default_factory=list)
    domain_scores: dict[str, float] = field(default_factory=dict)
    domain_question_counts: dict[str, int] = field(default_factory=dict)
    layer_health: dict[str, float] = field(default_factory=dict)
    total_questions: int = 0
    total_correct: int = 0

    @property
    def overall_score(self) -> float:
        if not self.domain_scores:
            return 0.0
        # Weight by question count so domains with more questions (and thus more
        # reliable scores) contribute proportionally. Domains with 5 questions
        # shouldn't have the same influence as domains with 200.
        if self.domain_question_counts:
            total_q = sum(self.domain_question_counts.values())
            if total_q > 0:
                return sum(
                    score * self.domain_question_counts.get(domain, 1)
                    for domain, score in self.domain_scores.items()
                ) / total_q
        return sum(self.domain_scores.values()) / len(self.domain_scores)


# Each domain -> subdomain -> list of parameterized templates
# Templates with {A},{B},{C} get randomized nouns for anti-memorization
QUESTION_TEMPLATES = {
    "reasoning": {
        "syllogism": [
            {"prompt": "If all {A} are {B} and all {B} are {C}, are all {A} {C}? Explain step by step.", "expected": "yes", "check_type": "contains"},
            {"prompt": "If all {A} are {B} and some {B} are {C}, are all {A} {C}? Explain step by step.", "expected": "no", "check_type": "contains"},
            {"prompt": "If no {A} are {B} and all {C} are {A}, are any {C} also {B}? Explain step by step.", "expected": "no", "check_type": "contains"},
            {"prompt": "If some {A} are {B} and some {B} are {C}, are some {A} necessarily {C}? Explain.", "expected": "no", "check_type": "contains"},
            {"prompt": "All {A} are {B}. No {B} are {C}. Are any {A} {C}? Prove it.", "expected": "no", "check_type": "contains"},
        ],
        "cognitive_bias": [
            {"prompt": "A bat and ball cost ${total} total. The bat costs ${diff} more than the ball. How much does the ball cost? Show your work.", "expected": "{ball_answer}", "check_type": "contains", "_params": "bat_ball"},
            {"prompt": "If it takes {n} machines {n} minutes to make {n} widgets, how long would it take {m} machines to make {m} widgets? Show your work.", "expected": "{n}", "check_type": "contains", "_params": "machines"},
            {"prompt": "A lily pad doubles in size every day. If it takes {days} days to cover the whole lake, how many days to cover half? Show reasoning.", "expected": "{half_answer}", "check_type": "contains", "_params": "lilypad"},
        ],
        "counterfactual": [
            {"prompt": "If the sun rose in the west, which direction would shadows point at noon in the northern hemisphere? Reason step by step.", "expected": "east", "check_type": "contains"},
            {"prompt": "If gravity were repulsive instead of attractive, what would happen to the ocean tides? Reason step by step.", "expected": "away", "check_type": "contains"},
        ],
        "causal": [
            {"prompt": "A plant dies. The soil is dry. Did the dry soil cause the plant to die, or did the dead plant cause the soil to dry? What additional information would you need?", "expected": "additional information", "check_type": "contains"},
            {"prompt": "Ice cream sales and drowning deaths both increase in summer. Does ice cream cause drowning? Explain the logical error.", "expected": "correlation", "check_type": "contains"},
        ],
    },
    "math": {
        "calculus": [
            {"prompt": "What is the derivative of x^{n} + {a}x^2 - {b}x + {c}? Show each step.", "expected": "{deriv_answer}", "check_type": "contains", "_params": "derivative"},
            {"prompt": "Find the integral of 2x*cos(x^2) dx. Show your substitution.", "expected": "sin(x^2)", "check_type": "math_equiv"},
            {"prompt": "Find the limit as x approaches 0 of sin(x)/x. Prove your answer.", "expected": "1", "check_type": "contains"},
            {"prompt": "Find the second derivative of e^(2x) + ln(x). Show all steps.", "expected": "4*exp(2*x) - 1/x**2", "check_type": "math_equiv"},
        ],
        "algebra": [
            {"prompt": "Solve: {base}^x = {result}. Show your reasoning.", "expected": "{exp_answer}", "check_type": "contains", "_params": "exponential"},
            {"prompt": "Factor completely: x^3 - {cube}. Show each step.", "expected": "(x - {root})", "check_type": "contains", "_params": "factor_cube"},
            {"prompt": "Solve the system: {a1}x + {b1}y = {c1}, {a2}x + {b2}y = {c2}. Show all work.", "expected": "x = {x_answer}", "check_type": "contains", "_params": "system"},
            {"prompt": "Solve the system: {a1}x + {b1}y = {c1}, {a2}x + {b2}y = {c2}. Find y. Show all work.", "expected": "y = {y_answer}", "check_type": "contains", "_params": "system"},
        ],
        "number_theory": [
            {"prompt": "Is {num} prime? Explain your reasoning step by step.", "expected": "{prime_answer}", "check_type": "contains", "_params": "primality"},
            {"prompt": "What is the GCD of {a} and {b}? Use the Euclidean algorithm and show each step.", "expected": "{gcd_answer}", "check_type": "contains", "_params": "gcd"},
        ],
        "probability": [
            {"prompt": "You flip a fair coin {n} times. What is the probability of getting exactly {k} heads? Show the calculation.", "expected": "{prob_answer}", "check_type": "contains", "_params": "coin_flip"},
        ],
    },
    "code": {
        "implementation": [
            {"prompt": "Write a Python function that checks if a string is a palindrome, handling edge cases (empty string, spaces, case).", "expected": "def", "check_type": "code_executes"},
            {"prompt": "Write a Python function to find all prime numbers up to n using the Sieve of Eratosthenes.", "expected": "def", "check_type": "code_executes"},
            {"prompt": "Write a Python function that merges two sorted lists into one sorted list without using built-in sort.", "expected": "def", "check_type": "code_executes"},
            {"prompt": "Write a Python function that implements binary search on a sorted list. Return the index or -1.", "expected": "def", "check_type": "code_executes"},
        ],
        "debugging": [
            {"prompt": "This Python code has a bug: `def fib(n): return fib(n-1) + fib(n-2)`. What's wrong and how do you fix it? Explain step by step.", "expected": "base case", "check_type": "contains"},
            {"prompt": "This code has a bug: `def avg(lst): return sum(lst) / len(lst)`. What happens with an empty list and how do you fix it?", "expected": "empty", "check_type": "contains"},
        ],
        "complexity": [
            {"prompt": "What is the time complexity of this: `for i in range(n): for j in range(i, n): pass`? Prove it.", "expected": "n^2", "check_type": "contains"},
            {"prompt": "What is the time complexity of binary search? Prove it mathematically.", "expected": "log", "check_type": "contains"},
        ],
    },
    "logic": {
        "fallacies": [
            {"prompt": "If it's raining, the ground is wet. The ground is wet. Is it necessarily raining? Name the fallacy.", "expected": "affirming the consequent", "check_type": "contains"},
            {"prompt": "Everyone who studied passed. Alice passed. Did Alice study? Explain the logical error.", "expected": "affirming the consequent", "check_type": "contains"},
            {"prompt": "'No true Scotsman would do X. But McGregor did X. He's not a true Scotsman.' What fallacy is this?", "expected": "no true scotsman", "check_type": "contains"},
            {"prompt": "'You can't prove ghosts don't exist, so they must exist.' What fallacy is this?", "expected": "burden of proof", "check_type": "contains"},
            {"prompt": "'Everyone is buying this product, so it must be good.' What fallacy is this?", "expected": "bandwagon", "check_type": "contains"},
        ],
        "propositional": [
            {"prompt": "Is 'If P then Q' logically equivalent to 'If not Q then not P'? Prove using a truth table.", "expected": "yes", "check_type": "contains"},
            {"prompt": "Is 'P or Q' the same as 'not (not P and not Q)'? Prove using De Morgan's law.", "expected": "yes", "check_type": "contains"},
        ],
        "modal": [
            {"prompt": "Is it possible for something to be necessarily true but not actually true? Explain using modal logic.", "expected": "no", "check_type": "contains"},
        ],
    },
    "science": {
        "physics": [
            {"prompt": "A ball is thrown straight up at {v} m/s. Ignoring air resistance, how high does it go? (g=10 m/s^2) Show all steps.", "expected": "{height_answer}", "check_type": "contains", "_params": "projectile"},
            {"prompt": "Two objects of mass {m1} kg and {m2} kg are {d} meters apart. What is the gravitational force between them? (G=6.67e-11) Show work.", "expected": "{grav_answer}", "check_type": "math_equiv", "_params": "gravity"},
        ],
        "biology": [
            {"prompt": "Explain why antibiotics don't work on viruses. Be specific about the mechanism.", "expected": "cell", "check_type": "contains"},
            {"prompt": "Explain how CRISPR-Cas9 gene editing works at the molecular level.", "expected": "guide RNA", "check_type": "contains"},
        ],
        "chemistry": [
            {"prompt": "Balance this equation: Fe + O2 -> Fe2O3. Show your work.", "expected": "4Fe", "check_type": "contains"},
            {"prompt": "What is the pH of a 0.01 M HCl solution? Show the calculation.", "expected": "2", "check_type": "contains"},
        ],
    },
    "common_sense": {
        "physical": [
            {"prompt": "Can you fit a basketball inside a mailbox? Explain your reasoning about the physical dimensions.", "expected": "no", "check_type": "contains"},
            {"prompt": "If you drop a bowling ball and a feather in a vacuum, which hits the ground first? Explain.", "expected": "same", "check_type": "contains"},
            {"prompt": "Can a person lift a car with one hand? Explain the physics involved.", "expected": "no", "check_type": "contains"},
            {"prompt": "If you pour water into a cup that already has a hole in the bottom, will the cup fill up? Explain.", "expected": "no", "check_type": "contains"},
            {"prompt": "Can sound travel through outer space? Explain why or why not.", "expected": "no", "check_type": "contains"},
        ],
        "social": [
            {"prompt": "If someone at a funeral is laughing loudly, is this likely appropriate? Explain.", "expected": "no", "check_type": "contains"},
            {"prompt": "Is it appropriate to clap after a surgeon finishes an operation? Explain the social norms.", "expected": "no", "check_type": "contains"},
            {"prompt": "If a stranger asks for your home address on the street, should you give it? Explain.", "expected": "no", "check_type": "contains"},
        ],
        "temporal": [
            {"prompt": "Can you eat breakfast before you wake up? Explain the logical impossibility or possibility.", "expected": "no", "check_type": "contains"},
            {"prompt": "Can yesterday come after tomorrow? Explain.", "expected": "no", "check_type": "contains"},
            {"prompt": "Can you remember events from the future? Explain why or why not.", "expected": "no", "check_type": "contains"},
            {"prompt": "If today is Monday, what day was it 3 days ago? Show your reasoning.", "expected": "friday", "check_type": "contains"},
        ],
        "spatial": [
            {"prompt": "If you are facing north and turn 90 degrees clockwise, which direction are you facing? Explain.", "expected": "east", "check_type": "contains"},
            {"prompt": "Can a cube have exactly 5 faces? Explain using geometry.", "expected": "no", "check_type": "contains"},
        ],
    },
    "language_understanding": {
        "ambiguity": [
            {"prompt": "What does 'I saw her duck' mean? List all possible interpretations.", "expected": "duck", "check_type": "contains"},
            {"prompt": "What does 'Time flies like an arrow; fruit flies like a banana' mean? Explain each interpretation.", "expected": "flies", "check_type": "contains"},
            {"prompt": "What does 'Visiting relatives can be boring' mean? List all interpretations.", "expected": "relatives", "check_type": "contains"},
            {"prompt": "What does 'The chicken is ready to eat' mean? Explain both interpretations.", "expected": "chicken", "check_type": "contains"},
        ],
        "implication": [
            {"prompt": "'Some students passed the exam.' Does this imply some failed? Explain pragmatic vs logical.", "expected": "not necessarily", "check_type": "contains"},
            {"prompt": "'He managed to finish on time.' Does 'managed' imply difficulty? Explain.", "expected": "difficulty", "check_type": "contains"},
            {"prompt": "'Not all birds can fly.' Does this mean some birds can fly? Explain the logic.", "expected": "some", "check_type": "contains"},
        ],
        "sentiment": [
            {"prompt": "'Oh great, another meeting.' Is this positive or negative? Explain sarcasm detection.", "expected": "sarcas", "check_type": "contains"},
            {"prompt": "'What a wonderful way to spend my Saturday — doing taxes.' Positive or negative? Explain.", "expected": "sarcas", "check_type": "contains"},
            {"prompt": "'I could care less about the result.' What does the speaker actually mean? Explain.", "expected": "couldn't care less", "check_type": "contains"},
        ],
        "reference": [
            {"prompt": "'The trophy would not fit in the suitcase because it was too big.' What does 'it' refer to? Explain.", "expected": "trophy", "check_type": "contains"},
            {"prompt": "'The city council refused the protesters a permit because they feared violence.' Who feared violence? Explain.", "expected": "council", "check_type": "contains"},
        ],
    },
    "abstraction": {
        "pattern": [
            {"prompt": "What comes next: 1, 1, 2, 3, 5, 8, 13, ...? Explain the pattern.", "expected": "21", "check_type": "contains"},
            {"prompt": "What comes next: 2, 6, 12, 20, 30, ...? Explain the pattern.", "expected": "42", "check_type": "contains"},
            {"prompt": "What comes next: 1, 4, 9, 16, 25, ...? Explain the pattern.", "expected": "36", "check_type": "contains"},
            {"prompt": "What comes next: 1, 3, 6, 10, 15, ...? Explain the pattern.", "expected": "21", "check_type": "contains"},
            {"prompt": "What comes next: 2, 3, 5, 7, 11, 13, ...? Explain the pattern.", "expected": "17", "check_type": "contains"},
            {"prompt": "What comes next: 1, 8, 27, 64, 125, ...? Explain the pattern.", "expected": "216", "check_type": "contains"},
        ],
        "analogy": [
            {"prompt": "Hot is to cold as light is to ___. Explain the relationship.", "expected": "dark", "check_type": "contains"},
            {"prompt": "Bird is to nest as human is to ___. Explain the analogy.", "expected": "house", "check_type": "contains"},
            {"prompt": "Painter is to canvas as writer is to ___. Explain the analogy.", "expected": "paper", "check_type": "contains"},
            {"prompt": "Fish is to water as bird is to ___. Explain the analogy.", "expected": "air", "check_type": "contains"},
        ],
        "classification": [
            {"prompt": "Which doesn't belong: apple, banana, carrot, grape? Explain your reasoning.", "expected": "carrot", "check_type": "contains"},
            {"prompt": "Which doesn't belong: circle, square, triangle, cube? Explain your reasoning.", "expected": "cube", "check_type": "contains"},
            {"prompt": "Which doesn't belong: 2, 3, 5, 9, 11? Explain your reasoning.", "expected": "9", "check_type": "contains"},
        ],
    },
}

# Module-level constants for Unicode superscript normalization in _check_answer
_SUPERSCRIPT_TABLE = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
_RE_SUPERSCRIPT = re.compile(r'(?<=[^⁰¹²³⁴⁵⁶⁷⁸⁹])[⁰¹²³⁴⁵⁶⁷⁸⁹]+')
_ANSWER_PREFIX_RE = re.compile(
    r'^(?:the\s+answer\s+is|final\s+answer(?:\s+is)?|answer\s*[:=]?|a\s*[:=]|=|equals|is)\s*[:=]?\s*',
    re.I,
)

_NOUNS = ["bloops", "razzles", "lazzles", "flinks", "morgs", "pliffs", "zents", "quargs",
          "dreps", "snivs", "wumps", "glorps", "brixes", "tazzles", "nimps", "florbs",
          "grepts", "shlinks", "vorps", "klems", "frobs", "glints", "praxes", "qualms"]

_LANG_TAGS = frozenset({"python", "python3", "py", "javascript", "js", "typescript", "ts",
                        "java", "go", "rust", "ruby", "bash", "sh", "sql",
                        "cpp", "c", "csharp", "html", "css", "json", "yaml"})


# -----------------------------------------------------------------------------
# Difficulty bands and programmatic generators
# -----------------------------------------------------------------------------
# Each programmatic generator returns a list of {prompt, expected, check_type,
# subdomain, difficulty} question dicts. They are deterministic given a seed,
# and produce large volumes of unique questions via parameter sampling.

_DIFFICULTIES = ("easy", "medium", "hard", "expert")


def _wilson_bounds(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. Returns (lower, upper)."""
    if n <= 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * ((p * (1 - p) / n) + (z * z / (4 * n * n))) ** 0.5) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _binomial_significance(successes: int, n: int, baseline: float = 0.5) -> float:
    """One-sided p-value-ish: probability of observing >= successes failures under baseline.

    Uses normal approximation with continuity correction. Lower = more significant.
    """
    if n <= 0:
        return 1.0
    import math
    p = successes / n
    if p <= baseline:
        return 1.0
    mean = baseline * n
    std = (baseline * (1 - baseline) * n) ** 0.5
    if std == 0:
        return 0.0 if successes > mean else 1.0
    z = (successes - 0.5 - mean) / std
    # Standard normal upper tail
    return 0.5 * math.erfc(z / math.sqrt(2))


# ---------- Math generators ----------

def _gen_arithmetic(rng: random.Random, difficulty: str) -> dict:
    if difficulty == "easy":
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        op = rng.choice(["+", "-"])
    elif difficulty == "medium":
        a, b = rng.randint(100, 999), rng.randint(10, 99)
        op = rng.choice(["+", "-", "*"])
    elif difficulty == "hard":
        a, b = rng.randint(100, 9999), rng.randint(10, 999)
        op = rng.choice(["*", "//"])
    else:  # expert
        a, b = rng.randint(1000, 99999), rng.randint(100, 999)
        op = rng.choice(["*", "//", "%"])
    expr = f"{a} {op} {b}"
    _OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul,
            "//": operator.floordiv, "%": operator.mod}
    ans = _OPS[op](a, b)
    return {
        "prompt": f"Compute {expr}. Show intermediate steps.",
        "expected": str(ans),
        "check_type": "numeric",
        "subdomain": "arithmetic",
        "difficulty": difficulty,
    }


def _gen_linear_equation(rng: random.Random, difficulty: str) -> dict:
    x = rng.randint(-10, 10) if difficulty in ("easy", "medium") else rng.randint(-50, 50)
    a = rng.randint(2, 9) if difficulty == "easy" else rng.randint(2, 20)
    b = rng.randint(-20, 20)
    c = a * x + b
    return {
        "prompt": f"Solve for x: {a}x + {b} = {c}. Show your work.",
        "expected": str(x),
        "check_type": "numeric",
        "subdomain": "algebra",
        "difficulty": difficulty,
    }


def _gen_quadratic(rng: random.Random, difficulty: str) -> dict:
    r1 = rng.randint(-6, 6)
    r2 = rng.randint(-6, 6)
    # (x - r1)(x - r2) = x^2 - (r1+r2)x + r1*r2
    b = -(r1 + r2)
    c = r1 * r2
    roots = sorted({r1, r2})
    expected = ", ".join(str(r) for r in roots)

    # Build clean prompt — omit zero-coefficient terms to avoid
    # nonsensical prompts like "x^2 + 0x + 0 = 0".
    terms = ["x^2"]
    if b != 0:
        sign_b = "+" if b > 0 else "-"
        terms.append(f"{sign_b} {abs(b)}x")
    if c != 0:
        sign_c = "+" if c > 0 else "-"
        terms.append(f"{sign_c} {abs(c)}")
    return {
        "prompt": f"Find all real roots of {' '.join(terms)} = 0. List them.",
        "expected": expected,
        "check_type": "numeric_set",
        "subdomain": "algebra",
        "difficulty": difficulty,
    }


def _gen_derivative(rng: random.Random, difficulty: str) -> dict:
    if difficulty == "easy":
        n = rng.randint(2, 4)
        a = rng.randint(1, 9)
        return {
            "prompt": f"Find d/dx of {a}x^{n}. Show your work.",
            "expected": f"{a*n}*x**{n-1}",
            "check_type": "math_equiv",
            "subdomain": "calculus",
            "difficulty": difficulty,
        }
    if difficulty == "medium":
        a, b, c = rng.randint(1, 6), rng.randint(1, 8), rng.randint(1, 10)
        return {
            "prompt": f"Find d/dx of {a}x^3 + {b}x^2 - {c}x. Show each step.",
            "expected": f"{3*a}*x**2 + {2*b}*x - {c}",
            "check_type": "math_equiv",
            "subdomain": "calculus",
            "difficulty": difficulty,
        }
    if difficulty == "hard":
        a = rng.randint(2, 5)
        return {
            "prompt": f"Find d/dx of sin({a}x) * cos(x). Use the product rule.",
            "expected": f"{a}*cos({a}*x)*cos(x) - sin({a}*x)*sin(x)",
            "check_type": "math_equiv",
            "subdomain": "calculus",
            "difficulty": difficulty,
        }
    # expert
    a = rng.randint(2, 4)
    return {
        "prompt": f"Find d/dx of exp({a}x) * ln(x). Use the product rule.",
        "expected": f"{a}*exp({a}*x)*log(x) + exp({a}*x)/x",
        "check_type": "math_equiv",
        "subdomain": "calculus",
        "difficulty": difficulty,
    }


def _gen_gcd(rng: random.Random, difficulty: str) -> dict:
    lo, hi = {"easy": (2, 50), "medium": (10, 500), "hard": (100, 5000), "expert": (1000, 100000)}[difficulty]
    a = rng.randint(lo, hi)
    b = rng.randint(lo, hi)
    return {
        "prompt": f"Compute GCD({a}, {b}) using the Euclidean algorithm. Show each step.",
        "expected": str(gcd(a, b)),
        "check_type": "numeric",
        "subdomain": "number_theory",
        "difficulty": difficulty,
    }


def _gen_modular(rng: random.Random, difficulty: str) -> dict:
    lo, hi = {"easy": (2, 20), "medium": (5, 200), "hard": (10, 2000), "expert": (100, 99999)}[difficulty]
    a = rng.randint(lo, hi)
    m = rng.randint(max(2, lo // 2), hi)
    return {
        "prompt": f"What is {a} mod {m}? Show the division.",
        "expected": str(a % m),
        "check_type": "numeric",
        "subdomain": "number_theory",
        "difficulty": difficulty,
    }


def _gen_probability(rng: random.Random, difficulty: str) -> dict:
    if difficulty == "easy":
        n = rng.choice([2, 3, 4])
    elif difficulty == "medium":
        n = rng.choice([5, 6, 7])
    elif difficulty == "hard":
        n = rng.choice([8, 9, 10])
    else:
        n = rng.choice([11, 12, 14, 16])
    k = rng.randint(1, n - 1)
    frac = Fraction(comb(n, k), 2 ** n)
    prob = float(frac)
    return {
        "prompt": f"Probability of exactly {k} heads in {n} fair coin flips. Give the exact fraction.",
        "expected": f"{frac.numerator}/{frac.denominator}",
        "check_type": "numeric",
        "subdomain": "probability",
        "difficulty": difficulty,
        "_numeric_value": float(prob),
    }


# ---------- Logic generators ----------

def _gen_syllogism(rng: random.Random, difficulty: str) -> dict:
    a, b, c = rng.sample(_NOUNS, 3)
    patterns = {
        "easy": [
            (f"All {a} are {b}. All {b} are {c}. Are all {a} {c}?", "yes"),
            (f"No {a} are {b}. All {c} are {a}. Are any {c} {b}?", "no"),
        ],
        "medium": [
            (f"All {a} are {b}. Some {b} are {c}. Are all {a} necessarily {c}?", "no"),
            (f"Some {a} are {b}. Some {b} are {c}. Are some {a} necessarily {c}?", "no"),
        ],
        "hard": [
            (f"All {a} are {b}. No {b} are {c}. Are any {a} {c}?", "no"),
            (f"If no {a} are {b} and no {b} are {c}, does it follow that no {a} are {c}?", "no"),
        ],
        "expert": [
            (f"All {a} that are not {b} are {c}. No {c} are {b}. Is every {a} either {b} or {c}?", "yes"),
            (f"If some {a} are {b}, and all {b} are {c}, but no {c} are {a}, is this consistent?", "no"),
        ],
    }
    prompt, expected = rng.choice(patterns[difficulty])
    return {
        "prompt": prompt + " Reason step by step.",
        "expected": expected,
        "check_type": "contains",
        "subdomain": "syllogism",
        "difficulty": difficulty,
    }


def _gen_truth_table(rng: random.Random, difficulty: str) -> dict:
    formulas = {
        "easy": [
            ("(P and Q) or (not P and not Q)", "P iff Q", "equivalent"),
            ("not (P and Q)", "not P or not Q", "equivalent"),
        ],
        "medium": [
            ("P implies Q", "not P or Q", "equivalent"),
            ("P implies Q", "Q implies P", "not equivalent"),
        ],
        "hard": [
            ("(P implies Q) and (Q implies R)", "P implies R", "equivalent"),
            ("not (P or Q)", "not P and not Q", "equivalent"),
        ],
        "expert": [
            ("(P iff Q) iff R", "P iff (Q iff R)", "equivalent"),
            ("(P implies (Q implies R))", "((P and Q) implies R)", "equivalent"),
        ],
    }
    f1, f2, label = rng.choice(formulas[difficulty])
    expected = "yes" if label == "equivalent" else "no"
    return {
        "prompt": f"Are '{f1}' and '{f2}' logically equivalent? Prove using a truth table.",
        "expected": expected,
        "check_type": "contains",
        "subdomain": "propositional",
        "difficulty": difficulty,
    }


# ---------- Reasoning / common sense ----------

def _gen_counting(rng: random.Random, difficulty: str) -> dict:
    if difficulty == "easy":
        n = rng.randint(3, 8)
        return {
            "prompt": f"How many handshakes occur if {n} people each shake hands with every other person exactly once?",
            "expected": str(n * (n - 1) // 2),
            "check_type": "numeric",
            "subdomain": "counting",
            "difficulty": difficulty,
        }
    if difficulty == "medium":
        n = rng.randint(4, 9)
        return {
            "prompt": f"How many distinct arrangements of the letters in a {n}-letter word with all distinct letters?",
            "expected": str(__import__("math").factorial(n)),
            "check_type": "numeric",
            "subdomain": "counting",
            "difficulty": difficulty,
        }
    if difficulty == "hard":
        n = rng.randint(6, 10)
        k = rng.randint(2, n - 2)
        return {
            "prompt": f"How many ways to choose {k} items from {n} distinct items (unordered)?",
            "expected": str(comb(n, k)),
            "check_type": "numeric",
            "subdomain": "counting",
            "difficulty": difficulty,
        }
    # expert
    n = rng.randint(5, 8)
    return {
        "prompt": f"How many ways to arrange {n} people around a round table (rotations considered equal)?",
        "expected": str(__import__("math").factorial(n - 1)),
        "check_type": "numeric",
        "subdomain": "counting",
        "difficulty": "expert",
    }


def _gen_date_reasoning(rng: random.Random, difficulty: str) -> dict:
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    start = rng.choice(days)
    if difficulty == "easy":
        offset = rng.randint(1, 6)
    elif difficulty == "medium":
        offset = rng.randint(7, 30)
    elif difficulty == "hard":
        offset = rng.randint(31, 365)
    else:
        offset = rng.randint(366, 3650)
    direction = rng.choice(["after", "before"])
    start_idx = days.index(start)
    if direction == "after":
        end = days[(start_idx + offset) % 7]
    else:
        end = days[(start_idx - offset) % 7]
    return {
        "prompt": f"If today is {start.capitalize()}, what day of the week will it be {offset} days {direction}?",
        "expected": end,
        "check_type": "contains",
        "subdomain": "temporal",
        "difficulty": difficulty,
    }


def _gen_spatial(rng: random.Random, difficulty: str) -> dict:
    directions = ["north", "east", "south", "west"]
    start = rng.choice(directions)
    turns = 1 if difficulty == "easy" else (2 if difficulty == "medium" else (3 if difficulty == "hard" else rng.randint(4, 7)))
    idx = directions.index(start)
    rotation = 0
    desc = []
    for _ in range(turns):
        deg = rng.choice([90, 180, 270])
        way = rng.choice(["clockwise", "counter-clockwise"])
        steps = deg // 90
        if way == "counter-clockwise":
            steps = -steps
        rotation += steps
        desc.append(f"turn {deg} degrees {way}")
    final = directions[(idx + rotation) % 4]
    turn_text = ", then ".join(desc)
    return {
        "prompt": f"You face {start}. You {turn_text}. What direction are you facing?",
        "expected": final,
        "check_type": "contains",
        "subdomain": "spatial",
        "difficulty": difficulty,
    }


# ---------- Abstraction ----------

def _gen_sequence(rng: random.Random, difficulty: str) -> dict:
    if difficulty == "easy":
        # arithmetic
        start = rng.randint(1, 10)
        step = rng.randint(2, 7)
        seq = [start + i * step for i in range(5)]
        nxt = start + 5 * step
    elif difficulty == "medium":
        # geometric
        start = rng.randint(1, 5)
        ratio = rng.randint(2, 4)
        seq = [start * (ratio ** i) for i in range(5)]
        nxt = start * (ratio ** 5)
    elif difficulty == "hard":
        # squares + constant
        c = rng.randint(0, 5)
        seq = [i * i + c for i in range(1, 6)]
        nxt = 36 + c
    else:  # expert — fibonacci-like with seed values
        a, b = rng.randint(1, 4), rng.randint(1, 5)
        seq = [a, b]
        for _ in range(4):
            seq.append(seq[-1] + seq[-2])
        nxt = seq[-1] + seq[-2]
        seq = seq[:6]
    return {
        "prompt": f"What comes next in the sequence: {', '.join(str(x) for x in seq)}, ...? Explain the pattern.",
        "expected": str(nxt),
        "check_type": "numeric",
        "subdomain": "pattern",
        "difficulty": difficulty,
    }


def _gen_analogy(rng: random.Random, difficulty: str) -> dict:
    easy = [("hot", "cold", "light", "dark"),
            ("day", "night", "sun", "moon"),
            ("big", "small", "tall", "short"),
            ("wet", "dry", "rain", "desert")]
    medium = [("bird", "nest", "human", "house"),
              ("fish", "water", "bird", "air"),
              ("painter", "canvas", "writer", "paper"),
              ("doctor", "hospital", "teacher", "school")]
    hard = [("novel", "chapter", "symphony", "movement"),
            ("atom", "molecule", "word", "sentence"),
            ("leaf", "tree", "feather", "bird")]
    expert = [("entropy", "disorder", "enthalpy", "energy"),
              ("syntax", "grammar", "morphology", "word"),
              ("genotype", "phenotype", "blueprint", "building")]
    pool = {"easy": easy, "medium": medium, "hard": hard, "expert": expert}[difficulty]
    a, b, c, d = rng.choice(pool)
    return {
        "prompt": f"{a.capitalize()} is to {b} as {c} is to ___. Explain the relationship.",
        "expected": d,
        "check_type": "contains",
        "subdomain": "analogy",
        "difficulty": difficulty,
    }


# ---------- Code ----------

def _gen_code_complexity(rng: random.Random, difficulty: str) -> dict:
    snippets = {
        "easy": [
            ("for i in range(n): print(i)", "n"),
            ("x = n + 1", "1"),
        ],
        "medium": [
            ("for i in range(n):\n  for j in range(n): pass", "n^2"),
            ("while n > 1: n //= 2", "log"),
        ],
        "hard": [
            ("for i in range(n):\n  for j in range(i, n):\n    for k in range(j, n): pass", "n^3"),
            ("def f(n):\n  if n<=1: return 1\n  return f(n-1)+f(n-1)", "2^n"),
        ],
        "expert": [
            ("def f(n):\n  if n<=1: return 1\n  return f(n//2)+f(n//2)+n", "n log"),
            ("for i in range(n):\n  j=1\n  while j<n: j*=2", "n log"),
        ],
    }
    code, expected = rng.choice(snippets[difficulty])
    return {
        "prompt": f"What is the time complexity of:\n{code}\nProve it.",
        "expected": expected,
        "check_type": "contains",
        "subdomain": "complexity",
        "difficulty": difficulty,
    }


def _gen_code_implementation(rng: random.Random, difficulty: str) -> dict:
    easy = [
        "Write a Python function `double(x)` that returns x doubled.",
        "Write a Python function `is_even(n)` that returns True iff n is even.",
        "Write a Python function `sum_list(lst)` that returns the sum of a list.",
    ]
    medium = [
        "Write a Python function `reverse_string(s)` that reverses s without using [::-1].",
        "Write a Python function `count_vowels(s)` that counts vowels.",
        "Write a Python function `fizzbuzz(n)` that returns the FizzBuzz sequence up to n as a list.",
    ]
    hard = [
        "Write a Python function `is_palindrome(s)` handling spaces, case, and punctuation.",
        "Write a Python function `sieve(n)` using the Sieve of Eratosthenes for primes up to n.",
        "Write a Python function `merge_sorted(a, b)` that merges two sorted lists without sort().",
        "Write a Python function `binary_search(lst, x)` returning the index or -1.",
    ]
    expert = [
        "Write a Python function `kth_smallest(lst, k)` using quickselect (expected O(n)).",
        "Write a Python function `longest_common_subseq(a, b)` returning LCS length via DP.",
        "Write a Python function `dijkstra(graph, start)` returning shortest distances (graph=dict of dicts).",
    ]
    prompt = rng.choice({"easy": easy, "medium": medium, "hard": hard, "expert": expert}[difficulty])
    return {
        "prompt": prompt,
        "expected": "def",
        "check_type": "code_executes",
        "subdomain": "implementation",
        "difficulty": difficulty,
    }


def _gen_code_debug(rng: random.Random, difficulty: str) -> dict:
    bugs = {
        "easy": [
            ("def add(a,b): return a-b", "subtraction"),
            ("def div(a,b): return a/b  # b could be 0", "zero"),
        ],
        "medium": [
            ("def avg(lst): return sum(lst)/len(lst)", "empty"),
            ("def fib(n): return fib(n-1)+fib(n-2)", "base case"),
        ],
        "hard": [
            ("def find(lst,x):\n  for i,v in enumerate(lst):\n    if v=x: return i", "syntax"),
            ("def rm_dup(lst):\n  for x in lst:\n    if lst.count(x)>1: lst.remove(x)", "mutat"),
        ],
        "expert": [
            ("def bsearch(a,x):\n  lo,hi=0,len(a)\n  while lo<=hi:\n    m=(lo+hi)//2\n    if a[m]==x: return m\n    elif a[m]<x: lo=m+1\n    else: hi=m-1\n  return -1", "off-by-one"),
        ],
    }
    code, kw = rng.choice(bugs[difficulty])
    return {
        "prompt": f"This Python code has a bug:\n{code}\nWhat is wrong and how do you fix it?",
        "expected": kw,
        "check_type": "semantic",
        "subdomain": "debugging",
        "difficulty": difficulty,
    }


# ---------- Science ----------

def _gen_physics_kinematics(rng: random.Random, difficulty: str) -> dict:
    v = rng.choice([10, 15, 20, 25, 30, 40, 50])
    height = v * v / 20.0  # g=10
    # Adjust precision expectation by difficulty
    if difficulty in ("easy", "medium"):
        expected = f"{height:.1f}"
    else:
        expected = f"{height:.3f}"
    return {
        "prompt": f"A ball is thrown straight up at {v} m/s (g=10 m/s^2, ignoring air resistance). What is the maximum height in meters? Show steps.",
        "expected": expected,
        "check_type": "numeric",
        "subdomain": "physics",
        "difficulty": difficulty,
        "_numeric_value": height,
    }


def _gen_ohms_law(rng: random.Random, difficulty: str) -> dict:
    v = rng.choice([5, 9, 12, 24, 120, 240])
    r = rng.choice([2, 10, 100, 470, 1000, 4700])
    i = v / r
    return {
        "prompt": f"A circuit has {v} V across a {r} ohm resistor. What is the current in amps?",
        "expected": f"{i:.6g}",
        "check_type": "numeric",
        "subdomain": "physics",
        "difficulty": difficulty,
        "_numeric_value": i,
    }


def _gen_chemistry_ph(rng: random.Random, difficulty: str) -> dict:
    import math
    conc = rng.choice([0.001, 0.01, 0.1, 1.0, 0.0001])
    ph = -math.log10(conc)
    return {
        "prompt": f"What is the pH of a {conc} M HCl solution? Show the calculation.",
        "expected": f"{ph:.2f}",
        "check_type": "numeric",
        "subdomain": "chemistry",
        "difficulty": difficulty,
        "_numeric_value": ph,
    }


# ---------- Language understanding (generated, not programmatic numeric) ----------

_AMBIGUOUS_SENTENCES = [
    ("I saw her duck.", "duck"),
    ("Visiting relatives can be boring.", "relatives"),
    ("The chicken is ready to eat.", "chicken"),
    ("Flying planes can be dangerous.", "flying"),
    ("They are cooking apples.", "apples"),
    ("The old man the boat.", "man"),
    ("The horse raced past the barn fell.", "raced"),
]

_REFERENCE_SENTENCES = [
    ("The trophy would not fit in the suitcase because it was too big.", "trophy"),
    ("The trophy would not fit in the suitcase because it was too small.", "suitcase"),
    ("The city council refused the protesters a permit because they feared violence.", "council"),
    ("The city council refused the protesters a permit because they advocated violence.", "protesters"),
    ("The sculptor hated the critic because he was arrogant.", "critic"),
]


def _gen_ambiguity(rng: random.Random, difficulty: str) -> dict:
    sentence, kw = rng.choice(_AMBIGUOUS_SENTENCES)
    return {
        "prompt": f"What does this sentence mean? List all interpretations: '{sentence}'",
        "expected": kw,
        "check_type": "contains",
        "subdomain": "ambiguity",
        "difficulty": difficulty,
    }


def _gen_coreference(rng: random.Random, difficulty: str) -> dict:
    sentence, referent = rng.choice(_REFERENCE_SENTENCES)
    return {
        "prompt": f"In the sentence: '{sentence}' — what does the pronoun refer to? Explain.",
        "expected": referent,
        "check_type": "contains",
        "subdomain": "reference",
        "difficulty": difficulty,
    }


# Registry: domain -> list of (generator, default_difficulty_set)
_PROGRAMMATIC_GENERATORS = {
    "math": [_gen_arithmetic, _gen_linear_equation, _gen_quadratic,
             _gen_derivative, _gen_gcd, _gen_modular, _gen_probability],
    "logic": [_gen_syllogism, _gen_truth_table],
    "reasoning": [_gen_syllogism, _gen_counting, _gen_date_reasoning],
    "common_sense": [_gen_date_reasoning, _gen_spatial],
    "abstraction": [_gen_sequence, _gen_analogy],
    "code": [_gen_code_complexity, _gen_code_implementation, _gen_code_debug],
    "science": [_gen_physics_kinematics, _gen_ohms_law, _gen_chemistry_ph],
    "language_understanding": [_gen_ambiguity, _gen_coreference],
}


# ---------- Semantic grading helpers ----------

_SYNONYMS = {
    "yes": {"yes", "correct", "true", "affirmative", "valid", "does follow", "it follows"},
    "no": {"no", "incorrect", "false", "not necessarily", "does not follow", "doesn't follow",
           "cannot be concluded", "not valid", "invalid"},
    "base case": {"base case", "base-case", "missing base", "no termination", "terminating condition",
                  "stopping condition", "recursion never terminates"},
    "empty": {"empty list", "empty input", "zero division", "zerodivisionerror", "division by zero",
              "len(lst) == 0", "len is 0"},
    "syntax": {"syntax", "syntaxerror", "typo", "= vs ==", "assignment instead of comparison"},
    "mutat": {"mutating", "mutation", "modifying while iterating", "skip elements", "in-place"},
    "off-by-one": {"off-by-one", "off by one", "boundary", "hi = len", "should be len-1", "inclusive"},
    "correlation": {"correlation", "not causation", "confounding", "confound", "lurking variable",
                    "spurious"},
    "affirming the consequent": {"affirming the consequent", "converse error"},
    "bandwagon": {"bandwagon", "appeal to popularity", "ad populum"},
    "burden of proof": {"burden of proof", "appeal to ignorance", "argumentum ad ignorantiam"},
    "no true scotsman": {"no true scotsman", "ad hoc rescue"},
}


def _semantic_match(response: str, expected: str) -> bool:
    r = response.lower()
    e = expected.lower().strip()
    syn_set = _SYNONYMS.get(e)
    candidates = syn_set if syn_set else {e}
    for cand in candidates:
        if cand in r:
            return True
    return False


def _numeric_match(response: str, expected: str, tol: float = 1e-3) -> bool:
    """Tolerant numeric match. Accepts answers in plausible forms.

    Supports: integers, decimals, fractions (a/b), scientific notation,
    negative numbers, and comma thousands separators.
    """
    try:
        exp_val = _parse_number(expected)
    except Exception:
        return False
    if exp_val is None:
        return False
    for m in re.finditer(r'-?\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?(?:\s*/\s*\d+)?', response):
        token = m.group(0).replace(",", "")
        try:
            val = _parse_number(token)
        except Exception:
            continue
        if val is None:
            continue
        # Use relative tolerance for large values
        denom = max(abs(exp_val), 1.0)
        if abs(val - exp_val) / denom <= tol:
            return True
    return False


def _parse_number(text: str) -> Optional[float]:
    text = text.strip().replace(",", "")
    if "/" in text:
        parts = text.split("/")
        if len(parts) == 2:
            try:
                return float(parts[0]) / float(parts[1])
            except (ValueError, ZeroDivisionError):
                return None
    try:
        return float(text)
    except ValueError:
        return None


def _numeric_set_match(response: str, expected: str, tol: float = 1e-3) -> bool:
    """Match each expected number appears somewhere in response."""
    targets = []
    for tok in expected.replace(",", " ").split():
        val = _parse_number(tok)
        if val is not None:
            targets.append(val)
    if not targets:
        return False
    seen = []
    for m in re.finditer(r'-?\d+(?:\.\d+)?', response):
        val = _parse_number(m.group(0))
        if val is not None:
            seen.append(val)
    for t in targets:
        denom = max(abs(t), 1.0)
        if not any(abs(s - t) / denom <= tol for s in seen):
            return False
    return True


def _generate_params(param_type: str, seed: int) -> dict:
    """Generate randomized parameters for question templates."""
    rng = random.Random(seed)

    if param_type == "bat_ball":
        ball = rng.choice([5, 10, 15, 20])  # cents
        diff = rng.choice([100, 200, 50])  # cents
        total = (ball * 2 + diff) / 100
        return {"total": f"{total:.2f}", "diff": f"{diff/100:.2f}", "ball_answer": f"{ball/100:.2f}"}

    if param_type == "machines":
        n = rng.choice([3, 5, 7, 10])
        m = rng.choice([50, 100, 200])
        return {"n": str(n), "m": str(m)}

    if param_type == "lilypad":
        days = rng.choice([30, 48, 60])
        return {"days": str(days), "half_answer": str(days - 1)}

    if param_type == "derivative":
        n = rng.randint(3, 5)  # min 3 so leading term is distinct from ax^2
        a = rng.randint(1, 6)
        b = rng.randint(1, 10)
        c = rng.randint(1, 10)
        # Full derivative: nx^(n-1) + 2ax - b
        # Check for the leading term in the model's response
        return {"n": str(n), "a": str(a), "b": str(b), "c": str(c),
                "deriv_answer": f"{n}x^{n-1}"}

    if param_type == "exponential":
        base = rng.choice([2, 3, 5])
        exp = rng.randint(2, 6)
        return {"base": str(base), "result": str(base**exp), "exp_answer": str(exp)}

    if param_type == "factor_cube":
        root = rng.choice([2, 3, 4, 5])
        return {"cube": str(root**3), "root": str(root)}

    if param_type == "system":
        x, y = rng.randint(1, 5), rng.randint(1, 5)
        # Ensure non-singular system (det != 0) so a unique solution exists.
        # Without this, some coefficient combos produce det=0 (e.g., a1=2,b1=1,a2=2,b2=-1),
        # and the "expected" answer is wrong — the system has no unique solution.
        for _ in range(20):
            a1, b1 = rng.randint(1, 4), rng.randint(1, 4)
            a2, b2 = rng.randint(1, 4), rng.randint(-3, -1)
            if a1 * b2 - a2 * b1 != 0:
                break
        else:
            # Fallback: guaranteed non-singular
            a1, b1, a2, b2 = 1, 1, 1, -1
        return {"a1": str(a1), "b1": str(b1), "c1": str(a1*x + b1*y),
                "a2": str(a2), "b2": str(b2), "c2": str(a2*x + b2*y),
                "x_answer": str(x), "y_answer": str(y)}

    if param_type == "primality":
        # Mix of primes and composites
        composites = [91, 57, 87, 51, 119, 133, 143, 161]
        primes = [97, 89, 83, 79, 71, 67, 61, 59]
        is_prime = rng.choice([True, False])
        num = rng.choice(primes if is_prime else composites)
        return {"num": str(num), "prime_answer": "yes" if is_prime else "no"}

    if param_type == "gcd":
        a = rng.randint(20, 200)
        b = rng.randint(20, 200)
        return {"a": str(a), "b": str(b), "gcd_answer": str(gcd(a, b))}

    if param_type == "coin_flip":
        n = rng.choice([3, 4, 5])
        k = rng.randint(1, n - 1)
        frac = Fraction(comb(n, k), 2 ** n)
        return {"n": str(n), "k": str(k), "prob_answer": f"{frac.numerator}/{frac.denominator}"}

    if param_type == "projectile":
        v = rng.choice([10, 20, 30, 40])
        height = v**2 // 20  # v^2/(2g)
        return {"v": str(v), "height_answer": str(height)}

    if param_type == "gravity":
        m1 = rng.choice([500, 1000, 2000, 5000])
        m2 = rng.choice([500, 1000, 2000, 5000])
        d = rng.choice([5, 10, 20, 50])
        force = 6.67e-11 * m1 * m2 / (d ** 2)
        # Use math_equiv check type so different notations all match.
        # Store the numeric value as a string for sympy comparison.
        return {"m1": str(m1), "m2": str(m2), "d": str(d),
                "grav_answer": f"{force:.4g}"}

    return {}


class DiagnosticsEngine:
    """Rapidly probes the model to find its weakest areas."""

    def __init__(self, config: DiagnosticsConfig, model_loader: ModelLoader):
        self.config = config
        self.model = model_loader
        self._seen_hashes: set[str] = set()
        self._model_generated_questions: dict[str, list[dict]] = {}  # from escalation
        self._model_score: float = 0.0  # current model performance for adaptive curriculum

    def set_model_score(self, score: float):
        """Set current model performance score for adaptive curriculum difficulty.

        Higher-performing models get harder questions to keep the diagnostic
        challenging and informative. This complements the cycle-based ramp
        in _curriculum_mix.
        """
        self._model_score = score

    def _generate_batch_with_oom_retry(
        self, prompts: list[str], max_new_tokens: int = 512,
        temperature: float = 0.0, max_retries: int = 3,
    ) -> list[str]:
        """Call generate_batch with OOM retry — halves batch on each failure.

        On OOM, splits prompts into smaller chunks and concatenates results.
        Max 3 retries before skipping remaining prompts (returns empty strings).
        """
        import gc
        chunk_size = len(prompts)
        for attempt in range(max_retries + 1):
            try:
                # Process in chunks of chunk_size
                all_responses: list[str] = []
                for start in range(0, len(prompts), chunk_size):
                    batch = prompts[start:start + chunk_size]
                    all_responses.extend(
                        self.model.generate_batch(batch, max_new_tokens=max_new_tokens, temperature=temperature)
                    )
                return all_responses
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                chunk_size = max(1, chunk_size // 2)
                if attempt < max_retries:
                    import logging
                    logger.warning(
                        f"OOM in diagnostics generate_batch (attempt {attempt+1}/{max_retries}), "
                        f"retrying with chunk_size={chunk_size}"
                    )
                else:
                    logger.error(
                        f"OOM in diagnostics after {max_retries} retries — skipping remaining prompts"
                    )
                    return [""] * len(prompts)
        return [""] * len(prompts)

    def run(self, cycle: int) -> DiagnosticResult:
        """Run full diagnostics."""
        # Reset seen hashes each cycle — prevents hash set from growing unboundedly
        # across 100+ cycles (which would block most question variants from generating).
        # Anti-memorization comes from randomized params per cycle, not cross-cycle dedup.
        self._seen_hashes.clear()
        result = DiagnosticResult(cycle=cycle, timestamp=time.time())

        for domain in self.config.domains:
            score, evidence = self._probe_domain(domain, cycle)
            result.domain_scores[domain] = score
            result.domain_question_counts[domain] = len(evidence)
            result.total_questions += len(evidence)
            result.total_correct += sum(1 for e in evidence if e["correct"])

            if score < self.config.confidence_threshold:
                subweaknesses = self._drill_down(domain, evidence)
                result.weaknesses.extend(subweaknesses)

        if self.config.activation_analysis:
            layer_health = self._analyze_layers_with_activations()
            result.layer_health = layer_health
            self._correlate_weak_layers(result)

        result.weaknesses.sort(key=lambda w: w.severity, reverse=True)
        # Free VRAM from diagnostic inferences (200+ generate calls across all
        # domains + activation probes). Without this, the CUDA allocator is
        # heavily fragmented when generation/training phases need large contiguous
        # blocks for KV cache and gradient checkpointing.
        torch.cuda.empty_cache()
        return result

    def generate_adaptive_questions(self, weak_domains: list[str], cycle: int = 0):
        """Escalation: ask the model to generate diagnostic questions for its weak areas.

        Uses cycle number in the prompt to get different questions each time.
        """
        if not weak_domains:
            return
        # Vary the difficulty focus each cycle to get diverse questions
        focus_areas = ["conceptual understanding", "edge cases and exceptions",
                       "multi-step reasoning", "common misconceptions", "applied problems"]
        # Batch all domain prompts for efficiency — sequential generate() calls
        # waste GPU time on kernel launch overhead and KV cache recomputation.
        prompts = []
        for domain in weak_domains:
            focus = focus_areas[cycle % len(focus_areas)]
            prompts.append(
                f"<|im_start|>system\n"
                f"You are an expert question writer for {domain}. "
                f"You produce self-contained questions with unambiguous correct answers.<|im_end|>\n"
                f"<|im_start|>user\n"
                f"Generate exactly 5 {domain} questions focused on {focus}.\n"
                f"Cycle: {cycle} (make them DIFFERENT from any previous round).\n\n"
                f"Requirements:\n"
                f"- Each question must have a single, definite correct answer.\n"
                f"- Make them progressively harder (question 1 = easiest, 5 = hardest).\n"
                f"- Do NOT include explanations or solutions in the question.\n\n"
                f"Output format — use EXACTLY this structure for each question:\n"
                f"Q: <question text>\n"
                f"A: <short answer — a single number, word, or phrase, nothing else>\n\n"
                f"Example:\n"
                f"Q: What is the derivative of x^3 + 2x?\n"
                f"A: 3x^2 + 2\n\n"
                f"Begin:<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        responses = self.model.generate_batch(prompts, max_new_tokens=1024, temperature=0.7)
        # Guard against length mismatch (same pattern as verifier)
        if len(responses) < len(weak_domains):
            responses.extend([""] * (len(weak_domains) - len(responses)))
        for domain, response in zip(weak_domains, responses):
            questions = self._parse_generated_questions(response, domain)
            # Only replace if we got questions — if parsing failed (model returned
            # unparseable output), keep the previous cycle's valid questions rather
            # than wiping them out with an empty list.
            if questions:
                self._model_generated_questions[domain] = questions

    def _parse_generated_questions(self, response: str, domain: str) -> list[dict]:
        """Parse model-generated questions into question format."""
        questions = []
        lines = response.split("\n")
        current_q = None
        for line in lines:
            stripped = line.strip()
            # Handle "Q:", "1. Q:", "1) Q:" — models often number their questions
            q_match = re.match(r'^(?:\d+[.)]\s*)?Q:\s*(.*)', stripped, re.IGNORECASE)
            a_match = re.match(r'^(?:\d+[.)]\s*)?A:\s*(.*)', stripped, re.IGNORECASE)
            if q_match:
                current_q = q_match.group(1).strip()
            elif a_match and current_q:
                answer = a_match.group(1).strip()
                # Sanitize: model answers are often verbose ("Yes, because X").
                # Extract only the core answer — take text before first comma,
                # period, or "because"/"since" clause. Long answers as expected
                # values almost never match with "contains" check.
                # Lower first since the expected value is lowered anyway — avoids
                # index mismatch from Unicode case folding changing string length.
                answer = answer.lower()
                # Strip common preamble phrases ("the answer is", "a:", "equals", etc.)
                # so the extracted keyword is the substantive answer, not filler.
                answer = _ANSWER_PREFIX_RE.sub("", answer).strip()
                for sep in [" because ", " since ", ". ", ", "]:
                    idx = answer.find(sep)
                    if idx != -1:
                        answer = answer[:idx]
                        break
                # Strip trailing punctuation/quotes/backticks left after truncation.
                answer = answer.strip(" .`\"'")
                # Skip answers that are still too long to be useful keywords
                if len(answer) > 80:
                    current_q = None
                    continue
                # Empty answers or pure-punctuation residue make "contains" match
                # everything — drop them.
                if not answer or not re.search(r"[A-Za-z0-9]", answer):
                    current_q = None
                    continue
                questions.append({
                    "prompt": current_q + " Show your reasoning step by step.",
                    "expected": answer.lower().strip(),
                    "check_type": "contains",
                    "subdomain": "model_generated",
                })
                current_q = None
        return questions

    def _probe_domain(self, domain: str, cycle: int) -> tuple[float, list[dict]]:
        """Probe a domain with randomized questions."""
        questions = self._generate_questions(domain, cycle)

        # Add model-generated questions if available (from escalation)
        if domain in self._model_generated_questions:
            questions.extend(self._model_generated_questions[domain])

        if not questions:
            return 1.0, []

        evidence = []
        batch_prompts = [q["prompt"] for q in questions]

        # OOM-resilient batched inference: if a single large batch OOMs,
        # halve the batch size and retry (up to 3 times). This lets the
        # default batch stay large for throughput while gracefully handling
        # memory pressure from long prompts or fragmentation.
        responses = self._generate_batch_with_oom_retry(batch_prompts, max_new_tokens=512, temperature=0.0)

        if len(responses) < len(batch_prompts):
            responses = list(responses) + [""] * (len(batch_prompts) - len(responses))

        for q, response in zip(questions, responses):
            correct = self._check_answer(response, q["expected"], q["check_type"])
            confidence = self._estimate_confidence(response)
            evidence.append({
                "question": q["prompt"],
                "expected": q["expected"],
                "response": response[:500],
                "correct": correct,
                "domain": domain,
                "subdomain": q.get("subdomain", "general"),
                "difficulty": q.get("difficulty", "medium"),
                "confidence": confidence,
                "check_type": q.get("check_type", "contains"),
            })

        correct_count = sum(1 for e in evidence if e["correct"])
        score = correct_count / len(evidence) if evidence else 0.0
        return score, evidence

    def _generate_questions(self, domain: str, cycle: int) -> list[dict]:
        """Generate questions for a domain with curriculum-mixed difficulties.

        Combines three sources:
        1. Existing parameterized QUESTION_TEMPLATES (marked `difficulty=medium`
           if not otherwise tagged — they are template-driven).
        2. Programmatic generators (volume + controllable difficulty).
        3. Topping-off from whichever source still has capacity.
        """
        templates = QUESTION_TEMPLATES.get(domain, {})
        seed_bytes = hashlib.md5(f"{domain}:{cycle}".encode()).digest()
        rng = random.Random(int.from_bytes(seed_bytes[:4], "big"))

        target = int(self.config.questions_per_domain)
        clamped = max(self.config.min_questions_per_domain,
                      min(target, self.config.max_questions_per_domain))
        if clamped != target and cycle == 1:
            logger.warning(
                f"diagnostics.questions_per_domain={target} clamped to {clamped} "
                f"(min={self.config.min_questions_per_domain}, "
                f"max={self.config.max_questions_per_domain})"
            )
        target = clamped

        questions: list[dict] = []

        # --- Template-based questions first (keeps behavior for tagged domains) ---
        for subdomain, template_list in templates.items():
            for template in template_list:
                variants = self._create_variants(template, subdomain, cycle)
                # Legacy templates are medium-difficulty by default
                for v in variants:
                    v.setdefault("difficulty", "medium")
                questions.extend(variants)

        # --- Programmatic generators ---
        if self.config.use_programmatic_generators:
            gens = _PROGRAMMATIC_GENERATORS.get(domain, [])
            if gens:
                # Difficulty curriculum: ramp harder as cycle grows.
                diff_mix = self._curriculum_mix(cycle)
                # Budget ~2/3 of target from programmatic generators
                prog_budget = max(0, target - len(questions))
                if prog_budget < target // 2:
                    prog_budget = target  # we'll cap later
                prog_count = 0
                attempts = 0
                max_attempts = prog_budget * 4 + 64
                while prog_count < prog_budget and attempts < max_attempts:
                    attempts += 1
                    difficulty = self._sample_difficulty(rng, diff_mix)
                    gen = rng.choice(gens)
                    try:
                        q = gen(rng, difficulty)
                    except Exception:
                        continue
                    if not q:
                        continue
                    q["_prog"] = True
                    q.setdefault("difficulty", difficulty)
                    h = hashlib.md5((q["prompt"] + "|" + str(q["expected"])).encode()).hexdigest()
                    if h in self._seen_hashes:
                        continue
                    self._seen_hashes.add(h)
                    questions.append(q)
                    prog_count += 1
                    if len(questions) >= target:
                        break

        # --- Top-off from templates if still short ---
        if len(questions) < target and templates:
            attempts = 0
            prev = -1
            while len(questions) < target and attempts < target * 2:
                if len(questions) == prev:
                    break
                prev = len(questions)
                attempts += 1
                for subdomain, template_list in templates.items():
                    if len(questions) >= target:
                        break
                    if not template_list:
                        continue
                    template = rng.choice(template_list)
                    extra = self._create_variants(template, subdomain, cycle * 1000 + attempts)
                    for v in extra:
                        v.setdefault("difficulty", "medium")
                    questions.extend(extra)

        # Nothing generated? Fall back to a single general question.
        if not questions:
            questions.append({
                "prompt": f"Explain a fundamental concept in {domain} with step-by-step reasoning.",
                "expected": "",
                "check_type": "nonempty",
                "subdomain": "general",
                "difficulty": "medium",
            })

        rng.shuffle(questions)
        return questions[:target]

    def _curriculum_mix(self, cycle: int) -> dict:
        """Return a difficulty weight dict for this cycle.

        Early cycles bias toward easy/medium; late cycles bias toward hard/expert.
        Additionally adapts to model performance: higher-scoring models get harder
        questions so the diagnostic remains informative as the model improves.
        """
        if not self.config.difficulty_curriculum:
            return dict(self.config.difficulty_mix)
        # Ramp parameter: 0 early, ~1 by cycle 20.
        t = min(1.0, cycle / 20.0)
        # Performance ramp: model score in [0, 1] shifts difficulty further.
        # A model scoring 0.8+ should face mostly hard/expert questions.
        p = max(0.0, min(1.0, self._model_score))
        # Combine cycle ramp and performance ramp (performance has stronger effect)
        t_combined = min(1.0, t * 0.4 + p * 0.6)
        base = {"easy": 0.45, "medium": 0.35, "hard": 0.15, "expert": 0.05}
        target = {"easy": 0.10, "medium": 0.25, "hard": 0.40, "expert": 0.25}
        mix = {k: base[k] * (1 - t_combined) + target[k] * t_combined for k in base}
        # Normalize
        s = sum(mix.values()) or 1.0
        return {k: v / s for k, v in mix.items()}

    @staticmethod
    def _sample_difficulty(rng: random.Random, mix: dict) -> str:
        r = rng.random()
        cum = 0.0
        for d in _DIFFICULTIES:
            cum += mix.get(d, 0.0)
            if r <= cum:
                return d
        return "medium"

    def _create_variants(self, template: dict, subdomain: str, cycle: int) -> list[dict]:
        """Create randomized variants of a question template."""
        variants = []
        prompt = template["prompt"]
        param_type = template.get("_params")

        if "{A}" in prompt:
            # Syllogism — random nouns, seeded for reproducibility
            seed_bytes = hashlib.md5(f"{subdomain}:{cycle}:{prompt[:50]}".encode()).digest()
            rng = random.Random(int.from_bytes(seed_bytes[:4], "big"))
            for i in range(3):
                nouns = rng.sample(_NOUNS, 3)
                variant_prompt = prompt.format(A=nouns[0], B=nouns[1], C=nouns[2])
                h = hashlib.md5(variant_prompt.encode()).hexdigest()
                if h not in self._seen_hashes:
                    self._seen_hashes.add(h)
                    variants.append({
                        "prompt": variant_prompt,
                        "expected": template["expected"],
                        "check_type": template["check_type"],
                        "subdomain": subdomain,
                    })
        elif param_type:
            # Parameterized template — generate 3 random variants (match syllogism count)
            for variant_idx in range(3):
                # Use template content for seed, not id() (memory address varies across runs)
                seed_bytes = hashlib.md5(f"{subdomain}:{cycle}:{template['prompt'][:50]}:{variant_idx}".encode()).digest()
                seed = int.from_bytes(seed_bytes[:4], "big")
                params = _generate_params(param_type, seed)
                if params:
                    try:
                        variant_prompt = prompt.format(**params)
                        expected = template["expected"].format(**params)
                        h = hashlib.md5(variant_prompt.encode()).hexdigest()
                        if h not in self._seen_hashes:
                            self._seen_hashes.add(h)
                            variants.append({
                                "prompt": variant_prompt,
                                "expected": expected,
                                "check_type": template["check_type"],
                                "subdomain": subdomain,
                            })
                    except (KeyError, ValueError, IndexError):
                        # Don't fall back to raw template with unsubstituted
                        # placeholders like "{base}^x = {result}" — the model
                        # would get tested on nonsensical literal brace text.
                        # Skip this variant (not the whole template — other seeds
                        # may produce valid params).
                        continue
        else:
            h = hashlib.md5(prompt.encode()).hexdigest()
            if h not in self._seen_hashes:
                self._seen_hashes.add(h)
                variants.append({
                    "prompt": prompt,
                    "expected": template["expected"],
                    "check_type": template["check_type"],
                    "subdomain": subdomain,
                })

        return variants

    def _drill_down(self, domain: str, evidence: list[dict]) -> list[WeaknessReport]:
        """Identify specific subdomains of failure with statistical rigor."""
        subdomain_results: dict[str, dict] = {}
        for e in evidence:
            sd = e["subdomain"]
            if sd not in subdomain_results:
                subdomain_results[sd] = {"total": 0, "failures": [], "all": [],
                                          "by_difficulty": {}}
            subdomain_results[sd]["total"] += 1
            subdomain_results[sd]["all"].append(e)
            diff = e.get("difficulty", "medium")
            subdomain_results[sd]["by_difficulty"].setdefault(diff, [0, 0])
            subdomain_results[sd]["by_difficulty"][diff][0] += 1  # total
            if not e["correct"]:
                subdomain_results[sd]["failures"].append(e)
                subdomain_results[sd]["by_difficulty"][diff][1] += 1  # failures

        baseline = 1.0 - self.config.confidence_threshold
        alpha = self.config.significance_alpha
        min_n = max(1, self.config.min_evidence_for_weakness)

        weaknesses = []
        for subdomain, data in subdomain_results.items():
            n = data["total"]
            f = len(data["failures"])
            if f == 0:
                continue
            raw_severity = f / max(n, 1)

            # Wilson 95% bounds on failure rate
            lo, hi = _wilson_bounds(f, n)
            # One-sided significance that failure rate exceeds baseline
            pval = _binomial_significance(f, n, baseline=baseline)

            # Discount severity for small n (shrink toward prior of baseline)
            if n < min_n:
                shrink = (n / min_n) ** 0.5
                severity = raw_severity * shrink + baseline * (1 - shrink)
            else:
                severity = raw_severity

            # Dominant difficulty bucket (the one with worst failure rate among
            # difficulties that have at least 2 samples)
            dominant = "mixed"
            worst_rate = -1.0
            for d, (tot, fail) in data["by_difficulty"].items():
                if tot >= 2:
                    rate = fail / tot
                    if rate > worst_rate:
                        worst_rate = rate
                        dominant = d

            # Calibrated confidence: model's mean confidence on WRONG answers.
            # High value = overconfident (worse calibration).
            wrong_confidences = [e.get("confidence", 0.0) for e in data["failures"]]
            calibrated = sum(wrong_confidences) / len(wrong_confidences) if wrong_confidences else 0.0

            # Only flag as weakness if significant OR enough evidence
            is_flagged = (n >= min_n and pval <= alpha) or (n < min_n and raw_severity >= 0.5)
            if not is_flagged and n >= min_n:
                # Not statistically significant — skip
                continue

            capped_evidence = data["failures"][:20]

            # Check-type distribution across ALL evidence (not just failures),
            # so verifier sees the true question-shape mix.
            ct_dist: dict[str, int] = {}
            for ev in data["all"]:
                ct = ev.get("check_type", "contains")
                ct_dist[ct] = ct_dist.get(ct, 0) + 1
            dominant_ct = max(ct_dist, key=ct_dist.get) if ct_dist else "contains"
            expected_answer_type = self._answer_type_from_check_type(dominant_ct)
            invariants = self._invariants_for(domain, subdomain, dominant_ct)

            weaknesses.append(WeaknessReport(
                domain=domain,
                subdomain=subdomain,
                severity=severity,
                evidence=capped_evidence,
                description=(f"Fails {raw_severity:.0%} of {subdomain} in {domain} "
                             f"({f}/{n}, 95% CI [{lo:.0%}, {hi:.0%}], "
                             f"p={pval:.3g}, dominant={dominant}, "
                             f"overconfidence={calibrated:.2f})"),
                difficulty=dominant,
                n_questions=n,
                n_failures=f,
                confidence_lower=lo,
                confidence_upper=hi,
                significance=pval,
                calibrated_confidence=calibrated,
                expected_answer_type=expected_answer_type,
                check_type_distribution=ct_dist,
                dominant_check_type=dominant_ct,
                invariants=invariants,
            ))
        return weaknesses

    @staticmethod
    def _answer_type_from_check_type(check_type: str) -> str:
        """Map a check_type to the verifier's expected_answer_type taxonomy."""
        return {
            "numeric": "numeric",
            "numeric_set": "numeric",
            "math_equiv": "symbolic",
            "code_executes": "code",
            "contains": "enum",
            "semantic": "enum",
            "regex": "enum",
            "exact": "enum",
            "nonempty": "freeform",
        }.get(check_type, "freeform")

    @staticmethod
    def _invariants_for(domain: str, subdomain: str, dominant_check: str) -> list[str]:
        invs: list[str] = []
        if dominant_check in ("numeric", "numeric_set", "math_equiv", "code_executes", "exact"):
            invs.append("deterministic")
            invs.append("has_unique_answer")
        if dominant_check in ("numeric", "numeric_set", "math_equiv"):
            invs.append("requires_calculation")
        if dominant_check == "code_executes":
            invs.append("requires_execution")
        if domain in ("logic", "reasoning"):
            invs.append("requires_logical_deduction")
        if domain == "math":
            invs.append("requires_calculation")
        return list(dict.fromkeys(invs))  # dedupe, preserve order

    @staticmethod
    def _estimate_confidence(response: str) -> float:
        """Estimate the model's confidence in its own answer via surface cues.

        Without access to logprobs, use hedge-word density as a proxy. Higher
        score = more confident. This feeds calibration: high confidence on a
        wrong answer = bad calibration.
        """
        if not response:
            return 0.0
        r = response.lower()
        hedges = ("maybe", "perhaps", "possibly", "might", "could be", "i think",
                  "i believe", "not sure", "unsure", "guess", "probably", "seems",
                  "appears to", "i'm not certain", "approximately", "roughly",
                  "it depends")
        certain = ("definitely", "certainly", "clearly", "obviously", "must be",
                   "therefore", "thus", "proves", "conclusively", "exactly")
        hedge_count = sum(r.count(h) for h in hedges)
        certain_count = sum(r.count(c) for c in certain)
        # Length-normalized
        words = max(len(r.split()), 1)
        hedge_rate = hedge_count / words * 100
        certain_rate = certain_count / words * 100
        score = 0.5 + 0.1 * certain_rate - 0.15 * hedge_rate
        return max(0.0, min(1.0, score))

    def _analyze_layers_with_activations(self) -> dict[str, float]:
        """Analyze layers using weight norms and activation health."""
        layer_info = self.model.get_layer_info()
        health = {}
        norms = {name: info["norm"] for name, info in layer_info.items()}
        if not norms:
            return health

        mean_norm = sum(norms.values()) / len(norms)
        std_norm = (sum((n - mean_norm) ** 2 for n in norms.values()) / len(norms)) ** 0.5

        for name, norm in norms.items():
            if std_norm > 0:
                # Signed z: only LOW norms flag as unhealthy. High norms are
                # typically healthy late layers and shouldn't be given extra
                # LoRA rank as if they were weak.
                z_signed = (norm - mean_norm) / std_norm
                if z_signed >= 0:
                    health[name] = 1.0
                else:
                    health[name] = max(0.0, 1.0 - (abs(z_signed) / 3.0))
            else:
                health[name] = 1.0

        # Multiple diverse probes for activation analysis. Tag probes by domain
        # so we can attribute weak layers to the domain whose probes lit them up.
        domain_probes: dict[str, list[str]] = {
            "math": [
                "Explain step by step why 2 + 2 = 4 from the Peano axioms.",
                "Derive the derivative of x^3 using the limit definition.",
            ],
            "code": [
                "Write a Python function to reverse a linked list. Explain your approach.",
                "Implement binary search in Python and explain each step.",
            ],
            "logic": [
                "Is this valid? All cats are mammals. Some mammals are pets. Therefore some cats are pets.",
                "Prove: if P implies Q and not Q, then not P. Use natural deduction.",
            ],
            "reasoning": [
                "If all roses are flowers and some flowers fade quickly, do all roses fade quickly?",
            ],
            "science": [
                "Explain why ice floats on water at the molecular level.",
            ],
            "language_understanding": [
                "What does 'the trophy doesn't fit in the suitcase because it is too big' mean? Explain 'it'.",
            ],
            "common_sense": [
                "Can you eat soup with a fork efficiently? Explain why or why not.",
            ],
            "abstraction": [
                "What comes next: 2, 6, 12, 20, 30, ...? Explain the pattern.",
            ],
        }
        per_probe = max(1, int(getattr(self.config, "activation_probes_per_domain", 2)))
        probes: list[tuple[str, str]] = []
        for domain in self.config.domains:
            for p in domain_probes.get(domain, [])[:per_probe]:
                probes.append((domain, p))
        if not probes:
            probes = [("general", "Explain a fundamental concept with step-by-step reasoning.")]

        # Accumulate health scores across probes and average.
        # Also track per-domain activation stats to allow correlating weak
        # layers with specific domains.
        probe_healths: dict[str, list[float]] = {}
        self._per_domain_layer_activation: dict[str, dict[str, float]] = {}
        # Batch all probe prompts in one generate_batch call for throughput,
        # but activation capture requires individual forward passes (hooks fire
        # per-call). Use generate_batch for speed when activation_analysis is
        # off; otherwise fall back to sequential for hook capture.
        for probe_domain, probe in probes:
            try:
                with self.model.capture_activations() as capture:
                    self.model.generate(probe, max_new_tokens=128, temperature=0.0)

                for act_name, stats in capture.activations.items():
                    act_health = 1.0
                    act_health -= stats.dead_ratio
                    if stats.std > 100:
                        act_health -= 0.3
                    elif stats.std < 0.01:
                        act_health -= 0.2
                    clamped = max(0.0, act_health)
                    if act_name not in probe_healths:
                        probe_healths[act_name] = []
                    probe_healths[act_name].append(clamped)

                    dom_map = self._per_domain_layer_activation.setdefault(probe_domain, {})
                    prev = dom_map.get(act_name)
                    dom_map[act_name] = clamped if prev is None else min(prev, clamped)
            except Exception:
                continue
        # Single cache clear after all probes instead of per-probe
        torch.cuda.empty_cache()

        # Apply averaged activation health to parameter health.
        # Use min(weight_health, blended) so activation analysis can LOWER health
        # (catching dead/exploding neurons that weight norms miss) but never RAISE
        # it above the weight-norm signal. A layer with abnormal weight norms
        # (health=0.0) and healthy activations (1.0) should stay unhealthy — the
        # weight issue is real even if activations happen to look OK on 3 probes.
        for act_name, scores in probe_healths.items():
            avg_act_health = sum(scores) / len(scores)
            prefix = act_name + "."
            for param_name in health:
                if param_name.startswith(prefix):
                    blended = 0.5 * health[param_name] + 0.5 * avg_act_health
                    health[param_name] = min(health[param_name], blended)

        return health

    def _correlate_weak_layers(self, result: DiagnosticResult):
        """Assign weak layers to weaknesses based on layer type heuristics.

        Instead of running N extra forward passes (one per weakness),
        use structural heuristics: attention layers correlate with reasoning
        weaknesses, MLP layers correlate with knowledge weaknesses.
        """
        if not result.layer_health or not result.weaknesses:
            return

        sorted_health = sorted(result.layer_health.items(), key=lambda x: x[1])
        cutoff_idx = max(1, int(len(sorted_health) * self.config.weak_layer_percentile))
        globally_weak = [name for name, _ in sorted_health[:cutoff_idx]]

        # Categorize weak layers by type
        attention_layers = [n for n in globally_weak if any(k in n for k in ("q_proj", "k_proj", "v_proj", "o_proj", "attn"))]
        mlp_layers = [n for n in globally_weak if any(k in n for k in ("gate_proj", "up_proj", "down_proj", "mlp", "fc"))]

        reasoning_domains = {"reasoning", "math", "logic", "abstraction", "code"}
        knowledge_domains = {"science", "common_sense", "language_understanding"}

        # Distribute layers across weaknesses of the same type so different
        # weaknesses get different (partially overlapping) layer sets.
        # Without this, every reasoning weakness gets identical layers → identical
        # LoRA ranks, defeating the purpose of per-weakness targeting.
        reasoning_weaknesses = [w for w in result.weaknesses if w.domain in reasoning_domains]
        knowledge_weaknesses = [w for w in result.weaknesses if w.domain in knowledge_domains]
        other_weaknesses = [w for w in result.weaknesses if w.domain not in reasoning_domains and w.domain not in knowledge_domains]

        def _distribute(weaknesses: list, layer_pool: list):
            if not weaknesses or not layer_pool:
                for w in weaknesses:
                    w.weak_layers = list(globally_weak)
                return
            # Sort by severity (highest first) so the worst weaknesses get
            # the unhealthiest layers. Each weakness gets a sliding window.
            weaknesses_sorted = sorted(weaknesses, key=lambda w: w.severity, reverse=True)
            # Minimum window size: at least 3 layers or 1/3 of pool, whichever is larger.
            # This ensures meaningful differentiation even with few layers.
            min_window = max(3, len(layer_pool) // 3)
            window = max(min_window, len(layer_pool) // max(len(weaknesses_sorted), 1))
            window = min(window, len(layer_pool))  # can't exceed pool
            # Stride between windows — if more weaknesses than layer pool permits,
            # stride=0 and all get the same layers (unavoidable).
            stride = max(1, (len(layer_pool) - window) // max(len(weaknesses_sorted) - 1, 1)) if len(weaknesses_sorted) > 1 else 0
            for i, w in enumerate(weaknesses_sorted):
                start = min(i * stride, len(layer_pool) - window)
                w.weak_layers = layer_pool[start:start + window]

        _distribute(reasoning_weaknesses, attention_layers)
        _distribute(knowledge_weaknesses, mlp_layers)
        _distribute(other_weaknesses, globally_weak)

        # Refine with real per-domain activation data when available.
        # For each weakness, if we have domain-specific activation stats,
        # augment weak_layers with layers flagged unhealthy DURING that domain's
        # probes specifically (strongest signal we have that that layer is
        # implicated in the domain's failures).
        per_dom = getattr(self, "_per_domain_layer_activation", {}) or {}
        for w in result.weaknesses:
            layer_acts = per_dom.get(w.domain)
            if not layer_acts:
                continue
            # Pick bottom 20% of activation health, intersect with parameter names
            sorted_acts = sorted(layer_acts.items(), key=lambda x: x[1])
            cutoff = max(1, int(len(sorted_acts) * self.config.weak_layer_percentile))
            weak_act_names = {name for name, _ in sorted_acts[:cutoff]}
            # Match parameter names starting with any weak activation layer name
            extra = []
            for param_name in result.layer_health:
                for act_name in weak_act_names:
                    if param_name.startswith(act_name + "."):
                        extra.append(param_name)
                        break
            if extra:
                # Merge — preserve existing layers, append domain-specific ones,
                # dedupe while preserving order.
                merged = list(dict.fromkeys(list(w.weak_layers) + extra))
                w.weak_layers = merged

    def _check_answer(self, response: str, expected: str, check_type: str) -> bool:
        response_lower = response.lower().strip()
        expected_lower = expected.lower().strip()

        if check_type == "nonempty":
            # Reject refusals/non-answers — must be substantive, not just long
            text = response.strip()
            if len(text) <= 20:
                return False
            refusal_markers = ("i don't know", "i'm not sure", "i cannot",
                               "i can't", "unable to", "not enough information")
            return not any(m in response_lower for m in refusal_markers)
        if check_type == "contains":
            # Normalize exponentiation syntax before matching — models write x^2,
            # x**2, x², and the expected answer uses ^. Normalize all to ^ so they match.
            response_lower = response_lower.replace("**", "^")
            # Unicode superscripts → ^ notation (only if any are present)
            if _RE_SUPERSCRIPT.search(response_lower):
                response_lower = _RE_SUPERSCRIPT.sub(
                    lambda m: '^' + m.group(0).translate(_SUPERSCRIPT_TABLE),
                    response_lower)

            # Word-boundary match to prevent "no" matching "know"/"notice",
            # "2" matching "12"/"200", etc.
            # Allow flexible spacing around '=' — models write "y=3", "y = 3",
            # "y =3" interchangeably. After re.escape (which doesn't escape = or space),
            # replace literal " = " with \s*=\s* so all variants match.
            escaped = re.escape(expected_lower)
            if '=' in escaped:
                escaped = re.sub(r'\s*=\s*', r'\\s*=\\s*', escaped)
            pattern = r'(?<!\w)' + escaped + r'(?!\w)'
            # For very short expected answers ("no", "yes"), restrict search to
            # conclusion-like parts of the response to reduce false positives.
            if len(expected_lower) <= 3:
                # Check conclusion markers first
                for marker in ("conclusion:", "answer:", "therefore,", "therefore ", "so,", "so the answer"):
                    if marker in response_lower:
                        conclusion_part = response_lower[response_lower.rindex(marker):]
                        if re.search(pattern, conclusion_part):
                            return True
                # Check first sentence and last few sentences. Checking ALL sentences
                # for "no"/"yes" creates false positives: "No simplification needed"
                # matches \bno\b even when the actual answer is "yes." Restrict to
                # positions where models typically state their final answer.
                sentences = [s.strip() for s in re.split(r'[.!?\n]', response_lower) if s.strip()]
                if sentences:
                    # Use dict.fromkeys to deduplicate while preserving order —
                    # if there's only 1 sentence, [:1] + [-3:] would search it twice.
                    check_parts = list(dict.fromkeys(sentences[:1] + sentences[-3:]))
                    return any(re.search(pattern, part) for part in check_parts)
                return False
            return bool(re.search(pattern, response_lower))
        if check_type == "exact":
            return response_lower == expected_lower
        if check_type == "math_equiv":
            return self._check_math_equivalence(response, expected)
        if check_type == "code_executes":
            return self._check_code_executes(response)
        if check_type == "numeric":
            return _numeric_match(response, expected)
        if check_type == "numeric_set":
            return _numeric_set_match(response, expected)
        if check_type == "semantic":
            if self.config.semantic_grading:
                return _semantic_match(response, expected)
            # fall back
            return expected_lower in response_lower
        if check_type == "regex":
            try:
                return bool(re.search(expected, response, re.IGNORECASE))
            except re.error:
                return False
        return False

    def _check_math_equivalence(self, response: str, expected: str) -> bool:
        # Normalize scientific notation variants (6.67 × 10^-6 → 6.67e-6)
        def _normalize_sci(text: str) -> str:
            text = text.replace("×", "*").replace("\\times", "*")
            # "6.67 * 10^{-6}" or "6.67 * 10^-6" → "6.67e-6"
            text = re.sub(r'(\d+\.?\d*)\s*\*\s*10\s*\^\s*\{?\s*(-?\d+)\s*\}?',
                          lambda m: f"{m.group(1)}e{m.group(2)}", text)
            return text
        response = _normalize_sci(response)
        expected = _normalize_sci(expected)
        # Normalize exponentiation syntax but preserve multiplication
        clean_response = response.replace("**", "^")
        clean_expected = expected.replace(" ", "").replace("**", "^")

        # Quick check: expected appears as a standalone token (word boundaries).
        # Plain substring would false-positive: expected "2" matching "12" or "32".
        expected_nospace = clean_expected.replace(" ", "")
        if re.search(r'(?<!\w)' + re.escape(expected_nospace) + r'(?!\w)',
                      clean_response.replace(" ", "")):
            return True

        # Extract a math expression from the response — look for common patterns
        # like "= sin(x^2)", "answer: sin(x^2)", or the last line with math symbols
        expr_text = None
        # Try after last "=" sign
        if "=" in clean_response:
            expr_text = clean_response.split("=")[-1].strip()
        # Try after "answer:" or "result:"
        if not expr_text:
            m = re.search(r'(?:answer|result|equals?)\s*:?\s*(.+)', clean_response, re.IGNORECASE)
            if m:
                expr_text = m.group(1).strip()
        # Fallback: last non-empty line
        if not expr_text:
            lines = [l.strip() for l in clean_response.split("\n") if l.strip()]
            expr_text = lines[-1] if lines else clean_response

        # Strip trailing prose (take only the math-looking prefix)
        expr_text = re.split(r'[,.](?:\s|$)', expr_text)[0].strip()

        try:
            from sympy.parsing.sympy_parser import parse_expr
            from sympy import simplify

            # Sanitize before parsing — parse_expr uses eval() internally and can
            # execute arbitrary Python. Strip anything that isn't math-like.
            def _sanitize(text: str) -> str:
                text = text.replace("^", "**").replace(" ", "")
                # Normalize "e**(...)" to "exp(...)" so sympy treats it as
                # Euler's number, not a symbol named 'e'. Models write "e^(2x)"
                # which becomes "e**(2*x)" — without this, sympy creates Symbol('e').
                text = re.sub(r'\be\*\*\(([^)]+)\)', r'exp(\1)', text)
                # Also handle "e**x" (no parens)
                text = re.sub(r'\be\*\*(\w+)', r'exp(\1)', text)
                # Reject if it contains dangerous names
                if re.search(r'(?:import|exec|eval|open|system|compile|getattr|setattr|__|lambda|exit|quit|input|print|breakpoint|srepr|Symbol|Function|sympify|Rational|Integer|Float|Derivative|Subs|Lambda|classmethod|staticmethod|globals|locals|vars)', text):
                    return ""
                # Reject excessively large exponents (can hang sympy)
                if re.search(r'\*\*\d{4,}', text):
                    return ""
                # Only allow: digits, word chars, operators, parens, period.
                # Comma is excluded — "1,2" parses as a tuple in sympy, causing
                # TypeError in complex() and nonsensical expand() results.
                if not re.match(r'^[\d\w+\-*/().eE]+$', text):
                    return ""
                # Length limit — very long expressions are likely garbage
                if len(text) > 200:
                    return ""
                return text

            safe_resp = _sanitize(expr_text)
            safe_exp = _sanitize(clean_expected)
            if not safe_resp or not safe_exp:
                return False

            # local_dict={} + global_dict={} prevents parse_expr from resolving
            # to arbitrary sympy names (defense in depth over the sanitizer).
            resp_expr = parse_expr(safe_resp, local_dict={}, global_dict={})
            exp_expr = parse_expr(safe_exp, local_dict={}, global_dict={})
            # simplify can hang on adversarial inputs — use a weaker but bounded check
            diff = resp_expr - exp_expr
            # Try numeric evaluation first (fast, handles most cases)
            try:
                import cmath
                num = complex(diff.evalf())
                if cmath.isnan(num) or cmath.isinf(num):
                    raise ValueError("non-finite diff")
                return abs(num) < 1e-6
            except (TypeError, ValueError):
                # simplify() can hang on complex expressions — impose a generous
                # but bounded attempt via expand() first (cheaper than simplify).
                from sympy import expand
                expanded = expand(diff)
                if expanded == 0:
                    return True
                # simplify() can hang on adversarial inputs — bound it.
                # signal.alarm only works on Unix main thread; fall back to
                # skipping simplify entirely if unavailable (expand already
                # handles most real math expressions).
                import signal
                import threading
                if sys.platform != "win32" and threading.current_thread() is threading.main_thread():
                    def _timeout_handler(signum, frame):
                        raise TimeoutError("sympy simplify timed out")
                    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                    signal.alarm(2)
                    try:
                        result = simplify(expanded) == 0
                    finally:
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, old_handler)
                    return result
                # Non-main thread or Windows: skip simplify, it's too risky
                return False
        except Exception:
            pass
        return False

    def _check_code_executes(self, response: str) -> bool:
        """Check if code compiles, runs, and the function actually works.

        Just defining a function always succeeds — we need to CALL it with
        basic test inputs to detect runtime errors.
        """
        from ..utils.sandbox import run_python_sandboxed

        code = self._extract_code(response)
        if not code:
            return False
        try:
            compile(code, "<string>", "exec")
        except SyntaxError:
            return False

        # Append a basic smoke-test call. Use the LAST top-level function defined,
        # not the first — models often define helpers (is_prime, swap, etc.) before
        # the main solution function. The last function is most likely the entry point.
        # Handle class-based solutions: if the function is a method (has `self`),
        # find the enclosing class and instantiate it first.
        all_funcs = list(re.finditer(r'def\s+(\w+)\s*\(([^)]*)\)', code))
        # Keep only top-level defs (column 0) — nested helpers would have leading
        # whitespace before "def" and would otherwise hijack the smoke test.
        top_level = [
            m for m in all_funcs
            if m.start() == 0 or code[m.start() - 1] == "\n"
        ]
        func_match = (top_level or all_funcs)[-1] if all_funcs else None
        if func_match:
            func_name = func_match.group(1).lower()
            raw_params = func_match.group(2).strip()
            param_list = [p.strip() for p in raw_params.split(",") if p.strip()] if raw_params else []
            # Strip type annotations and defaults for matching — "self: 'Cls'" → "self",
            # "n: int = 5" → "n". Only need the bare name for method detection and counting.
            bare_names = [re.split(r'[=:]', p)[0].strip() for p in param_list]
            is_method = bare_names and bare_names[0] in ("self", "cls")
            param_count = len([n for n in bare_names if n not in ("self", "cls")])

            # For class methods, find enclosing class and call via instance
            fname = func_match.group(1)
            if is_method:
                class_match = re.search(r'class\s+(\w+)', code)
                if class_match:
                    fname = f'{class_match.group(1)}().{func_match.group(1)}'
                # else: standalone def with self param — unusual but just try it
            if any(k in func_name for k in ("palindrome", "string", "str")):
                # Use truthiness, not == True — function may return 1, "racecar", etc.
                test_call = f'assert {fname}("racecar"); assert not {fname}("hello")'
            elif any(k in func_name for k in ("prime", "sieve", "primes")):
                test_call = f'r = {fname}(30); assert 2 in r and 29 in r and 4 not in r'
            elif any(k in func_name for k in ("merge", "sort")):
                test_call = f'assert {fname}([1,3,5], [2,4,6]) == [1,2,3,4,5,6]'
            elif any(k in func_name for k in ("search", "binary")):
                # Expect index 2. Don't include True in the tuple — True == 1 in
                # Python, so `1 in (2, True)` is True, letting index-1 returns pass.
                test_call = f'r = {fname}([1,2,3,4,5], 3); assert r == 2 or r is True, f"got {{r}}"'
            elif param_count == 1:
                test_call = f'{fname}([1,2,3])'
            elif param_count == 0:
                test_call = f'{fname}()'
            else:
                test_call = f'{fname}([1,2,3], [4,5,6])'

            # Only swallow TypeError for generic fallback calls where we're
            # guessing the arg types. For heuristic-matched tests (palindrome,
            # primes, etc.) a TypeError IS a real bug — the function signature
            # doesn't match what it claims to implement.
            is_heuristic_match = any(k in func_name for k in
                ("palindrome", "string", "str", "prime", "sieve", "primes",
                 "merge", "sort", "search", "binary"))
            if is_heuristic_match:
                code = (f"{code}\n\n# Smoke test\n{test_call}\n")
            else:
                code = (f"{code}\n\n# Smoke test\n"
                        f"import sys as _sys\n"
                        f"try:\n    {test_call}\n"
                        f"except TypeError:\n    _sys.exit(0)  # arg type guess, not a real bug\n")

        # Use the shared sandbox (RLIMITS, audit hooks, scrubbed env, ephemeral cwd)
        # instead of duplicating subprocess logic here.
        ok, _detail = run_python_sandboxed(
            code, timeout_s=self.config.code_execution_timeout, memory_mb=256,
        )
        return ok

    def _extract_code(self, response: str) -> str:
        # Prefer python-tagged blocks; if multiple, take the longest (most likely real code)
        if "```python" in response:
            parts = response.split("```python")[1:]
            blocks = [p.split("```")[0].strip() for p in parts]
            blocks = [b for b in blocks if b]
            if blocks:
                return max(blocks, key=len)
        if "```" in response:
            parts = response.split("```")
            # Odd-indexed parts are inside fences
            blocks = []
            for i in range(1, len(parts), 2):
                block = parts[i]
                # Strip language tag on the first line (e.g., "python\n...", "js\n...")
                first_nl = block.find("\n")
                if first_nl != -1 and first_nl < 20 and block[:first_nl].strip().lower() in _LANG_TAGS:
                    block = block[first_nl + 1:]
                blocks.append(block.strip())
            blocks = [b for b in blocks if b]
            if blocks:
                return max(blocks, key=len)
        if "def " in response:
            lines = response.split("\n")
            all_code_lines = []
            current_func_lines = []
            in_code = False
            base_indent = 0
            for line in lines:
                # Also capture class definitions so class-based solutions include
                # the class header. Without this, methods are extracted without
                # their class, and smoke tests can't instantiate ClassName().
                # Capture decorators — buffer @lines until we see a def/class.
                # Must check BEFORE class/def so "@dataclass\nclass Foo:" works.
                if line.strip().startswith("@") and not in_code:
                    if not current_func_lines:
                        current_func_lines = []
                    current_func_lines.append(line)
                    continue
                if line.strip().startswith("class ") and not in_code:
                    # Same logic as def — check if current_func_lines has a
                    # complete def/class (not just decorators) before flushing.
                    has_def_or_class = any(
                        l.strip().startswith(("def ", "class ")) for l in current_func_lines)
                    if current_func_lines and has_def_or_class:
                        all_code_lines.extend(current_func_lines)
                        all_code_lines.append("")
                        current_func_lines = []
                    in_code = True
                    base_indent = len(line) - len(line.lstrip())
                    current_func_lines.append(line)
                    continue
                if line.strip().startswith("def "):
                    # If current_func_lines has only decorators (no def yet),
                    # they belong to this def — don't flush them.
                    has_def = any(l.strip().startswith(("def ", "class ")) for l in current_func_lines)
                    if current_func_lines and has_def:
                        all_code_lines.extend(current_func_lines)
                        all_code_lines.append("")  # blank line between functions
                        current_func_lines = []
                    in_code = True
                    base_indent = len(line) - len(line.lstrip())
                    current_func_lines.append(line)
                    continue
                if in_code:
                    if not line.strip():
                        current_func_lines.append(line)
                        continue
                    current_indent = len(line) - len(line.lstrip())
                    if current_indent <= base_indent and not line.strip().startswith("def"):
                        in_code = False
                        # Don't break — there may be more functions
                        continue
                    current_func_lines.append(line)
            if current_func_lines:
                all_code_lines.extend(current_func_lines)
            code_text = "\n".join(all_code_lines)
            # Dedent if the code was embedded inside indented prose
            if all_code_lines and all_code_lines[0] != all_code_lines[0].lstrip():
                code_text = textwrap.dedent(code_text)
            return code_text
        return ""
