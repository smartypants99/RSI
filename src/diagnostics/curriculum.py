"""Open-ended curriculum — generates NEW problems at the Zone of Proximal Development.

Frontier labs evaluate on fixed benchmarks. Proper RSI requires problems that
track capability: solved-at-100% gets retired, solved-at-0% gets broken down,
solved at 30-70% is where learning happens. This module implements that.

Core objects:
- ProblemClass: a family of problems parameterized by a difficulty knob.
- CurriculumState: per-class rolling solve-rates, active/retired/frontier sets,
  and frontier selection (70/20/10 mix). Persists across cycles via to_dict/
  from_dict so the engine's checkpoint carries it.

Usage:
    state = CurriculumState.from_dict(checkpoint.get("curriculum", {}))
    questions = state.pick_frontier(domain="math", n=50, rng=rng)
    # ... run diagnostics, grade ...
    state.record_results(results)   # results: list[(class_id, difficulty, solved)]
    checkpoint["curriculum"] = state.to_dict()
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Problem generators
# ---------------------------------------------------------------------------
#
# Each generator takes (rng, difficulty: int) -> (prompt, canonical_answer,
# check_type). Difficulty is an integer knob the class interprets — typically
# 1..10, but a class can expand past its original ceiling when it hits it.


def _gen_linear_system(rng: random.Random, difficulty: int):
    """N equations, N unknowns over integers. Difficulty scales N and coefficient range."""
    n = min(2 + difficulty // 2, 6)
    hi = 3 + difficulty * 2
    # Pick canonical solution, then build random nonsingular coefficient matrix.
    solution = [rng.randint(-hi, hi) for _ in range(n)]
    for _ in range(20):
        A = [[rng.randint(-hi, hi) for _ in range(n)] for _ in range(n)]
        if _det(A) != 0:
            break
    else:
        A = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    b = [sum(A[i][j] * solution[j] for j in range(n)) for i in range(n)]
    vars_ = ["x", "y", "z", "w", "u", "v"][:n]
    lines = []
    for i in range(n):
        terms = " + ".join(f"{A[i][j]}*{vars_[j]}" for j in range(n))
        lines.append(f"{terms} = {b[i]}")
    prompt = ("Solve the linear system (integers). Give each variable.\n"
              + "\n".join(lines))
    expected = ", ".join(f"{vars_[i]}={solution[i]}" for i in range(n))
    return prompt, expected, "contains"


def _det(M):
    n = len(M)
    if n == 1:
        return M[0][0]
    if n == 2:
        return M[0][0] * M[1][1] - M[0][1] * M[1][0]
    s = 0
    for j in range(n):
        minor = [row[:j] + row[j + 1:] for row in M[1:]]
        s += ((-1) ** j) * M[0][j] * _det(minor)
    return s


def _gen_modular_arith(rng: random.Random, difficulty: int):
    m = 5 + difficulty * 7
    a = rng.randint(2, 10 + difficulty * 20)
    k = rng.randint(2, 3 + difficulty)
    ans = pow(a, k, m)
    return f"Compute {a}^{k} mod {m}.", str(ans), "contains"


def _gen_gcd_chain(rng: random.Random, difficulty: int):
    count = 2 + difficulty // 2
    hi = 20 + difficulty * 40
    nums = [rng.randint(2, hi) for _ in range(count)]
    g = nums[0]
    for x in nums[1:]:
        g = math.gcd(g, x)
    return f"Compute gcd({', '.join(map(str, nums))}).", str(g), "contains"


def _gen_polynomial_eval(rng: random.Random, difficulty: int):
    deg = 2 + difficulty // 2
    coeffs = [rng.randint(-5 - difficulty, 5 + difficulty) for _ in range(deg + 1)]
    x = rng.randint(-3 - difficulty, 3 + difficulty)
    val = sum(c * (x ** i) for i, c in enumerate(coeffs))
    poly = " + ".join(f"{c}*x^{i}" for i, c in enumerate(coeffs) if c)
    return f"Evaluate p(x) = {poly} at x = {x}.", str(val), "contains"


def _gen_fraction_arith(rng: random.Random, difficulty: int):
    def f():
        return Fraction(rng.randint(-5 - difficulty, 5 + difficulty),
                        rng.randint(1, 3 + difficulty))
    a, b, c = f(), f(), f()
    op1, op2 = rng.choice(["+", "-", "*"]), rng.choice(["+", "-", "*"])
    def apply(x, y, op):
        return x + y if op == "+" else (x - y if op == "-" else x * y)
    result = apply(apply(a, b, op1), c, op2)
    prompt = (f"Evaluate ({a}) {op1} ({b}) {op2} ({c}). "
              f"Give the answer as a single reduced fraction p/q or integer.")
    return prompt, f"{result.numerator}/{result.denominator}" if result.denominator != 1 else str(result.numerator), "contains"


def _gen_counting_combinatorics(rng: random.Random, difficulty: int):
    n = 4 + difficulty
    k = rng.randint(2, min(n - 1, 2 + difficulty))
    ans = math.comb(n, k)
    return f"How many ways to choose {k} items from {n} distinct items?", str(ans), "contains"


def _gen_sequence_pattern(rng: random.Random, difficulty: int):
    # Geometric-with-offset: a*r^i + c
    r = rng.randint(2, 2 + difficulty)
    a = rng.randint(1, 3 + difficulty)
    c = rng.randint(-5, 5)
    terms = [a * (r ** i) + c for i in range(4)]
    nxt = a * (r ** 4) + c
    prompt = f"Next term in the sequence: {terms[0]}, {terms[1]}, {terms[2]}, {terms[3]}, ?"
    return prompt, str(nxt), "contains"


def _gen_logic_sat(rng: random.Random, difficulty: int):
    # Conjunctive truth-value problem over k variables.
    k = 3 + difficulty // 2
    assign = {v: rng.choice([True, False]) for v in "abcdefgh"[:k]}
    clauses = []
    for _ in range(k + difficulty):
        v = rng.choice(list(assign.keys()))
        negate = rng.choice([True, False])
        lit = f"NOT {v}" if negate else v
        truth_of_lit = (not assign[v]) if negate else assign[v]
        clauses.append((lit, truth_of_lit))
    # Ask for the AND of all clauses under the given assignment.
    truth = all(t for _, t in clauses)
    setting = ", ".join(f"{v}={'T' if assign[v] else 'F'}" for v in assign)
    expr = " AND ".join(lit for lit, _ in clauses)
    return (f"Given {setting}, evaluate: {expr}. Answer T or F.",
            "T" if truth else "F", "contains")


def _gen_word_problem_rates(rng: random.Random, difficulty: int):
    # Two workers/pipes filling at different rates.
    t1 = rng.randint(2, 4 + difficulty)
    t2 = rng.randint(t1 + 1, t1 + 5 + difficulty)
    # Combined rate 1/t1 + 1/t2 ⇒ combined time = t1*t2/(t1+t2)
    f = Fraction(t1 * t2, t1 + t2)
    ans = f"{f.numerator}/{f.denominator}" if f.denominator != 1 else str(f.numerator)
    prompt = (f"Pipe A fills a tank in {t1} hours, pipe B in {t2} hours. "
              f"Together, how many hours? Answer as a reduced fraction p/q or integer.")
    return prompt, ans, "contains"


def _gen_code_output(rng: random.Random, difficulty: int):
    # Predict the output of a small python snippet.
    n = rng.randint(3 + difficulty, 6 + difficulty * 2)
    op = rng.choice(["sum_sq", "sum_odd", "max_minus_min"])
    xs = [rng.randint(-5 - difficulty, 10 + difficulty) for _ in range(n)]
    if op == "sum_sq":
        ans = sum(x * x for x in xs)
        snippet = f"sum(x*x for x in {xs})"
    elif op == "sum_odd":
        ans = sum(x for x in xs if x % 2 != 0)
        snippet = f"sum(x for x in {xs} if x % 2 != 0)"
    else:
        ans = max(xs) - min(xs)
        snippet = f"max({xs}) - min({xs})"
    return f"What is the value of: {snippet}?", str(ans), "contains"


def _gen_base_conversion(rng: random.Random, difficulty: int):
    n = rng.randint(8, 8 + difficulty * 50)
    base = rng.choice([2, 8, 16])
    if base == 16:
        ans = hex(n)[2:].upper()
    elif base == 8:
        ans = oct(n)[2:]
    else:
        ans = bin(n)[2:]
    return f"Convert decimal {n} to base {base} (no prefix).", ans, "contains"


# ---------------------------------------------------------------------------
# ProblemClass
# ---------------------------------------------------------------------------


GeneratorFn = Callable[[random.Random, int], tuple[str, str, str]]


@dataclass
class ProblemClass:
    """A family of problems + difficulty knob + canonical-answer computer.

    `generate(difficulty, rng)` returns a question dict compatible with the
    diagnostics engine (prompt / expected / check_type / subdomain /
    difficulty). Difficulty is an integer — a class has a current ceiling
    that can be expanded by `expand_ceiling()` when mastered.
    """
    id: str
    domain: str
    subdomain: str
    generator: GeneratorFn
    min_difficulty: int = 1
    max_difficulty: int = 10
    # Ceiling can grow past max_difficulty when the model masters the class.
    ceiling: int = 10
    # Generation number — bumped each time the class is expanded. New
    # expansions inherit the base generator but with cranked difficulty
    # offset applied when sampling.
    generation: int = 0

    def generate(self, difficulty: int, rng: random.Random) -> dict:
        d = max(self.min_difficulty, min(difficulty, self.ceiling))
        prompt, expected, check_type = self.generator(rng, d)
        return {
            "prompt": prompt,
            "expected": expected,
            "check_type": check_type,
            "subdomain": self.subdomain,
            "difficulty": _bucket(d),
            "_class_id": self.id,
            "_difficulty_int": d,
        }

    def expand_ceiling(self) -> None:
        """Crank difficulty up when the class is saturated."""
        self.ceiling += 3
        self.generation += 1


def _bucket(d: int) -> str:
    if d <= 2:
        return "easy"
    if d <= 5:
        return "medium"
    if d <= 8:
        return "hard"
    return "expert"


# 10+ canonical classes covering math / code / logic / reasoning.
DEFAULT_CLASSES: list[ProblemClass] = [
    ProblemClass("math.linear_system", "math", "algebra", _gen_linear_system),
    ProblemClass("math.modular", "math", "number_theory", _gen_modular_arith),
    ProblemClass("math.gcd_chain", "math", "number_theory", _gen_gcd_chain),
    ProblemClass("math.polynomial_eval", "math", "algebra", _gen_polynomial_eval),
    ProblemClass("math.fraction_arith", "math", "arithmetic", _gen_fraction_arith),
    ProblemClass("math.combinatorics", "math", "combinatorics", _gen_counting_combinatorics),
    ProblemClass("reasoning.sequence", "reasoning", "pattern", _gen_sequence_pattern),
    ProblemClass("reasoning.logic_sat", "reasoning", "logic", _gen_logic_sat),
    ProblemClass("reasoning.word_rates", "reasoning", "word_problem", _gen_word_problem_rates),
    ProblemClass("code.predict_output", "code", "prediction", _gen_code_output),
    ProblemClass("code.base_conversion", "code", "bit_manipulation", _gen_base_conversion),
]


# ---------------------------------------------------------------------------
# CurriculumState
# ---------------------------------------------------------------------------


@dataclass
class _Stats:
    """Rolling solve-rate tracker for a single (class, difficulty) bucket."""
    attempts: int = 0
    solved: int = 0
    # Cycle-level history — list of (solved, attempts) per cycle, capped.
    history: list[tuple[int, int]] = field(default_factory=list)

    def rate(self) -> float:
        return self.solved / self.attempts if self.attempts else 0.0

    def update(self, solved: int, attempts: int, window: int) -> None:
        self.history.append((solved, attempts))
        if len(self.history) > window:
            self.history = self.history[-window:]
        self.solved = sum(s for s, _ in self.history)
        self.attempts = sum(a for _, a in self.history)


FRONTIER_LO = 0.30
FRONTIER_HI = 0.70
RETIRE_RATE = 0.95
FLOOR_RATE = 0.05
HISTORY_WINDOW = 5  # rolling over last N cycles
MIN_ATTEMPTS_BEFORE_JUDGMENT = 6


@dataclass
class CurriculumState:
    """Persistent curriculum state — active classes, solve rates, retirement."""
    # class_id -> difficulty_int -> _Stats
    solve_rate: dict[str, dict[int, _Stats]] = field(default_factory=dict)
    active_classes: list[str] = field(default_factory=list)
    retired_classes: list[str] = field(default_factory=list)
    # class_id -> ceiling, generation (for reconstruction after expansion)
    class_meta: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def fresh(cls, classes: Optional[list[ProblemClass]] = None) -> "CurriculumState":
        classes = classes or DEFAULT_CLASSES
        s = cls()
        for c in classes:
            s.active_classes.append(c.id)
            s.solve_rate[c.id] = {}
            s.class_meta[c.id] = {"ceiling": c.ceiling, "generation": c.generation}
        return s

    # -- Class registry ------------------------------------------------------

    def _class_by_id(self, class_id: str,
                     registry: list[ProblemClass]) -> Optional[ProblemClass]:
        for c in registry:
            if c.id == class_id:
                meta = self.class_meta.get(class_id, {})
                c.ceiling = meta.get("ceiling", c.ceiling)
                c.generation = meta.get("generation", c.generation)
                return c
        return None

    # -- Frontier classification --------------------------------------------

    def _class_rate(self, class_id: str) -> tuple[float, int]:
        """Aggregate solve rate across all difficulties for this class."""
        bucket = self.solve_rate.get(class_id, {})
        solved = sum(s.solved for s in bucket.values())
        attempts = sum(s.attempts for s in bucket.values())
        return (solved / attempts if attempts else 0.0), attempts

    def classify(self) -> dict[str, str]:
        """Each active class -> 'frontier' | 'easy' | 'probe_harder' | 'insufficient'."""
        out: dict[str, str] = {}
        for cid in self.active_classes:
            rate, attempts = self._class_rate(cid)
            if attempts < MIN_ATTEMPTS_BEFORE_JUDGMENT:
                out[cid] = "insufficient"
            elif FRONTIER_LO <= rate <= FRONTIER_HI:
                out[cid] = "frontier"
            elif rate > FRONTIER_HI:
                out[cid] = "probe_harder"
            else:
                out[cid] = "easy"  # below frontier: needs easier variants
        return out

    # -- Selection ----------------------------------------------------------

    def pick_frontier(
        self,
        n: int,
        rng: random.Random,
        registry: Optional[list[ProblemClass]] = None,
        domain: Optional[str] = None,
    ) -> list[dict]:
        """Sample n questions with frontier/retention/probe mix (70/20/10)."""
        registry = registry or DEFAULT_CLASSES
        classification = self.classify()

        def in_domain(cid: str) -> bool:
            if not domain:
                return True
            cls = self._class_by_id(cid, registry)
            return cls is not None and cls.domain == domain

        frontier = [c for c, k in classification.items() if k == "frontier" and in_domain(c)]
        easy = [c for c, k in classification.items() if k == "easy" and in_domain(c)]
        probe = [c for c, k in classification.items()
                 if k in ("probe_harder", "insufficient") and in_domain(c)]

        # When frontier is empty, promote insufficient classes so the model
        # actually gets data to classify them.
        if not frontier:
            frontier = probe or easy
            probe = []

        target_frontier = max(1, int(round(n * 0.70)))
        target_easy = int(round(n * 0.20))
        target_probe = n - target_frontier - target_easy

        questions: list[dict] = []
        questions += self._sample_from(frontier, target_frontier, rng, registry,
                                        mode="frontier")
        questions += self._sample_from(easy, target_easy, rng, registry,
                                        mode="retention")
        questions += self._sample_from(probe, target_probe, rng, registry,
                                        mode="probe_harder")

        # Pad if a bucket was empty.
        if len(questions) < n:
            fallback = frontier + easy + probe
            questions += self._sample_from(fallback, n - len(questions), rng,
                                            registry, mode="frontier")
        rng.shuffle(questions)
        return questions[:n]

    def _sample_from(
        self,
        class_ids: list[str],
        n: int,
        rng: random.Random,
        registry: list[ProblemClass],
        mode: str,
    ) -> list[dict]:
        if n <= 0 or not class_ids:
            return []
        out: list[dict] = []
        attempts = 0
        while len(out) < n and attempts < n * 6:
            attempts += 1
            cid = rng.choice(class_ids)
            cls = self._class_by_id(cid, registry)
            if cls is None:
                continue
            d = self._pick_difficulty(cid, cls, mode, rng)
            try:
                q = cls.generate(d, rng)
            except Exception:
                continue
            out.append(q)
        return out

    def _pick_difficulty(
        self, class_id: str, cls: ProblemClass, mode: str, rng: random.Random
    ) -> int:
        """Pick a difficulty — frontier targets the band nearest 50% solve rate."""
        bucket = self.solve_rate.get(class_id, {})
        # Find the difficulty whose observed solve-rate is closest to 0.5.
        if bucket and mode == "frontier":
            scored = [(d, abs(s.rate() - 0.5), s.attempts)
                      for d, s in bucket.items() if s.attempts >= 2]
            if scored:
                scored.sort(key=lambda x: (x[1], -x[2]))
                base = scored[0][0]
                return max(cls.min_difficulty,
                           min(cls.ceiling, base + rng.randint(-1, 1)))
        if mode == "retention":
            # Easier end — well-mastered difficulties.
            easy_d = [d for d, s in bucket.items()
                      if s.attempts >= 2 and s.rate() >= 0.85]
            if easy_d:
                return rng.choice(easy_d)
            return max(cls.min_difficulty, cls.ceiling // 3)
        if mode == "probe_harder":
            hardest_known = max(bucket) if bucket else cls.min_difficulty
            return min(cls.ceiling, hardest_known + rng.randint(1, 2))
        # Default — middle of range.
        return (cls.min_difficulty + cls.ceiling) // 2

    # -- Recording ----------------------------------------------------------

    def record_results(
        self,
        results: list[tuple[str, int, bool]],
        registry: Optional[list[ProblemClass]] = None,
    ) -> None:
        """Update rolling stats from this cycle's graded outcomes.

        `results` is a list of (class_id, difficulty_int, solved_bool).
        After updating, retires classes solved at >95% and expands ceilings
        for classes that have saturated their current ceiling.
        """
        registry = registry or DEFAULT_CLASSES
        # Aggregate this cycle's attempts before pushing into history.
        per_bucket: dict[tuple[str, int], list[int]] = {}
        for cid, diff, solved in results:
            per_bucket.setdefault((cid, diff), []).append(1 if solved else 0)
        for (cid, diff), flags in per_bucket.items():
            self.solve_rate.setdefault(cid, {})
            stats = self.solve_rate[cid].setdefault(diff, _Stats())
            stats.update(sum(flags), len(flags), HISTORY_WINDOW)

        # Retirement + expansion decisions.
        for cid in list(self.active_classes):
            rate, attempts = self._class_rate(cid)
            if attempts < MIN_ATTEMPTS_BEFORE_JUDGMENT:
                continue
            if rate >= RETIRE_RATE:
                self._retire(cid)
                continue
            # Ceiling-hit detection: top difficulty is being solved consistently.
            cls = self._class_by_id(cid, registry)
            if cls is None:
                continue
            top_stats = self.solve_rate[cid].get(cls.ceiling)
            if top_stats and top_stats.attempts >= 3 and top_stats.rate() >= 0.8:
                cls.expand_ceiling()
                self.class_meta[cid] = {
                    "ceiling": cls.ceiling, "generation": cls.generation,
                }

    def _retire(self, class_id: str) -> None:
        if class_id in self.active_classes:
            self.active_classes.remove(class_id)
        if class_id not in self.retired_classes:
            self.retired_classes.append(class_id)

    # -- Persistence --------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "active_classes": list(self.active_classes),
            "retired_classes": list(self.retired_classes),
            "class_meta": {k: dict(v) for k, v in self.class_meta.items()},
            "solve_rate": {
                cid: {
                    str(d): {
                        "attempts": s.attempts,
                        "solved": s.solved,
                        "history": list(s.history),
                    }
                    for d, s in buckets.items()
                }
                for cid, buckets in self.solve_rate.items()
            },
        }

    @classmethod
    def from_dict(
        cls, data: dict, classes: Optional[list[ProblemClass]] = None
    ) -> "CurriculumState":
        classes = classes or DEFAULT_CLASSES
        if not data:
            return cls.fresh(classes)
        s = cls()
        s.active_classes = list(data.get("active_classes", [c.id for c in classes]))
        s.retired_classes = list(data.get("retired_classes", []))
        s.class_meta = {k: dict(v) for k, v in data.get("class_meta", {}).items()}
        for cid, buckets in data.get("solve_rate", {}).items():
            s.solve_rate[cid] = {}
            for d_str, payload in buckets.items():
                try:
                    d = int(d_str)
                except (TypeError, ValueError):
                    continue
                hist = [tuple(x) for x in payload.get("history", [])]
                s.solve_rate[cid][d] = _Stats(
                    attempts=int(payload.get("attempts", 0)),
                    solved=int(payload.get("solved", 0)),
                    history=hist,
                )
        # Any newly-added default class absent from the restored state becomes active.
        known = set(s.active_classes) | set(s.retired_classes)
        for c in classes:
            if c.id not in known:
                s.active_classes.append(c.id)
                s.class_meta.setdefault(
                    c.id, {"ceiling": c.ceiling, "generation": c.generation}
                )
        return s
