"""Procedural problem generator: infinite supply of novel coding problems
with deterministically verified canonical solutions.

Why this exists: when external benchmarks (HumanEval, MBPP, etc.) saturate,
the loop runs out of clean training signal. Procedural generators create
fresh problems on demand using Python — the canonical solution and test
cases are computed by trusted reference implementations, so the training
data is 100% ground truth.

Each generator function:
    generate_<category>(seed: int) -> dict with keys
        prompt           — natural language problem statement
        entry_point      — function name the solution should expose
        canonical_code   — reference solution (str, runnable Python)
        tests            — list of `assert <call> == <expected>` strings
        difficulty       — float in [0.1, 0.9]
        category         — generator name (for telemetry)

The generators are seeded so the same seed produces the same problem,
which lets us deterministically sample without overlap across cycles.

Categories shipped:
    array_basic        — list operations (sort variants, sum, max, find)
    array_window       — sliding window (max in window, sum in window)
    string_basic       — string transformations (reverse, count, palindrome)
    number_theory      — gcd, primes, divisors, modular arithmetic
    sequence_dp        — fibonacci-style recurrences
    graph_simple       — BFS/DFS over adjacency dicts
    bitwise            — bitcount, parity, bit-flip ops
    arithmetic_seq     — sum-of-arithmetic, geometric ops

To add a category: write a generator function returning the dict above and
register it in `_GENERATORS`. Compose new categories by combining: the
random seed + category name uniquely determines the problem instance.
"""
from __future__ import annotations
import random
from typing import Callable


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _r(seed: int) -> random.Random:
    return random.Random(seed)


def _format_tests(call_strs: list[str]) -> list[str]:
    return [f"assert {c}" for c in call_strs]


# ──────────────────────────────────────────────────────────────────────
# Generators
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# Tier scaling (#61): difficulty_tier ∈ [1, ∞). Each tier roughly doubles
# the size/complexity dimension that's hardest for the model. tier=1
# matches the original fixed-difficulty problems; tier=10 has 1024-element
# arrays, fib(>1000), graphs with 100s of nodes, etc.
# Capability-tier anchor metric (also #61) computes "highest tier model
# passes ≥80% of N=20 sampled at" — UNBOUNDED, doesn't saturate.
# ──────────────────────────────────────────────────────────────────────

def _tier_scale(tier: int, base: int = 8, growth: float = 1.6) -> int:
    """Convert tier to a size dimension. tier=1 ≈ base; tier=10 ≈ base*100."""
    if tier <= 1:
        return base
    return int(base * (growth ** (tier - 1)))


def gen_array_basic(seed: int, tier: int = 1) -> dict:
    """Sort/sum/find operations on integer arrays. Difficulty scales with
    array size — tier=1 ≈ 5-12 elements, tier=10 ≈ 800+ elements."""
    rng = _r(seed)
    op = rng.choice([
        "max_element", "second_max", "sum_positive", "count_distinct",
        "reverse_sort", "median",
    ])
    n_min = max(2, _tier_scale(tier, base=5))
    n_max = max(n_min + 1, _tier_scale(tier, base=12))
    n = rng.randint(n_min, n_max)
    val_range = max(50, _tier_scale(tier, base=50, growth=1.4))
    arr = [rng.randint(-val_range, val_range) for _ in range(n)]
    if op == "max_element":
        prompt = f"Given a list of integers `nums`, return the maximum element. Assume `nums` is non-empty."
        entry = "solve"
        code = "def solve(nums):\n    return max(nums)"
        tests = [f"solve({arr}) == {max(arr)}"]
        for _ in range(2):
            a2 = [rng.randint(-100, 100) for _ in range(rng.randint(3, 8))]
            tests.append(f"solve({a2}) == {max(a2)}")
    elif op == "second_max":
        # Strictly second-largest distinct value.
        u = sorted(set(arr), reverse=True)
        if len(u) < 2:
            arr.append(arr[0] + 1)
            u = sorted(set(arr), reverse=True)
        prompt = "Given a list of integers `nums`, return the second-largest DISTINCT value. Assume there are at least 2 distinct values."
        entry = "solve"
        code = "def solve(nums):\n    s = sorted(set(nums), reverse=True)\n    return s[1]"
        tests = [f"solve({arr}) == {u[1]}"]
        a2 = list({rng.randint(-50, 50) for _ in range(rng.randint(4, 10))})
        if len(a2) >= 2:
            tests.append(f"solve({a2}) == {sorted(a2, reverse=True)[1]}")
    elif op == "sum_positive":
        prompt = "Given a list of integers `nums`, return the sum of strictly-positive elements only. Empty list returns 0."
        entry = "solve"
        code = "def solve(nums):\n    return sum(x for x in nums if x > 0)"
        tests = [f"solve({arr}) == {sum(x for x in arr if x > 0)}"]
        tests.append("solve([]) == 0")
        a2 = [rng.randint(-30, 30) for _ in range(6)]
        tests.append(f"solve({a2}) == {sum(x for x in a2 if x > 0)}")
    elif op == "count_distinct":
        prompt = "Given a list `nums`, return the number of distinct values."
        entry = "solve"
        code = "def solve(nums):\n    return len(set(nums))"
        tests = [f"solve({arr}) == {len(set(arr))}", "solve([]) == 0"]
    elif op == "reverse_sort":
        prompt = "Given a list of integers `nums`, return a new list sorted descending. Original list must not be mutated."
        entry = "solve"
        code = "def solve(nums):\n    return sorted(nums, reverse=True)"
        tests = [f"solve({arr}) == {sorted(arr, reverse=True)}"]
    else:  # median
        s = sorted(arr)
        med = s[len(s) // 2] if len(s) % 2 else (s[len(s)//2 - 1] + s[len(s)//2]) / 2
        prompt = "Given a list of integers `nums` (non-empty), return the median. For even-length lists return the mean of the two middle elements (a float)."
        entry = "solve"
        code = (
            "def solve(nums):\n"
            "    s = sorted(nums)\n"
            "    n = len(s)\n"
            "    return s[n // 2] if n % 2 else (s[n//2 - 1] + s[n//2]) / 2"
        )
        tests = [f"solve({arr}) == {med}"]
    return {
        "prompt": prompt, "entry_point": entry, "canonical_code": code,
        "tests": _format_tests(tests), "difficulty": 0.20, "category": "array_basic",
    }


def gen_string_basic(seed: int, tier: int = 1) -> dict:
    rng = _r(seed)
    s_min = max(8, _tier_scale(tier, base=8))
    s_max = max(s_min + 1, _tier_scale(tier, base=20))
    op = rng.choice(["reverse", "is_palindrome", "count_vowels", "longest_word"])
    s = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz ") for _ in range(rng.randint(s_min, s_max))).strip()
    s = s if s else "hello world"
    if op == "reverse":
        prompt = "Given a string `s`, return s reversed."
        entry = "solve"
        code = "def solve(s):\n    return s[::-1]"
        tests = [f'solve({s!r}) == {s[::-1]!r}', 'solve("") == ""']
    elif op == "is_palindrome":
        # ignore spaces and case
        clean = "".join(c.lower() for c in s if c.isalnum())
        ans = clean == clean[::-1]
        prompt = "Given a string `s`, return True if `s` is a palindrome (ignoring case and non-alphanumeric chars), else False."
        entry = "solve"
        code = (
            'def solve(s):\n'
            '    c = "".join(ch.lower() for ch in s if ch.isalnum())\n'
            '    return c == c[::-1]'
        )
        tests = [f'solve({s!r}) == {ans}', 'solve("racecar") == True', 'solve("abc") == False']
    elif op == "count_vowels":
        v = sum(1 for c in s.lower() if c in "aeiou")
        prompt = "Given a string `s`, count vowels (a,e,i,o,u) ignoring case."
        entry = "solve"
        code = 'def solve(s):\n    return sum(1 for c in s.lower() if c in "aeiou")'
        tests = [f'solve({s!r}) == {v}', 'solve("") == 0']
    else:  # longest_word
        words = s.split()
        longest = max(words, key=len) if words else ""
        prompt = "Given a string `s` of space-separated words, return the longest word. Tie-break: first occurrence wins."
        entry = "solve"
        code = (
            'def solve(s):\n'
            '    ws = s.split()\n'
            '    if not ws: return ""\n'
            '    return max(ws, key=len)'
        )
        tests = [f'solve({s!r}) == {longest!r}', 'solve("") == ""']
    return {
        "prompt": prompt, "entry_point": entry, "canonical_code": code,
        "tests": _format_tests(tests), "difficulty": 0.25, "category": "string_basic",
    }


def gen_number_theory(seed: int, tier: int = 1) -> dict:
    rng = _r(seed)
    op = rng.choice(["gcd", "is_prime", "count_divisors", "digit_sum"])
    n_max = max(200, _tier_scale(tier, base=200, growth=2.5))
    if op == "gcd":
        a, b = rng.randint(2, n_max), rng.randint(2, n_max)
        from math import gcd as _g
        prompt = "Given two non-negative integers `a` and `b`, return their greatest common divisor."
        entry = "solve"
        code = "def solve(a, b):\n    while b:\n        a, b = b, a % b\n    return a"
        tests = [f"solve({a}, {b}) == {_g(a, b)}", "solve(0, 5) == 5", "solve(48, 18) == 6"]
    elif op == "is_prime":
        n = rng.randint(2, max(1000, n_max * 5))
        def _isprime(x):
            if x < 2: return False
            if x < 4: return True
            if x % 2 == 0: return False
            i = 3
            while i*i <= x:
                if x % i == 0: return False
                i += 2
            return True
        ans = _isprime(n)
        prompt = "Given a non-negative integer `n`, return True if n is prime, else False."
        entry = "solve"
        code = (
            "def solve(n):\n"
            "    if n < 2: return False\n"
            "    if n < 4: return True\n"
            "    if n % 2 == 0: return False\n"
            "    i = 3\n"
            "    while i*i <= n:\n"
            "        if n % i == 0: return False\n"
            "        i += 2\n"
            "    return True"
        )
        tests = [f"solve({n}) == {ans}", "solve(2) == True", "solve(9) == False", "solve(0) == False"]
    elif op == "count_divisors":
        n = rng.randint(2, max(500, n_max * 2))
        cnt = sum(1 for d in range(1, n + 1) if n % d == 0)
        prompt = "Given a positive integer `n`, return the count of its positive divisors."
        entry = "solve"
        code = (
            "def solve(n):\n"
            "    c = 0\n"
            "    i = 1\n"
            "    while i*i <= n:\n"
            "        if n % i == 0:\n"
            "            c += 2 if i != n // i else 1\n"
            "        i += 1\n"
            "    return c"
        )
        tests = [f"solve({n}) == {cnt}", "solve(1) == 1", "solve(12) == 6"]
    else:  # digit_sum
        n = rng.randint(0, max(1_000_000, n_max * 100_000))
        ans = sum(int(d) for d in str(n))
        prompt = "Given a non-negative integer `n`, return the sum of its decimal digits."
        entry = "solve"
        code = "def solve(n):\n    return sum(int(d) for d in str(n))"
        tests = [f"solve({n}) == {ans}", "solve(0) == 0", "solve(123) == 6"]
    return {
        "prompt": prompt, "entry_point": entry, "canonical_code": code,
        "tests": _format_tests(tests), "difficulty": 0.30, "category": "number_theory",
    }


def gen_sequence_dp(seed: int, tier: int = 1) -> dict:
    rng = _r(seed)
    op = rng.choice(["fib", "stair_climb", "longest_increasing"])
    fib_max = max(30, _tier_scale(tier, base=30, growth=1.5))
    arr_max = max(10, _tier_scale(tier, base=10))
    if op == "fib":
        n = rng.randint(2, fib_max)
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        prompt = "Given a non-negative integer `n`, return the nth Fibonacci number where fib(0)=0, fib(1)=1."
        entry = "solve"
        code = (
            "def solve(n):\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n):\n"
            "        a, b = b, a + b\n"
            "    return a"
        )
        tests = [f"solve({n}) == {a}", "solve(0) == 0", "solve(1) == 1", "solve(10) == 55"]
    elif op == "stair_climb":
        n = rng.randint(1, max(20, _tier_scale(tier, base=20, growth=1.5)))
        a, b = 1, 1
        for _ in range(n):
            a, b = b, a + b
        ways = a
        prompt = "Given a positive integer `n`, return the number of ways to climb n stairs taking 1 or 2 steps at a time."
        entry = "solve"
        code = (
            "def solve(n):\n"
            "    if n <= 1: return 1\n"
            "    a, b = 1, 1\n"
            "    for _ in range(n):\n"
            "        a, b = b, a + b\n"
            "    return a"
        )
        tests = [f"solve({n}) == {ways}", "solve(1) == 1", "solve(2) == 2", "solve(5) == 8"]
    else:  # longest_increasing
        n = rng.randint(4, arr_max)
        arr_range = max(20, _tier_scale(tier, base=20, growth=1.4))
        arr = [rng.randint(-arr_range, arr_range) for _ in range(n)]
        # Compute LIS length (O(n^2) reference)
        lis = [1] * n
        for i in range(1, n):
            for j in range(i):
                if arr[j] < arr[i]:
                    lis[i] = max(lis[i], lis[j] + 1)
        ans = max(lis) if lis else 0
        prompt = "Given a list of integers `nums`, return the length of the longest STRICTLY-increasing subsequence (not necessarily contiguous)."
        entry = "solve"
        code = (
            "def solve(nums):\n"
            "    n = len(nums)\n"
            "    if n == 0: return 0\n"
            "    lis = [1] * n\n"
            "    for i in range(1, n):\n"
            "        for j in range(i):\n"
            "            if nums[j] < nums[i]:\n"
            "                lis[i] = max(lis[i], lis[j] + 1)\n"
            "    return max(lis)"
        )
        tests = [f"solve({arr}) == {ans}", "solve([]) == 0", "solve([1]) == 1"]
    return {
        "prompt": prompt, "entry_point": entry, "canonical_code": code,
        "tests": _format_tests(tests), "difficulty": 0.45, "category": "sequence_dp",
    }


def gen_bitwise(seed: int, tier: int = 1) -> dict:
    rng = _r(seed)
    n_range = max(1_000_000, _tier_scale(tier, base=1_000_000, growth=2.0))
    op = rng.choice(["popcount", "is_power_of_two", "single_number"])
    if op == "popcount":
        n = rng.randint(0, n_range)
        ans = bin(n).count("1")
        prompt = "Given a non-negative integer `n`, return the number of 1-bits in its binary representation."
        entry = "solve"
        code = "def solve(n):\n    return bin(n).count('1')"
        tests = [f"solve({n}) == {ans}", "solve(0) == 0", "solve(7) == 3"]
    elif op == "is_power_of_two":
        n = rng.choice([rng.choice([1, 2, 4, 8, 16, 64, 1024]), rng.randint(3, 1023)])
        ans = (n > 0) and (n & (n - 1) == 0)
        prompt = "Given a non-negative integer `n`, return True if n is a positive power of 2 (1,2,4,8,...), else False."
        entry = "solve"
        code = "def solve(n):\n    return n > 0 and (n & (n - 1)) == 0"
        tests = [f"solve({n}) == {ans}", "solve(0) == False", "solve(1) == True", "solve(64) == True", "solve(63) == False"]
    else:  # single_number
        n = rng.randint(2, 6)
        pairs = []
        for _ in range(n):
            v = rng.randint(1, 50)
            pairs.extend([v, v])
        unique = rng.randint(51, 100)
        arr = pairs + [unique]
        rng.shuffle(arr)
        prompt = "Given a list `nums` where every element appears twice except ONE, return the unique element. O(n) time, O(1) space."
        entry = "solve"
        code = "def solve(nums):\n    r = 0\n    for x in nums:\n        r ^= x\n    return r"
        tests = [f"solve({arr}) == {unique}", "solve([1,1,2,2,3]) == 3"]
    return {
        "prompt": prompt, "entry_point": entry, "canonical_code": code,
        "tests": _format_tests(tests), "difficulty": 0.35, "category": "bitwise",
    }


def gen_array_window(seed: int, tier: int = 1) -> dict:
    rng = _r(seed)
    n_min = max(6, _tier_scale(tier, base=6))
    n_max = max(n_min + 2, _tier_scale(tier, base=14))
    n = rng.randint(n_min, n_max)
    k = rng.randint(2, max(2, n // 2))
    arr = [rng.randint(-20, 20) for _ in range(n)]
    op = rng.choice(["max_in_window", "sum_in_window"])
    if op == "max_in_window":
        ans = [max(arr[i:i+k]) for i in range(n - k + 1)]
        prompt = f"Given a list `nums` and integer `k`, return a list where the i-th element is the maximum of nums[i:i+k]. Length of output is len(nums) - k + 1."
        entry = "solve"
        code = (
            "def solve(nums, k):\n"
            "    return [max(nums[i:i+k]) for i in range(len(nums) - k + 1)]"
        )
        tests = [f"solve({arr}, {k}) == {ans}"]
    else:
        ans = [sum(arr[i:i+k]) for i in range(n - k + 1)]
        prompt = f"Given a list `nums` and integer `k`, return a list of contiguous-window sums of length k."
        entry = "solve"
        code = (
            "def solve(nums, k):\n"
            "    return [sum(nums[i:i+k]) for i in range(len(nums) - k + 1)]"
        )
        tests = [f"solve({arr}, {k}) == {ans}"]
    return {
        "prompt": prompt, "entry_point": entry, "canonical_code": code,
        "tests": _format_tests(tests), "difficulty": 0.35, "category": "array_window",
    }


# ──────────────────────────────────────────────────────────────────────
# Registry + sampler
# ──────────────────────────────────────────────────────────────────────

_GENERATORS: dict[str, Callable[[int, int], dict]] = {
    "array_basic": gen_array_basic,
    "string_basic": gen_string_basic,
    "number_theory": gen_number_theory,
    "sequence_dp": gen_sequence_dp,
    "bitwise": gen_bitwise,
    "array_window": gen_array_window,
}


def sample_problems(n: int, seed: int, tier: int = 1) -> list[dict]:
    """Return n procedurally-generated problems at the given difficulty
    tier. Categories cycled round-robin, seeds derived from `seed` so
    same (n, seed, tier) → same set. Tier ≥ 1; higher = harder.
    """
    if n <= 0:
        return []
    cats = sorted(_GENERATORS.keys())
    out: list[dict] = []
    rng = random.Random(seed)
    for i in range(n):
        cat = cats[i % len(cats)]
        sub_seed = rng.randint(0, 2**31 - 1)
        try:
            problem = _GENERATORS[cat](sub_seed, tier)
            problem["tier"] = tier
            out.append(problem)
        except Exception:
            continue
    return out


# Capability-tier anchor metric (#61): unbounded measurement of model
# capability. Returns the highest tier at which the model passes ≥ pass_thresh
# fraction of N=samples_per_tier procedural problems. Probes tier 1,
# tier 2, ... until pass-rate drops below threshold. Genuinely unbounded
# — there's no benchmark saturation cap because procedural difficulty
# scales arbitrarily.
def capability_tier(
    grade_fn: Callable[[dict], bool],
    *,
    samples_per_tier: int = 20,
    pass_thresh: float = 0.8,
    max_tier: int = 100,
    seed: int = 0xCAFE,
) -> int:
    """Probe tiers ascending; return highest tier where grade_fn passes
    ≥ pass_thresh of samples_per_tier problems.

    grade_fn(problem) -> bool. Caller is responsible for invoking the
    model on problem["prompt"] and checking the response against
    problem["tests"] / problem["canonical_code"].
    """
    last_passing = 0
    for tier in range(1, max_tier + 1):
        probs = sample_problems(
            n=samples_per_tier, seed=seed + tier, tier=tier,
        )
        if not probs:
            break
        passes = 0
        for p in probs:
            try:
                if grade_fn(p):
                    passes += 1
            except Exception:
                continue
        rate = passes / max(1, len(probs))
        if rate >= pass_thresh:
            last_passing = tier
        else:
            # Two-strikes: if the previous tier also failed, we're past
            # the model's frontier — return the last passing.
            if tier - last_passing >= 2:
                return last_passing
    return last_passing


# Quick self-test invocable manually: ensures every canonical_code passes
# every test it claims to pass. Run: python -m src.generator.procedural_problems
def _selftest() -> None:
    import sys
    failures = 0
    for tier in (1, 3, 5):
        for cat, gen in _GENERATORS.items():
            for s in range(3):
                p = gen(s + cat.__hash__() % 1000, tier)
                ns = {}
                try:
                    exec(p["canonical_code"], ns)
                    for t in p["tests"]:
                        exec(t, ns)
                except Exception as e:
                    print(f"FAIL tier={tier} {cat}/seed={s}: {e}")
                    failures += 1
    if failures:
        print(f"{failures} failures")
        sys.exit(1)
    n_total = 3 * len(_GENERATORS) * 3  # 3 tiers × 6 cats × 3 seeds
    print(f"PASS — {n_total} problems across {len(_GENERATORS)} categories × 3 tiers (1, 3, 5)")


if __name__ == "__main__":
    _selftest()
