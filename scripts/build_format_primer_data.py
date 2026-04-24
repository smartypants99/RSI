"""Build SFT pair dataset teaching R1 the proposer output schema.

Each pair = (prompt=CODE_PROPOSAL_TEMPLATE, completion=valid schema-compliant
task). 20 pedagogically diverse entries spanning DP, graphs, strings, math,
recursion, two-pointer, greedy, simulation. SFT on these for a few epochs
teaches the format without teaching any specific problem.

Run: python scripts/build_format_primer_data.py
Emits: data/format_primer_pairs.jsonl
"""
from __future__ import annotations
import json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.generator.task_synthesizer import CODE_PROPOSAL_TEMPLATE


# Each entry: problem text, entry-point name, reference code (body only, we
# wrap in fences), tests (list of assert strings), sample_input (JSON list of
# args), empty_input (JSON list of args), expected_type, difficulty, reason.
_TASKS: list[dict] = [
    {
        "problem": "Given a list of integers `nums`, return the length of the longest strictly increasing contiguous subarray. Return 0 if `nums` is empty.",
        "entry": "solve",
        "ref": "def solve(nums):\n    if not nums:\n        return 0\n    best = cur = 1\n    for i in range(1, len(nums)):\n        cur = cur + 1 if nums[i] > nums[i-1] else 1\n        if cur > best: best = cur\n    return best",
        "tests": [
            "assert solve([1,2,3,2,4,5,6]) == 4",
            "assert solve([]) == 0",
            "assert solve([5]) == 1",
            "assert solve([5,4,3,2]) == 1",
        ],
        "sample_input": [[1, 2, 3, 2, 4, 5, 6]],
        "empty_input": [[]],
        "expected_type": "int",
        "difficulty": 0.2,
        "reason": "Scan with reset-on-break; off-by-one on single element is common.",
    },
    {
        "problem": "Given a non-empty list of integers `nums` where every element appears twice except one, return the element that appears exactly once. O(n) time, O(1) space.",
        "entry": "solve",
        "ref": "def solve(nums):\n    acc = 0\n    for x in nums:\n        acc ^= x\n    return acc",
        "tests": [
            "assert solve([4,1,2,1,2]) == 4",
            "assert solve([7]) == 7",
            "assert solve([2,2,9,3,3]) == 9",
        ],
        "sample_input": [[4, 1, 2, 1, 2]],
        "empty_input": [[0]],
        "expected_type": "int",
        "difficulty": 0.35,
        "reason": "XOR trick — non-obvious without the duplicate-pair insight.",
    },
    {
        "problem": "Given a directed graph as an adjacency dict `graph` and two nodes `src` and `dst`, return True if there is any path from `src` to `dst`, else False. A node reaches itself.",
        "entry": "solve",
        "ref": "def solve(graph, src, dst):\n    if src == dst: return True\n    seen, stack = {src}, [src]\n    while stack:\n        u = stack.pop()\n        for v in graph.get(u, []):\n            if v == dst: return True\n            if v not in seen:\n                seen.add(v); stack.append(v)\n    return False",
        "tests": [
            "assert solve({'a':['b'],'b':['c']}, 'a','c') == True",
            "assert solve({'a':['b'],'b':['c']}, 'c','a') == False",
            "assert solve({}, 'x','x') == True",
            "assert solve({'a':['a']}, 'a','a') == True",
        ],
        "sample_input": [{"a": ["b"], "b": ["c"]}, "a", "c"],
        "empty_input": [{}, "x", "x"],
        "expected_type": "bool",
        "difficulty": 0.4,
        "reason": "DFS with cycle guard; self-loop and trivial src==dst edge cases.",
    },
    {
        "problem": "Given a list `coins` of positive ints and an integer `amount` in 0..5000, return the minimum number of coins summing to `amount`, or -1 if impossible. Unlimited supply.",
        "entry": "solve",
        "ref": "def solve(coins, amount):\n    INF = float('inf')\n    dp = [0] + [INF]*amount\n    for a in range(1, amount+1):\n        for c in coins:\n            if c <= a and dp[a-c]+1 < dp[a]:\n                dp[a] = dp[a-c]+1\n    return dp[amount] if dp[amount] != INF else -1",
        "tests": [
            "assert solve([1,2,5], 11) == 3",
            "assert solve([2], 3) == -1",
            "assert solve([1], 0) == 0",
            "assert solve([3,7], 14) == 2",
        ],
        "sample_input": [[1, 2, 5], 11],
        "empty_input": [[1], 0],
        "expected_type": "int",
        "difficulty": 0.5,
        "reason": "Classic DP; greedy fails, off-by-one on dp[0], -1 impossibility.",
    },
    {
        "problem": "Given a string `s`, return the length of the longest substring with no repeating characters.",
        "entry": "solve",
        "ref": "def solve(s):\n    last = {}; start = 0; best = 0\n    for i, c in enumerate(s):\n        if c in last and last[c] >= start:\n            start = last[c] + 1\n        last[c] = i\n        if i - start + 1 > best: best = i - start + 1\n    return best",
        "tests": [
            "assert solve('abcabcbb') == 3",
            "assert solve('') == 0",
            "assert solve('bbbb') == 1",
            "assert solve('pwwkew') == 3",
        ],
        "sample_input": ["abcabcbb"],
        "empty_input": [""],
        "expected_type": "int",
        "difficulty": 0.4,
        "reason": "Sliding window with last-seen map; resetting start incorrectly is common.",
    },
    {
        "problem": "Given a positive integer `n`, return the number of distinct ways to climb a staircase of `n` steps, taking 1 or 2 steps at a time.",
        "entry": "solve",
        "ref": "def solve(n):\n    if n <= 1: return 1\n    a, b = 1, 1\n    for _ in range(n-1):\n        a, b = b, a + b\n    return b",
        "tests": [
            "assert solve(1) == 1",
            "assert solve(2) == 2",
            "assert solve(5) == 8",
            "assert solve(0) == 1",
        ],
        "sample_input": [5],
        "empty_input": [0],
        "expected_type": "int",
        "difficulty": 0.2,
        "reason": "Fibonacci recurrence; naive recursion is exponential.",
    },
    {
        "problem": "Given a list of intervals `ivs` as [start, end] pairs, return the minimum number of intervals that must be removed so the remaining intervals are non-overlapping. Intervals sharing only an endpoint do not overlap.",
        "entry": "solve",
        "ref": "def solve(ivs):\n    if not ivs: return 0\n    ivs = sorted(ivs, key=lambda x: x[1])\n    keep = 1; end = ivs[0][1]\n    for s, e in ivs[1:]:\n        if s >= end:\n            keep += 1; end = e\n    return len(ivs) - keep",
        "tests": [
            "assert solve([[1,2],[2,3],[3,4],[1,3]]) == 1",
            "assert solve([[1,2],[1,2],[1,2]]) == 2",
            "assert solve([]) == 0",
            "assert solve([[1,2],[2,3]]) == 0",
        ],
        "sample_input": [[[1, 2], [2, 3], [3, 4], [1, 3]]],
        "empty_input": [[]],
        "expected_type": "int",
        "difficulty": 0.55,
        "reason": "Greedy by earliest end-time; sorting by start instead is a classic wrong-turn.",
    },
    {
        "problem": "Given a list of integers `nums` and an integer `k`, return the number of contiguous subarrays whose sum equals `k`. O(n) time.",
        "entry": "solve",
        "ref": "def solve(nums, k):\n    from collections import defaultdict\n    prefix = 0; counts = defaultdict(int); counts[0] = 1; ans = 0\n    for x in nums:\n        prefix += x\n        ans += counts[prefix - k]\n        counts[prefix] += 1\n    return ans",
        "tests": [
            "assert solve([1,1,1], 2) == 2",
            "assert solve([1,2,3], 3) == 2",
            "assert solve([], 0) == 0",
            "assert solve([0,0,0], 0) == 6",
        ],
        "sample_input": [[1, 1, 1], 2],
        "empty_input": [[], 0],
        "expected_type": "int",
        "difficulty": 0.55,
        "reason": "Prefix-sum + hashmap; O(n^2) naive works but times out; zero-sum handling subtle.",
    },
    {
        "problem": "Given a positive integer `n`, return True if `n` is a prime, else False. Handle n < 2 by returning False.",
        "entry": "solve",
        "ref": "def solve(n):\n    if n < 2: return False\n    if n < 4: return True\n    if n % 2 == 0: return False\n    i = 3\n    while i*i <= n:\n        if n % i == 0: return False\n        i += 2\n    return True",
        "tests": [
            "assert solve(2) == True",
            "assert solve(9) == False",
            "assert solve(1) == False",
            "assert solve(97) == True",
        ],
        "sample_input": [97],
        "empty_input": [0],
        "expected_type": "bool",
        "difficulty": 0.2,
        "reason": "Trial division to sqrt(n); forgetting n<2 and n==2 edge cases is common.",
    },
    {
        "problem": "Given a binary tree as nested dict with keys 'val', 'left', 'right' (None-leaves), return its maximum depth. Empty tree has depth 0.",
        "entry": "solve",
        "ref": "def solve(root):\n    if root is None: return 0\n    return 1 + max(solve(root.get('left')), solve(root.get('right')))",
        "tests": [
            "assert solve(None) == 0",
            "assert solve({'val':1,'left':None,'right':None}) == 1",
            "assert solve({'val':1,'left':{'val':2,'left':None,'right':None},'right':None}) == 2",
        ],
        "sample_input": [{"val": 1, "left": {"val": 2, "left": None, "right": None}, "right": None}],
        "empty_input": [None],
        "expected_type": "int",
        "difficulty": 0.3,
        "reason": "Recursive tree walk; None-root edge and leaf vs internal distinction.",
    },
    {
        "problem": "Given a list of non-negative integers `heights`, each representing bar height at unit width, return the amount of rainwater trapped between bars.",
        "entry": "solve",
        "ref": "def solve(h):\n    if not h: return 0\n    l, r = 0, len(h)-1; lm = rm = 0; ans = 0\n    while l < r:\n        if h[l] < h[r]:\n            lm = max(lm, h[l]); ans += lm - h[l]; l += 1\n        else:\n            rm = max(rm, h[r]); ans += rm - h[r]; r -= 1\n    return ans",
        "tests": [
            "assert solve([0,1,0,2,1,0,1,3,2,1,2,1]) == 6",
            "assert solve([]) == 0",
            "assert solve([4,2,0,3,2,5]) == 9",
            "assert solve([1,1,1]) == 0",
        ],
        "sample_input": [[0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]],
        "empty_input": [[]],
        "expected_type": "int",
        "difficulty": 0.65,
        "reason": "Two-pointer with running maxes; naive O(n^2) works, O(n) is the insight.",
    },
    {
        "problem": "Given a list of integers `nums`, return a new list with the elements in reverse order WITHOUT using slicing or the reversed() builtin.",
        "entry": "solve",
        "ref": "def solve(nums):\n    out = []\n    for i in range(len(nums)-1, -1, -1):\n        out.append(nums[i])\n    return out",
        "tests": [
            "assert solve([1,2,3]) == [3,2,1]",
            "assert solve([]) == []",
            "assert solve([7]) == [7]",
        ],
        "sample_input": [[1, 2, 3]],
        "empty_input": [[]],
        "expected_type": "list",
        "difficulty": 0.15,
        "reason": "Index iteration without reversed(); off-by-one on range bounds.",
    },
    {
        "problem": "Given a positive integer `n`, return the nth Catalan number. Definition: C(0)=1, C(n)=sum_{i=0..n-1} C(i)*C(n-1-i).",
        "entry": "solve",
        "ref": "def solve(n):\n    c = [0]*(n+1); c[0] = 1\n    for k in range(1, n+1):\n        c[k] = sum(c[i]*c[k-1-i] for i in range(k))\n    return c[n]",
        "tests": [
            "assert solve(0) == 1",
            "assert solve(1) == 1",
            "assert solve(4) == 14",
            "assert solve(5) == 42",
        ],
        "sample_input": [5],
        "empty_input": [0],
        "expected_type": "int",
        "difficulty": 0.45,
        "reason": "Convolution DP; forgetting C(0)=1 base case breaks everything.",
    },
    {
        "problem": "Given a string `s` containing only '(' and ')', return True iff the parentheses are balanced.",
        "entry": "solve",
        "ref": "def solve(s):\n    d = 0\n    for c in s:\n        d += 1 if c == '(' else -1\n        if d < 0: return False\n    return d == 0",
        "tests": [
            "assert solve('(())') == True",
            "assert solve(')(') == False",
            "assert solve('') == True",
            "assert solve('((') == False",
        ],
        "sample_input": ["(())"],
        "empty_input": [""],
        "expected_type": "bool",
        "difficulty": 0.2,
        "reason": "Counter with negativity guard; returning `d==0` without the guard misses ')('.",
    },
    {
        "problem": "Given a sorted list of integers `nums` (ascending, possibly empty) and a target `t`, return the index where `t` is found, or the index where it would be inserted to keep sorted order.",
        "entry": "solve",
        "ref": "def solve(nums, t):\n    lo, hi = 0, len(nums)\n    while lo < hi:\n        m = (lo+hi)//2\n        if nums[m] < t: lo = m+1\n        else: hi = m\n    return lo",
        "tests": [
            "assert solve([1,3,5,6], 5) == 2",
            "assert solve([1,3,5,6], 2) == 1",
            "assert solve([], 5) == 0",
            "assert solve([1,3,5,6], 7) == 4",
        ],
        "sample_input": [[1, 3, 5, 6], 5],
        "empty_input": [[], 0],
        "expected_type": "int",
        "difficulty": 0.3,
        "reason": "Lower-bound binary search; `hi = len(nums)` not `len-1` is the subtle bit.",
    },
    {
        "problem": "Given a positive integer `n`, return the list of its prime factors in ascending order (with multiplicity).",
        "entry": "solve",
        "ref": "def solve(n):\n    out = []; d = 2\n    while d*d <= n:\n        while n % d == 0:\n            out.append(d); n //= d\n        d += 1\n    if n > 1: out.append(n)\n    return out",
        "tests": [
            "assert solve(12) == [2,2,3]",
            "assert solve(7) == [7]",
            "assert solve(1) == []",
            "assert solve(60) == [2,2,3,5]",
        ],
        "sample_input": [60],
        "empty_input": [1],
        "expected_type": "list",
        "difficulty": 0.35,
        "reason": "Trial division with multiplicity; forgetting the trailing `if n>1` misses large primes.",
    },
    {
        "problem": "Given a list of integers `nums`, return True iff some value appears at least twice.",
        "entry": "solve",
        "ref": "def solve(nums):\n    seen = set()\n    for x in nums:\n        if x in seen: return True\n        seen.add(x)\n    return False",
        "tests": [
            "assert solve([1,2,3,1]) == True",
            "assert solve([1,2,3]) == False",
            "assert solve([]) == False",
        ],
        "sample_input": [[1, 2, 3, 1]],
        "empty_input": [[]],
        "expected_type": "bool",
        "difficulty": 0.15,
        "reason": "Set-membership scan; using list instead of set is a common perf trap.",
    },
    {
        "problem": "Given a list of integers `nums` of length ≥ 1, return the maximum sum of any contiguous non-empty subarray.",
        "entry": "solve",
        "ref": "def solve(nums):\n    best = cur = nums[0]\n    for x in nums[1:]:\n        cur = max(x, cur + x)\n        if cur > best: best = cur\n    return best",
        "tests": [
            "assert solve([-2,1,-3,4,-1,2,1,-5,4]) == 6",
            "assert solve([-1]) == -1",
            "assert solve([5,4,-1,7,8]) == 23",
            "assert solve([-3,-2,-5]) == -2",
        ],
        "sample_input": [[-2, 1, -3, 4, -1, 2, 1, -5, 4]],
        "empty_input": [[0]],
        "expected_type": "int",
        "difficulty": 0.4,
        "reason": "Kadane's algorithm; all-negative case trips naive `max(0, cur)` variants.",
    },
    {
        "problem": "Given a list `words` of lowercase strings, group anagrams together. Return a list of groups; groups may be returned in any order, and words within a group in any order.",
        "entry": "solve",
        "ref": "def solve(words):\n    from collections import defaultdict\n    groups = defaultdict(list)\n    for w in words:\n        groups[''.join(sorted(w))].append(w)\n    return [sorted(g) for g in groups.values()]",
        "tests": [
            "assert sorted(solve(['eat','tea','tan','ate','nat','bat'])) == [['ate','eat','tea'],['bat'],['nat','tan']]",
            "assert solve([]) == []",
            "assert solve(['']) == [['']]",
        ],
        "sample_input": [["eat", "tea", "tan", "ate", "nat", "bat"]],
        "empty_input": [[]],
        "expected_type": "list",
        "difficulty": 0.4,
        "reason": "Hash by sorted-letters signature; naive O(n^2) compare-all is quadratic.",
    },
    {
        "problem": "Given a list `nums` and an integer `k`, rotate the list to the right by `k` steps in place, then return it. `k` may exceed len(nums).",
        "entry": "solve",
        "ref": "def solve(nums, k):\n    n = len(nums)\n    if n == 0: return nums\n    k %= n\n    nums[:] = nums[-k:] + nums[:-k] if k else nums\n    return nums",
        "tests": [
            "assert solve([1,2,3,4,5], 2) == [4,5,1,2,3]",
            "assert solve([1,2,3], 5) == [2,3,1]",
            "assert solve([], 3) == []",
            "assert solve([1,2,3], 0) == [1,2,3]",
        ],
        "sample_input": [[1, 2, 3, 4, 5], 2],
        "empty_input": [[], 0],
        "expected_type": "list",
        "difficulty": 0.3,
        "reason": "k%=n handling; k=0 and empty-list edges trip zero-division-by-len variants.",
    },
]


def _format_completion(t: dict) -> str:
    tests_str = "\n".join(f"- {line}" for line in t["tests"])
    return (
        f"PROBLEM: {t['problem']}\n"
        f"ENTRY: {t['entry']}\n"
        "REFERENCE:\n"
        "```python\n"
        f"{t['ref']}\n"
        "```\n"
        f"TESTS:\n{tests_str}\n"
        f"EMPTY_INPUT: {json.dumps(t['empty_input'])}\n"
        f"SAMPLE_INPUT: {json.dumps(t['sample_input'])}\n"
        f"EXPECTED_TYPE: {t['expected_type']}\n"
        f"DIFFICULTY: {t['difficulty']}\n"
        f"DIFFICULTY_REASON: {t['reason']}\n"
    )


def main() -> None:
    out_path = pathlib.Path(__file__).resolve().parent.parent / "data" / "format_primer_pairs.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for t in _TASKS:
            rec = {"prompt": CODE_PROPOSAL_TEMPLATE, "completion": _format_completion(t)}
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(_TASKS)} pairs to {out_path}")


if __name__ == "__main__":
    main()
