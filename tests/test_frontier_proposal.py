"""Smoke tests for the frontier-biased code-proposal path.

Covers the three behavior changes introduced to raise proposal difficulty:

1. CODE_PROPOSAL_TEMPLATE carries an explicitly-harder few-shot example
   (coin-change DP, not "sum of even numbers") and an explicit frontier
   framing + DIFFICULTY field.
2. parse_code_proposal parses a DIFFICULTY float and rejects proposals
   below 0.3 (§3.2.1 pre-commit frontier threshold).
3. propose_batch_code runs a differential self-solve gate that drops
   proposals the model can solve blind against the proposal's own tests.

These are pure-Python tests (no GPU, no model) — they inject a stub
generate_fn to exercise the proposal plumbing end to end.
"""

from __future__ import annotations

import re

from src.generator.task_synthesizer import (
    CODE_PROPOSAL_TEMPLATE,
    TaskSynthesizer,
    _build_failure_seeded_prompt,
    parse_code_proposal,
)
from src.utils.config import SynthesisConfig


# --- Template content ------------------------------------------------------


def test_code_proposal_template_is_not_trivial():
    """The canonical few-shot example must not be the old trivial
    'sum of even numbers' example — that was the direct cause of the
    regression (model proposing problems it already trivially knows)."""
    tpl = CODE_PROPOSAL_TEMPLATE.lower()
    # Old trivial few-shot must be gone.
    assert "sum of all even numbers" not in tpl
    # New few-shot must exercise real algorithmic depth.
    assert "coins" in tpl and "amount" in tpl
    # Must mention algorithmic depth keywords.
    assert "dynamic programming" in tpl or "dp" in tpl.split()
    # Must explicitly demand frontier framing and ≥0.3 difficulty.
    assert "frontier" in tpl
    assert "0.3" in tpl
    # Must include the DIFFICULTY label the parser now requires.
    assert "DIFFICULTY:" in CODE_PROPOSAL_TEMPLATE


def test_failure_seeded_prompt_includes_inspiration():
    seeded = _build_failure_seeded_prompt(
        "Compute the number of distinct subsequences of S equal to T."
    )
    assert "INSPIRATION" in seeded
    assert "distinct subsequences" in seeded
    assert seeded.rstrip().endswith("YOUR OUTPUT:")


# --- Parser ----------------------------------------------------------------


_HARD_PROPOSAL = """\
PROBLEM: Given n intervals, return the minimum number of rooms needed.
ENTRY: solve
REFERENCE:
```python
def solve(intervals):
    import heapq
    intervals.sort()
    heap = []
    for s, e in intervals:
        if heap and heap[0] <= s:
            heapq.heappop(heap)
        heapq.heappush(heap, e)
    return len(heap)
```
TESTS:
- assert solve([[0, 30], [5, 10], [15, 20]]) == 2
- assert solve([]) == 0
- assert solve([[1, 2]]) == 1
EMPTY_INPUT: []
SAMPLE_INPUT: [[0, 30], [5, 10], [15, 20]]
EXPECTED_TYPE: int
DIFFICULTY: 0.55
DIFFICULTY_REASON: Requires recognizing the sweep-line / min-heap pattern.
"""


def test_parse_accepts_frontier_proposal():
    p = parse_code_proposal(_HARD_PROPOSAL)
    assert p.ok, p.issues
    assert abs(p.difficulty - 0.55) < 1e-6
    assert "sweep-line" in p.difficulty_reason
    assert len(p.tests) == 3


def test_parse_rejects_below_frontier_difficulty():
    raw = _HARD_PROPOSAL.replace("DIFFICULTY: 0.55", "DIFFICULTY: 0.10")
    p = parse_code_proposal(raw)
    assert not p.ok
    assert any("difficulty_below_frontier" in i for i in p.issues)


def test_parse_rejects_missing_difficulty_as_below_frontier():
    raw = re.sub(r"DIFFICULTY:.*\n", "", _HARD_PROPOSAL)
    raw = re.sub(r"DIFFICULTY_REASON:.*", "", raw)
    p = parse_code_proposal(raw)
    assert not p.ok
    assert any("difficulty_below_frontier" in i for i in p.issues)


# --- Differential self-solve gate -----------------------------------------


def _make_synth(gen_fn, *, gate: bool) -> TaskSynthesizer:
    cfg = SynthesisConfig()
    s = TaskSynthesizer(cfg, model_loader=None, generate_fn=gen_fn, run_vov=False)
    s._frontier_self_solve_gate = gate
    return s


def test_self_solve_gate_drops_already_known_problems():
    """If the model can blind-solve the problem and pass its own tests,
    the proposal is inside the model's capability and must be dropped."""
    # Model emits a well-formed HARD-LOOKING proposal whose tests the
    # blind solve trivially satisfies (identity function against
    # identity assertions) — simulates "model proposes a problem it
    # already knows how to solve".
    calls = {"n": 0}
    easy_proposal = """\
PROBLEM: Return the input unchanged.
ENTRY: solve
REFERENCE:
```python
def solve(x):
    return x
```
TESTS:
- assert solve(1) == 1
- assert solve(2) == 2
DIFFICULTY: 0.5
DIFFICULTY_REASON: pretend-hard
"""
    blind_solve = "```python\ndef solve(x):\n    return x\n```\n"

    def gen_fn(prompt: str) -> str:
        calls["n"] += 1
        # First N calls = proposals, remainder = blind self-solves.
        if "Solve the following problem" in prompt:
            return blind_solve
        return easy_proposal

    s = _make_synth(gen_fn, gate=True)
    kept = s.propose_batch_code(2)
    assert kept == [], "frontier gate should drop blind-solvable proposals"


def test_self_solve_gate_keeps_frontier_problems():
    """If the blind solve FAILS the proposal's tests, the proposal is
    at the frontier and must be kept."""
    proposal = """\
PROBLEM: Return twice the input.
ENTRY: solve
REFERENCE:
```python
def solve(x):
    return 2 * x
```
TESTS:
- assert solve(1) == 2
- assert solve(3) == 6
DIFFICULTY: 0.4
DIFFICULTY_REASON: frontier-simulated
"""
    # Blind solve returns the WRONG function (returns x instead of 2x),
    # so the proposal's tests will fail → proposal is kept.
    blind_solve = "```python\ndef solve(x):\n    return x\n```\n"

    def gen_fn(prompt: str) -> str:
        if "Solve the following problem" in prompt:
            return blind_solve
        return proposal

    s = _make_synth(gen_fn, gate=True)
    kept = s.propose_batch_code(1)
    assert len(kept) == 1
    assert kept[0].declared_difficulty == 0.4


def test_gate_can_be_disabled():
    """With the gate off, even a model-already-knows proposal is kept
    (useful for unit tests that don't want the extra generate call)."""
    easy_proposal = """\
PROBLEM: Return the input unchanged.
ENTRY: solve
REFERENCE:
```python
def solve(x):
    return x
```
TESTS:
- assert solve(1) == 1
- assert solve(2) == 2
DIFFICULTY: 0.5
DIFFICULTY_REASON: pretend
"""

    def gen_fn(prompt: str) -> str:
        return easy_proposal

    s = _make_synth(gen_fn, gate=False)
    kept = s.propose_batch_code(1)
    assert len(kept) == 1
