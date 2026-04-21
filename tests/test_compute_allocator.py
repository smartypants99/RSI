"""Tests for self-compute-allocation (bandit over allocation strategies)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.orchestrator.compute_allocator import (
    AllocationOutcome,
    AllocationStrategy,
    ComputeAllocator,
    allocator_from_history,
    append_outcome,
    default_strategies,
    load_outcomes,
)


def _s(name: str, **over) -> AllocationStrategy:
    base = dict(
        k_candidates=4,
        token_budget=1024,
        branch_vs_confirm="branch",
        fast_student=False,
        train_mode="sft",
    )
    base.update(over)
    return AllocationStrategy(name=name, **base)


# ------------------------------ strategy validation ----------------------


def test_strategy_rejects_bad_knobs():
    with pytest.raises(ValueError):
        _s("x", k_candidates=0)
    with pytest.raises(ValueError):
        _s("x", token_budget=0)
    with pytest.raises(ValueError):
        _s("x", branch_vs_confirm="sideways")
    with pytest.raises(ValueError):
        _s("x", train_mode="ppo")


def test_strategy_cost_grows_with_knobs():
    a = _s("a", k_candidates=4, token_budget=1024)
    b = _s("b", k_candidates=8, token_budget=1024)
    assert b.expected_cost() > a.expected_cost()


def test_strategy_fast_student_is_cheaper():
    slow = _s("slow", fast_student=False)
    fast = _s("fast", fast_student=True)
    assert fast.expected_cost() < slow.expected_cost()


# ------------------------------ allocator --------------------------------


def test_allocator_requires_unique_names():
    with pytest.raises(ValueError):
        ComputeAllocator(strategies=(_s("a"), _s("a")))


def test_allocator_untried_strategy_wins_first():
    strategies = (_s("a"), _s("b"), _s("c"))
    alloc = ComputeAllocator(strategies=strategies)
    picked = alloc.select(remaining_budget=1e12)
    assert picked.name == "a"  # first untried


def test_allocator_prefers_higher_reward_once_all_tried():
    strategies = (_s("a"), _s("b"))
    alloc = ComputeAllocator(strategies=strategies)
    # Give b strong positive rewards, a negative.
    for i in range(5):
        alloc.record(AllocationOutcome(cycle_id=i, strategy_name="a",
                                       held_out_delta=-0.001, compute_used=1000.0))
        alloc.record(AllocationOutcome(cycle_id=i, strategy_name="b",
                                       held_out_delta=0.01, compute_used=1000.0))
    picked = alloc.select(remaining_budget=1e12)
    assert picked.name == "b"


def test_allocator_filters_over_budget_strategies():
    cheap = _s("cheap", k_candidates=1, token_budget=100)
    expensive = _s("expensive", k_candidates=16, token_budget=8192)
    alloc = ComputeAllocator(strategies=(cheap, expensive))
    picked = alloc.select(remaining_budget=cheap.expected_cost())
    assert picked.name == "cheap"


def test_allocator_falls_back_to_cheapest_when_all_over_budget():
    a = _s("a", k_candidates=2, token_budget=500)
    b = _s("b", k_candidates=4, token_budget=1024)
    alloc = ComputeAllocator(strategies=(a, b))
    picked = alloc.select(remaining_budget=1.0)  # nothing fits
    # cheapest expected_cost wins
    assert picked.name == "a"


def test_allocator_zero_budget_returns_cheapest():
    a = _s("a", k_candidates=2, token_budget=500)
    b = _s("b", k_candidates=4, token_budget=1024)
    alloc = ComputeAllocator(strategies=(a, b))
    picked = alloc.select(remaining_budget=0)
    assert picked.name == "a"


def test_allocator_ranking_orders_by_mean():
    alloc = ComputeAllocator(strategies=(_s("a"), _s("b"), _s("c")))
    alloc.record(AllocationOutcome(0, "a", 0.02, 1000.0))
    alloc.record(AllocationOutcome(1, "b", 0.001, 1000.0))
    alloc.record(AllocationOutcome(2, "c", -0.01, 1000.0))
    ranking = alloc.ranking()
    names_in_order = [r[0] for r in ranking]
    assert names_in_order == ["a", "b", "c"]


def test_outcome_reward_handles_zero_compute():
    o = AllocationOutcome(cycle_id=0, strategy_name="x",
                          held_out_delta=0.1, compute_used=0.0)
    assert o.reward() == 0.0


def test_outcome_reward_is_clipped():
    o = AllocationOutcome(cycle_id=0, strategy_name="x",
                          held_out_delta=1e9, compute_used=1.0)
    assert o.reward() == 1.0  # REWARD_CLIP hi
    o2 = AllocationOutcome(cycle_id=0, strategy_name="x",
                           held_out_delta=-1e9, compute_used=1.0)
    assert o2.reward() == -1.0


# ------------------------------ persistence ------------------------------


def test_outcomes_roundtrip(tmp_path: Path):
    path = tmp_path / "outcomes.jsonl"
    outs = [
        AllocationOutcome(0, "a", 0.01, 1000.0, wall_time_s=12.3),
        AllocationOutcome(1, "b", -0.005, 800.0),
    ]
    for o in outs:
        append_outcome(path, o)
    loaded = load_outcomes(path)
    assert len(loaded) == 2
    assert loaded[0].strategy_name == "a"
    assert loaded[1].held_out_delta == pytest.approx(-0.005)


def test_load_outcomes_ignores_garbage_lines(tmp_path: Path):
    path = tmp_path / "o.jsonl"
    path.write_text("not json\n" + json.dumps({
        "cycle_id": 0, "strategy_name": "a",
        "held_out_delta": 0.1, "compute_used": 1.0,
    }) + "\n\n")
    loaded = load_outcomes(path)
    assert len(loaded) == 1


def test_allocator_from_history_replays_known_strategies(tmp_path: Path):
    path = tmp_path / "o.jsonl"
    append_outcome(path, AllocationOutcome(0, "baseline_sft", 0.01, 1000.0))
    append_outcome(path, AllocationOutcome(1, "ghost_strategy", 0.5, 1000.0))
    alloc = allocator_from_history(path)
    # ghost strategy filtered out because it's not in default_strategies().
    assert len(alloc.history) == 1
    assert alloc.history[0].strategy_name == "baseline_sft"


def test_default_strategies_nonempty_and_unique():
    ds = default_strategies()
    assert len(ds) >= 3
    names = [s.name for s in ds]
    assert len(set(names)) == len(names)
