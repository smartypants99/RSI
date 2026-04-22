"""Tests for DifficultyTracker (curriculum escalation)."""

from __future__ import annotations

import json
from pathlib import Path

from src.diagnostics.difficulty_tracker import DifficultyTracker


def _pq(domain, subdomain, correct):
    return {"domain": domain, "subdomain": subdomain, "correct": correct}


def test_record_heldout_tracks_subdomain_stats():
    t = DifficultyTracker()
    t.record_heldout([
        _pq("code", "implementation", True),
        _pq("code", "implementation", False),
        _pq("math", "calculus", False),
    ])
    stats = t.to_dict()["subdomain_stats"]
    assert stats["code/implementation"] == {"attempts": 2, "correct": 1}
    assert stats["math/calculus"] == {"attempts": 1, "correct": 0}
    assert t.cycles_recorded == 1


def test_frontier_picks_highest_accuracy_among_failed():
    t = DifficultyTracker()
    # Build history: code/impl strong, math/calc weak. Last cycle fails both.
    for _ in range(9):
        t.record_heldout([_pq("code", "implementation", True)])
        t.record_heldout([_pq("math", "calculus", False)])
    # Now a cycle where both subdomains are wrong -> frontier = easier one.
    t.record_heldout([
        _pq("code", "implementation", False),
        _pq("math", "calculus", False),
    ])
    assert t.frontier() == "code/implementation"


def test_frontier_empty_when_no_data():
    assert DifficultyTracker().frontier() == ""


def test_frontier_fallback_to_weakest_when_all_last_correct():
    t = DifficultyTracker()
    # code perfect, math partial; last cycle all right → fallback = weakest non-perfect.
    t.record_heldout([_pq("code", "impl", True), _pq("math", "calc", False)])
    t.record_heldout([_pq("code", "impl", True), _pq("math", "calc", True)])
    # Last cycle has no wrong; fallback to lowest-accuracy (math/calc = 0.5).
    assert t.frontier() == "math/calc"


def test_ratchet_raises_floor_on_improvement():
    t = DifficultyTracker()
    assert t.difficulty_floor == 0.0
    new = t.update_ratchet(0.02, cycle=1)
    assert abs(new - 0.05) < 1e-9
    assert len(t.ratchet_history) == 1


def test_ratchet_lowers_floor_on_regression():
    t = DifficultyTracker()
    t.difficulty_floor = 0.3
    new = t.update_ratchet(-0.02, cycle=2)
    assert abs(new - 0.25) < 1e-9


def test_ratchet_no_change_within_deadband():
    t = DifficultyTracker()
    t.difficulty_floor = 0.4
    assert t.update_ratchet(0.005) == 0.4
    assert t.update_ratchet(-0.009) == 0.4
    assert t.ratchet_history == []


def test_ratchet_caps_at_090_and_floors_at_000():
    t = DifficultyTracker()
    t.difficulty_floor = 0.88
    assert abs(t.update_ratchet(0.5) - 0.9) < 1e-9
    # Further improvement cannot push above the cap.
    assert abs(t.update_ratchet(0.5) - 0.9) < 1e-9
    t.difficulty_floor = 0.03
    assert abs(t.update_ratchet(-0.5) - 0.0) < 1e-9
    assert abs(t.update_ratchet(-0.5) - 0.0) < 1e-9


def test_record_proposals_accumulates():
    t = DifficultyTracker()
    t.record_proposals(5, 3)
    t.record_proposals(2, 7)
    assert t.last_accepted == 2
    assert t.last_rejected == 7
    assert t.proposals_accepted_total == 7
    assert t.proposals_rejected_total == 10


def test_persistence_roundtrip(tmp_path: Path):
    path = tmp_path / "difficulty_state.json"
    t = DifficultyTracker(state_path=path)
    t.record_heldout([_pq("code", "impl", True), _pq("math", "calc", False)])
    t.record_proposals(4, 1)
    t.update_ratchet(0.02, cycle=1)
    t.save()
    assert path.exists()

    t2 = DifficultyTracker.load_or_new(path)
    assert t2.difficulty_floor == t.difficulty_floor
    assert t2.proposals_accepted_total == 4
    assert t2.cycles_recorded == 1
    assert t2.frontier() == "math/calc"  # the subdomain wrong last cycle


def test_load_or_new_missing_file_returns_fresh(tmp_path: Path):
    path = tmp_path / "does_not_exist.json"
    t = DifficultyTracker.load_or_new(path)
    assert t.difficulty_floor == 0.0
    assert t.state_path == path


def test_load_or_new_corrupt_file_falls_back(tmp_path: Path):
    path = tmp_path / "corrupt.json"
    path.write_text("{not valid json")
    t = DifficultyTracker.load_or_new(path)
    assert t.difficulty_floor == 0.0
    assert t.state_path == path


def test_frontier_domain_scope_filters_cross_domain_drift():
    """domain='code' must never return a math/* subdomain even when math
    has the globally lowest accuracy. Regression guard for the overnight-run
    bug where frontier() returned math/percentage and was spliced into the
    code-only propose_batch_code prompt."""
    t = DifficultyTracker()
    # Build history: math has many more failures than code.
    for _ in range(10):
        t.record_heldout([_pq("math", "percentage", False)])
        t.record_heldout([_pq("code", "implementation", False)])
        t.record_heldout([_pq("code", "implementation", True)])
    # Last cycle: both wrong. Global frontier would pick whichever has
    # higher historical accuracy (code/implementation is ~33% vs math 0%),
    # but we explicitly request domain="code" anyway to lock the contract.
    t.record_heldout([
        _pq("code", "implementation", False),
        _pq("math", "percentage", False),
    ])
    # Unscoped call: picks higher-accuracy failing subdomain (code).
    assert t.frontier().startswith("code/")
    # Scoped call: must stay in code regardless of math stats.
    assert t.frontier(domain="code").startswith("code/")
    # Scoped to math: must return math.
    assert t.frontier(domain="math").startswith("math/")
    # Scoped to a domain with NO data: empty string (no spurious match).
    assert t.frontier(domain="physics") == ""


def test_frontier_domain_scope_aggregate_fallback():
    """When last-cycle-wrong has nothing in the requested domain, the
    aggregate-stats fallback must also respect the domain scope rather
    than leaking the unscoped weakest subdomain."""
    t = DifficultyTracker()
    # code has a weak subdomain in history; last cycle only math fails.
    for _ in range(5):
        t.record_heldout([_pq("code", "recursion", False)])
        t.record_heldout([_pq("math", "algebra", True)])
    t.record_heldout([_pq("math", "algebra", False)])  # last-cycle wrong is math only
    # Scoped to code: last-cycle-wrong has no code entry → falls back to
    # aggregate weakest code subdomain, which must still be code/*.
    assert t.frontier(domain="code") == "code/recursion"


def test_ratchet_history_records_deltas():
    t = DifficultyTracker()
    t.update_ratchet(0.02, cycle=1)
    t.update_ratchet(-0.02, cycle=2)
    assert len(t.ratchet_history) == 2
    assert t.ratchet_history[0]["floor_after"] > t.ratchet_history[0]["floor_before"]
    assert t.ratchet_history[1]["floor_after"] < t.ratchet_history[1]["floor_before"]
