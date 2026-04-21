"""Tests for src.orchestrator.meta_meta — contribution estimation, tier
promotion, and revert-on-negative."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.orchestrator import meta_meta as mm


def _rec(cycle: int, delta: float, **active: bool) -> mm.CycleRecord:
    comps = {k: False for k in mm.COMPONENT_KEYS}
    comps.update(active)
    tier = active.pop("_tier", None) if "_tier" in active else None
    return mm.CycleRecord(
        cycle_id=cycle,
        components_active=comps,
        held_out_delta=delta,
        self_edit_tier=tier,
    )


def _se(cycle: int, delta: float, tier: int = 0, **others: bool) -> mm.CycleRecord:
    comps = {k: False for k in mm.COMPONENT_KEYS}
    comps["self_edit"] = True
    comps.update(others)
    return mm.CycleRecord(
        cycle_id=cycle,
        components_active=comps,
        held_out_delta=delta,
        self_edit_tier=tier,
    )


# --- Contribution estimation --------------------------------------------------


def test_contribution_difference_of_means():
    history = [
        _rec(1, 0.05, fast_student=True),
        _rec(2, 0.04, fast_student=True),
        _rec(3, 0.01),
        _rec(4, -0.01),
    ]
    out = mm.component_contributions(history, components=["fast_student"])
    assert out["fast_student"]["active_n"] == 2
    assert out["fast_student"]["inactive_n"] == 2
    assert out["fast_student"]["active_mean"] == pytest.approx(0.045)
    assert out["fast_student"]["inactive_mean"] == pytest.approx(0.0)
    assert out["fast_student"]["contribution"] == pytest.approx(0.045)


def test_contribution_none_when_arm_empty():
    history = [_rec(1, 0.05, fast_student=True)]
    out = mm.component_contributions(history, components=["fast_student", "ood"])
    assert out["fast_student"]["contribution"] is None  # no inactive arm
    # ood never active in history: active arm empty
    assert out["ood"]["active_n"] == 0
    assert out["ood"]["contribution"] is None


# --- Tier promotion -----------------------------------------------------------


def test_tier2_unlocks_after_5_successful_self_edits():
    history = [_se(i, 0.01, tier=0) for i in range(1, 6)]
    state = mm.compute_tier_state(history, current_cycle=6)
    assert 0 in state.unlocked
    assert 2 in state.unlocked
    assert 3 not in state.unlocked


def test_tier2_not_unlocked_below_threshold_count():
    history = [_se(i, 0.01, tier=0) for i in range(1, 5)]  # only 4 wins
    state = mm.compute_tier_state(history, current_cycle=5)
    assert state.unlocked == [0]


def test_tier2_not_unlocked_if_mean_delta_too_small():
    history = [_se(i, 0.0005, tier=0) for i in range(1, 10)]  # 9 wins but tiny mean
    state = mm.compute_tier_state(history, current_cycle=10)
    assert 2 not in state.unlocked


def test_tier_graduation_sequential():
    history = [_se(i, 0.01, tier=0) for i in range(1, 11)]
    state = mm.compute_tier_state(history, current_cycle=11)
    # 10 successful cycles -> tier 2 AND tier 3 unlocked
    assert 2 in state.unlocked
    assert 3 in state.unlocked
    assert 4 not in state.unlocked


def test_effective_allow_list_tier0_only_initially():
    assert mm.effective_allow_list([], current_cycle=0) == mm.TIER_0


def test_effective_allow_list_expands_with_tier2():
    history = [_se(i, 0.01, tier=0) for i in range(1, 6)]
    globs = mm.effective_allow_list(history, current_cycle=6)
    for g in mm.TIER_0 + mm.TIER_2:
        assert g in globs


def test_effective_allow_list_never_overlaps_hard_deny():
    from src.orchestrator.self_edit import HARD_DENY_LIST
    history = [_se(i, 0.01, tier=0) for i in range(1, 25)]
    globs = mm.effective_allow_list(history, current_cycle=26)
    banned_prefixes = (
        "src/orchestrator/loop.py", "src/utils/config.py", "src/trainer/",
        "src/safety/", "src/orchestrator/self_edit.py",
        "src/orchestrator/meta_meta.py", "tests/", "run.sh",
    )
    for g in globs:
        for bp in banned_prefixes:
            assert not g.startswith(bp), f"tier glob {g} collides with HARD_DENY {bp}"
    assert "src/orchestrator/loop.py" in HARD_DENY_LIST
    assert "src/trainer/*" in HARD_DENY_LIST


def test_tier4_is_human_gated_not_auto_unlocked():
    history = [_se(i, 0.01, tier=0) for i in range(1, 25)]
    qualified = mm.qualified_tiers(history, current_cycle=26)
    assert 4 in qualified
    globs = mm.effective_allow_list(history, current_cycle=26)
    for g in mm.TIER_4_PROPOSAL:
        assert g not in globs
    assert mm.tier_requires_human_approval(4) is True
    assert mm.tier_requires_human_approval(2) is False


def test_audit_log_records_tier_events(tmp_path: Path):
    ap = tmp_path / "update-log.txt"
    mm.append_audit_log(ap, cycle_id=7, event="tier_unlocked", tier=2)
    mm.append_audit_log(ap, cycle_id=9, event="tier_reverted", tier=2, detail="2x<0")
    text = ap.read_text()
    assert "event=tier_unlocked tier=2" in text
    assert "event=tier_reverted tier=2" in text
    assert "cycle=9" in text


# --- Revert on negative -------------------------------------------------------


def test_tier_reverted_after_two_consecutive_bad_patches():
    # Unlock tier 2, then two bad tier-2 patches in a row.
    history = [_se(i, 0.01, tier=0) for i in range(1, 6)]
    # After cycle 6 tier 2 is unlocked. Now two consecutive bad tier-2.
    history.append(_se(7, -0.02, tier=2))
    history.append(_se(8, -0.03, tier=2))
    state = mm.compute_tier_state(history, current_cycle=9)
    assert 2 not in state.unlocked
    assert state.cooldown_until.get(2) == 8 + mm.REVERT_COOLDOWN


def test_tier_stays_unlocked_if_bad_not_consecutive():
    history = [_se(i, 0.01, tier=0) for i in range(1, 6)]
    history.append(_se(7, -0.02, tier=2))
    history.append(_se(8, 0.01, tier=2))  # good patch breaks the streak
    history.append(_se(9, -0.02, tier=2))
    state = mm.compute_tier_state(history, current_cycle=10)
    assert 2 in state.unlocked


def test_cooldown_blocks_re_promotion():
    # Build up, revert, then try to re-qualify immediately — should stay locked
    # until the cooldown window elapses.
    history = [_se(i, 0.01, tier=0) for i in range(1, 6)]
    history.append(_se(7, -0.02, tier=2))
    history.append(_se(8, -0.03, tier=2))  # tier 2 reverted at cycle 8
    # Add more wins inside the cooldown window (until cycle 13).
    for i in range(9, 13):
        history.append(_se(i, 0.01, tier=0))
    state = mm.compute_tier_state(history, current_cycle=13)
    assert 2 not in state.unlocked


# --- Persistence --------------------------------------------------------------


def test_record_and_load_roundtrip(tmp_path: Path):
    hp = tmp_path / "meta_meta.jsonl"
    mm.record_cycle(hp, cycle_id=1, components_active={"fast_student": True}, held_out_delta=0.03)
    mm.record_cycle(hp, cycle_id=2, components_active={"self_edit": True}, held_out_delta=-0.01, self_edit_tier=0)
    loaded = mm.load_history(hp)
    assert len(loaded) == 2
    assert loaded[0].components_active["fast_student"] is True
    assert loaded[1].self_edit_tier == 0
    assert loaded[1].held_out_delta == pytest.approx(-0.01)


def test_wall_time_record_and_trend(tmp_path: Path):
    wp = tmp_path / "wall.jsonl"
    # 10 cycles: older 5 avg 1000ms/cycle, newer 5 avg 500ms/cycle → down 50%.
    for c in range(1, 6):
        mm.record_wall_time(wp, c, "solve", 1000.0)
    for c in range(6, 11):
        mm.record_wall_time(wp, c, "solve", 500.0)
    recs = mm.load_wall_time(wp)
    assert len(recs) == 10
    trend = mm.wall_time_trend(recs, window=10)
    assert trend is not None
    assert trend["pct_change_down"] == pytest.approx(50.0)
    assert trend["mean_ms_older"] == pytest.approx(1000.0)
    assert trend["mean_ms_newer"] == pytest.approx(500.0)


def test_wall_time_trend_returns_none_below_window(tmp_path: Path):
    wp = tmp_path / "wall.jsonl"
    for c in range(1, 5):
        mm.record_wall_time(wp, c, "x", 10.0)
    assert mm.wall_time_trend(mm.load_wall_time(wp), window=10) is None


def test_wall_time_aggregates_phases_per_cycle(tmp_path: Path):
    wp = tmp_path / "wall.jsonl"
    # Two phases per cycle, each contributes to the cycle total.
    mm.record_wall_time(wp, 1, "solve", 300.0)
    mm.record_wall_time(wp, 1, "verify", 200.0)
    recs = mm.load_wall_time(wp)
    assert len(recs) == 2
    # Trend requires ≥window cycles; assert aggregation via direct sum.
    total = sum(r.ms for r in recs if r.cycle_id == 1)
    assert total == pytest.approx(500.0)


def test_load_history_tolerates_corrupt_lines(tmp_path: Path):
    hp = tmp_path / "meta_meta.jsonl"
    hp.write_text(
        '{"cycle_id": 1, "components_active": {"fast_student": true}, "held_out_delta": 0.02}\n'
        "not-json\n"
        '{"cycle_id": 2, "components_active": {}, "held_out_delta": -0.01}\n'
    )
    loaded = mm.load_history(hp)
    assert [r.cycle_id for r in loaded] == [1, 2]


def test_current_self_edit_tier_reflects_unlocks():
    history = [_se(i, 0.01, tier=0) for i in range(1, 11)]
    assert mm.current_self_edit_tier(history, current_cycle=11) == 3
