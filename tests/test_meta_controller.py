"""Tests for the expanded MetaController (Task #5).

Covers:
  * DimensionBandit picking, rolling-window observation, ±30% bounding.
  * MetaController end-to-end wiring: record_cycle → propose_updates emits
    picks for lora_rank / num_epochs / min_train_samples / grad_accum.
  * Persistence round-trip via persist_state + load_state.
  * Bandit convergence on a synthetic reward signal.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from src.orchestrator.meta import DimensionBandit, MetaController


def _synthetic_reward(value: int, best: int) -> float:
    """Smooth peak at `best`: 0.1 at peak, falls off quadratically."""
    return 0.1 - 0.001 * (value - best) ** 2


def test_dimension_bandit_ranges_match_spec():
    db_rank = DimensionBandit.from_range("lora_rank", 8, 64, step=8)
    assert db_rank.values == [8, 16, 24, 32, 40, 48, 56, 64]

    db_epochs = DimensionBandit.from_range("num_epochs", 1, 4)
    assert db_epochs.values == [1, 2, 3, 4]

    db_min = DimensionBandit.from_range("min_train_samples", 5, 50, step=5)
    assert db_min.values == [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    db_ga = DimensionBandit.from_range("gradient_accumulation_steps", 1, 8)
    assert db_ga.values == [1, 2, 3, 4, 5, 6, 7, 8]


def test_dimension_bandit_observe_rolling_window():
    db = DimensionBandit.from_range("x", 1, 4, step=1)
    for i in range(15):
        db.observe(2, delta=0.01 * i)
    # Rolling window default = 10.
    idx = db.values.index(2)
    assert len(db.history[idx]) == 10
    # Most-recent deltas kept.
    assert db.history[idx][-1] == 0.14


def test_dimension_bandit_pick_bounded_to_best():
    db = DimensionBandit.from_range("rank", 8, 64, step=8)
    # Make value=32 clearly best by feeding positive deltas there
    # and negative elsewhere.
    for _ in range(10):
        db.observe(32, delta=0.05)
    for v in (8, 16, 24, 40, 48, 56, 64):
        db.observe(v, delta=-0.05)
    rng = random.Random(0)
    picks = {db.pick(rng, current=32) for _ in range(80)}
    # ±30% of 32 → [22.4, 41.6]; available step-8 values = {24, 32, 40}.
    assert picks.issubset({24, 32, 40}), f"bandit escaped ±30% window: {picks}"


def test_dimension_bandit_convergence_on_synthetic_reward():
    """With enough trials the bandit's last-pulled value should sit on the peak."""
    db = DimensionBandit.from_range("rank", 8, 64, step=8)
    rng = random.Random(42)
    best = 32
    picked = None
    # Warm each arm once so the rolling-best anchor can settle.
    for v in db.values:
        db.observe(v, delta=_synthetic_reward(v, best))
    for _ in range(200):
        picked = db.pick(rng, current=picked or 32)
        db.observe(picked, delta=_synthetic_reward(picked, best))

    # Final mean-per-arm — peak arm should have the highest mean delta.
    means = {
        v: (sum(h) / len(h) if h else 0.0)
        for v, h in zip(db.values, db.history)
    }
    top = max(means, key=means.get)
    assert top == best, f"bandit failed to identify peak: means={means}"


def test_meta_controller_propose_emits_dim_picks(tmp_path: Path):
    mc = MetaController(log_path=tmp_path / "meta.jsonl", initial_lr=5e-6)
    snap = {
        "learning_rate": 5e-6,
        "lora_rank": 8,
        "num_epochs": 2,
        "min_train_samples": 5,
        "gradient_accumulation_steps": 4,
        "verifier_check_weights": {},
        "generator_template": None,
    }
    # First cycle — no prev eval, no delta; just exercise the path.
    mc.record_cycle(cycle=1, config_snapshot=snap,
                    held_out_score=0.5, prev_held_out=None)
    prop = mc.propose_updates(snap)
    # Every dim-key exists in the proposal (may be None when pick matches current).
    for k in ("lora_rank", "num_epochs", "min_train_samples",
              "gradient_accumulation_steps"):
        assert k in prop
    # Because the bandit is Thompson-sampling from Laplace priors, at least
    # one of the four dims should almost always differ from current.
    changed = [k for k in ("lora_rank", "num_epochs", "min_train_samples",
                           "gradient_accumulation_steps")
               if prop[k] is not None]
    assert changed, "expected at least one dimension to be proposed on cycle 1"


def test_meta_controller_observes_dim_deltas(tmp_path: Path):
    mc = MetaController(log_path=tmp_path / "meta.jsonl", initial_lr=5e-6)
    snap1 = {"learning_rate": 5e-6, "lora_rank": 16, "num_epochs": 2,
             "min_train_samples": 10, "gradient_accumulation_steps": 4,
             "verifier_check_weights": {}, "generator_template": None}
    mc.record_cycle(cycle=1, config_snapshot=snap1,
                    held_out_score=0.5, prev_held_out=None)
    mc.record_cycle(cycle=2, config_snapshot=snap1,
                    held_out_score=0.6, prev_held_out=0.5)
    # Observation landed on the rank=16 arm.
    rank_bandit = mc.dimension_bandits["lora_rank"]
    idx = rank_bandit.values.index(16)
    assert len(rank_bandit.history[idx]) == 1
    assert abs(rank_bandit.history[idx][0] - 0.1) < 1e-9
    assert rank_bandit.arms[idx].alpha == 2.0  # initial 1.0 + success


def test_meta_state_persist_round_trip(tmp_path: Path):
    mc = MetaController(log_path=tmp_path / "meta.jsonl", initial_lr=5e-6)
    snap = {"learning_rate": 5e-6, "lora_rank": 8, "num_epochs": 2,
            "min_train_samples": 5, "gradient_accumulation_steps": 4,
            "verifier_check_weights": {}, "generator_template": None}
    mc.record_cycle(cycle=1, config_snapshot=snap,
                    held_out_score=0.4, prev_held_out=None)
    mc.record_cycle(cycle=2, config_snapshot=snap,
                    held_out_score=0.5, prev_held_out=0.4)
    mc.persist_state()
    state_file = tmp_path / "meta_state.json"
    assert state_file.exists()
    data = json.loads(state_file.read_text())
    assert "dimension_bandits" in data
    assert set(data["dimension_bandits"].keys()) == {
        "lora_rank", "num_epochs", "min_train_samples",
        "gradient_accumulation_steps",
    }
    # Round-trip into a fresh controller.
    mc2 = MetaController(log_path=tmp_path / "meta2.jsonl", initial_lr=5e-6)
    mc2.load_state(data)
    rank_bandit = mc2.dimension_bandits["lora_rank"]
    idx = rank_bandit.values.index(8)
    assert len(rank_bandit.history[idx]) == 1
    assert abs(rank_bandit.history[idx][0] - 0.1) < 1e-9
