"""Tests for src/generator/reasoning_strategies.py (Task #1B)."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from src.generator.reasoning_strategies import (
    ReasoningStrategy,
    SEED_STRATEGIES,
    StrategyLibrary,
    parse_strategies_from_model_output,
    wilson_lower_bound,
)
from src.utils.config import SynthesisConfig


# ───────── config additions ─────────

def test_synthesis_config_has_strategy_fields():
    c = SynthesisConfig()
    assert hasattr(c, "strategy_library_enabled")
    assert hasattr(c, "strategy_ab_holdout_size")
    assert c.strategy_library_enabled is False   # default off
    assert c.strategy_ab_holdout_size >= 0


def test_synthesis_config_validates_strategy_fields():
    with pytest.raises(ValueError):
        SynthesisConfig(strategy_ab_holdout_size=-1)
    with pytest.raises(ValueError):
        SynthesisConfig(strategy_library_k_few_shot=-1)


# ───────── ReasoningStrategy dataclass ─────────

def test_strategy_basic_construction_fills_hash():
    s = ReasoningStrategy(name="my-strat", template="do X, then Y")
    assert s.template_hash
    assert len(s.template_hash) == 16


def test_strategy_rejects_bad_name():
    with pytest.raises(ValueError):
        ReasoningStrategy(name="BadName!", template="t")


def test_strategy_rejects_empty_template():
    with pytest.raises(ValueError):
        ReasoningStrategy(name="a-b", template="   ")


def test_strategy_rejects_oversize_template():
    with pytest.raises(ValueError):
        ReasoningStrategy(name="a-b", template="x" * 5000)


def test_strategy_rejects_bad_origin():
    with pytest.raises(ValueError):
        ReasoningStrategy(name="a-b", template="t", origin="weird")


def test_strategy_scores_handle_zero_trials():
    s = ReasoningStrategy(name="zero", template="t")
    assert s.holdout_score == 0.0
    assert s.prod_score == 0.0
    assert s.blended_score == 0.0


def test_wilson_lower_bound_ordering():
    # More trials with same rate → higher lower bound.
    a = wilson_lower_bound(5, 10)
    b = wilson_lower_bound(50, 100)
    assert 0.0 < a < b


# ───────── StrategyLibrary ─────────

def test_library_seeds_when_empty(tmp_path: Path):
    lib = StrategyLibrary(tmp_path / "strats.jsonl").load()
    assert len(lib) == len(SEED_STRATEGIES)
    names = {s.name for s in lib}
    for seed in SEED_STRATEGIES:
        assert seed["name"] in names


def test_library_persists_roundtrip(tmp_path: Path):
    path = tmp_path / "strats.jsonl"
    lib = StrategyLibrary(path).load()
    ok = lib.propose_from_model("new-strat", "Novel template text.")
    assert ok is not None
    lib.save()
    lib2 = StrategyLibrary(path, seed_if_empty=False).load()
    assert lib2.get("new-strat") is not None
    assert lib2.get("new-strat").origin == "model"


def test_library_dedupes_identical_templates(tmp_path: Path):
    lib = StrategyLibrary(tmp_path / "s.jsonl").load()
    first = lib.propose_from_model("variant-a", "EXACT SAME TEMPLATE BODY")
    assert first is not None
    second = lib.propose_from_model("variant-b", "EXACT SAME TEMPLATE BODY")
    assert second is None  # rejected as duplicate hash


def test_library_propose_rejects_malformed(tmp_path: Path):
    lib = StrategyLibrary(tmp_path / "s.jsonl").load()
    assert lib.propose_from_model("Bad Name", "t") is None
    assert lib.propose_from_model("good-name", "") is None


def test_library_record_result_and_scores(tmp_path: Path):
    lib = StrategyLibrary(tmp_path / "s.jsonl").load()
    lib.propose_from_model("alpha", "Try X first then Y.")
    for _ in range(4):
        lib.record_result("alpha", success=True, holdout=True)
    s = lib.get("alpha")
    assert s.holdout_trials == 4
    assert s.holdout_successes == 4
    assert s.holdout_score > 0.3


def test_library_ab_admit_requires_min_trials(tmp_path: Path):
    lib = StrategyLibrary(tmp_path / "s.jsonl", ab_holdout_size=4).load()
    lib.propose_from_model("beta", "Try X first then Y.")
    # 2 successful trials < min 4 → not admitted yet
    for _ in range(2):
        lib.record_result("beta", success=True, holdout=True)
    assert lib.ab_admit("beta") is False
    for _ in range(2):
        lib.record_result("beta", success=True, holdout=True)
    assert lib.ab_admit("beta") is True
    assert lib.get("beta").origin == "hybrid"


def test_library_ab_admit_unknown_strategy(tmp_path: Path):
    lib = StrategyLibrary(tmp_path / "s.jsonl").load()
    assert lib.ab_admit("nonexistent") is False


def test_library_top_k_and_prefix(tmp_path: Path):
    lib = StrategyLibrary(tmp_path / "s.jsonl", k_few_shot=2).load()
    # Promote one strategy with trials so it outranks fresh seeds.
    seed_name = SEED_STRATEGIES[0]["name"]
    for _ in range(8):
        lib.record_result(seed_name, success=True, holdout=True)
    top = lib.top_k()
    assert len(top) == 2
    assert top[0].name == seed_name
    prefix = lib.few_shot_prefix()
    assert seed_name in prefix
    assert "PROVEN REASONING STRATEGIES" in prefix


def test_library_top_k_empty(tmp_path: Path):
    lib = StrategyLibrary(tmp_path / "s.jsonl", seed_if_empty=False).load()
    assert lib.top_k() == []
    assert lib.few_shot_prefix() == ""


def test_library_holdout_slice_deterministic(tmp_path: Path):
    lib = StrategyLibrary(tmp_path / "s.jsonl", ab_holdout_size=3).load()
    items = [f"p{i}" for i in range(10)]
    assert lib.holdout_slice(items) == items[:3]


def test_library_holdout_slice_random(tmp_path: Path):
    lib = StrategyLibrary(tmp_path / "s.jsonl", ab_holdout_size=3).load()
    items = [f"p{i}" for i in range(10)]
    rng1 = random.Random(42)
    rng2 = random.Random(42)
    assert lib.holdout_slice(items, rng1) == lib.holdout_slice(items, rng2)


# ───────── parser ─────────

def test_parse_strategies_from_model_output_basic():
    raw = (
        "Here's my idea:\n\n"
        "STRATEGY: try-backwards\n"
        "TEMPLATE: work from the answer to the question.\n\n"
        "STRATEGY: guess-and-check\n"
        "TEMPLATE: propose candidate answers, then verify.\n"
    )
    pairs = parse_strategies_from_model_output(raw)
    names = [n for n, _ in pairs]
    assert "try-backwards" in names
    assert "guess-and-check" in names


def test_parse_strategies_empty_input():
    assert parse_strategies_from_model_output("") == []
    assert parse_strategies_from_model_output("no strategies here") == []
