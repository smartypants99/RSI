from unittest.mock import MagicMock
from timedilate.improver import ImprovementEngine
from timedilate.config import TimeDilateConfig


def make_mock_engine(responses: list[str]):
    engine = MagicMock()
    engine.generate = MagicMock(side_effect=responses)
    engine.estimate_tokens = MagicMock(return_value=100)
    return engine


def test_branch_generates_n_variants():
    engine = make_mock_engine([
        "variant_1", "variant_2", "variant_3",
        "90", "85", "70",
    ])
    config = TimeDilateConfig(branch_factor=3)
    improver = ImprovementEngine(engine, config)
    best, score, idx = improver.run_cycle(
        original_prompt="Write hello world",
        current_best="print('hi')",
        current_score=50,
        directive="Improve this code.",
    )
    assert best == "variant_1"
    assert score == 90


def test_keeps_current_if_no_improvement():
    engine = make_mock_engine([
        "worse_v1", "worse_v2", "worse_v3",
        "30", "20", "10",
    ])
    config = TimeDilateConfig(branch_factor=3)
    improver = ImprovementEngine(engine, config)
    best, score, idx = improver.run_cycle(
        original_prompt="Write hello world",
        current_best="print('hello world')",
        current_score=95,
        directive="Improve this code.",
    )
    assert best == "print('hello world')"
    assert score == 95


def test_single_branch_mode():
    engine = make_mock_engine(["improved", "88"])
    config = TimeDilateConfig(branch_factor=1)
    improver = ImprovementEngine(engine, config)
    best, score, idx = improver.run_cycle(
        original_prompt="test",
        current_best="original",
        current_score=50,
        directive="Improve.",
    )
    assert best == "improved"
    assert score == 88


def test_handles_generation_failure():
    """If all generations fail, returns current best."""
    engine = MagicMock()
    engine.generate = MagicMock(side_effect=RuntimeError("model crashed"))
    engine.estimate_tokens = MagicMock(return_value=100)
    config = TimeDilateConfig(branch_factor=2)
    improver = ImprovementEngine(engine, config)
    best, score, idx = improver.run_cycle(
        original_prompt="test",
        current_best="original",
        current_score=50,
        directive="Improve.",
    )
    assert best == "original"
    assert score == 50
    assert idx == -1


def test_handles_empty_generation():
    """Empty outputs are skipped."""
    engine = make_mock_engine(["", "good variant", "85"])
    config = TimeDilateConfig(branch_factor=2)
    improver = ImprovementEngine(engine, config)
    best, score, idx = improver.run_cycle(
        original_prompt="test",
        current_best="original",
        current_score=50,
        directive="Improve.",
    )
    assert best == "good variant"
    assert score == 85


def test_variant_index_tracking():
    engine = make_mock_engine(["v1", "v2", "v3", "60", "90", "70"])
    config = TimeDilateConfig(branch_factor=3)
    improver = ImprovementEngine(engine, config)
    best, score, idx = improver.run_cycle(
        original_prompt="test",
        current_best="original",
        current_score=50,
        directive="Improve.",
    )
    assert idx == 1  # v2 scored highest
    assert best == "v2"


def test_history_summary_in_prompt():
    """History summary is included in the improvement prompt."""
    engine = make_mock_engine(["improved", "88"])
    config = TimeDilateConfig(branch_factor=1)
    improver = ImprovementEngine(engine, config)
    best, score, idx = improver.run_cycle(
        original_prompt="test",
        current_best="original",
        current_score=50,
        directive="Improve.",
        history_summary='- Cycle 1: "Fix bugs" -> no improvement (score 50->50)',
    )
    # Should still work, the history is passed through to the prompt
    assert best == "improved"
    assert score == 88


def test_comparative_overrule_on_close_score():
    """When scores are close (<=5), comparative check can overrule."""
    # variant scores 53 (only 3 above current 50), comparative says A is better
    engine = make_mock_engine(["variant", "53", "A"])
    config = TimeDilateConfig(branch_factor=1)
    improver = ImprovementEngine(engine, config)
    best, score, idx = improver.run_cycle(
        original_prompt="test",
        current_best="original",
        current_score=50,
        directive="Improve.",
    )
    assert best == "original"  # overruled by comparative
    assert score == 50
    assert idx == -1


def test_branch_temperature_diversity():
    """Different branches should get different temperatures."""
    config = TimeDilateConfig(branch_factor=3, temperature=0.7)
    engine = MagicMock()
    engine.estimate_tokens = MagicMock(return_value=100)
    improver = ImprovementEngine(engine, config)
    t0 = improver._branch_temperature(0)
    t1 = improver._branch_temperature(1)
    t2 = improver._branch_temperature(2)
    assert t0 == 0.7  # base temperature
    assert t1 != t2  # different temperatures
    assert 0.3 <= t1 <= 1.0
    assert 0.3 <= t2 <= 1.0


def test_single_branch_no_diversity():
    config = TimeDilateConfig(branch_factor=1, temperature=0.5)
    engine = MagicMock()
    improver = ImprovementEngine(engine, config)
    assert improver._branch_temperature(0) == 0.5


def test_no_comparative_on_large_delta():
    """When score improvement is large (>5), skip comparative check."""
    engine = make_mock_engine(["variant", "80"])  # delta=30, no comparison call
    config = TimeDilateConfig(branch_factor=1)
    improver = ImprovementEngine(engine, config)
    best, score, idx = improver.run_cycle(
        original_prompt="test",
        current_best="original",
        current_score=50,
        directive="Improve.",
    )
    assert best == "variant"
    assert score == 80
    assert idx == 0


def test_tournament_select_picks_winner():
    """Tournament selection with 4+ variants uses pairwise comparisons."""
    engine = MagicMock()
    engine.estimate_tokens = MagicMock(return_value=100)
    # 4 variants generated, then tournament: (v1 vs v2) -> B, (v3 vs v4) -> A, (v2 vs v3) -> B
    # Winner: v3. Then score v3 -> 85
    engine.generate = MagicMock(side_effect=[
        "v1", "v2", "v3", "v4",   # 4 variants
        "B",                        # v1 vs v2 -> v2 wins
        "A",                        # v3 vs v4 -> v3 wins
        "B",                        # v2 vs v3 -> v3 wins
        "85",                       # score the winner (v3)
    ])
    config = TimeDilateConfig(branch_factor=4)
    improver = ImprovementEngine(engine, config)
    best, score, idx = improver.run_cycle(
        original_prompt="test",
        current_best="original",
        current_score=50,
        directive="Improve.",
    )
    assert best == "v3"
    assert score == 85
    assert idx == 2


def test_tournament_with_odd_number():
    """Tournament handles odd number of variants (bye for last one)."""
    engine = MagicMock()
    engine.estimate_tokens = MagicMock(return_value=100)
    config = TimeDilateConfig(branch_factor=5)
    improver = ImprovementEngine(engine, config)
    # 5 variants: (0v1, 1v2)->"A", (2v3, 3v4)->"B", 4v5 gets bye
    # Round 2: (0v1, 3v4)->"A", 4v5 gets bye
    # Round 3: (0v1, 4v5)->"A" -> 0v1 wins
    variants = ["v1", "v2", "v3", "v4", "v5"]
    indexed = list(enumerate(variants))
    # Just test the method directly
    engine.generate = MagicMock(side_effect=["A", "B", "A", "A"])
    winner, idx = improver._tournament_select("test", variants)
    assert winner == "v1"
    assert idx == 0
