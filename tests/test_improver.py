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


def test_reflection_based_generation():
    """With use_reflection=True and score>=60, branch 0 uses reflect-then-act."""
    engine = make_mock_engine([
        "Should fix error handling and add tests",  # reflection
        "improved with reflection",                   # variant from reflection
        "92",                                         # score
    ])
    config = TimeDilateConfig(branch_factor=1, use_reflection=True)
    improver = ImprovementEngine(engine, config)
    best, score, idx = improver.run_cycle(
        original_prompt="test",
        current_best="original",
        current_score=70,  # >= 60, triggers reflection
        directive="Improve.",
    )
    assert best == "improved with reflection"
    assert score == 92
    # Should have 3 calls: reflection, variant, score
    assert engine.generate.call_count == 3


def test_reflection_skipped_below_threshold():
    """Reflection is skipped when score < 60 even if enabled."""
    engine = make_mock_engine(["improved", "80"])
    config = TimeDilateConfig(branch_factor=1, use_reflection=True)
    improver = ImprovementEngine(engine, config)
    best, score, idx = improver.run_cycle(
        original_prompt="test",
        current_best="original",
        current_score=40,  # < 60, no reflection
        directive="Improve.",
    )
    assert best == "improved"
    assert engine.generate.call_count == 2  # just variant + score


def test_fresh_attempt():
    """Fresh attempt generates from scratch, not from current best."""
    engine = make_mock_engine(["fresh solution", "85"])
    config = TimeDilateConfig(branch_factor=1)
    improver = ImprovementEngine(engine, config)
    output, score = improver.fresh_attempt("Write hello world", "Be creative")
    assert output == "fresh solution"
    assert score == 85
    # Check prompt doesn't reference current best
    call_args = engine.generate.call_args_list[0]
    prompt_text = call_args[0][0]
    assert "current" not in prompt_text.lower() or "Current solution" not in prompt_text


def test_fresh_attempt_failure():
    """Fresh attempt handles errors gracefully."""
    engine = MagicMock()
    engine.generate = MagicMock(side_effect=RuntimeError("crash"))
    engine.estimate_tokens = MagicMock(return_value=100)
    config = TimeDilateConfig(branch_factor=1)
    improver = ImprovementEngine(engine, config)
    output, score = improver.fresh_attempt("test", "directive")
    assert output is None
    assert score == 0


def test_prompt_trimming_on_overflow():
    """History is trimmed when prompt exceeds context budget."""
    engine = MagicMock()
    # Return high token count to trigger trimming
    engine.estimate_tokens = MagicMock(return_value=20000)
    engine.generate = MagicMock(side_effect=["improved", "80"])
    config = TimeDilateConfig(branch_factor=1, context_window=32768)
    improver = ImprovementEngine(engine, config)
    best, score, idx = improver.run_cycle(
        original_prompt="test",
        current_best="original",
        current_score=50,
        directive="Improve.",
        history_summary="long history...",
        score_feedback="feedback...",
    )
    # Should still work — history/feedback trimmed
    assert best == "improved"


def test_score_feedback_in_improvement_prompt():
    """Score feedback is included in the improvement prompt."""
    engine = make_mock_engine(["improved", "88"])
    config = TimeDilateConfig(branch_factor=1)
    improver = ImprovementEngine(engine, config)
    best, score, idx = improver.run_cycle(
        original_prompt="test",
        current_best="original",
        current_score=50,
        directive="Improve.",
        score_feedback="1. Missing error handling\n2. Poor variable names",
    )
    # Check that the prompt included the feedback
    call_args = engine.generate.call_args_list[0]
    prompt_text = call_args[0][0]
    assert "Missing error handling" in prompt_text
    assert best == "improved"


def test_crossover_combines_variants():
    """Crossover generates a combined output from two variants."""
    engine = make_mock_engine(["combined solution"])
    config = TimeDilateConfig(branch_factor=1)
    improver = ImprovementEngine(engine, config)
    result = improver._crossover("test task", "solution A", "solution B")
    assert result == "combined solution"
    call_args = engine.generate.call_args_list[0][0][0]
    assert "Solution A" in call_args
    assert "Solution B" in call_args


def test_crossover_in_score_select():
    """When top 2 variants are close, crossover is attempted."""
    # v1=80, v2=75 (within 10), crossover="merged", cross_score=90
    engine = make_mock_engine([
        "v1", "v2",
        "SCORE: 80", "SCORE: 75",  # CoT scores
        "merged",                    # crossover
        "SCORE: 90",                 # crossover score
    ])
    config = TimeDilateConfig(branch_factor=2)
    improver = ImprovementEngine(engine, config)
    best, score, idx = improver.run_cycle(
        original_prompt="test",
        current_best="original",
        current_score=50,
        directive="Improve.",
    )
    assert best == "merged"
    assert score == 90
    assert idx == -2  # crossover indicator


def test_validate_variant_rejects_echo():
    """Variant that echoes the prompt is rejected."""
    config = TimeDilateConfig(branch_factor=1)
    engine = MagicMock()
    engine.estimate_tokens = MagicMock(return_value=100)
    improver = ImprovementEngine(engine, config)
    assert improver._validate_variant("Write hello world", "print('hello')", "Write hello world") is False


def test_validate_variant_rejects_too_short():
    """Variant much shorter than current best is rejected."""
    config = TimeDilateConfig(branch_factor=1)
    engine = MagicMock()
    engine.estimate_tokens = MagicMock(return_value=100)
    improver = ImprovementEngine(engine, config)
    current = "x" * 200
    assert improver._validate_variant("short", current, "test") is False


def test_similarity_ratio():
    config = TimeDilateConfig(branch_factor=1)
    engine = MagicMock()
    engine.estimate_tokens = MagicMock(return_value=100)
    improver = ImprovementEngine(engine, config)
    assert improver._similarity_ratio("hello", "hello") == 1.0
    assert improver._similarity_ratio("hello", "world") < 0.5
    assert improver._similarity_ratio("hello world", "hello earth") > 0.4


def test_validate_variant_rejects_near_duplicate():
    config = TimeDilateConfig(branch_factor=1)
    engine = MagicMock()
    engine.estimate_tokens = MagicMock(return_value=100)
    improver = ImprovementEngine(engine, config)
    current = "def sort(lst): return sorted(lst)"
    # Identical is rejected
    assert improver._validate_variant(current, current, "test") is False


def test_validate_variant_accepts_good():
    config = TimeDilateConfig(branch_factor=1)
    engine = MagicMock()
    engine.estimate_tokens = MagicMock(return_value=100)
    improver = ImprovementEngine(engine, config)
    assert improver._validate_variant("good output here", "original output", "test prompt") is True


def test_score_retry_on_zero():
    """Scoring retries when result is 0."""
    engine = make_mock_engine(["no score here", "75"])  # first returns 0, retry returns 75
    config = TimeDilateConfig(branch_factor=1)
    improver = ImprovementEngine(engine, config)
    score = improver._score_variant("test", "variant", retries=1)
    assert score == 75
    assert engine.generate.call_count == 2


def test_score_retry_on_exception():
    """Scoring retries on exception."""
    engine = MagicMock()
    engine.estimate_tokens = MagicMock(return_value=100)
    engine.generate = MagicMock(side_effect=[RuntimeError("fail"), "80"])
    config = TimeDilateConfig(branch_factor=1)
    improver = ImprovementEngine(engine, config)
    score = improver._score_variant("test", "variant", retries=1)
    assert score == 80


def test_ensemble_scoring():
    """Ensemble scoring averages normal and CoT scores."""
    engine = make_mock_engine([
        "80",             # normal score
        "SCORE: 90",      # CoT score
    ])
    config = TimeDilateConfig(branch_factor=1)
    improver = ImprovementEngine(engine, config)
    score = improver._score_variant("test", "variant", ensemble=True)
    assert score == 85  # (80 + 90) // 2


def test_cot_scoring_with_multi_branch():
    """Multi-branch scoring uses CoT format for better discrimination."""
    engine = make_mock_engine([
        "v1", "v2",
        "Correctness: 20\nSCORE: 75",  # CoT score for v1
        "Correctness: 22\nSCORE: 85",  # CoT score for v2
    ])
    config = TimeDilateConfig(branch_factor=2)
    improver = ImprovementEngine(engine, config)
    best, score, idx = improver.run_cycle(
        original_prompt="test",
        current_best="original",
        current_score=50,
        directive="Improve.",
    )
    assert best == "v2"
    assert score == 85
    assert idx == 1
