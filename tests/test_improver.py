from unittest.mock import MagicMock, patch
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


@patch("timedilate.improver.time.sleep")
def test_handles_generation_failure(mock_sleep):
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


def test_stagnation_boost_widens_temperature():
    """With stagnation_boost, branch temperatures are higher."""
    config = TimeDilateConfig(branch_factor=3, temperature=0.7)
    engine = MagicMock()
    engine.estimate_tokens = MagicMock(return_value=100)
    improver = ImprovementEngine(engine, config)

    normal_t2 = improver._branch_temperature(2)
    improver.stagnation_boost = True
    boosted_t2 = improver._branch_temperature(2)
    assert boosted_t2 > normal_t2  # boosted should be higher
    assert boosted_t2 <= 1.3


def test_stagnation_boost_single_branch():
    """With single branch and stagnation_boost, temperature is raised."""
    config = TimeDilateConfig(branch_factor=1, temperature=0.7)
    engine = MagicMock()
    improver = ImprovementEngine(engine, config)
    assert improver._branch_temperature(0) == 0.7
    improver.stagnation_boost = True
    assert abs(improver._branch_temperature(0) - 0.9) < 0.01  # 0.7 + 0.2


def test_single_branch_no_diversity():
    config = TimeDilateConfig(branch_factor=1, temperature=0.5)
    engine = MagicMock()
    improver = ImprovementEngine(engine, config)
    assert improver._branch_temperature(0) == 0.5


def test_comparative_wider_threshold_at_high_score():
    """At high scores (>=80), comparative threshold widens to 10."""
    # Score goes from 85 to 93 (delta=8, within threshold=10 at high score)
    # Comparative says A is better → overrule
    engine = make_mock_engine(["variant", "93", "A"])
    config = TimeDilateConfig(branch_factor=1)
    improver = ImprovementEngine(engine, config)
    best, score, idx = improver.run_cycle(
        original_prompt="test",
        current_best="original",
        current_score=85,
        directive="Improve.",
    )
    assert best == "original"  # overruled by comparative
    assert score == 85


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


def test_fresh_attempt_with_insights():
    """Fresh attempt uses best directive and score history when provided."""
    engine = make_mock_engine(["fresh solution", "90"])
    config = TimeDilateConfig(branch_factor=1)
    improver = ImprovementEngine(engine, config)
    output, score = improver.fresh_attempt(
        "Write hello world", "Be creative",
        best_directive="Add error handling",
        score_history=[50, 60, 65, 65, 65],
    )
    assert output == "fresh solution"
    assert score == 90
    # Check insights were included in the prompt
    prompt_text = engine.generate.call_args_list[0][0][0]
    assert "error handling" in prompt_text.lower()
    assert "65" in prompt_text


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


def test_variant_diversity():
    config = TimeDilateConfig(branch_factor=1)
    engine = MagicMock()
    engine.estimate_tokens = MagicMock(return_value=100)
    improver = ImprovementEngine(engine, config)
    # Identical variants = 0 diversity
    assert improver._variant_diversity(["hello", "hello"]) == 0.0
    # Completely different = high diversity
    div = improver._variant_diversity(["abcdef", "xyz123"])
    assert div > 0.5
    # Single variant = 1.0
    assert improver._variant_diversity(["single"]) == 1.0


def test_deduplicate_variants():
    config = TimeDilateConfig(branch_factor=1)
    engine = MagicMock()
    engine.estimate_tokens = MagicMock(return_value=100)
    improver = ImprovementEngine(engine, config)
    variants = ["hello world", "hello world", "something different"]
    result = improver._deduplicate_variants(variants)
    assert len(result) == 2  # one "hello world" removed
    assert result[0][1] == "hello world"
    assert result[1][1] == "something different"


def test_crossover_with_scores():
    """Crossover includes score context when scores provided."""
    engine = make_mock_engine(["combined"])
    config = TimeDilateConfig(branch_factor=1)
    improver = ImprovementEngine(engine, config)
    result = improver._crossover("test", "sol A", "sol B", score_a=85, score_b=78)
    assert result == "combined"
    prompt_text = engine.generate.call_args_list[0][0][0]
    assert "85/100" in prompt_text
    assert "78/100" in prompt_text
    assert "higher-scoring" in prompt_text


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


def test_diversity_tiebreaker():
    """When top variants score within 3 points, prefer the more different one."""
    # v1 is similar to current_best, v2 is very different. Both score 80.
    engine = make_mock_engine([
        "original_tweaked",  # v1: similar to current_best
        "completely_new_approach_xyz",  # v2: very different
        "80",  # score for v1 (CoT used with 2 variants)
        "79",  # score for v2 (within 3 points)
    ])
    config = TimeDilateConfig(branch_factor=2)
    improver = ImprovementEngine(engine, config)
    best, score, idx = improver.run_cycle(
        original_prompt="test",
        current_best="original_version",
        current_score=50,
        directive="Improve.",
    )
    # Should prefer v2 (more different) even though v1 scored 1 point higher
    assert best == "completely_new_approach_xyz"
    assert score == 79


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


@patch("timedilate.improver.time.sleep")
def test_score_retry_on_zero(mock_sleep):
    """Scoring retries when result is 0."""
    engine = make_mock_engine(["no score here", "75"])  # first returns 0, retry returns 75
    config = TimeDilateConfig(branch_factor=1)
    improver = ImprovementEngine(engine, config)
    score = improver._score_variant("test", "variant", retries=1)
    assert score == 75
    assert engine.generate.call_count == 2


@patch("timedilate.improver.time.sleep")
def test_score_retry_on_exception(mock_sleep):
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


def test_task_aware_scoring_uses_code_rubric():
    """When task_type is 'code', scoring uses code-specific rubric."""
    engine = make_mock_engine(["improved", "85"])
    config = TimeDilateConfig(branch_factor=1)
    improver = ImprovementEngine(engine, config)
    improver.task_type = "code"
    best, score, idx = improver.run_cycle(
        original_prompt="Write a sort function",
        current_best="def sort(lst): return sorted(lst)",
        current_score=50,
        directive="Improve.",
    )
    assert best == "improved"
    assert score == 85
    # Verify the scoring prompt used code-specific rubric
    score_call = engine.generate.call_args_list[1]
    score_prompt = score_call[0][0]
    assert "edge cases" in score_prompt  # from CODE_RUBRIC_ADDENDUM


def test_progressive_harshness_at_high_score():
    """When current_score >= 75, scoring prompt gets harshness addendum."""
    engine = make_mock_engine(["improved", "82"])
    config = TimeDilateConfig(branch_factor=1)
    improver = ImprovementEngine(engine, config)
    best, score, idx = improver.run_cycle(
        original_prompt="test",
        current_best="original",
        current_score=80,
        directive="Improve.",
    )
    assert best == "improved"
    # Verify harshness addendum was included
    score_call = engine.generate.call_args_list[1]
    score_prompt = score_call[0][0]
    assert "EXTRA critical" in score_prompt
    assert "80/100" in score_prompt


def test_score_aware_guidance_high():
    """At high scores (>=85), prompt uses surgical guidance."""
    engine = make_mock_engine(["improved", "90"])
    config = TimeDilateConfig(branch_factor=1)
    improver = ImprovementEngine(engine, config)
    best, score, idx = improver.run_cycle(
        original_prompt="test", current_best="original",
        current_score=88, directive="Polish.",
    )
    gen_call = engine.generate.call_args_list[0]
    assert "surgical" in gen_call[0][0].lower()


def test_urgency_in_prompt_when_few_cycles_left():
    """When cycles_remaining <= 3, urgency note appears in prompt."""
    engine = make_mock_engine(["improved", "85"])
    config = TimeDilateConfig(branch_factor=1)
    improver = ImprovementEngine(engine, config)
    improver.cycles_remaining = 2
    best, score, idx = improver.run_cycle(
        original_prompt="test", current_best="original",
        current_score=70, directive="Improve.",
    )
    gen_call = engine.generate.call_args_list[0]
    assert "2 improvement cycle(s) remaining" in gen_call[0][0]


def test_no_urgency_when_many_cycles_left():
    """When cycles_remaining > 3, no urgency note."""
    engine = make_mock_engine(["improved", "85"])
    config = TimeDilateConfig(branch_factor=1)
    improver = ImprovementEngine(engine, config)
    improver.cycles_remaining = 10
    best, score, idx = improver.run_cycle(
        original_prompt="test", current_best="original",
        current_score=70, directive="Improve.",
    )
    gen_call = engine.generate.call_args_list[0]
    assert "remaining" not in gen_call[0][0]


def test_score_aware_guidance_low():
    """At low scores (<65), prompt encourages bold changes."""
    engine = make_mock_engine(["improved", "60"])
    config = TimeDilateConfig(branch_factor=1)
    improver = ImprovementEngine(engine, config)
    best, score, idx = improver.run_cycle(
        original_prompt="test", current_best="original",
        current_score=30, directive="Fix.",
    )
    gen_call = engine.generate.call_args_list[0]
    assert "bold" in gen_call[0][0].lower()


def test_no_harshness_at_low_score():
    """When current_score < 75, no harshness addendum."""
    engine = make_mock_engine(["improved", "70"])
    config = TimeDilateConfig(branch_factor=1)
    improver = ImprovementEngine(engine, config)
    best, score, idx = improver.run_cycle(
        original_prompt="test",
        current_best="original",
        current_score=40,
        directive="Improve.",
    )
    assert best == "improved"
    score_call = engine.generate.call_args_list[1]
    score_prompt = score_call[0][0]
    assert "EXTRA critical" not in score_prompt


def test_weighted_scoring_uses_detailed_rubric():
    """When score_weights is set, scoring uses detailed rubric with weighted totals."""
    engine = make_mock_engine(["improved", "C:20 K:15 Q:10 E:5"])
    config = TimeDilateConfig(
        branch_factor=1,
        score_weights={"correctness": 60, "completeness": 20, "quality": 10, "elegance": 10},
    )
    improver = ImprovementEngine(engine, config)
    best, score, idx = improver.run_cycle(
        original_prompt="test",
        current_best="original",
        current_score=30,
        directive="Improve.",
    )
    # Weighted: (20*60 + 15*20 + 10*10 + 5*10) / (25*100) * 100 = heavy on correctness
    assert best == "improved"
    assert score > 0
    assert idx == 0


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


def test_cot_scoring_uses_task_type():
    """CoT scoring should include task-specific rubric addenda."""
    engine = make_mock_engine(["variant", "SCORE: 75"])
    config = TimeDilateConfig(branch_factor=1)
    improver = ImprovementEngine(engine, config)
    improver.task_type = "code"
    score = improver._score_variant("Write a sort fn", "def sort(): pass", use_cot=True)
    # Check that the scoring prompt included code-specific criteria
    scoring_call = engine.generate.call_args_list[0]
    prompt_text = scoring_call[0][0]
    assert "compile/run without errors" in prompt_text
    assert score == 75


def test_cot_scoring_applies_harshness():
    """CoT scoring should apply progressive harshness at high scores."""
    engine = make_mock_engine(["SCORE: 80"])
    config = TimeDilateConfig(branch_factor=1)
    improver = ImprovementEngine(engine, config)
    score = improver._score_variant("test", "output", use_cot=True, current_score=80)
    scoring_call = engine.generate.call_args_list[0]
    prompt_text = scoring_call[0][0]
    assert "EXTRA critical" in prompt_text


def test_cot_scoring_applies_antibloat():
    """CoT scoring should penalize bloated outputs."""
    engine = make_mock_engine(["SCORE: 70"])
    config = TimeDilateConfig(branch_factor=1)
    improver = ImprovementEngine(engine, config)
    improver.initial_output_length = 100
    # Variant is 3x the initial length (>2.5x threshold)
    long_variant = "x" * 300
    score = improver._score_variant("test", long_variant, use_cot=True)
    scoring_call = engine.generate.call_args_list[0]
    prompt_text = scoring_call[0][0]
    assert "Penalize unnecessary verbosity" in prompt_text


def test_reflection_prompt_code_specific():
    """Code tasks get code-specific reflection analysis points."""
    engine = make_mock_engine([])
    config = TimeDilateConfig(branch_factor=1)
    improver = ImprovementEngine(engine, config)
    improver.task_type = "code"
    prompt = improver._build_reflection_prompt("Write sort", "def sort(): pass", "Fix bugs", 70)
    assert "edge cases" in prompt
    assert "introducing a bug" in prompt


def test_reflection_prompt_prose_specific():
    """Prose tasks get prose-specific reflection analysis points."""
    engine = make_mock_engine([])
    config = TimeDilateConfig(branch_factor=1)
    improver = ImprovementEngine(engine, config)
    improver.task_type = "prose"
    prompt = improver._build_reflection_prompt("Write essay", "An essay...", "Improve", 70)
    assert "paragraph" in prompt
    assert "argument" in prompt


def test_reflection_threshold_lower_for_code():
    """Reflection activates at score 50 for code, 60 for other types."""
    engine = make_mock_engine(["reflection", "variant", "80"])
    config = TimeDilateConfig(branch_factor=1, use_reflection=True)
    improver = ImprovementEngine(engine, config)
    improver.task_type = "code"
    # Score 55: should use reflection for code
    best, score, idx = improver.run_cycle("Write sort", "def sort(): pass", 55, "Fix bugs")
    # First call should be reflection prompt (has "Before making changes")
    first_call = engine.generate.call_args_list[0][0][0]
    assert "Before making changes" in first_call


def test_fallback_scoring_on_zero():
    """When primary scoring returns 0 after retries, fallback scoring kicks in."""
    engine = make_mock_engine([
        "not a number",  # primary returns 0
        "still not",     # retry returns 0
        "75",            # fallback succeeds
    ])
    config = TimeDilateConfig(branch_factor=1)
    improver = ImprovementEngine(engine, config)
    score = improver._score_variant("test", "output", retries=1)
    assert score == 75


def test_fallback_scoring_on_exception():
    """When primary scoring raises exceptions, fallback scoring kicks in."""
    engine = MagicMock()
    engine.estimate_tokens = MagicMock(return_value=100)
    engine.generate = MagicMock(side_effect=[
        RuntimeError("gpu error"),  # primary fails
        RuntimeError("gpu error"),  # retry fails
        "60",                       # fallback succeeds
    ])
    config = TimeDilateConfig(branch_factor=1)
    improver = ImprovementEngine(engine, config)
    score = improver._score_variant("test", "output", retries=1)
    assert score == 60
