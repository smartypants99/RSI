from timedilate.scorer import Scorer, DetailedScore


def test_build_scoring_prompt():
    scorer = Scorer()
    prompt = scorer.build_scoring_prompt("Write a sort function", "def sort(lst): return sorted(lst)")
    assert "sort" in prompt.lower()
    assert "elegance" in prompt.lower()


def test_parse_score_valid():
    scorer = Scorer()
    assert scorer.parse_score("85") == 85
    assert scorer.parse_score("Score: 72") == 72
    assert scorer.parse_score("I'd give this a 90 out of 100") == 90


def test_parse_score_explicit_patterns():
    scorer = Scorer()
    assert scorer.parse_score("Score: 75") == 75
    assert scorer.parse_score("score=68") == 68
    assert scorer.parse_score("Rating: 82") == 82
    assert scorer.parse_score("75/100") == 75
    assert scorer.parse_score("I'd rate it 88 out of 100") == 88
    # Explicit pattern takes priority over other numbers in text
    assert scorer.parse_score("There are 3 issues. Score: 65") == 65


def test_parse_score_invalid():
    scorer = Scorer()
    assert scorer.parse_score("no number here") == 0
    assert scorer.parse_score("") == 0


def test_parse_cot_subscores():
    scorer = Scorer()
    raw = "Correctness: 20/25\nCompleteness: 18/25\nQuality: 15/25\nElegance: 22/25\nSCORE: 75"
    subs = scorer.parse_cot_subscores(raw)
    assert subs["correctness"] == 20
    assert subs["completeness"] == 18
    assert subs["quality"] == 15
    assert subs["elegance"] == 22


def test_parse_cot_subscores_partial():
    scorer = Scorer()
    raw = "Correctness = 20. The code is good. SCORE: 70"
    subs = scorer.parse_cot_subscores(raw)
    assert subs["correctness"] == 20
    assert "completeness" not in subs


def test_parse_cot_subscores_empty():
    scorer = Scorer()
    assert scorer.parse_cot_subscores("just a number 75") == {}


def test_parse_score_clamps():
    scorer = Scorer()
    assert scorer.parse_score("150") == 100
    assert scorer.parse_score("-5") == 5  # regex extracts digits, no negative scores exist


def test_build_comparative_prompt():
    scorer = Scorer()
    prompt = scorer.build_comparative_prompt("Sort a list", "output A", "output B")
    assert "Output A" in prompt
    assert "Output B" in prompt


def test_comparative_prompt_task_aware():
    scorer = Scorer()
    code_prompt = scorer.build_comparative_prompt("Sort a list", "A", "B", task_type="code")
    assert "correctness" in code_prompt.lower()
    prose_prompt = scorer.build_comparative_prompt("Write essay", "A", "B", task_type="prose")
    assert "argument" in prose_prompt.lower()
    general_prompt = scorer.build_comparative_prompt("Do thing", "A", "B", task_type="general")
    assert "edge case" not in general_prompt.lower()  # no code-specific hints


def test_parse_comparison():
    scorer = Scorer()
    assert scorer.parse_comparison("A") == "A"
    assert scorer.parse_comparison("B") == "B"
    assert scorer.parse_comparison("TIE") == "TIE"
    assert scorer.parse_comparison("a is better") == "A"
    assert scorer.parse_comparison("b wins") == "B"
    assert scorer.parse_comparison("something else") == "TIE"


def test_parse_comparison_verbose():
    scorer = Scorer()
    assert scorer.parse_comparison("Output A is better overall") == "A"
    assert scorer.parse_comparison("Output B is superior in every way") == "B"
    assert scorer.parse_comparison("They are roughly equal in quality") == "TIE"
    assert scorer.parse_comparison("After analysis, Output A wins") == "A"
    # Edge case: verbose text that starts with non-A/B
    assert scorer.parse_comparison("Overall, I think they are about the same") == "TIE"


def test_detailed_scoring_includes_task_type():
    scorer = Scorer()
    prompt = scorer.build_detailed_scoring_prompt("Write sort", "def sort(): pass", "code")
    assert "edge cases" in prompt or "run correctly" in prompt
    prompt = scorer.build_detailed_scoring_prompt("Write essay", "An essay...", "prose")
    assert "argument" in prompt or "clarity" in prompt
    # General has no addendum
    prompt = scorer.build_detailed_scoring_prompt("Do something", "output", "general")
    assert "edge cases" not in prompt


def test_scoring_rubric_has_calibration():
    scorer = Scorer()
    assert "harsh" in scorer.RUBRIC.lower() or "40-75" in scorer.RUBRIC


def test_parse_detailed_score():
    scorer = Scorer()
    result = scorer.parse_detailed_score("C:20 K:18 Q:15 E:22")
    assert result.correctness == 20
    assert result.completeness == 18
    assert result.quality == 15
    assert result.elegance == 22
    assert result.total == 75


def test_parse_detailed_score_clamps():
    scorer = Scorer()
    result = scorer.parse_detailed_score("C:30 K:0 Q:25 E:25")
    assert result.correctness == 25  # clamped to 25
    assert result.completeness == 0


def test_detailed_score_weakest_aspect():
    score = DetailedScore(correctness=20, completeness=10, quality=18, elegance=22)
    assert score.weakest_aspect == "completeness"


def test_weighted_total_default():
    score = DetailedScore(correctness=20, completeness=18, quality=15, elegance=22)
    assert score.weighted_total() == 75  # same as total


def test_weighted_total_custom():
    score = DetailedScore(correctness=25, completeness=0, quality=0, elegance=0)
    # Weight correctness 100%, others 0%
    result = score.weighted_total({"correctness": 100, "completeness": 0, "quality": 0, "elegance": 0})
    assert result == 100  # 25/25 * 100%


def test_weighted_total_heavy_correctness():
    score = DetailedScore(correctness=20, completeness=10, quality=10, elegance=10)
    equal = score.weighted_total()  # 50
    heavy_c = score.weighted_total({"correctness": 60, "completeness": 20, "quality": 10, "elegance": 10})
    # With correctness weighted higher, score should be higher since correctness is highest
    assert heavy_c > equal


def test_detailed_score_to_dict():
    score = DetailedScore(correctness=20, completeness=18, quality=15, elegance=22)
    d = score.to_dict()
    assert d["total"] == 75
    assert d["correctness"] == 20


def test_build_detailed_scoring_prompt():
    scorer = Scorer()
    prompt = scorer.build_detailed_scoring_prompt("test task", "test output")
    assert "C:##" in prompt
    assert "Correctness" in prompt


def test_build_cot_scoring_prompt():
    scorer = Scorer()
    prompt = scorer.build_cot_scoring_prompt("test task", "test output")
    assert "step by step" in prompt.lower()
    assert "SCORE:" in prompt


def test_parse_cot_score_with_reasoning():
    scorer = Scorer()
    raw = (
        "Correctness: 20/25 - logic is sound\n"
        "Completeness: 18/25 - missing edge cases\n"
        "Quality: 15/25 - needs better structure\n"
        "Elegance: 12/25 - could be cleaner\n\n"
        "SCORE: 65"
    )
    assert scorer.parse_cot_score(raw) == 65


def test_parse_cot_score_fallback():
    scorer = Scorer()
    # No SCORE: line, falls back to parse_score
    assert scorer.parse_cot_score("72") == 72


def test_parse_cot_score_clamps():
    scorer = Scorer()
    assert scorer.parse_cot_score("SCORE: 150") == 100
    assert scorer.parse_cot_score("SCORE: 0") == 0


def test_sanity_check_plausible():
    scorer = Scorer()
    assert scorer.sanity_check_score(75, 70, "longer output", "short") is True


def test_sanity_check_huge_jump():
    scorer = Scorer()
    assert scorer.sanity_check_score(95, 50, "output", "original") is False  # +45 > 40


def test_sanity_check_shorter_with_big_jump():
    scorer = Scorer()
    old = "x" * 200
    new = "x" * 50  # much shorter
    assert scorer.sanity_check_score(90, 60, new, old) is False


def test_sanity_check_near_identical_big_jump():
    scorer = Scorer()
    old = "def sort(lst):\n    return sorted(lst)\n" + "# padding " * 10
    new = old[:-1] + "."  # trivially different
    assert scorer.sanity_check_score(90, 60, new, old) is False  # +30 on near-identical


def test_sanity_check_no_improvement():
    scorer = Scorer()
    assert scorer.sanity_check_score(50, 60, "any", "any") is True


def test_progressive_scoring_harsh_at_high_score():
    scorer = Scorer()
    prompt = scorer.build_progressive_scoring_prompt("test", "output", current_score=85)
    assert "EXTRA critical" in prompt
    assert "85" in prompt


def test_progressive_scoring_normal_at_low_score():
    scorer = Scorer()
    prompt = scorer.build_progressive_scoring_prompt("test", "output", current_score=50)
    assert "EXTRA critical" not in prompt


def test_task_aware_scoring_code():
    scorer = Scorer()
    prompt = scorer.build_task_aware_scoring_prompt("write sort", "def sort(): pass", "code")
    assert "compile" in prompt.lower() or "edge case" in prompt.lower()


def test_task_aware_scoring_prose():
    scorer = Scorer()
    prompt = scorer.build_task_aware_scoring_prompt("write essay", "Climate change...", "prose")
    assert "argument" in prompt.lower() or "flow" in prompt.lower()


def test_task_aware_scoring_general():
    scorer = Scorer()
    prompt = scorer.build_task_aware_scoring_prompt("do thing", "result", "general")
    assert "Correctness" in prompt  # base rubric only


def test_build_feedback_scoring_prompt():
    scorer = Scorer()
    prompt = scorer.build_feedback_scoring_prompt("test task", "test output")
    assert "actionable" in prompt.lower()
    assert "SCORE:" in prompt


def test_parse_feedback_score():
    scorer = Scorer()
    raw = (
        "1. Missing error handling for empty input\n"
        "2. Variable names are unclear\n"
        "SCORE: 58"
    )
    score, feedback = scorer.parse_feedback_score(raw)
    assert score == 58
    assert "error handling" in feedback
    assert "SCORE" not in feedback


def test_parse_feedback_score_no_feedback():
    scorer = Scorer()
    score, feedback = scorer.parse_feedback_score("SCORE: 75")
    assert score == 75
    assert feedback == ""


def test_parse_strengths_weaknesses():
    scorer = Scorer()
    raw = (
        "STRENGTHS:\n"
        "- Clean error handling\n"
        "- Good variable names\n"
        "WEAKNESSES:\n"
        "1. Missing edge case for empty input\n"
        "2. No docstring\n"
        "SCORE: 72"
    )
    strengths, weaknesses = scorer.parse_strengths_weaknesses(raw)
    assert "Clean error handling" in strengths
    assert "Good variable names" in strengths
    assert "Missing edge case" in weaknesses
    assert "No docstring" in weaknesses


def test_parse_strengths_weaknesses_empty():
    scorer = Scorer()
    strengths, weaknesses = scorer.parse_strengths_weaknesses("Just some text\nSCORE: 50")
    assert strengths == ""
    assert weaknesses == ""


def test_cot_scoring_prompt_includes_task_addendum():
    scorer = Scorer()
    prompt = scorer.build_cot_scoring_prompt("task", "output", task_type="code")
    assert "compile/run without errors" in prompt
    prompt_prose = scorer.build_cot_scoring_prompt("task", "output", task_type="prose")
    assert "argument clear" in prompt_prose
    prompt_general = scorer.build_cot_scoring_prompt("task", "output", task_type="general")
    assert "compile/run" not in prompt_general
