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


def test_parse_score_invalid():
    scorer = Scorer()
    assert scorer.parse_score("no number here") == 0
    assert scorer.parse_score("") == 0


def test_parse_score_clamps():
    scorer = Scorer()
    assert scorer.parse_score("150") == 100
    assert scorer.parse_score("-5") == 5  # regex extracts digits, no negative scores exist


def test_build_comparative_prompt():
    scorer = Scorer()
    prompt = scorer.build_comparative_prompt("Sort a list", "output A", "output B")
    assert "Output A" in prompt
    assert "Output B" in prompt


def test_parse_comparison():
    scorer = Scorer()
    assert scorer.parse_comparison("A") == "A"
    assert scorer.parse_comparison("B") == "B"
    assert scorer.parse_comparison("TIE") == "TIE"
    assert scorer.parse_comparison("a is better") == "A"
    assert scorer.parse_comparison("b wins") == "B"
    assert scorer.parse_comparison("something else") == "TIE"


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
