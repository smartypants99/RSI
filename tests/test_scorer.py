from timedilate.scorer import Scorer


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
