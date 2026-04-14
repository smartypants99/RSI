from unittest.mock import MagicMock
from timedilate.controller import DilationController
from timedilate.config import TimeDilateConfig


def make_mock_engine(responses: list[str]):
    engine = MagicMock()
    engine.generate = MagicMock(side_effect=responses)
    engine.estimate_tokens = MagicMock(return_value=100)
    return engine


def test_full_pipeline_dilation_2x():
    config = TimeDilateConfig(dilation_factor=2, branch_factor=1)
    mock_engine = make_mock_engine([
        "def sort(lst): return sorted(lst)",
        "75",
        "def sort(lst):\n    if not lst:\n        return []\n    return sorted(lst)",
        "90",
    ])
    controller = DilationController(config, mock_engine)
    result = controller.run("Write a sort function")
    assert result.cycles_completed == 1
    assert result.score == 90
    assert "if not lst" in result.output


def test_full_pipeline_dilation_5x():
    config = TimeDilateConfig(dilation_factor=5, branch_factor=1)
    responses = [
        "v0", "50", "50",  # initial + feedback score + consistency re-score
        "v1", "60",        # cycle 0
        "v2", "70",        # cycle 1
        "v3", "80",        # cycle 2
        # cycle 3 (index 3): feedback scoring at 3%3==0
        "STRENGTHS:\n- Good\nWEAKNESSES:\n1. Bad\nSCORE: 80",
        "v4", "90",        # cycle 3 variant + score
    ]
    mock_engine = make_mock_engine(responses)
    controller = DilationController(config, mock_engine)
    result = controller.run("test prompt")
    assert result.cycles_completed == 4
    assert result.score == 90
    assert result.output == "v4"


def test_full_pipeline_dilation_1x_no_refinement():
    config = TimeDilateConfig(dilation_factor=1)
    mock_engine = make_mock_engine(["direct output"])
    controller = DilationController(config, mock_engine)
    result = controller.run("test")
    assert result.output == "direct output"
    assert result.cycles_completed == 0
    assert result.score == 0
