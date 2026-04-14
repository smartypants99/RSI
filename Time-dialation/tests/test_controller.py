"""Tests for DilationController."""
import sys
from unittest.mock import MagicMock
from timedilate.config import TimeDilateConfig

if "vllm" not in sys.modules or not isinstance(sys.modules["vllm"], MagicMock):
    sys.modules["vllm"] = MagicMock()
mock_vllm = sys.modules["vllm"]

from timedilate.controller import DilationController, DilationResult


def _mock_engine(responses):
    """Create a mock engine that returns responses in sequence."""
    engine = MagicMock()
    engine.generate = MagicMock(side_effect=list(responses))
    return engine


def test_factor_1_single_pass():
    engine = _mock_engine(["initial output"])
    config = TimeDilateConfig(dilation_factor=1.0)
    controller = DilationController(config, engine)
    result = controller.run("test")
    assert result.output == "initial output"
    assert result.cycles_completed == 0


def test_factor_2_runs_cycles():
    """Factor 2 = 2 cycles. Each cycle: score + critique + refine + score."""
    engine = _mock_engine([
        "initial output",     # initial generation
        "75",                 # score initial
        # cycle 1:
        "needs work here",    # critique
        "improved v1",        # refine
        "85",                 # score v1
        # cycle 2:
        "still needs X",      # critique
        "improved v2",        # refine
        "90",                 # score v2
    ])
    config = TimeDilateConfig(dilation_factor=2, convergence_patience=10)
    controller = DilationController(config, engine)
    result = controller.run("test")
    assert result.cycles_completed == 2
    assert result.score == 90


def test_keeps_best_output():
    """If a cycle produces worse output, best is preserved."""
    engine = _mock_engine([
        "initial",        # gen
        "80",             # score
        # cycle 1: worse
        "critique",       # critique
        "worse version",  # refine
        "60",             # score (worse — not adopted)
    ])
    config = TimeDilateConfig(dilation_factor=1.5, convergence_patience=10)
    controller = DilationController(config, engine)
    result = controller.run("test")
    assert result.score == 80
    assert result.output == "initial"


def test_convergence_triggers_fresh_attempt():
    responses = ["initial", "70"]  # gen + score
    # 5 cycles with no improvement
    for _ in range(5):
        responses.extend(["critique", "no better", "60"])
    # fresh attempt after convergence
    responses.extend(["fresh approach", "85"])
    # remaining cycles
    for _ in range(10):
        responses.extend(["critique", "variant", "80"])
    engine = _mock_engine(responses)
    config = TimeDilateConfig(dilation_factor=10, convergence_patience=5)
    controller = DilationController(config, engine)
    result = controller.run("test")
    assert result.convergence_resets >= 1


def test_result_to_report():
    result = DilationResult(
        output="test", dilation_factor=100, cycles_completed=100,
        total_cycles=100, elapsed_seconds=5.0, model_used="test", score=85,
    )
    report = result.to_report()
    assert report["dilation_factor"] == 100
    assert report["score"] == 85
    assert "version" in report


def test_result_to_report_with_config():
    result = DilationResult(
        output="x", dilation_factor=10, cycles_completed=10,
        total_cycles=10, elapsed_seconds=1.0, model_used="test", score=70,
    )
    config = TimeDilateConfig(dilation_factor=10)
    report = result.to_report(config)
    assert "config" in report


def test_on_cycle_callback():
    engine = _mock_engine([
        "initial", "70",
        "critique", "improved", "80",
        "critique2", "improved2", "85",
    ])
    config = TimeDilateConfig(dilation_factor=2, convergence_patience=10)
    callbacks = []
    controller = DilationController(config, engine)
    controller.run("test", on_cycle=lambda c, t, s, e: callbacks.append((c, t, s)))
    assert len(callbacks) == 2


def test_infinite_factor_config():
    """Config accepts any factor — no artificial ceiling."""
    config = TimeDilateConfig(dilation_factor=1_000_000_000)
    assert config.num_cycles == 1_000_000_000


def test_time_budget_subjective_time():
    """5s budget * 1M factor = 5M seconds subjective."""
    config = TimeDilateConfig(dilation_factor=1_000_000, time_budget_seconds=5)
    assert config.subjective_time == 5_000_000
    assert config.num_cycles == 0  # unlimited in time-budget mode


def test_time_budget_mode_runs():
    """Time budget mode runs cycles until wall clock expires."""
    import time as _time

    call_count = [0]
    def mock_generate(prompt, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return "initial"
        if "scoring" in prompt.lower() or "rate it" in prompt.lower():
            return "75"
        if "reviewing" in prompt.lower() or "critique" in prompt.lower():
            return "fix stuff"
        return "refined output"

    engine = MagicMock()
    engine.generate = MagicMock(side_effect=mock_generate)

    # Very short budget so it stops quickly
    config = TimeDilateConfig(dilation_factor=1000, time_budget_seconds=0.001)
    controller = DilationController(config, engine)
    result = controller.run("test")
    assert result.elapsed_seconds < 1.0  # should respect budget
