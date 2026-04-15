"""Branch diversity: different temperatures must produce different outputs.

Mocks the engine at the controller level so we can observe per-call
temperature values and emit different outputs for each.
"""
from unittest.mock import MagicMock

from timedilate.config import TimeDilateConfig
from timedilate.controller import DilationController


def _temperature_aware_engine():
    """Engine mock that returns output varying with the temperature kwarg."""
    call_seq = {"count": 0}

    def generate(prompt, max_tokens=None, temperature=None, retries=2, stop=None):
        call_seq["count"] += 1
        lower = prompt.lower()
        if "rate the response" in lower:
            # Score branch: reward lower-temperature outputs more
            # Parse the RESPONSE line
            for line in prompt.splitlines():
                if line.startswith("branch@t="):
                    t = float(line.split("=", 1)[1])
                    # Higher temperature -> higher score here (arbitrary but
                    # deterministic), so temperature-spread produces a winner.
                    return str(int(60 + t * 50))
            # initial gen scoring
            return "60"
        if "weaknesses" in lower or "reviewing an ai response" in lower:
            return "improve clarity"
        if "improved version" in lower or "previously produced" in lower:
            # Refine: stamp the temperature into the output so tests can
            # observe per-branch variation.
            t = temperature if temperature is not None else 0.7
            return f"branch@t={t:.4f}"
        # initial generation
        return "initial output"

    engine = MagicMock()
    engine.generate = MagicMock(side_effect=generate)
    return engine


def test_branch_temperatures_vary_across_branches():
    """With branch_factor=3 and non-zero spread, the three refine calls
    should receive three distinct temperature values."""
    engine = _temperature_aware_engine()
    config = TimeDilateConfig(
        dilation_factor=1.5, branch_factor=3,
        branch_temperature_spread=0.3, convergence_patience=10,
        temperature=0.7,
    )
    controller = DilationController(config, engine)
    controller.run("task")

    refine_calls = [
        c for c in engine.generate.call_args_list
        if "previously produced" in c.args[0]
    ]
    assert len(refine_calls) == 3
    temps = sorted(c.kwargs.get("temperature") for c in refine_calls)
    assert len(set(temps)) == 3, f"expected 3 distinct temps, got {temps}"
    # Spread is symmetric around base 0.7, bounds [0.4, 1.0]
    assert temps[0] == pytest.approx(0.4)
    assert temps[-1] == pytest.approx(1.0)


def test_branch_outputs_differ_when_temperatures_differ():
    """Since the engine mock stamps temperature into output, the three
    refine outputs must be distinct."""
    engine = _temperature_aware_engine()
    config = TimeDilateConfig(
        dilation_factor=1.5, branch_factor=3,
        branch_temperature_spread=0.4, convergence_patience=10,
    )
    controller = DilationController(config, engine)
    result = controller.run("task")
    # Winner has the highest temperature stamp because score ∝ t in the mock
    assert result.output.startswith("branch@t=")
    # The winner's temperature is the highest -> ends near base + spread
    t_val = float(result.output.split("=", 1)[1])
    assert t_val == max(
        float(c.args[0].split("branch@t=", 1)[1]) if False else
        c.kwargs.get("temperature")
        for c in engine.generate.call_args_list
        if "previously produced" in c.args[0]
    )


def test_zero_spread_keeps_all_branches_same_temperature():
    engine = _temperature_aware_engine()
    config = TimeDilateConfig(
        dilation_factor=1.5, branch_factor=3,
        branch_temperature_spread=0.0, convergence_patience=10,
        temperature=0.5,
    )
    controller = DilationController(config, engine)
    controller.run("task")
    refine_calls = [
        c for c in engine.generate.call_args_list
        if "previously produced" in c.args[0]
    ]
    temps = [c.kwargs.get("temperature") for c in refine_calls]
    assert len(set(temps)) == 1
    assert temps[0] == 0.5


def test_branch_factor_1_uses_config_temperature():
    engine = _temperature_aware_engine()
    config = TimeDilateConfig(
        dilation_factor=1.5, branch_factor=1,
        branch_temperature_spread=0.5, convergence_patience=10,
        temperature=0.3,
    )
    controller = DilationController(config, engine)
    controller.run("task")
    refine_calls = [
        c for c in engine.generate.call_args_list
        if "previously produced" in c.args[0]
    ]
    assert len(refine_calls) == 1
    assert refine_calls[0].kwargs.get("temperature") == 0.3


# Need pytest import for approx
import pytest  # noqa: E402
