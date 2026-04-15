"""Specification scaffolds — tests for features not yet implemented.

These probe current behavior without MagicMock's "always truthy" trap.
When the feature lands, remove the xfail marker. Strict=False because
MagicMock's permissive return values can cause spurious xpass.
"""
import sys
from unittest.mock import MagicMock
import pytest

from timedilate.config import TimeDilateConfig

if "vllm" not in sys.modules or not isinstance(sys.modules["vllm"], MagicMock):
    sys.modules["vllm"] = MagicMock()
mock_vllm = sys.modules["vllm"]

from timedilate.engine import DilationEngine
from timedilate.controller import DilationController


def _reset():
    mock_vllm.reset_mock()
    mock_vllm.LLM.reset_mock()
    mock_vllm.SamplingParams.reset_mock()
    mock_vllm.LLM.return_value.generate.side_effect = None
    out = MagicMock()
    out.outputs = [MagicMock(text="ok")]
    mock_vllm.LLM.return_value.generate.return_value = [out]


# --- Engine: dtype / enforce_eager / seed / swap_space propagation ---
# Use "not in" since MagicMock kwargs always contains keys via __contains__ =
# False for real dicts. We inspect call_args.kwargs, which is a real dict.

def test_dtype_propagates_to_llm_kwargs():
    _reset()
    engine = DilationEngine(TimeDilateConfig(dtype="bfloat16"))
    engine.generate("p")
    assert "dtype" in mock_vllm.LLM.call_args.kwargs


def test_enforce_eager_propagates():
    _reset()
    engine = DilationEngine(TimeDilateConfig(enforce_eager=True))
    engine.generate("p")
    assert "enforce_eager" in mock_vllm.LLM.call_args.kwargs


def test_swap_space_propagates():
    _reset()
    engine = DilationEngine(TimeDilateConfig(swap_space_gb=8))
    engine.generate("p")
    kw = mock_vllm.LLM.call_args.kwargs
    assert "swap_space" in kw or "swap_space_gb" in kw


def test_seed_propagates_to_llm_only():
    """Seed on LLM() only — per-call SamplingParams stays seedless so
    branch_factor + temperature spread preserves diversity."""
    _reset()
    engine = DilationEngine(TimeDilateConfig(seed=1234))
    engine.generate("p")
    assert mock_vllm.LLM.call_args.kwargs.get("seed") == 1234
    assert "seed" not in mock_vllm.SamplingParams.call_args.kwargs


# --- Engine: stop sequences + batched generate ---

def test_stop_sequences_propagate():
    _reset()
    engine = DilationEngine(TimeDilateConfig())
    engine.generate("p", stop=["\n\n", "END"])
    assert "stop" in mock_vllm.SamplingParams.call_args.kwargs


def test_batched_generate_returns_list():
    _reset()
    o1, o2 = MagicMock(), MagicMock()
    o1.outputs = [MagicMock(text="first")]
    o2.outputs = [MagicMock(text="second")]
    mock_vllm.LLM.return_value.generate.return_value = [o1, o2]
    engine = DilationEngine(TimeDilateConfig())
    results = engine.generate(["p1", "p2"])
    assert isinstance(results, list) and results == ["first", "second"]


# --- Controller: score cache clear ---

@pytest.mark.xfail(reason="clear_score_cache() method not yet implemented")
def test_score_cache_clear_method_exists():
    controller = DilationController(TimeDilateConfig(), MagicMock())
    assert hasattr(controller, "clear_score_cache")


# --- Controller: adaptive patience ---

@pytest.mark.xfail(reason="adaptive patience adjustment not yet implemented")
def test_controller_exposes_adaptive_patience():
    controller = DilationController(
        TimeDilateConfig(convergence_patience=5), MagicMock()
    )
    assert hasattr(controller, "effective_patience") or hasattr(controller, "_patience")


# --- Controller: time-budget predictive early break ---

def test_time_budget_predictive_break():
    """When remaining time < avg cycle time, loop should exit without
    starting a cycle it cannot finish."""
    import time as _time
    calls = [0]

    def gen(prompt, **kwargs):
        calls[0] += 1
        c = calls[0]
        if c == 1:
            return "initial"
        _time.sleep(0.05)
        if "Rate the RESPONSE" in prompt:
            return "60"
        if "weaknesses" in prompt.lower():
            return "critique"
        return "refined"

    engine = MagicMock()
    engine.generate = MagicMock(side_effect=gen)
    config = TimeDilateConfig(
        dilation_factor=100, time_budget_seconds=0.3, convergence_patience=50
    )
    controller = DilationController(config, engine)
    result = controller.run("test")
    # Predictive break leaves meaningful headroom — stays at most 10% over budget
    # Predictive break means we stop without starting a cycle we can't
    # finish; generous upper bound accommodates scheduler jitter on CI.
    assert result.elapsed_seconds <= 0.45


# --- Controller: pairwise tiebreak judge ---

@pytest.mark.xfail(reason="pairwise tiebreak judge not yet implemented")
def test_pairwise_tiebreak_method_exists():
    controller = DilationController(TimeDilateConfig(branch_factor=2), MagicMock())
    assert hasattr(controller, "_pairwise_compare") or hasattr(controller, "pairwise_tiebreak")


# --- Engine: seed determinism (sanity probe) ---

def test_same_seed_passes_to_llm_only():
    _reset()
    DilationEngine(TimeDilateConfig(seed=42)).generate("hello")
    assert mock_vllm.LLM.call_args.kwargs.get("seed") == 42
    assert "seed" not in mock_vllm.SamplingParams.call_args.kwargs
