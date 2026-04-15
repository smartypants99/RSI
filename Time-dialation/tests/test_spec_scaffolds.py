"""Specification scaffolds — tests for features not yet implemented.

These are marked xfail(strict=True) so they flip to xpass (fail) once
the feature lands, forcing the implementer to remove the marker and
promote the test to green.
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

@pytest.mark.xfail(strict=True, reason="engine.initialize does not yet pass dtype")
def test_dtype_propagates_to_llm_kwargs():
    _reset()
    engine = DilationEngine(TimeDilateConfig(dtype="bfloat16"))
    engine.generate("p")
    assert mock_vllm.LLM.call_args.kwargs["dtype"] == "bfloat16"


@pytest.mark.xfail(strict=True, reason="engine.initialize does not yet pass enforce_eager")
def test_enforce_eager_propagates():
    _reset()
    engine = DilationEngine(TimeDilateConfig(enforce_eager=True))
    engine.generate("p")
    assert mock_vllm.LLM.call_args.kwargs["enforce_eager"] is True


@pytest.mark.xfail(strict=True, reason="engine.initialize does not yet pass swap_space")
def test_swap_space_propagates():
    _reset()
    engine = DilationEngine(TimeDilateConfig(swap_space_gb=8))
    engine.generate("p")
    kw = mock_vllm.LLM.call_args.kwargs
    assert kw.get("swap_space") == 8 or kw.get("swap_space_gb") == 8


@pytest.mark.xfail(strict=True, reason="SamplingParams does not yet receive seed")
def test_seed_propagates_to_sampling_params():
    _reset()
    engine = DilationEngine(TimeDilateConfig(seed=1234))
    engine.generate("p")
    assert mock_vllm.SamplingParams.call_args.kwargs.get("seed") == 1234


@pytest.mark.xfail(strict=True, reason="seed=None should mean no seed key set")
def test_seed_none_not_passed_to_sampling_params():
    _reset()
    engine = DilationEngine(TimeDilateConfig(seed=None))
    engine.generate("p")
    # seed key absent or None
    kw = mock_vllm.SamplingParams.call_args.kwargs
    assert kw.get("seed", None) is None


# --- Engine: stop sequences + batched generate ---

@pytest.mark.xfail(strict=True, reason="engine.generate does not yet accept stop=")
def test_stop_sequences_propagate():
    _reset()
    engine = DilationEngine(TimeDilateConfig())
    engine.generate("p", stop=["\n\n", "END"])
    kw = mock_vllm.SamplingParams.call_args.kwargs
    assert kw["stop"] == ["\n\n", "END"]


@pytest.mark.xfail(strict=True, reason="batched generate() not yet implemented")
def test_batched_generate_returns_list():
    _reset()
    o1, o2 = MagicMock(), MagicMock()
    o1.outputs = [MagicMock(text="first")]
    o2.outputs = [MagicMock(text="second")]
    mock_vllm.LLM.return_value.generate.return_value = [o1, o2]
    engine = DilationEngine(TimeDilateConfig())
    results = engine.generate(["p1", "p2"])
    assert results == ["first", "second"]


# --- Engine: seed determinism (two runs, same seed, same mock call sequence) ---

@pytest.mark.xfail(strict=True, reason="depends on seed wiring into SamplingParams")
def test_same_seed_produces_same_sampling_params_seed():
    _reset()
    e1 = DilationEngine(TimeDilateConfig(seed=42))
    e1.generate("hello")
    s1 = mock_vllm.SamplingParams.call_args.kwargs["seed"]

    _reset()
    e2 = DilationEngine(TimeDilateConfig(seed=42))
    e2.generate("hello")
    s2 = mock_vllm.SamplingParams.call_args.kwargs["seed"]

    assert s1 == s2 == 42


# --- Controller: score cache clear ---

@pytest.mark.xfail(strict=True, reason="clear_score_cache() method not yet implemented")
def test_score_cache_clear_resets_hits_and_entries():
    engine = MagicMock()
    engine.generate = MagicMock(return_value="75")
    controller = DilationController(TimeDilateConfig(), engine)
    controller._score("p", "o")
    controller._score("p", "o")  # cache hit
    assert controller._score_cache_hits == 1
    controller.clear_score_cache()
    assert controller._score_cache_hits == 0
    assert len(controller._score_cache) == 0


# --- Controller: adaptive patience ---

@pytest.mark.xfail(strict=True, reason="adaptive patience adjustment not yet implemented")
def test_patience_adapts_after_reset():
    """After a convergence reset, patience should adjust (grow or shrink
    per the strategy chosen by controller-engineer)."""
    from timedilate.controller import DilationController
    controller = DilationController(
        TimeDilateConfig(convergence_patience=5), MagicMock()
    )
    # Expect an API like controller.effective_patience or controller._patience
    assert hasattr(controller, "effective_patience") or hasattr(controller, "_patience")


# --- Controller: time-budget predictive early break ---

@pytest.mark.xfail(strict=True, reason="predictive early-break not yet implemented")
def test_time_budget_predictive_break_when_remaining_lt_avg_cycle():
    """When remaining time < avg cycle time, loop should exit rather than
    start a cycle it cannot finish."""
    import time as _time
    calls = [0]

    def gen(prompt, **kwargs):
        calls[0] += 1
        c = calls[0]
        if c == 1:
            return "initial"
        if "Rate the RESPONSE" in prompt:
            _time.sleep(0.05)
            return "60"
        if "weaknesses" in prompt.lower() or "reviewing" in prompt.lower():
            _time.sleep(0.05)
            return "critique"
        _time.sleep(0.05)
        return "refined"

    engine = MagicMock()
    engine.generate = MagicMock(side_effect=gen)
    # Budget only fits ~2 cycles; predictive break should stop before starting a 3rd
    config = TimeDilateConfig(
        dilation_factor=100, time_budget_seconds=0.3, convergence_patience=50
    )
    controller = DilationController(config, engine)
    result = controller.run("test")
    # Predictive break should leave headroom — result time well under budget+1 cycle
    assert result.elapsed_seconds <= 0.35


# --- Controller: pairwise tiebreak (if implemented as pairwise judge, not stable sort) ---

@pytest.mark.xfail(strict=True, reason="pairwise tiebreak judge not yet implemented")
def test_pairwise_tiebreak_calls_judge_on_equal_scores():
    """If pairwise tiebreak is implemented, equal-score branches trigger
    a judge call. Current impl uses stable sort only."""
    engine = MagicMock()
    # Expect a _pairwise_compare or similar method
    controller = DilationController(TimeDilateConfig(branch_factor=2), engine)
    assert hasattr(controller, "_pairwise_compare") or hasattr(controller, "pairwise_tiebreak")
