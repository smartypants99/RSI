"""Finalized engine tests against stable API.

API: generate(prompt | list[str], max_tokens=None, temperature=None,
retries=2, stop=None) — str in -> str out, list in -> list out.
initialize() auto-falls-back through 0.60 -> 0.45 -> 0.30 on OOM.
stats exposes oom_retries, effective_gpu_util, token counts.
health_check() tiny-ping probe.
"""
import sys
from unittest.mock import MagicMock
import pytest

from timedilate.config import TimeDilateConfig

if "vllm" not in sys.modules or not isinstance(sys.modules["vllm"], MagicMock):
    sys.modules["vllm"] = MagicMock()
mock_vllm = sys.modules["vllm"]

from timedilate.engine import (
    DilationEngine, InferenceError, _looks_like_oom, _OOM_FALLBACK_UTILS,
)


def _make_output(text, token_ids=None, prompt_token_ids=None):
    o = MagicMock()
    inner = MagicMock()
    inner.text = text
    inner.token_ids = token_ids if token_ids is not None else [0] * 3
    o.outputs = [inner]
    o.prompt_token_ids = prompt_token_ids if prompt_token_ids is not None else [0] * 2
    return o


def _reset():
    mock_vllm.reset_mock()
    mock_vllm.LLM.reset_mock()
    mock_vllm.LLM.side_effect = None
    mock_vllm.SamplingParams.reset_mock()
    mock_vllm.SamplingParams.side_effect = None
    mock_vllm.LLM.return_value.generate.side_effect = None
    mock_vllm.LLM.return_value.generate.return_value = [_make_output("ok")]


# --- OOM init fallback chain ---

def test_oom_fallback_utils_constant():
    assert _OOM_FALLBACK_UTILS == (0.45, 0.30)


def test_looks_like_oom_recognizer():
    assert _looks_like_oom(RuntimeError("CUDA out of memory"))
    assert _looks_like_oom(RuntimeError("cuda oom"))
    assert _looks_like_oom(RuntimeError("No Memory available"))
    assert not _looks_like_oom(RuntimeError("network error"))
    # class-name based recognition
    class OutOfMemoryError(Exception):
        pass
    assert _looks_like_oom(OutOfMemoryError("x"))


def test_init_falls_back_once_on_oom():
    _reset()
    good_llm = MagicMock()
    good_llm.generate.return_value = [_make_output("ok")]
    mock_vllm.LLM.side_effect = [RuntimeError("CUDA out of memory"), good_llm]
    engine = DilationEngine(TimeDilateConfig(gpu_memory_utilization=0.60))
    engine.initialize()
    assert engine._effective_gpu_util == 0.45
    assert engine.stats["oom_retries"] == 1
    assert engine.stats["effective_gpu_util"] == 0.45


def test_init_falls_back_twice_on_repeated_oom():
    _reset()
    good_llm = MagicMock()
    good_llm.generate.return_value = [_make_output("ok")]
    mock_vllm.LLM.side_effect = [
        RuntimeError("CUDA out of memory"),
        RuntimeError("CUDA OOM"),
        good_llm,
    ]
    engine = DilationEngine(TimeDilateConfig(gpu_memory_utilization=0.60))
    engine.initialize()
    assert engine._effective_gpu_util == 0.30
    assert engine.stats["oom_retries"] == 2


def test_init_non_oom_error_does_not_fall_back():
    _reset()
    mock_vllm.LLM.side_effect = RuntimeError("totally different problem")
    engine = DilationEngine(TimeDilateConfig())
    with pytest.raises(InferenceError) as ei:
        engine.initialize()
    assert "totally different problem" in str(ei.value)
    assert engine.stats["oom_retries"] == 0


def test_init_all_fallbacks_exhausted_raises():
    _reset()
    mock_vllm.LLM.side_effect = RuntimeError("CUDA out of memory")
    engine = DilationEngine(TimeDilateConfig())
    with pytest.raises(InferenceError):
        engine.initialize()
    # 2 retries at fallback utils (primary + 2 fallbacks = 3 attempts, 2 oom_retries)
    assert engine.stats["oom_retries"] == 2


# --- Single-prompt API preserves str return ---

def test_str_prompt_returns_str():
    _reset()
    mock_vllm.LLM.return_value.generate.return_value = [_make_output("hello")]
    engine = DilationEngine(TimeDilateConfig())
    result = engine.generate("p")
    assert isinstance(result, str)
    assert result == "hello"


# --- Batched API returns list ---

def test_list_prompt_returns_list():
    _reset()
    mock_vllm.LLM.return_value.generate.return_value = [
        _make_output("a"), _make_output("b"), _make_output("c"),
    ]
    engine = DilationEngine(TimeDilateConfig())
    result = engine.generate(["p1", "p2", "p3"])
    assert isinstance(result, list)
    assert result == ["a", "b", "c"]


def test_empty_batch_returns_empty_list():
    _reset()
    engine = DilationEngine(TimeDilateConfig())
    assert engine.generate([]) == []


def test_generate_batch_explicit_always_list():
    _reset()
    mock_vllm.LLM.return_value.generate.return_value = [_make_output("only")]
    engine = DilationEngine(TimeDilateConfig())
    result = engine.generate_batch(["solo"])
    assert isinstance(result, list) and result == ["only"]


# --- Stop sequences ---

def test_stop_sequences_passed_as_list():
    _reset()
    engine = DilationEngine(TimeDilateConfig())
    engine.generate("p", stop=["\n\n", "END"])
    assert mock_vllm.SamplingParams.call_args.kwargs["stop"] == ["\n\n", "END"]


def test_stop_none_not_added_to_sampling_params():
    _reset()
    engine = DilationEngine(TimeDilateConfig())
    engine.generate("p")
    assert "stop" not in mock_vllm.SamplingParams.call_args.kwargs


def test_stop_tuple_accepted_as_sequence():
    _reset()
    engine = DilationEngine(TimeDilateConfig())
    engine.generate("p", stop=("STOP",))
    assert mock_vllm.SamplingParams.call_args.kwargs["stop"] == ["STOP"]


# --- Seed propagation ---

def test_seed_set_propagates_to_llm_only_preserves_branch_diversity():
    """Seed goes to LLM() (reproducible base sampler) but NOT to SamplingParams
    per-call — otherwise branch_factor with temperature spread would collapse
    to a single output."""
    _reset()
    engine = DilationEngine(TimeDilateConfig(seed=7))
    engine.generate("p")
    assert mock_vllm.LLM.call_args.kwargs["seed"] == 7
    assert "seed" not in mock_vllm.SamplingParams.call_args.kwargs


def test_seed_none_not_in_kwargs():
    _reset()
    engine = DilationEngine(TimeDilateConfig(seed=None))
    engine.generate("p")
    assert "seed" not in mock_vllm.LLM.call_args.kwargs
    assert "seed" not in mock_vllm.SamplingParams.call_args.kwargs


def test_seed_determinism_two_engines_same_llm_seed():
    _reset()
    DilationEngine(TimeDilateConfig(seed=42)).generate("hello")
    first_llm = mock_vllm.LLM.call_args.kwargs["seed"]

    _reset()
    DilationEngine(TimeDilateConfig(seed=42)).generate("hello")
    second_llm = mock_vllm.LLM.call_args.kwargs["seed"]

    assert first_llm == second_llm == 42


# --- dtype / enforce_eager / swap_space ---

def test_dtype_propagates():
    _reset()
    DilationEngine(TimeDilateConfig(dtype="bfloat16")).generate("p")
    assert mock_vllm.LLM.call_args.kwargs["dtype"] == "bfloat16"


def test_enforce_eager_propagates():
    _reset()
    DilationEngine(TimeDilateConfig(enforce_eager=True)).generate("p")
    assert mock_vllm.LLM.call_args.kwargs["enforce_eager"] is True


def test_swap_space_propagates():
    _reset()
    DilationEngine(TimeDilateConfig(swap_space_gb=12)).generate("p")
    assert mock_vllm.LLM.call_args.kwargs["swap_space"] == 12


# --- Token counting / usage tracking ---

def test_last_usage_populated_after_generate():
    _reset()
    out = _make_output("hello", token_ids=[1, 2, 3, 4, 5], prompt_token_ids=[10, 20])
    mock_vllm.LLM.return_value.generate.return_value = [out]
    engine = DilationEngine(TimeDilateConfig())
    engine.generate("p")
    usage = engine.last_usage
    assert usage["output_tokens"] == 5
    assert usage["input_tokens"] == 2
    assert usage["total_tokens"] == 7


def test_last_token_counts_per_prompt():
    _reset()
    outs = [
        _make_output("a", token_ids=[1, 2]),
        _make_output("bb", token_ids=[3, 4, 5]),
    ]
    mock_vllm.LLM.return_value.generate.return_value = outs
    engine = DilationEngine(TimeDilateConfig())
    engine.generate(["p1", "p2"])
    assert engine.last_token_counts == [2, 3]


def test_total_tokens_accumulate_across_calls():
    _reset()
    mock_vllm.LLM.return_value.generate.return_value = [
        _make_output("x", token_ids=[1, 2, 3], prompt_token_ids=[9]),
    ]
    engine = DilationEngine(TimeDilateConfig())
    engine.generate("p1")
    engine.generate("p2")
    assert engine.stats["total_output_tokens"] == 6
    assert engine.stats["total_input_tokens"] == 2


def test_generate_with_usage_returns_tuple():
    _reset()
    out = _make_output("result", token_ids=[1, 2], prompt_token_ids=[3])
    mock_vllm.LLM.return_value.generate.return_value = [out]
    engine = DilationEngine(TimeDilateConfig())
    text, usage = engine.generate_with_usage("p")
    assert text == "result"
    assert usage["output_tokens"] == 2
    assert usage["input_tokens"] == 1


# --- Health check ---

def test_health_check_ok():
    _reset()
    engine = DilationEngine(TimeDilateConfig())
    assert engine.health_check() is True


def test_health_check_returns_false_on_exception():
    _reset()
    mock_vllm.LLM.side_effect = RuntimeError("non-OOM init fail")
    engine = DilationEngine(TimeDilateConfig())
    assert engine.health_check() is False


# --- Retries still work with new signature ---

def test_retries_on_empty_batched():
    _reset()
    empty_batch = [_make_output("")]
    good_batch = [_make_output("recovered")]
    mock_vllm.LLM.return_value.generate.side_effect = [empty_batch, good_batch]
    engine = DilationEngine(TimeDilateConfig())
    assert engine.generate("p", retries=1) == "recovered"


def test_stats_exposes_new_fields():
    _reset()
    engine = DilationEngine(TimeDilateConfig())
    engine.generate("p")
    stats = engine.stats
    assert "oom_retries" in stats
    assert "effective_gpu_util" in stats
    assert "total_output_tokens" in stats
    assert "total_input_tokens" in stats
    assert stats["effective_gpu_util"] == pytest.approx(0.60)
