"""Engine round-2 correctness tests.

Covers:
- Batched retry re-issues only empty subset (not whole batch)
- Stats not double-counted on empty retries
- _looks_like_oom tightening (word boundaries; "room" does NOT fire)
- raise-from preserves __cause__
- Public package exports
"""
import pytest

import timedilate
from timedilate.config import TimeDilateConfig
from timedilate.engine import (
    DilationEngine, InferenceError, _looks_like_oom, _OOM_PATTERN,
)


# --- Batched retry re-issues only empty subset ---

def test_batched_retry_reissues_only_empty_subset(vllm_mock, make_vllm_output):
    """First call: [ok1, EMPTY, ok3]. Retry should include only prompt[1]."""
    first = [
        make_vllm_output("ok1"),
        make_vllm_output(""),
        make_vllm_output("ok3"),
    ]
    second = [make_vllm_output("recovered2")]
    vllm_mock.LLM.return_value.generate.side_effect = [first, second]

    engine = DilationEngine(TimeDilateConfig())
    result = engine.generate(["p1", "p2", "p3"], retries=1)

    assert result == ["ok1", "recovered2", "ok3"]
    # Second generate() call's first arg must be only the pending prompt
    second_call_prompts = vllm_mock.LLM.return_value.generate.call_args_list[1].args[0]
    assert second_call_prompts == ["p2"]


def test_batched_retry_multiple_empty_indices(vllm_mock, make_vllm_output):
    first = [
        make_vllm_output(""),
        make_vllm_output("ok"),
        make_vllm_output(""),
        make_vllm_output("ok4"),
    ]
    second = [make_vllm_output("fix0"), make_vllm_output("fix2")]
    vllm_mock.LLM.return_value.generate.side_effect = [first, second]

    engine = DilationEngine(TimeDilateConfig())
    result = engine.generate(["a", "b", "c", "d"], retries=1)
    assert result == ["fix0", "ok", "fix2", "ok4"]
    retry_prompts = vllm_mock.LLM.return_value.generate.call_args_list[1].args[0]
    assert retry_prompts == ["a", "c"]


def test_batched_retry_all_succeed_no_retry(vllm_mock, make_vllm_output):
    """If no slots empty on first attempt, no retry call happens."""
    vllm_mock.LLM.return_value.generate.return_value = [
        make_vllm_output("a"), make_vllm_output("b"),
    ]
    engine = DilationEngine(TimeDilateConfig())
    engine.generate(["p1", "p2"], retries=2)
    assert vllm_mock.LLM.return_value.generate.call_count == 1


# --- Stats not double-counted across retries ---

def test_stats_not_double_counted_on_empty_retry(vllm_mock, make_vllm_output):
    """After one empty-retry cycle and success, total_calls should reflect
    the single logical generate() request, not retry attempts."""
    first = [make_vllm_output("")]
    second = [make_vllm_output("recovered", token_ids=[1, 2, 3])]
    vllm_mock.LLM.return_value.generate.side_effect = [first, second]

    engine = DilationEngine(TimeDilateConfig())
    engine.generate("p", retries=1)

    # One logical call: total_calls counts outputs, not attempts
    assert engine.stats["total_calls"] == 1
    assert engine.stats["total_output_tokens"] == 3


def test_stats_token_counts_do_not_include_empty_attempt(vllm_mock, make_vllm_output):
    """Token totals should sum only the final-winning attempt's tokens,
    not the sum of all attempts (would double-count prompt tokens)."""
    empty = [make_vllm_output("", token_ids=[], prompt_token_ids=[1, 2, 3])]
    good = [make_vllm_output("ok", token_ids=[9, 9], prompt_token_ids=[1, 2, 3])]
    vllm_mock.LLM.return_value.generate.side_effect = [empty, good]

    engine = DilationEngine(TimeDilateConfig())
    engine.generate("p", retries=1)
    # Input tokens counted once (not 6 from doubling)
    assert engine.stats["total_input_tokens"] == 3
    assert engine.stats["total_output_tokens"] == 2


# --- _looks_like_oom tightening (word boundaries) ---

def test_oom_false_positive_room_not_oom():
    """The substring 'oom' should not match inside other words like 'room'."""
    assert not _looks_like_oom(RuntimeError("mushroom kingdom"))
    assert not _looks_like_oom(RuntimeError("gloom and doom"))
    assert not _looks_like_oom(RuntimeError("Zoom meeting failed"))


def test_oom_true_positive_vllm_kv_cache():
    assert _looks_like_oom(RuntimeError("Not enough KV cache blocks available"))
    assert _looks_like_oom(RuntimeError("not enough kv cache for this request"))


def test_oom_true_positive_no_available_memory():
    assert _looks_like_oom(RuntimeError("No available memory on device"))


def test_oom_pattern_requires_word_boundary():
    """Sanity: regex uses \\b to avoid substring false positives."""
    assert _OOM_PATTERN.search("CUDA out of memory") is not None
    assert _OOM_PATTERN.search("plenty of room") is None


def test_oom_case_insensitive():
    assert _looks_like_oom(RuntimeError("CUDA OUT OF MEMORY"))
    assert _looks_like_oom(RuntimeError("cuda out of memory"))


# --- raise-from preserves __cause__ ---

def test_init_failure_preserves_cause(vllm_mock):
    original = RuntimeError("original cuda failure")
    vllm_mock.LLM.side_effect = original
    engine = DilationEngine(TimeDilateConfig())
    with pytest.raises(InferenceError) as exc_info:
        engine.initialize()
    assert exc_info.value.__cause__ is original


def test_oom_exhausted_chain_preserves_last_cause(vllm_mock):
    oom1 = RuntimeError("CUDA out of memory #1")
    oom2 = RuntimeError("CUDA out of memory #2")
    oom3 = RuntimeError("CUDA out of memory #3")
    vllm_mock.LLM.side_effect = [oom1, oom2, oom3]
    engine = DilationEngine(TimeDilateConfig())
    with pytest.raises(InferenceError) as exc_info:
        engine.initialize()
    # After exhausting fallbacks the last OOM is the cause
    assert exc_info.value.__cause__ is oom3


# --- Public package exports ---

def test_public_exports_configured():
    assert hasattr(timedilate, "TimeDilateConfig")
    assert hasattr(timedilate, "ConfigError")
    assert hasattr(timedilate, "DilationController")
    assert hasattr(timedilate, "DilationResult")
    assert hasattr(timedilate, "CycleRecord")
    assert hasattr(timedilate, "DilationEngine")
    assert hasattr(timedilate, "InferenceError")
    assert hasattr(timedilate, "__version__")


def test_all_list_is_exhaustive():
    expected = {
        "__version__", "TimeDilateConfig", "ConfigError",
        "DilationController", "DilationResult", "CycleRecord",
        "DilationEngine", "InferenceError",
    }
    assert set(timedilate.__all__) == expected


def test_version_is_string():
    assert isinstance(timedilate.__version__, str)
    assert timedilate.__version__


def test_error_types_are_exceptions():
    assert issubclass(timedilate.InferenceError, RuntimeError)
    assert issubclass(timedilate.ConfigError, ValueError)
