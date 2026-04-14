"""Tests for DilationEngine."""
import sys
from unittest.mock import MagicMock
from timedilate.config import TimeDilateConfig

# Mock vllm before importing engine — reuse existing mock if present
if "vllm" not in sys.modules or not isinstance(sys.modules["vllm"], MagicMock):
    sys.modules["vllm"] = MagicMock()
mock_vllm = sys.modules["vllm"]

from timedilate.engine import DilationEngine, InferenceError


def _make_engine(config=None, response_text="generated output"):
    """Helper to create a DilationEngine with mocked vllm."""
    mock_vllm.LLM.reset_mock()
    mock_vllm.LLM.return_value.generate.side_effect = None
    mock_output = MagicMock()
    mock_output.outputs = [MagicMock(text=response_text)]
    mock_vllm.LLM.return_value.generate.return_value = [mock_output]

    config = config or TimeDilateConfig()
    engine = DilationEngine(config)
    return engine


def test_engine_init():
    engine = _make_engine()
    assert engine._total_calls == 0
    assert not engine._initialized


def test_engine_generate():
    engine = _make_engine(response_text="hello world")
    result = engine.generate("Say hello")
    assert result == "hello world"
    assert engine._total_calls == 1


def test_engine_tracks_stats():
    engine = _make_engine(response_text="output text here")
    engine.generate("test 1")
    engine.generate("test 2")
    stats = engine.stats
    assert stats["total_calls"] == 2
    assert stats["failed_calls"] == 0
    assert stats["total_latency_s"] >= 0


def test_engine_retries_on_empty():
    mock_vllm.LLM.reset_mock()
    empty = MagicMock()
    empty.outputs = [MagicMock(text="")]
    good = MagicMock()
    good.outputs = [MagicMock(text="result")]
    mock_vllm.LLM.return_value.generate.side_effect = [[empty], [good]]

    engine = DilationEngine(TimeDilateConfig())
    result = engine.generate("test", retries=1)
    assert result == "result"


def test_engine_raises_after_retries():
    mock_vllm.LLM.reset_mock()
    mock_vllm.LLM.return_value.generate.side_effect = RuntimeError("OOM")

    engine = DilationEngine(TimeDilateConfig())
    try:
        engine.generate("test", retries=0)
        assert False, "Should have raised"
    except InferenceError as e:
        assert "OOM" in str(e)


def test_engine_token_budget_compression():
    """When token_budget_ratio < 1.0, effective max_tokens should be reduced."""
    config = TimeDilateConfig(max_tokens=4096, token_budget_ratio=0.25)
    engine = _make_engine(config)
    engine.generate("test")
    # Check that SamplingParams was called with reduced max_tokens
    call_args = mock_vllm.SamplingParams.call_args
    assert call_args[1]["max_tokens"] == 1024  # 4096 * 0.25


def test_engine_prompt_compression():
    """Long prompts should be compressed when prompt_compression is enabled."""
    config = TimeDilateConfig(prompt_compression=True)
    engine = _make_engine(config)
    long_prompt = "x" * 5000
    engine.generate(long_prompt)
    # The actual prompt passed to the model should be shorter
    model_call = mock_vllm.LLM.return_value.generate.call_args
    actual_prompt = model_call[0][0][0]
    assert len(actual_prompt) < len(long_prompt)
    assert "compressed" in actual_prompt.lower()


def test_engine_prompt_compression_short_passthrough():
    """Short prompts should NOT be compressed."""
    config = TimeDilateConfig(prompt_compression=True)
    engine = _make_engine(config)
    short_prompt = "Write hello world"
    engine.generate(short_prompt)
    model_call = mock_vllm.LLM.return_value.generate.call_args
    actual_prompt = model_call[0][0][0]
    assert actual_prompt == short_prompt


def test_engine_speculative_decoding_configured():
    """Speculative decoding should be configured in vllm kwargs."""
    config = TimeDilateConfig(speculative_tokens=20, draft_model="small-model")
    engine = _make_engine(config)
    engine.generate("test")
    llm_call = mock_vllm.LLM.call_args
    assert llm_call[1]["speculative_model"] == "small-model"
    assert llm_call[1]["num_speculative_tokens"] == 20


def test_estimate_tokens():
    engine = _make_engine()
    assert engine.estimate_tokens("a" * 400) == 100


def test_avg_latency_no_calls():
    engine = _make_engine()
    assert engine.avg_latency == 0.0
