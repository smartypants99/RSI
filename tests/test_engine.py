import sys
from unittest.mock import MagicMock, patch
from timedilate.config import TimeDilateConfig

# Mock vllm before importing engine
mock_vllm = MagicMock()
sys.modules["vllm"] = mock_vllm


from timedilate.engine import InferenceEngine


def test_engine_init():
    config = TimeDilateConfig()
    mock_vllm.LLM.reset_mock()
    engine = InferenceEngine(config)
    assert engine.config == config


def test_generate_calls_model():
    config = TimeDilateConfig()
    mock_vllm.LLM.reset_mock()
    mock_output = MagicMock()
    mock_output.outputs = [MagicMock(text="def hello(): pass")]
    mock_vllm.LLM.return_value.generate.return_value = [mock_output]

    engine = InferenceEngine(config)
    result = engine.generate("Write a function")
    assert result == "def hello(): pass"


def test_estimate_tokens():
    config = TimeDilateConfig()
    mock_vllm.LLM.reset_mock()
    engine = InferenceEngine(config)
    assert engine.estimate_tokens("a" * 400) == 100


def test_engine_stats():
    config = TimeDilateConfig()
    mock_vllm.LLM.reset_mock()
    mock_output = MagicMock()
    mock_output.outputs = [MagicMock(text="hello world")]
    mock_vllm.LLM.return_value.generate.return_value = [mock_output]

    engine = InferenceEngine(config)
    engine.generate("test")
    engine.generate("test2")
    stats = engine.stats
    assert stats["total_calls"] == 2
    assert stats["total_tokens_generated"] > 0


def test_engine_retries_on_empty():
    """Engine retries when model returns empty."""
    config = TimeDilateConfig()
    mock_vllm.LLM.reset_mock()
    empty_output = MagicMock()
    empty_output.outputs = [MagicMock(text="")]
    good_output = MagicMock()
    good_output.outputs = [MagicMock(text="good result")]
    mock_vllm.LLM.return_value.generate.side_effect = [[empty_output], [good_output]]

    engine = InferenceEngine(config)
    result = engine.generate("test", retries=1)
    assert result == "good result"


def test_engine_stats_include_latency_and_failures():
    config = TimeDilateConfig()
    mock_vllm.LLM.reset_mock()
    mock_output = MagicMock()
    mock_output.outputs = [MagicMock(text="hello")]
    mock_vllm.LLM.return_value.generate.side_effect = None
    mock_vllm.LLM.return_value.generate.return_value = [mock_output]

    engine = InferenceEngine(config)
    engine.generate("test")
    stats = engine.stats
    assert "avg_latency_s" in stats
    assert "failed_calls" in stats
    assert stats["failed_calls"] == 0
    assert stats["avg_latency_s"] >= 0


def test_engine_tracks_failed_calls():
    config = TimeDilateConfig()
    mock_vllm.LLM.reset_mock()
    good_output = MagicMock()
    good_output.outputs = [MagicMock(text="ok")]
    mock_vllm.LLM.return_value.generate.side_effect = [
        RuntimeError("fail"), [good_output]
    ]
    engine = InferenceEngine(config)
    result = engine.generate("test", retries=1)
    assert result == "ok"
    assert engine.stats["failed_calls"] == 1


def test_engine_raises_after_retries():
    """Engine raises InferenceError after exhausting retries."""
    from timedilate.engine import InferenceError
    config = TimeDilateConfig()
    mock_vllm.LLM.reset_mock()
    mock_vllm.LLM.return_value.generate.side_effect = RuntimeError("OOM")

    engine = InferenceEngine(config)
    try:
        engine.generate("test", retries=0)
        assert False, "Should have raised"
    except InferenceError as e:
        assert "OOM" in str(e)
