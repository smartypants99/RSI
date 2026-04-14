"""Tests for DilationEngine."""
import sys
from unittest.mock import MagicMock
from timedilate.config import TimeDilateConfig

# Mock vllm before importing engine
if "vllm" not in sys.modules or not isinstance(sys.modules["vllm"], MagicMock):
    sys.modules["vllm"] = MagicMock()
mock_vllm = sys.modules["vllm"]

from timedilate.engine import DilationEngine, InferenceError


def _make_engine(config=None, response_text="generated output"):
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


def test_engine_retries_on_empty():
    mock_vllm.LLM.reset_mock()
    mock_vllm.LLM.return_value.generate.side_effect = None
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
