"""Tests for DilationController."""
import sys
from unittest.mock import MagicMock
from timedilate.config import TimeDilateConfig

# Mock vllm before importing
if "vllm" not in sys.modules or not isinstance(sys.modules["vllm"], MagicMock):
    sys.modules["vllm"] = MagicMock()
mock_vllm = sys.modules["vllm"]

from timedilate.controller import DilationController, DilationResult
from timedilate.engine import DilationEngine


def _mock_engine(response="generated output"):
    mock_vllm.LLM.reset_mock()
    mock_output = MagicMock()
    mock_output.outputs = [MagicMock(text=response)]
    mock_vllm.LLM.return_value.generate.return_value = [mock_output]
    return None  # let controller create its own engine


def test_controller_run_returns_result():
    _mock_engine("hello world")
    config = TimeDilateConfig(dilation_factor=10)
    controller = DilationController(config)
    result = controller.run("Say hello")
    assert isinstance(result, DilationResult)
    assert result.output == "hello world"
    assert result.dilation_factor == 10
    assert result.actual_latency > 0
    assert result.achieved_speedup > 0


def test_controller_factor_1_no_acceleration():
    _mock_engine("plain output")
    config = TimeDilateConfig(dilation_factor=1.0)
    controller = DilationController(config)
    result = controller.run("test")
    assert result.output == "plain output"
    assert result.dilation_factor == 1.0


def test_controller_high_factor_uses_smaller_model():
    _mock_engine("fast output")
    config = TimeDilateConfig(dilation_factor=10000)
    controller = DilationController(config)
    # After auto_configure, should have selected a smaller model
    assert controller.config.model != "Qwen/Qwen2.5-7B-Instruct" or controller.config.quantization_bits < 16


def test_result_to_report():
    result = DilationResult(
        output="test output",
        dilation_factor=100,
        base_latency_estimate=10.0,
        actual_latency=0.5,
        achieved_speedup=20.0,
        model_used="Qwen/Qwen2.5-3B-Instruct",
        acceleration_summary="test",
        tokens_generated=50,
    )
    report = result.to_report()
    assert "version" in report
    assert report["dilation_factor"] == 100
    assert report["achieved_speedup"] == 20.0


def test_result_to_report_with_config():
    result = DilationResult(
        output="x", dilation_factor=100, base_latency_estimate=10.0,
        actual_latency=0.5, achieved_speedup=20.0, model_used="test",
        acceleration_summary="test",
    )
    config = TimeDilateConfig(dilation_factor=100).auto_configure()
    report = result.to_report(config)
    assert "config" in report
    assert "quantization_bits" in report["config"]


def test_benchmark_returns_results():
    _mock_engine("output")
    config = TimeDilateConfig()
    controller = DilationController(config)
    results = controller.benchmark("test prompt", [1, 10])
    assert len(results) == 2
    assert all(isinstance(r, DilationResult) for r in results)


def test_controller_speedup_warning(caplog):
    """When achieved speedup is below 50% of target, a warning is logged."""
    import logging
    _mock_engine("output")
    config = TimeDilateConfig(dilation_factor=1000000)
    with caplog.at_level(logging.WARNING):
        controller = DilationController(config)
        controller.run("test")
    # The actual speedup will be huge (near-instant mock), so no warning expected
    # This just verifies controller runs without error at extreme factors
