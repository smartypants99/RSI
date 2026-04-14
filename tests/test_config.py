"""Tests for TimeDilateConfig and auto-configuration."""
import pytest
from timedilate.config import TimeDilateConfig, ConfigError


def test_default_config():
    config = TimeDilateConfig()
    assert config.dilation_factor == 1.0
    assert config.model == "Qwen/Qwen2.5-7B-Instruct"


def test_validate_rejects_negative_factor():
    config = TimeDilateConfig(dilation_factor=0.5)
    with pytest.raises(ConfigError):
        config.validate()


def test_validate_rejects_bad_temperature():
    config = TimeDilateConfig(temperature=3.0)
    with pytest.raises(ConfigError):
        config.validate()


def test_validate_rejects_bad_gpu_memory():
    config = TimeDilateConfig(gpu_memory_gb=0)
    with pytest.raises(ConfigError):
        config.validate()


def test_auto_configure_factor_1():
    """Factor 1 should not change anything."""
    config = TimeDilateConfig(dilation_factor=1.0).auto_configure()
    assert config.dilation_factor == 1.0


def test_auto_configure_low_factor():
    """Low factor (10x) should use speculative decoding + maybe quantization."""
    config = TimeDilateConfig(dilation_factor=10).auto_configure()
    assert config.speculative_tokens >= 5


def test_auto_configure_medium_factor():
    """Medium factor (100x) should cascade to smaller model + quantization."""
    config = TimeDilateConfig(dilation_factor=100).auto_configure()
    assert config.quantization_bits is not None
    assert config.quantization_bits < 16


def test_auto_configure_high_factor():
    """High factor (10000x) should use aggressive settings."""
    config = TimeDilateConfig(dilation_factor=10000).auto_configure()
    assert config.quantization_bits is not None
    assert config.quantization_bits <= 4
    # At 10000x, cascades may fully cover it; model should be small
    assert "0.5B" in config.model or "1.5B" in config.model


def test_auto_configure_extreme_factor():
    """Extreme factor (1000000x) should stack everything."""
    config = TimeDilateConfig(dilation_factor=1_000_000).auto_configure()
    assert config.token_budget_ratio < 0.5
    assert config.quantization_bits <= 4


def test_auto_configure_preserves_factor():
    """auto_configure should preserve the original dilation_factor."""
    config = TimeDilateConfig(dilation_factor=500).auto_configure()
    assert config.dilation_factor == 500


def test_describe_acceleration_basic():
    config = TimeDilateConfig(dilation_factor=1.0)
    desc = config.describe_acceleration()
    assert "1.0x" in desc


def test_describe_acceleration_with_quantization():
    config = TimeDilateConfig(dilation_factor=100).auto_configure()
    desc = config.describe_acceleration()
    assert "bit" in desc.lower() or "Quantization" in desc


def test_model_cascade_selects_smaller():
    """Higher factors should select smaller models from cascade."""
    config_low = TimeDilateConfig(dilation_factor=2).auto_configure()
    config_high = TimeDilateConfig(dilation_factor=10000).auto_configure()
    # High factor should use a different (smaller) model or same with more acceleration
    assert config_high.model != config_low.model or config_high.quantization_bits < config_low.quantization_bits


def test_token_budget_decreases_with_factor():
    """Higher factors should reduce token budget."""
    config_low = TimeDilateConfig(dilation_factor=10).auto_configure()
    config_extreme = TimeDilateConfig(dilation_factor=1_000_000).auto_configure()
    assert config_extreme.token_budget_ratio <= config_low.token_budget_ratio
