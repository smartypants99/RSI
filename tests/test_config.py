"""Tests for TimeDilateConfig."""
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


def test_num_cycles_factor_1():
    config = TimeDilateConfig(dilation_factor=1.0)
    assert config.num_cycles == 0


def test_num_cycles_factor_10():
    config = TimeDilateConfig(dilation_factor=10)
    assert config.num_cycles == 10


def test_num_cycles_factor_1000():
    config = TimeDilateConfig(dilation_factor=1000)
    assert config.num_cycles == 1000


def test_num_cycles_factor_million():
    config = TimeDilateConfig(dilation_factor=1_000_000)
    assert config.num_cycles == 1_000_000


def test_num_cycles_scales_infinitely():
    """No ceiling — any factor works."""
    config = TimeDilateConfig(dilation_factor=1_000_000_000_000)
    assert config.num_cycles == 1_000_000_000_000


def test_describe():
    config = TimeDilateConfig(dilation_factor=100)
    desc = config.describe()
    assert "100" in desc
    assert "100" in desc  # cycles


def test_describe_with_branch_factor():
    config = TimeDilateConfig(dilation_factor=10, branch_factor=3)
    desc = config.describe()
    assert "3" in desc
