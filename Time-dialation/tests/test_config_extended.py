"""Extended config validation tests."""
import pytest
from timedilate.config import TimeDilateConfig, ConfigError


def test_validate_accepts_factor_1():
    TimeDilateConfig(dilation_factor=1.0).validate()


def test_validate_accepts_large_factor():
    TimeDilateConfig(dilation_factor=1_000_000).validate()


def test_validate_rejects_zero_max_tokens():
    with pytest.raises(ConfigError):
        TimeDilateConfig(max_tokens=0).validate()


def test_validate_rejects_negative_temperature():
    with pytest.raises(ConfigError):
        TimeDilateConfig(temperature=-0.1).validate()


def test_validate_accepts_temperature_boundary_zero():
    TimeDilateConfig(temperature=0.0).validate()


def test_validate_accepts_temperature_boundary_two():
    TimeDilateConfig(temperature=2.0).validate()


def test_validate_rejects_gpu_util_too_low():
    with pytest.raises(ConfigError):
        TimeDilateConfig(gpu_memory_utilization=0.05).validate()


def test_validate_rejects_gpu_util_too_high():
    with pytest.raises(ConfigError):
        TimeDilateConfig(gpu_memory_utilization=1.0).validate()


def test_validate_rejects_bad_dtype():
    with pytest.raises(ConfigError):
        TimeDilateConfig(dtype="int8").validate()


def test_validate_accepts_all_valid_dtypes():
    for d in ("auto", "float16", "bfloat16", "float32"):
        TimeDilateConfig(dtype=d).validate()


def test_validate_rejects_negative_swap_space():
    with pytest.raises(ConfigError):
        TimeDilateConfig(swap_space_gb=-1).validate()


def test_validate_accepts_zero_swap_space():
    TimeDilateConfig(swap_space_gb=0).validate()


def test_validate_rejects_zero_max_model_len():
    with pytest.raises(ConfigError):
        TimeDilateConfig(max_model_len=0).validate()


def test_validate_accepts_none_max_model_len():
    TimeDilateConfig(max_model_len=None).validate()


def test_validate_rejects_zero_patience():
    with pytest.raises(ConfigError):
        TimeDilateConfig(convergence_patience=0).validate()


def test_validate_rejects_early_stop_score_out_of_range():
    with pytest.raises(ConfigError):
        TimeDilateConfig(early_stop_score=101).validate()
    with pytest.raises(ConfigError):
        TimeDilateConfig(early_stop_score=-1).validate()


def test_validate_rejects_negative_branch_spread():
    with pytest.raises(ConfigError):
        TimeDilateConfig(branch_temperature_spread=-0.1).validate()


def test_subjective_time_none_without_budget():
    assert TimeDilateConfig().subjective_time is None


def test_subjective_time_multiplies():
    c = TimeDilateConfig(dilation_factor=100, time_budget_seconds=3)
    assert c.subjective_time == 300


def test_num_cycles_time_budget_mode_returns_zero():
    c = TimeDilateConfig(dilation_factor=100, time_budget_seconds=5)
    assert c.num_cycles == 0


def test_describe_with_time_budget_mentions_subjective_time():
    c = TimeDilateConfig(dilation_factor=1000, time_budget_seconds=5)
    desc = c.describe()
    assert "Subjective time" in desc
    assert "Time budget" in desc


def test_describe_includes_patience():
    desc = TimeDilateConfig().describe()
    assert "patience" in desc.lower()
