import pytest
from timedilate.config import TimeDilateConfig, ConfigError


def test_valid_config():
    config = TimeDilateConfig()
    config.validate()  # should not raise


def test_invalid_dilation_factor():
    with pytest.raises(ConfigError, match="dilation_factor"):
        TimeDilateConfig(dilation_factor=0).validate()


def test_invalid_branch_factor():
    with pytest.raises(ConfigError, match="branch_factor"):
        TimeDilateConfig(branch_factor=0).validate()


def test_invalid_temperature():
    with pytest.raises(ConfigError, match="temperature"):
        TimeDilateConfig(temperature=3.0).validate()


def test_invalid_budget():
    with pytest.raises(ConfigError, match="budget_seconds"):
        TimeDilateConfig(budget_seconds=-1).validate()


def test_invalid_score_weights_keys():
    with pytest.raises(ConfigError, match="Invalid score_weights"):
        TimeDilateConfig(score_weights={"correctness": 50, "bogus": 50}).validate()


def test_invalid_task_type_override():
    with pytest.raises(ConfigError, match="task_type_override"):
        TimeDilateConfig(task_type_override="invalid").validate()


def test_valid_task_type_override():
    config = TimeDilateConfig(task_type_override="code")
    config.validate()  # should not raise


def test_valid_score_weights():
    config = TimeDilateConfig(score_weights={"correctness": 60, "completeness": 20, "quality": 10, "elegance": 10})
    config.validate()  # should not raise
