from dataclasses import dataclass


class ConfigError(ValueError):
    """Raised when configuration is invalid."""
    pass


@dataclass
class TimeDilateConfig:
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    draft_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dilation_factor: int = 2
    budget_seconds: float = 30.0
    branch_factor: int = 3
    max_tokens: int = 4096
    temperature: float = 0.7
    scoring_temperature: float = 0.0
    checkpoint_dir: str = ".timedilate_checkpoints"
    checkpoint_interval: int = 10
    convergence_threshold: int = 5
    context_window: int = 32768
    use_reflection: bool = False
    use_meta_learning: bool = True
    task_type_override: str | None = None  # Override auto-detection: "code", "prose", "general"
    score_weights: dict | None = None  # e.g. {"correctness": 40, "completeness": 20, "quality": 20, "elegance": 20}

    def validate(self) -> None:
        """Validate configuration, raising ConfigError on issues."""
        if self.dilation_factor < 1:
            raise ConfigError(f"dilation_factor must be >= 1, got {self.dilation_factor}")
        if self.branch_factor < 1:
            raise ConfigError(f"branch_factor must be >= 1, got {self.branch_factor}")
        if self.budget_seconds <= 0:
            raise ConfigError(f"budget_seconds must be > 0, got {self.budget_seconds}")
        if self.max_tokens < 1:
            raise ConfigError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if not (0.0 <= self.temperature <= 2.0):
            raise ConfigError(f"temperature must be 0.0-2.0, got {self.temperature}")
        if self.convergence_threshold < 1:
            raise ConfigError(f"convergence_threshold must be >= 1, got {self.convergence_threshold}")
        if self.context_window < 1024:
            raise ConfigError(f"context_window must be >= 1024, got {self.context_window}")
        if self.task_type_override is not None:
            valid_types = {"code", "prose", "general"}
            if self.task_type_override not in valid_types:
                raise ConfigError(f"task_type_override must be one of {valid_types}, got '{self.task_type_override}'")
        if self.score_weights is not None:
            valid_keys = {"correctness", "completeness", "quality", "elegance"}
            invalid = set(self.score_weights.keys()) - valid_keys
            if invalid:
                raise ConfigError(f"Invalid score_weights keys: {invalid}")
