"""Configuration for the Time Dilation Runtime.

Time dilation = giving the AI more subjective thinking time.
Factor 1 = single pass. Factor 1000 = 1000x more reasoning cycles.
Infinite scaling, no quality loss — more thinking only adds quality.
"""
from dataclasses import dataclass


class ConfigError(ValueError):
    """Raised when configuration is invalid."""
    pass


@dataclass
class TimeDilateConfig:
    # Core
    model: str = "Qwen/Qwen3-8B"
    draft_model: str = "Qwen/Qwen3-8B"
    dilation_factor: float = 1.0  # 1.0 = single pass, 1000 = 1000x more thinking

    # Hardware
    device: str = "cuda"
    gpu_memory_gb: float = 48.0
    gpu_memory_utilization: float = 0.60  # fraction of GPU memory vLLM may reserve
    max_model_len: int | None = None  # cap context length to reduce KV cache memory
    dtype: str = "auto"  # "auto" | "float16" | "bfloat16" | "float32"
    enforce_eager: bool = False  # skip CUDA graph capture — saves VRAM on small GPUs
    swap_space_gb: float = 4.0  # CPU swap space for KV cache overflow (GiB)
    seed: int | None = None  # RNG seed for deterministic sampling; None = nondeterministic

    # Generation
    max_tokens: int = 4096
    temperature: float = 0.7

    # Time budget — how long the AI gets in real wall-clock seconds.
    # Combined with dilation_factor, this gives subjective time:
    #   subjective_time = time_budget * dilation_factor
    # e.g., 5s budget * 1M factor = 5M seconds (57 days) of AI thinking
    time_budget_seconds: float | None = None  # None = no time limit, run all cycles

    # Dilation strategy
    branch_factor: int = 1  # how many parallel variants to explore per cycle
    convergence_patience: int = 5  # cycles without improvement before trying new strategy
    use_self_critique: bool = True  # AI critiques its own output each cycle
    use_chain_of_thought: bool = True  # deeper reasoning per cycle

    # Early stopping & exploration
    early_stop_score: int = 98  # stop if score >= this (saves tokens on perfect answers)
    branch_temperature_spread: float = 0.3  # +/- spread for branch diversification

    def validate(self) -> None:
        if self.dilation_factor < 1.0:
            raise ConfigError(f"dilation_factor must be >= 1.0, got {self.dilation_factor}")
        if self.max_tokens < 1:
            raise ConfigError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if not (0.0 <= self.temperature <= 2.0):
            raise ConfigError(f"temperature must be 0.0-2.0, got {self.temperature}")
        if self.gpu_memory_gb <= 0:
            raise ConfigError(f"gpu_memory_gb must be > 0, got {self.gpu_memory_gb}")
        if not (0.1 <= self.gpu_memory_utilization <= 0.99):
            raise ConfigError(
                f"gpu_memory_utilization must be in 0.1-0.99, got {self.gpu_memory_utilization}"
            )
        if self.dtype not in ("auto", "float16", "bfloat16", "float32"):
            raise ConfigError(
                f"dtype must be one of auto/float16/bfloat16/float32, got {self.dtype}"
            )
        if self.swap_space_gb < 0:
            raise ConfigError(f"swap_space_gb must be >= 0, got {self.swap_space_gb}")
        if self.max_model_len is not None and self.max_model_len < 1:
            raise ConfigError(f"max_model_len must be >= 1 or None, got {self.max_model_len}")
        if self.branch_factor < 1:
            raise ConfigError(f"branch_factor must be >= 1, got {self.branch_factor}")
        if self.convergence_patience < 1:
            raise ConfigError(f"convergence_patience must be >= 1, got {self.convergence_patience}")
        if not (0 <= self.early_stop_score <= 100):
            raise ConfigError(f"early_stop_score must be in 0-100, got {self.early_stop_score}")
        if self.branch_temperature_spread < 0:
            raise ConfigError(
                f"branch_temperature_spread must be >= 0, got {self.branch_temperature_spread}"
            )

    @property
    def num_cycles(self) -> int:
        """Total reasoning cycles the AI gets. More dilation = more cycles.
        When time_budget is set, cycles are unlimited (controlled by wall clock)."""
        if self.dilation_factor <= 1.0:
            return 0
        if self.time_budget_seconds is not None:
            return 0  # unlimited — controller uses time budget instead
        return max(1, int(self.dilation_factor))

    @property
    def subjective_time(self) -> float | None:
        """How much subjective thinking time the AI gets (seconds).
        time_budget * dilation_factor. None if no time budget set."""
        if self.time_budget_seconds is None:
            return None
        return self.time_budget_seconds * self.dilation_factor

    def describe(self) -> str:
        """Human-readable description of the dilation configuration."""
        lines = [f"Time Dilation: {self.dilation_factor}x"]
        if self.time_budget_seconds is not None:
            sub = self.subjective_time
            lines.append(f"  Time budget: {self.time_budget_seconds}s real time")
            lines.append(f"  Subjective time: {sub:,.0f}s ({sub/86400:,.1f} days) of AI thinking")
            lines.append("  Cycles: unlimited (runs until time budget expires)")
        else:
            lines.append(f"  Reasoning cycles: {self.num_cycles}")
        lines.append(f"  Model: {self.model}")
        if self.branch_factor > 1:
            lines.append(f"  Branch factor: {self.branch_factor} variants per cycle")
        if self.use_self_critique:
            lines.append("  Self-critique: enabled")
        if self.use_chain_of_thought:
            lines.append("  Chain-of-thought: enabled")
        lines.append(f"  Convergence patience: {self.convergence_patience}")
        return "\n".join(lines)
