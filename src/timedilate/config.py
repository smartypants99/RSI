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
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    draft_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dilation_factor: float = 1.0  # 1.0 = single pass, 1000 = 1000x more thinking

    # Hardware
    device: str = "cuda"
    gpu_memory_gb: float = 48.0

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

    def validate(self) -> None:
        if self.dilation_factor < 1.0:
            raise ConfigError(f"dilation_factor must be >= 1.0, got {self.dilation_factor}")
        if self.max_tokens < 1:
            raise ConfigError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if not (0.0 <= self.temperature <= 2.0):
            raise ConfigError(f"temperature must be 0.0-2.0, got {self.temperature}")
        if self.gpu_memory_gb <= 0:
            raise ConfigError(f"gpu_memory_gb must be > 0, got {self.gpu_memory_gb}")

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
