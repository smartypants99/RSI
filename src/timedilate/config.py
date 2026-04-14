"""Configuration for the Time Dilation Runtime."""
from dataclasses import dataclass, field


class ConfigError(ValueError):
    """Raised when configuration is invalid."""
    pass


# Model cascade: ordered from largest/slowest to smallest/fastest.
# The runtime selects the appropriate tier based on dilation factor.
DEFAULT_MODEL_CASCADE = [
    {"name": "Qwen/Qwen2.5-7B-Instruct", "speed_factor": 1.0, "quality": 1.0},
    {"name": "Qwen/Qwen2.5-3B-Instruct", "speed_factor": 2.5, "quality": 0.85},
    {"name": "Qwen/Qwen2.5-1.5B-Instruct", "speed_factor": 5.0, "quality": 0.7},
    {"name": "Qwen/Qwen2.5-0.5B-Instruct", "speed_factor": 15.0, "quality": 0.5},
]

# Quantization levels: ordered from highest quality to fastest.
QUANTIZATION_CASCADE = [
    {"bits": 16, "method": None, "speed_factor": 1.0, "quality": 1.0},
    {"bits": 8, "method": "int8", "speed_factor": 1.8, "quality": 0.98},
    {"bits": 4, "method": "awq", "speed_factor": 3.5, "quality": 0.92},
    {"bits": 2, "method": "gptq", "speed_factor": 6.0, "quality": 0.80},
]


@dataclass
class TimeDilateConfig:
    # Core
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    draft_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dilation_factor: float = 1.0  # 1.0 = normal speed, 1000.0 = 1000x faster

    # Hardware
    device: str = "cuda"
    gpu_memory_gb: float = 48.0  # A6000 default

    # Generation
    max_tokens: int = 4096
    temperature: float = 0.7

    # Acceleration knobs (auto-configured from dilation_factor if not set)
    quantization_bits: int | None = None  # None = auto-select
    model_tier: int | None = None  # None = auto-select from cascade
    speculative_tokens: int = 5  # tokens predicted per draft step
    kv_cache_compression: float = 1.0  # 1.0 = none, 0.5 = 50% compression
    token_budget_ratio: float = 1.0  # 1.0 = full output, 0.1 = 10% tokens
    prompt_compression: bool = False  # compress long prompts
    parallel_decode_width: int = 1  # >1 enables parallel token generation

    # Model cascade
    model_cascade: list[dict] = field(default_factory=lambda: list(DEFAULT_MODEL_CASCADE))
    quantization_cascade: list[dict] = field(default_factory=lambda: list(QUANTIZATION_CASCADE))

    def validate(self) -> None:
        if self.dilation_factor < 1.0:
            raise ConfigError(f"dilation_factor must be >= 1.0, got {self.dilation_factor}")
        if self.max_tokens < 1:
            raise ConfigError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if not (0.0 <= self.temperature <= 2.0):
            raise ConfigError(f"temperature must be 0.0-2.0, got {self.temperature}")
        if self.gpu_memory_gb <= 0:
            raise ConfigError(f"gpu_memory_gb must be > 0, got {self.gpu_memory_gb}")

    def auto_configure(self) -> "TimeDilateConfig":
        """Auto-select acceleration settings based on dilation_factor.
        Each technique multiplies the speedup. The system stacks them
        to reach the target factor, turning on more aggressive techniques
        as the factor increases.

        Returns a new config with settings filled in."""
        factor = self.dilation_factor
        if factor <= 1.0:
            return self  # no acceleration needed

        remaining_factor = factor
        bits = 16
        model_tier = 0
        spec_tokens = self.speculative_tokens
        kv_comp = 1.0
        token_ratio = 1.0
        prompt_comp = False
        parallel_width = 1

        # Layer 1: Speculative decoding (always on, 2-5x depending on spec_tokens)
        # More speculative tokens = more speedup but diminishing returns
        if remaining_factor >= 2:
            spec_tokens = min(int(5 + remaining_factor / 10), 40)
            spec_speedup = min(1.0 + spec_tokens * 0.3, 5.0)
            remaining_factor /= spec_speedup

        # Layer 2: Quantization cascade
        for q in self.quantization_cascade:
            if remaining_factor <= 1.0:
                break
            if q["speed_factor"] >= remaining_factor or q is self.quantization_cascade[-1]:
                bits = q["bits"]
                remaining_factor /= q["speed_factor"]
                break
            bits = q["bits"]
            remaining_factor /= q["speed_factor"]

        # Layer 3: Model cascade — drop to smaller models
        for i, tier in enumerate(self.model_cascade):
            if remaining_factor <= 1.0:
                break
            if tier["speed_factor"] >= remaining_factor or i == len(self.model_cascade) - 1:
                model_tier = i
                remaining_factor /= tier["speed_factor"]
                break
            model_tier = i
            remaining_factor /= tier["speed_factor"]

        # Layer 4: KV-cache compression (1.5-3x)
        if remaining_factor > 1.0:
            kv_comp = max(0.1, 1.0 / min(remaining_factor, 3.0))
            kv_speedup = 1.0 / kv_comp
            remaining_factor /= min(kv_speedup, 3.0)

        # Layer 5: Token budget compression — produce fewer tokens
        if remaining_factor > 1.0:
            token_ratio = max(0.01, 1.0 / remaining_factor)
            remaining_factor /= (1.0 / token_ratio)

        # Layer 6: Prompt compression — reduce prefill time
        if remaining_factor > 1.0:
            prompt_comp = True
            remaining_factor /= 2.0

        # Layer 7: Parallel decoding
        if remaining_factor > 1.0:
            parallel_width = max(1, min(int(remaining_factor), 16))
            remaining_factor /= parallel_width

        selected_model = self.model_cascade[model_tier]["name"] if model_tier < len(self.model_cascade) else self.model

        return TimeDilateConfig(
            model=selected_model,
            draft_model=self.draft_model,
            dilation_factor=self.dilation_factor,
            device=self.device,
            gpu_memory_gb=self.gpu_memory_gb,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            quantization_bits=bits,
            model_tier=model_tier,
            speculative_tokens=spec_tokens,
            kv_cache_compression=kv_comp,
            token_budget_ratio=token_ratio,
            prompt_compression=prompt_comp,
            parallel_decode_width=parallel_width,
            model_cascade=self.model_cascade,
            quantization_cascade=self.quantization_cascade,
        )

    def describe_acceleration(self) -> str:
        """Human-readable description of what acceleration is active."""
        lines = [f"Dilation factor: {self.dilation_factor}x"]
        if self.quantization_bits and self.quantization_bits < 16:
            lines.append(f"  Quantization: {self.quantization_bits}-bit")
        if self.model_tier and self.model_tier > 0:
            lines.append(f"  Model tier: {self.model_tier} ({self.model})")
        if self.speculative_tokens > 5:
            lines.append(f"  Speculative tokens: {self.speculative_tokens}")
        if self.kv_cache_compression < 1.0:
            lines.append(f"  KV-cache compression: {self.kv_cache_compression:.0%}")
        if self.token_budget_ratio < 1.0:
            lines.append(f"  Token budget: {self.token_budget_ratio:.0%}")
        if self.prompt_compression:
            lines.append("  Prompt compression: enabled")
        if self.parallel_decode_width > 1:
            lines.append(f"  Parallel decode width: {self.parallel_decode_width}")
        return "\n".join(lines)
