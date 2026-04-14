"""Time Dilation Controller — the main entry point.

Takes a prompt and dilation factor, auto-configures acceleration,
runs inference, and returns the result with timing metrics.
"""
import logging
import time
from dataclasses import dataclass

from timedilate.config import TimeDilateConfig
from timedilate.engine import DilationEngine

logger = logging.getLogger(__name__)


@dataclass
class DilationResult:
    output: str
    dilation_factor: float
    base_latency_estimate: float  # estimated time without dilation
    actual_latency: float  # actual wall-clock time
    achieved_speedup: float  # base_latency / actual_latency
    model_used: str
    acceleration_summary: str
    tokens_generated: int = 0

    def to_report(self, config: TimeDilateConfig | None = None) -> dict:
        from timedilate import __version__
        report = {
            "version": __version__,
            "timestamp": time.time(),
            "dilation_factor": self.dilation_factor,
            "base_latency_estimate_s": round(self.base_latency_estimate, 3),
            "actual_latency_s": round(self.actual_latency, 3),
            "achieved_speedup": round(self.achieved_speedup, 2),
            "model_used": self.model_used,
            "tokens_generated": self.tokens_generated,
            "output_length": len(self.output),
            "acceleration_summary": self.acceleration_summary,
        }
        if config:
            report["config"] = {
                "quantization_bits": config.quantization_bits,
                "speculative_tokens": config.speculative_tokens,
                "kv_cache_compression": config.kv_cache_compression,
                "token_budget_ratio": config.token_budget_ratio,
                "prompt_compression": config.prompt_compression,
                "parallel_decode_width": config.parallel_decode_width,
                "model_tier": config.model_tier,
            }
        return report


class DilationController:
    """Orchestrates time-dilated inference.

    Usage:
        config = TimeDilateConfig(dilation_factor=1000)
        controller = DilationController(config)
        result = controller.run("Write a Python sort function")
        print(result.output)
        print(f"Speedup: {result.achieved_speedup}x")
    """

    def __init__(self, config: TimeDilateConfig, engine: DilationEngine | None = None):
        config.validate()
        # Auto-configure acceleration based on dilation factor
        self.config = config.auto_configure()
        self.engine = engine or DilationEngine(self.config)
        logger.info("Controller initialized: %s", self.config.describe_acceleration())

    def run(self, prompt: str) -> DilationResult:
        """Run time-dilated inference on a prompt.

        The dilation factor determines how much faster the response should be
        compared to the base model at full precision. The controller stacks
        acceleration techniques to approach the target speedup.
        """
        start = time.time()

        # Estimate what the base model latency would be without any acceleration.
        # This is a rough estimate based on prompt length and max_tokens.
        prompt_tokens = len(prompt) // 4
        output_tokens = self.config.max_tokens
        # Base model: ~30 tokens/sec for 7B on A6000 at FP16
        base_tokens_per_sec = 30.0
        base_latency = (prompt_tokens + output_tokens) / base_tokens_per_sec

        logger.info("Starting dilated inference (target: %.0fx, base est: %.1fs)",
                     self.config.dilation_factor, base_latency)
        logger.info("Acceleration stack:\n%s", self.config.describe_acceleration())

        # Run the actual inference with all acceleration applied
        output = self.engine.generate(prompt)

        actual_latency = time.time() - start
        tokens_generated = len(output) // 4
        achieved_speedup = base_latency / actual_latency if actual_latency > 0 else float('inf')

        logger.info(
            "Inference complete: %.3fs actual (est base: %.1fs) = %.1fx speedup",
            actual_latency, base_latency, achieved_speedup,
        )

        if achieved_speedup < self.config.dilation_factor * 0.5:
            logger.warning(
                "Achieved speedup (%.1fx) is below target (%.0fx). "
                "Hardware may be the bottleneck.",
                achieved_speedup, self.config.dilation_factor,
            )

        return DilationResult(
            output=output,
            dilation_factor=self.config.dilation_factor,
            base_latency_estimate=base_latency,
            actual_latency=actual_latency,
            achieved_speedup=achieved_speedup,
            model_used=self.config.model,
            acceleration_summary=self.config.describe_acceleration(),
            tokens_generated=tokens_generated,
        )

    def benchmark(self, prompt: str, factors: list[float] | None = None) -> list[DilationResult]:
        """Run the same prompt at multiple dilation factors for comparison."""
        if factors is None:
            factors = [1, 10, 100, 1000]
        results = []
        for factor in factors:
            cfg = TimeDilateConfig(
                model=self.config.model_cascade[0]["name"],
                draft_model=self.config.draft_model,
                dilation_factor=factor,
                gpu_memory_gb=self.config.gpu_memory_gb,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            ctrl = DilationController(cfg)
            result = ctrl.run(prompt)
            results.append(result)
            logger.info("Factor %sx: %.3fs (%.1fx achieved)",
                        factor, result.actual_latency, result.achieved_speedup)
        return results
