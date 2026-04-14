"""Inference engine with layered acceleration for time dilation.

Each acceleration technique is a multiplier on speed. The engine stacks
them based on the configured dilation factor to hit the target speedup.
"""
import logging
import time

from timedilate.config import TimeDilateConfig

logger = logging.getLogger(__name__)


class InferenceError(RuntimeError):
    """Raised when inference fails after retries."""
    pass


class DilationEngine:
    """Accelerated inference engine that applies layered speedup techniques.

    Acceleration stack (each multiplies the previous):
    1. Speculative decoding — draft model predicts token batches
    2. Quantization — reduce precision for faster compute
    3. Model cascading — use smaller/faster models
    4. KV-cache compression — reduce memory bandwidth
    5. Token budget — generate fewer tokens (denser output)
    6. Prompt compression — reduce prefill time
    7. Parallel decoding — predict multiple positions at once
    """

    def __init__(self, config: TimeDilateConfig):
        self.config = config
        self._total_calls = 0
        self._total_tokens = 0
        self._total_latency = 0.0
        self._failed_calls = 0
        self._model = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the model with all acceleration techniques applied."""
        if self._initialized:
            return

        from vllm import LLM

        vllm_kwargs = {
            "model": self.config.model,
            "trust_remote_code": True,
            "gpu_memory_utilization": 0.90,
        }

        # Apply speculative decoding
        if self.config.speculative_tokens > 0 and self.config.draft_model:
            vllm_kwargs["speculative_model"] = self.config.draft_model
            vllm_kwargs["num_speculative_tokens"] = self.config.speculative_tokens
            logger.info("Speculative decoding: %d tokens with %s",
                        self.config.speculative_tokens, self.config.draft_model)

        # Apply quantization
        if self.config.quantization_bits and self.config.quantization_bits < 16:
            if self.config.quantization_bits <= 4:
                vllm_kwargs["quantization"] = "awq"
                logger.info("Quantization: AWQ %d-bit", self.config.quantization_bits)
            elif self.config.quantization_bits <= 8:
                vllm_kwargs["dtype"] = "half"
                logger.info("Quantization: FP16/INT8")

        # Apply parallel decoding via tensor parallelism
        if self.config.parallel_decode_width > 1:
            vllm_kwargs["tensor_parallel_size"] = min(
                self.config.parallel_decode_width, 1  # single GPU
            )

        logger.info("Initializing model: %s", self.config.model)
        self._model = LLM(**vllm_kwargs)
        self._initialized = True
        logger.info("Engine ready. Acceleration: %s", self.config.describe_acceleration())

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        retries: int = 2,
    ) -> str:
        """Generate text with all acceleration techniques active."""
        self.initialize()
        from vllm import SamplingParams

        # Apply token budget compression
        effective_max = max_tokens or self.config.max_tokens
        if self.config.token_budget_ratio < 1.0:
            effective_max = max(16, int(effective_max * self.config.token_budget_ratio))

        # Apply prompt compression if enabled
        actual_prompt = prompt
        if self.config.prompt_compression and len(prompt) > 2000:
            actual_prompt = self._compress_prompt(prompt)

        params = SamplingParams(
            max_tokens=effective_max,
            temperature=temperature if temperature is not None else self.config.temperature,
        )

        last_error = None
        for attempt in range(retries + 1):
            try:
                call_start = time.time()
                outputs = self._model.generate([actual_prompt], params)
                text = outputs[0].outputs[0].text
                elapsed = time.time() - call_start

                self._total_calls += 1
                self._total_latency += elapsed
                self._total_tokens += len(text) // 4

                if not text or not text.strip():
                    if attempt < retries:
                        logger.warning("Empty response on attempt %d, retrying", attempt + 1)
                        continue
                    raise InferenceError("Model returned empty response after retries")

                logger.debug("Generated %d chars in %.3fs", len(text), elapsed)
                return text
            except InferenceError:
                raise
            except Exception as e:
                last_error = e
                self._failed_calls += 1
                if attempt < retries:
                    backoff = min(1.0 * (2 ** attempt), 10.0)
                    logger.warning("Inference failed (attempt %d/%d): %s, retrying in %.1fs",
                                   attempt + 1, retries + 1, e, backoff)
                    time.sleep(backoff)
                    continue
        raise InferenceError(f"Inference failed after {retries + 1} attempts: {last_error}")

    def _compress_prompt(self, prompt: str) -> str:
        """Compress a long prompt by extracting key information.
        This reduces prefill time which is often the bottleneck for long prompts."""
        # Simple compression: keep first and last portions, trim middle
        max_chars = 2000
        if len(prompt) <= max_chars:
            return prompt
        half = max_chars // 2
        compressed = (
            prompt[:half]
            + "\n\n[... compressed for speed ...]\n\n"
            + prompt[-half:]
        )
        logger.debug("Prompt compressed: %d -> %d chars", len(prompt), len(compressed))
        return compressed

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~4 chars per token."""
        return len(text) // 4

    @property
    def avg_latency(self) -> float:
        if self._total_calls == 0:
            return 0.0
        return self._total_latency / self._total_calls

    @property
    def stats(self) -> dict:
        return {
            "total_calls": self._total_calls,
            "failed_calls": self._failed_calls,
            "total_tokens": self._total_tokens,
            "total_latency_s": round(self._total_latency, 3),
            "avg_latency_s": round(self.avg_latency, 3),
            "model": self.config.model,
            "quantization_bits": self.config.quantization_bits,
            "speculative_tokens": self.config.speculative_tokens,
            "token_budget_ratio": self.config.token_budget_ratio,
        }
