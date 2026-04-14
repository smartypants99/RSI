"""Inference engine — thin wrapper around vLLM.

No acceleration tricks, no quality tradeoffs. Just runs the model
at full precision and full output. The dilation (more thinking)
happens in the controller, not here.
"""
import logging
import time

from timedilate.config import TimeDilateConfig

logger = logging.getLogger(__name__)


class InferenceError(RuntimeError):
    """Raised when inference fails after retries."""
    pass


class DilationEngine:
    """Runs model inference. Full precision, full output, no shortcuts."""

    def __init__(self, config: TimeDilateConfig):
        self.config = config
        self._total_calls = 0
        self._total_latency = 0.0
        self._failed_calls = 0
        self._model = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the vLLM model."""
        if self._initialized:
            return

        from vllm import LLM

        logger.info("Initializing model: %s", self.config.model)
        self._model = LLM(
            model=self.config.model,
            trust_remote_code=True,
            gpu_memory_utilization=0.90,
        )
        self._initialized = True

    def generate(self, prompt: str, max_tokens: int | None = None,
                 temperature: float | None = None, retries: int = 2) -> str:
        """Generate text from a prompt. Full output, no truncation."""
        self.initialize()
        from vllm import SamplingParams

        params = SamplingParams(
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature if temperature is not None else self.config.temperature,
        )

        last_error = None
        for attempt in range(retries + 1):
            try:
                start = time.time()
                outputs = self._model.generate([prompt], params)
                text = outputs[0].outputs[0].text
                elapsed = time.time() - start

                self._total_calls += 1
                self._total_latency += elapsed

                if not text or not text.strip():
                    if attempt < retries:
                        logger.warning("Empty response on attempt %d, retrying", attempt + 1)
                        continue
                    raise InferenceError("Model returned empty response after retries")

                return text
            except InferenceError:
                raise
            except Exception as e:
                last_error = e
                self._failed_calls += 1
                if attempt < retries:
                    logger.warning("Inference failed (attempt %d/%d): %s",
                                   attempt + 1, retries + 1, e)
                    continue
        raise InferenceError(f"Inference failed after {retries + 1} attempts: {last_error}")

    @property
    def stats(self) -> dict:
        return {
            "total_calls": self._total_calls,
            "failed_calls": self._failed_calls,
            "total_latency_s": round(self._total_latency, 3),
        }
