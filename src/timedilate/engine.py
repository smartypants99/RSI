import logging
import time

from timedilate.config import TimeDilateConfig

logger = logging.getLogger(__name__)


class InferenceError(RuntimeError):
    """Raised when inference fails after retries."""
    pass


class InferenceEngine:
    def __init__(self, config: TimeDilateConfig):
        from vllm import LLM
        self.config = config
        self.llm = LLM(
            model=config.model,
            speculative_model=config.draft_model,
            num_speculative_tokens=5,
            trust_remote_code=True,
        )
        self._total_calls = 0
        self._total_tokens_generated = 0
        self._failed_calls = 0
        self._total_latency = 0.0

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        retries: int = 2,
    ) -> str:
        from vllm import SamplingParams
        params = SamplingParams(
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature if temperature is not None else self.config.temperature,
        )
        last_error = None
        for attempt in range(retries + 1):
            try:
                call_start = time.time()
                outputs = self.llm.generate([prompt], params)
                text = outputs[0].outputs[0].text
                self._total_calls += 1
                self._total_latency += time.time() - call_start
                self._total_tokens_generated += len(text) // 4  # rough estimate
                if not text or not text.strip():
                    if attempt < retries:
                        logger.warning("Empty response on attempt %d, retrying", attempt + 1)
                        time.sleep(0.5)
                        continue
                    raise InferenceError("Model returned empty response after retries")
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

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~4 chars per token."""
        return len(text) // 4

    @property
    def avg_latency(self) -> float:
        """Average latency per successful call in seconds."""
        if self._total_calls == 0:
            return 0.0
        return self._total_latency / self._total_calls

    @property
    def stats(self) -> dict:
        """Return engine usage statistics."""
        return {
            "total_calls": self._total_calls,
            "failed_calls": self._failed_calls,
            "total_tokens_generated": self._total_tokens_generated,
            "total_latency_s": round(self._total_latency, 2),
            "avg_latency_s": round(self.avg_latency, 3),
        }
