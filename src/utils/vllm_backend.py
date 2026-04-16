"""Optional vLLM backend for 5-10x faster inference across all phases.

Strategy: vLLM and HF model can't coexist on one GPU (both need ~16GB for 8B).
So we swap between them:
  - Inference phases (diagnose, generate, verify): vLLM loaded, HF unloaded
  - Training phase: vLLM destroyed, HF loaded, LoRA trained+merged
  - After training: HF unloaded, vLLM reloaded (with merged weights from saved checkpoint)
"""

from __future__ import annotations

import gc
import logging
import shutil
import time
import torch
from pathlib import Path

logger = logging.getLogger(__name__)

_VLLM_AVAILABLE = None

def _check_vllm():
    """Lazy check for vLLM availability. Cached after first call."""
    global _VLLM_AVAILABLE
    if _VLLM_AVAILABLE is None:
        try:
            import vllm  # noqa: F401
            _VLLM_AVAILABLE = True
        except ImportError:
            _VLLM_AVAILABLE = False
    return _VLLM_AVAILABLE


class VLLMModelLoader:
    """Drop-in replacement for ModelLoader that uses vLLM for inference
    and swaps to HF only for training. Has the same interface as ModelLoader."""

    def __init__(self, model_path: str, dtype: str = "bfloat16",
                 max_model_len: int = 4096, gpu_memory_utilization: float = 0.90,
                 allow_remote_code: bool = False,
                 quantization_config: dict | None = None):
        self.model_path = model_path
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.allow_remote_code = bool(allow_remote_code)
        self.quantization_config = quantization_config
        self._llm = None
        self._tokenizer = None
        self._sampling_params_cls = None
        # HF model/tokenizer for training phases
        self._hf_model = None
        self._hf_tokenizer = None
        # Track the current model path (changes after checkpoint save)
        self._current_model_path = model_path
        # Expose a config-like object so components that read model_loader.config
        # (e.g., CustomLoRATrainer accessing config.max_seq_length) work uniformly.
        from .config import ModelConfig
        self.config = ModelConfig(
            model_path=model_path,
            dtype=dtype,
            max_seq_length=max_model_len,
            allow_remote_code=allow_remote_code,
            quantization_config=quantization_config,
        )

    def load(self):
        """Load vLLM engine (called by ImprovementLoop._setup).

        Falls back to HF if vLLM fails to load (GPU incompatibility, OOM, etc.).
        """
        if not _check_vllm():
            logger.warning("vLLM not installed. Falling back to HF backend.")
            self._load_hf()
            return self

        if not torch.cuda.is_available():
            logger.warning("CUDA not available. Falling back to HF backend.")
            self._load_hf()
            return self

        try:
            self._load_vllm()
        except Exception as e:
            logger.warning(f"vLLM failed to load ({type(e).__name__}: {e}). Falling back to HF backend.")
            self._llm = None
            self._load_hf()
        return self

    def _load_vllm(self):
        """Load vLLM engine."""
        from vllm import LLM, SamplingParams
        self._sampling_params_cls = SamplingParams

        logger.info(f"Loading model with vLLM: {self._current_model_path}")
        self._llm = LLM(
            model=self._current_model_path,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=self.allow_remote_code,
            disable_log_stats=True,
        )
        self._tokenizer = self._llm.get_tokenizer()
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        logger.info("vLLM backend ready")

    def _unload_vllm(self):
        """Destroy vLLM engine and free GPU memory."""
        if self._llm is not None:
            logger.info("Unloading vLLM to free GPU memory for training")
            import contextlib
            with contextlib.suppress(Exception):
                from vllm.distributed.parallel_state import (
                    destroy_model_parallel,
                    destroy_distributed_environment,
                )
                destroy_model_parallel()
                destroy_distributed_environment()
            with contextlib.suppress(Exception):
                import ray
                if ray.is_initialized():
                    ray.shutdown()
            del self._llm
            self._llm = None
            # Clear vLLM tokenizer — it's tied to the destroyed engine.
            # _load_hf will set _hf_tokenizer, and _load_vllm will set _tokenizer.
            self._tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _load_hf(self):
        """Load HF model for training."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        logger.info(f"Loading HF model for training: {self._current_model_path}")

        self._hf_tokenizer = AutoTokenizer.from_pretrained(
            self._current_model_path, trust_remote_code=self.allow_remote_code)
        if self._hf_tokenizer.pad_token is None:
            self._hf_tokenizer.pad_token = self._hf_tokenizer.eos_token

        target_dtype = getattr(torch, self.dtype)
        from transformers import __version__ as _tf_version
        dtype_key = "dtype" if int(_tf_version.split(".")[0]) >= 5 else "torch_dtype"
        load_kwargs = {
            "device_map": "auto",
            "trust_remote_code": self.allow_remote_code,
            dtype_key: target_dtype,
            "attn_implementation": "flash_attention_2",
        }
        if self.quantization_config:
            try:
                from transformers import BitsAndBytesConfig
                qc = dict(self.quantization_config)
                if "bnb_4bit_compute_dtype" in qc and isinstance(qc["bnb_4bit_compute_dtype"], str):
                    qc["bnb_4bit_compute_dtype"] = getattr(torch, qc["bnb_4bit_compute_dtype"])
                load_kwargs["quantization_config"] = BitsAndBytesConfig(**qc)
            except ImportError:
                logger.warning("bitsandbytes not installed — ignoring quantization_config")
        # Retry with backoff for transient network errors during model download.
        last_err = None
        for attempt in range(3):
            try:
                self._hf_model = AutoModelForCausalLM.from_pretrained(
                    self._current_model_path,
                    **load_kwargs,
                )
                break
            except (ValueError, ImportError):
                # Flash attention not available — fall back to default
                load_kwargs.pop("attn_implementation", None)
                self._hf_model = AutoModelForCausalLM.from_pretrained(
                    self._current_model_path,
                    **load_kwargs,
                )
                break
            except (OSError, ConnectionError, TimeoutError) as e:
                last_err = e
                if attempt < 2:
                    wait = 2 ** attempt * 5
                    logger.warning(f"HF model load failed (attempt {attempt+1}/3): {e}. Retrying in {wait}s.")
                    time.sleep(wait)
        else:
            raise RuntimeError(f"HF model load failed after 3 attempts: {last_err}") from last_err

    def _unload_hf(self):
        """Unload HF model and free GPU memory."""
        if self._hf_model is not None:
            logger.info("Unloading HF model to free GPU memory for vLLM")
            for p in self._hf_model.parameters():
                p.grad = None
            del self._hf_model
            self._hf_model = None
        if self._hf_tokenizer is not None:
            del self._hf_tokenizer
            self._hf_tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Inference interface (used by diagnostics, generator, verifier) ----

    @property
    def tokenizer(self):
        tok = self._tokenizer or self._hf_tokenizer
        if tok is None:
            raise RuntimeError("No tokenizer loaded. Call load() first.")
        return tok

    @property
    def model(self):
        """HF model (only available during training phase)."""
        if self._hf_model is not None:
            return self._hf_model
        raise RuntimeError("HF model not loaded — currently in vLLM inference mode")

    @property
    def device(self):
        if self._hf_model is not None:
            return next(self._hf_model.parameters()).device
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def generate(self, prompt: str, max_new_tokens: int = 2048,
                 temperature: float = 0.7, top_p: float = 0.9) -> str:
        results = self.generate_batch([prompt], max_new_tokens, temperature, top_p)
        return results[0] if results else ""

    def generate_batch(self, prompts: list[str], max_new_tokens: int = 2048,
                       temperature: float = 0.7, top_p: float = 0.9) -> list[str]:
        if not prompts:
            return []

        if self._llm is not None:
            # vLLM path (fast)
            greedy = temperature <= 0
            params = self._sampling_params_cls(
                max_tokens=max_new_tokens,
                temperature=0.0 if greedy else temperature,
                top_p=1.0 if greedy else top_p,
            )
            outputs = self._llm.generate(prompts, params)
            return [o.outputs[0].text for o in outputs]
        else:
            # HF fallback (during training phase eval)
            return self._hf_generate_batch(prompts, max_new_tokens, temperature, top_p)

    def _hf_generate_batch(self, prompts, max_new_tokens=2048, temperature=0.7, top_p=0.9):
        """HF generate_batch fallback."""
        if not self._hf_model:
            raise RuntimeError("Neither vLLM nor HF model is loaded")
        original_side = self._hf_tokenizer.padding_side
        try:
            self._hf_tokenizer.padding_side = "left"
            inputs = self._hf_tokenizer(
                prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=self.max_model_len,
            ).to(self.device)
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "pad_token_id": self._hf_tokenizer.pad_token_id,
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p
            with torch.no_grad():
                outputs = self._hf_model.generate(**inputs, **gen_kwargs)
            responses = []
            for i, output in enumerate(outputs):
                prompt_len = inputs["attention_mask"][i].sum()
                responses.append(self._hf_tokenizer.decode(
                    output[prompt_len:], skip_special_tokens=True))
            del inputs, outputs
            return responses
        finally:
            self._hf_tokenizer.padding_side = original_side

    # ---- Training swap interface ----

    def swap_to_hf_for_training(self) -> None:
        """Unload vLLM, load HF model for LoRA training."""
        self._unload_vllm()
        self._load_hf()

    def swap_to_vllm_after_training(self, checkpoint_path: str | None = None) -> None:
        """Unload HF, reload vLLM with merged weights. Falls back to HF on failure."""
        if checkpoint_path:
            self._current_model_path = checkpoint_path
            self.config.model_path = checkpoint_path
        self._unload_hf()
        # If vLLM is already loaded (e.g. resume path called before training),
        # unload it first so _load_vllm doesn't leak the prior engine's GPU mem.
        self._unload_vllm()

        if not _check_vllm() or not torch.cuda.is_available():
            logger.warning("vLLM or CUDA not available after training. Falling back to HF.")
            self._load_hf()
            return

        try:
            self._load_vllm()
        except Exception as e:
            logger.warning(f"vLLM reload failed ({type(e).__name__}: {e}). Falling back to HF.")
            self._llm = None
            self._load_hf()

    # ---- Compatibility with ModelLoader interface ----

    def capture_activations(self, layer_names=None):
        """Activation capture requires HF model — used during diagnosis.
        For vLLM mode, return a dummy context manager with empty activations."""
        from contextlib import contextmanager
        from .model_loader import ActivationCapture

        @contextmanager
        def dummy():
            yield ActivationCapture()

        if self._hf_model is not None:
            capture = ActivationCapture()
            hf_model = self._hf_model
            @contextmanager
            def real():
                capture.register(hf_model, layer_names)
                try:
                    yield capture
                finally:
                    capture.remove_hooks()
            return real()
        else:
            return dummy()

    def get_layer_info(self) -> dict:
        """Layer info requires HF model. Return empty dict in vLLM mode."""
        if self._hf_model is not None:
            from .model_loader import _chunked_norm
            skip_suffixes = ("embed_tokens.weight", "lm_head.weight",
                             "layernorm.weight", "layer_norm.weight",
                             "norm.weight", "ln_f.weight")
            layers = {}
            for name, param in self._hf_model.named_parameters():
                if any(name.endswith(s) for s in skip_suffixes):
                    continue
                layers[name] = {
                    "shape": list(param.shape),
                    "dtype": str(param.dtype),
                    "requires_grad": param.requires_grad,
                    "norm": _chunked_norm(param.data),
                }
            return layers
        return {}

    def save_checkpoint(self, path: Path, cycle: int) -> None:
        """Save model checkpoint (only works in HF mode)."""
        if self._hf_model is None:
            logger.warning("Cannot save checkpoint — HF model not loaded")
            return
        save_path = path / f"cycle_{cycle}"
        save_path.mkdir(parents=True, exist_ok=True)
        # Check disk space — warn if less than 1GB free.
        try:
            free = shutil.disk_usage(save_path).free
            if free < 1024 * 1024 * 1024:
                logger.warning(f"Low disk space ({free // 1024 // 1024}MB free) — checkpoint save may fail")
        except OSError:
            pass
        self._hf_model.save_pretrained(save_path)
        self._hf_tokenizer.save_pretrained(save_path)

    def load_from_checkpoint(self, checkpoint_path: str) -> None:
        """Reload model from a saved checkpoint. Works in both vLLM and HF mode."""
        if self._llm is not None:
            # vLLM mode: swap to the new checkpoint
            self.swap_to_vllm_after_training(checkpoint_path)
        else:
            # HF mode: reload HF model
            self._current_model_path = checkpoint_path
            self.config.model_path = checkpoint_path
            self._unload_hf()
            self._load_hf()
