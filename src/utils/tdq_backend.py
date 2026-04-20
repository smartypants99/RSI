"""TDQ-backed model loader — uses TDQModelHF to decompress a .tdq file into
a standard HuggingFace model, then exposes the same interface as ModelLoader
and VLLMModelLoader so the RSI orchestrator/trainer can use it unchanged.

Why not vLLM here: vLLM can't load TDQ files. We lose vLLM's batched-KV
throughput; inference via HF .generate() with left-padded batches is slower
(rough factor 2–4x on typical batch sizes), but acceptable for RSI where
cycle correctness matters more than raw tokens/sec.

Architecture:
  - `load()` calls TDQModelHF.load_full() which returns a real
    AutoModelForCausalLM (fp16). Standard peft/LoRA wraps this directly.
  - `generate(prompt, ...)` and `generate_batch(prompts, ...)` use HF's
    native .generate() with the same left-padded-batch convention as
    ModelLoader.
  - `swap_to_hf_for_training()` / `swap_to_vllm_after_training()` are
    no-ops — we never unload the model; LoRA is applied/merged/stripped
    in-place via the trainer's existing custom_lora paths.
  - `save_checkpoint()` saves a LoRA adapter (not a merged base), because
    TDQ base can't be re-serialized from in-memory fp16 weights without
    re-running the TDQ quantizer. Next cycle simply re-loads the TDQ
    base + adapter.

Requirements at import time:
  - Either copy td_inference.py into this repo, OR set PYTHONPATH to
    include the directory where td_inference.py lives (default:
    /Users/milannarula/Desktop/ai_quatinization/final).
"""

from __future__ import annotations

import gc
import logging
import os
import sys
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


_TDQ_IMPORTED = False


def _import_td_inference():
    """Lazy-import td_inference from ai_quatinization/final.

    We add the directory to sys.path exactly once. Configurable via
    env var TDQ_INFERENCE_DIR if the path moves.
    """
    global _TDQ_IMPORTED
    if _TDQ_IMPORTED:
        return
    default_dir = "/Users/milannarula/Desktop/ai_quatinization/final"
    tdq_dir = os.environ.get("TDQ_INFERENCE_DIR", default_dir)
    if tdq_dir and tdq_dir not in sys.path:
        sys.path.insert(0, tdq_dir)
    # Probe import so we fail loudly at backend construction, not mid-cycle.
    import td_inference  # noqa: F401
    _TDQ_IMPORTED = True


class TDQModelLoader:
    """Drop-in replacement for ModelLoader / VLLMModelLoader, backed by TDQ.

    Interface contract (mirrors src/utils/model_loader.py and vllm_backend.py):
      - model_path: path to the .tdq file
      - device, dtype, max_seq_length
      - model, tokenizer properties
      - generate(prompt, max_new_tokens, temperature, top_p) -> str
      - generate_batch(prompts, max_new_tokens, temperature, top_p) -> list[str]
      - save_checkpoint(output_dir, cycle) — saves LoRA adapter only
      - load_from_checkpoint(path) — loads LoRA adapter on top of TDQ base
      - swap_to_hf_for_training() / swap_to_vllm_after_training() — no-ops
      - capture_activations() — context manager for diagnostics
      - get_layer_info() — layer norm probes for diagnostics

    The trainer (src/trainer/custom_lora.py) calls into `.model` directly and
    manages LoRA injection / merge / strip in-place, same as it does for
    ModelLoader. We don't need to change the trainer.
    """

    def __init__(self, model_path: str, dtype: str = "float16",
                 max_seq_length: int = 4096,
                 allow_remote_code: bool = True,
                 quantization_config: dict | None = None,
                 **_ignored):
        _import_td_inference()
        self.model_path = model_path
        self.dtype = dtype
        self.max_seq_length = max_seq_length
        self.allow_remote_code = bool(allow_remote_code)
        self.quantization_config = quantization_config  # unused; kept for API compat
        self._model = None
        self._tokenizer = None
        self._tdq = None
        # For diagnostics/registry code that reads .config.max_seq_length
        from .config import ModelConfig
        self.config = ModelConfig(
            model_path=model_path,
            dtype=dtype,
            max_seq_length=max_seq_length,
            allow_remote_code=allow_remote_code,
            quantization_config=quantization_config,
        )

    @property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("Call load() before accessing model.")
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            raise RuntimeError("Call load() before accessing tokenizer.")
        return self._tokenizer

    # ---- Load / unload ----

    def load(self) -> None:
        """Decompress the TDQ file into a real HF model in VRAM."""
        if self._model is not None:
            return
        from td_inference import TDQModelHF
        logger.info(f"TDQ: loading {self.model_path}")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self._tdq = TDQModelHF(self.model_path, device=dev)
        self._model = self._tdq.load_full()
        # TDQModelHF.load_full() returns the HF model in fp16. Ensure the
        # tokenizer is accessible in our attribute and has a sensible pad.
        self._tokenizer = self._tdq.tokenizer
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        logger.info(
            f"TDQ: model ready on {self.device}, "
            f"vram={_vram_gb():.1f} GB"
        )

    # swap_* are no-ops — TDQ base stays in VRAM always; LoRA is applied
    # in-place via the trainer.
    def swap_to_hf_for_training(self) -> None:
        return

    def swap_to_vllm_after_training(self, checkpoint_path=None) -> None:
        return

    # ---- Generation ----

    def generate(self, prompt: str, max_new_tokens: int = 2048,
                 temperature: float = 0.7, top_p: float = 0.9) -> str:
        if self._model is None:
            self.load()
        inputs = self._tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=self.max_seq_length,
        ).to(self.device)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        with torch.no_grad():
            out = self._model.generate(**inputs, **gen_kwargs)
        text = self._tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return text

    def generate_batch(self, prompts: list[str], max_new_tokens: int = 2048,
                        temperature: float = 0.7, top_p: float = 0.9) -> list[str]:
        if not prompts:
            return []
        if self._model is None:
            self.load()
        original_padding = self._tokenizer.padding_side
        try:
            self._tokenizer.padding_side = "left"
            inputs = self._tokenizer(
                prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=self.max_seq_length,
            ).to(self.device)
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "pad_token_id": self._tokenizer.pad_token_id,
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p
            with torch.no_grad():
                out = self._model.generate(**inputs, **gen_kwargs)
            # Under left-padding every generated sequence starts at the
            # padded input length, identical to model_loader.generate_batch
            # after its post-run-6 fix.
            input_len = inputs["input_ids"].shape[1]
            responses = [
                self._tokenizer.decode(o[input_len:], skip_special_tokens=True)
                for o in out
            ]
            del inputs, out
            return responses
        finally:
            self._tokenizer.padding_side = original_padding

    # ---- Checkpoint save/load: adapter-only ----

    def save_checkpoint(self, output_root: Path, cycle: int) -> None:
        """Save ONLY the LoRA adapter. TDQ base is static on disk at self.model_path;
        merging a fp16 LoRA delta into TDQ-compressed tensors isn't supported,
        so we persist the adapter and rely on next cycle re-loading base+adapter.
        """
        output_root = Path(output_root)
        cycle_dir = output_root / f"cycle_{cycle}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        # Let the trainer save the adapter (it already knows how via
        # save_lora_weights). This method just ensures the directory exists
        # so the orchestrator's bookkeeping (history.json, etc.) works.
        return

    def load_from_checkpoint(self, path: str) -> None:
        """No-op — LoRA adapters are loaded/applied by the trainer's own
        paths, not through the model loader."""
        return

    # ---- Diagnostics interfaces ----

    def capture_activations(self, layer_names=None):
        """Activation capture via the existing HF hook utilities."""
        from contextlib import contextmanager
        from .model_loader import ActivationCapture

        @contextmanager
        def _ctx():
            cap = ActivationCapture()
            hooks = []
            if self._model is not None and layer_names:
                for name, mod in self._model.named_modules():
                    if name in layer_names:
                        hooks.append(mod.register_forward_hook(cap.make_hook(name)))
            try:
                yield cap
            finally:
                for h in hooks:
                    h.remove()
        return _ctx()

    def get_layer_info(self) -> dict:
        """Return {layer_name: {norm: float}} for the diagnostics engine.
        Mirrors ModelLoader.get_layer_info — just walks named_parameters and
        computes L2 norm per transformer layer's attention + MLP matrices.
        """
        if self._model is None:
            return {}
        info: dict[str, dict] = {}
        for name, p in self._model.named_parameters():
            if ".layers." in name and any(
                k in name for k in ("q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj")
            ):
                info[name] = {"norm": float(p.detach().float().norm().cpu())}
        return info


def _vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / (1024 ** 3)
