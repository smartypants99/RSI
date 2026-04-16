"""Model loading and management utilities."""

import logging
import shutil
import time
import torch
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

from .config import ModelConfig

logger = logging.getLogger(__name__)


class ActivationStats:
    """Pre-computed stats from an activation tensor (avoids storing full tensor)."""
    __slots__ = ("dead_ratio", "std", "mean", "max_abs")

    def __init__(self, tensor: torch.Tensor):
        numel = tensor.numel()
        if numel == 0:
            self.dead_ratio = 0.0
            self.std = 0.0
            self.mean = 0.0
            self.max_abs = 0.0
            return
        # Compute dead_ratio and max_abs from abs values, then discard.
        # Avoid keeping abs_tensor alive while also computing float stats —
        # for (1, 512, 8192) that's 3 tensors alive simultaneously (original,
        # abs, float), tripling peak memory per hook invocation.
        abs_tensor = tensor.abs()
        self.dead_ratio = (abs_tensor < 1e-6).sum().item() / numel
        self.max_abs = abs_tensor.max().item()
        del abs_tensor
        # Compute std/mean in float32 — bfloat16 squaring in variance computation
        # can underflow (small activations) or overflow (large activations),
        # producing NaN or wildly inaccurate std values.
        if numel > 1:
            if numel < 1_000_000:
                float_tensor = tensor.float()
                self.std = float_tensor.std().item()
                self.mean = float_tensor.mean().item()
                del float_tensor
            else:
                # Two-pass chunked: pass 1 for mean, pass 2 for variance.
                # Accumulate on GPU (torch scalar) to minimize GPU→CPU syncs.
                flat = tensor.contiguous().view(-1)
                chunk_size = 1_000_000
                total_sum = torch.tensor(0.0, device=tensor.device)
                for start in range(0, numel, chunk_size):
                    total_sum += flat[start:start + chunk_size].float().sum()
                mean_val = (total_sum / numel).item()
                self.mean = mean_val
                var_sum = torch.tensor(0.0, device=tensor.device)
                for start in range(0, numel, chunk_size):
                    chunk = flat[start:start + chunk_size].float()
                    var_sum += (chunk - mean_val).pow(2).sum()
                self.std = (var_sum.item() / numel) ** 0.5
        else:
            self.std = 0.0
            self.mean = tensor.float().item()


class ActivationCapture:
    """Captures activation STATS during forward passes for diagnostics.

    Does NOT store full tensors — for 70B+ models, storing every transformer
    block's output would OOM. Instead, computes stats (dead ratio, std, mean)
    in the hook and stores only the lightweight ActivationStats object.
    """

    def __init__(self):
        self.activations: dict[str, ActivationStats] = {}
        self._hooks = []

    def register(self, model, layer_names: Optional[list[str]] = None):
        """Register forward hooks on specified layers."""
        target_names = set(layer_names) if layer_names else None

        for name, module in model.named_modules():
            if target_names is not None:
                if name not in target_names:
                    continue
            else:
                parts = name.split(".")
                if len(parts) < 2:
                    continue
                if not (parts[-1].isdigit() and any(k in name for k in ("layers", "h", "blocks"))):
                    continue

            hook = module.register_forward_hook(self._make_hook(name))
            self._hooks.append(hook)

    def _make_hook(self, name: str):
        def hook(module, input, output):
            with torch.no_grad():
                tensor = None
                if isinstance(output, torch.Tensor):
                    tensor = output.detach()
                elif isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
                    tensor = output[0].detach()
                if tensor is not None:
                    self.activations[name] = ActivationStats(tensor)
        return hook

    def clear(self):
        self.activations.clear()

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


def _chunked_norm(tensor: torch.Tensor, chunk_size: int = 1_000_000) -> float:
    """Compute L2 norm without materializing a full float32 copy.

    Uses contiguous() + view() instead of reshape() — reshape on non-contiguous
    tensors (e.g., transposed weights) silently allocates a full contiguous copy,
    defeating the purpose of chunking. contiguous() is explicit about the copy,
    but for model weights it's almost always already contiguous (no copy needed).
    """
    if tensor.numel() == 0:
        return 0.0
    flat = tensor.contiguous().view(-1)
    accum = 0.0
    for start in range(0, flat.numel(), chunk_size):
        chunk = flat[start:start + chunk_size].float()
        accum += chunk.pow(2).sum().item()
    return accum ** 0.5


class ModelLoader:
    """Handles loading, quantization, and lifecycle of target models."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._model = None
        self._tokenizer = None

    def load(self):
        """Load model and tokenizer from path."""
        if not self.config.model_path:
            raise ValueError("model_path must be set before loading")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        trust = bool(getattr(self.config, "allow_remote_code", False))
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=trust,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        target_dtype = getattr(torch, self.config.dtype)
        # transformers 5.x renamed torch_dtype → dtype
        from transformers import __version__ as _tf_version
        dtype_key = "dtype" if int(_tf_version.split(".")[0]) >= 5 else "torch_dtype"
        load_kwargs = {
            "device_map": self.config.device_map,
            dtype_key: target_dtype,
            "trust_remote_code": trust,
            "attn_implementation": "flash_attention_2",
        }

        if self.config.quantization_config:
            load_kwargs["quantization_config"] = self._build_quant_config()

        # Retry model download with exponential backoff for transient network errors.
        last_err = None
        for attempt in range(3):
            try:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_path,
                    **load_kwargs,
                )
                break
            except (ValueError, ImportError):
                # Flash attention not available — fall back to default
                load_kwargs.pop("attn_implementation", None)
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_path,
                    **load_kwargs,
                )
                break
            except (OSError, ConnectionError, TimeoutError) as e:
                last_err = e
                if attempt < 2:
                    wait = 2 ** attempt * 5
                    logger.warning(f"Model download failed (attempt {attempt+1}/3): {e}. Retrying in {wait}s.")
                    time.sleep(wait)
        else:
            raise RuntimeError(f"Model download failed after 3 attempts: {last_err}") from last_err
        # KV cache is enabled for inference speed. Trainer disables it during training.
        return self

    def _build_quant_config(self):
        """Build quantization config from dict. Returns BitsAndBytesConfig or raw dict."""
        qc = self.config.quantization_config
        if not qc:
            return None
        try:
            from transformers import BitsAndBytesConfig
            qc = dict(qc)
            if "bnb_4bit_compute_dtype" in qc and isinstance(qc["bnb_4bit_compute_dtype"], str):
                qc["bnb_4bit_compute_dtype"] = getattr(torch, qc["bnb_4bit_compute_dtype"])
            return BitsAndBytesConfig(**qc)
        except (ImportError, TypeError):
            # Fallback: user may have custom quantization that takes raw dict
            return qc

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model

    @property
    def device(self):
        """Get the device of the model's first parameter.
        Reliable even with device_map='auto' (unlike model.device which may return 'meta').
        """
        return next(self._model.parameters()).device

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load() first.")
        return self._tokenizer

    def generate(self, prompt: str, max_new_tokens: int = 2048, temperature: float = 0.7, top_p: float = 0.9) -> str:
        """Generate a single response from the model."""
        inputs = self._tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=self.config.max_seq_length,
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
            outputs = self._model.generate(**inputs, **gen_kwargs)

        response = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return response

    def generate_batch(self, prompts: list[str], max_new_tokens: int = 2048, temperature: float = 0.7, top_p: float = 0.9) -> list[str]:
        """Generate responses for multiple prompts efficiently."""
        if not prompts:
            return []
        original_padding_side = self._tokenizer.padding_side
        try:
            self._tokenizer.padding_side = "left"
            inputs = self._tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length,
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
                outputs = self._model.generate(**inputs, **gen_kwargs)

            responses = []
            for i, output in enumerate(outputs):
                # Use attention_mask to count actual prompt tokens — more reliable
                # than comparing to pad_token_id (which may equal eos_token_id).
                prompt_len = inputs["attention_mask"][i].sum()
                response = self._tokenizer.decode(
                    output[prompt_len:],
                    skip_special_tokens=True,
                )
                responses.append(response)

            # Eagerly delete large tensors so the CUDA allocator can reuse the
            # memory for the next batch. Don't call empty_cache() here — it's a
            # sync point that stalls the pipeline, and the allocator already
            # reuses freed blocks within the process.
            del inputs, outputs

            return responses
        finally:
            self._tokenizer.padding_side = original_padding_side

    @contextmanager
    def capture_activations(self, layer_names: Optional[list[str]] = None):
        """Context manager to capture layer activations during inference."""
        capture = ActivationCapture()
        capture.register(self._model, layer_names)
        try:
            yield capture
        finally:
            capture.remove_hooks()

    def get_layer_info(self) -> dict:
        """Extract layer information for diagnostics.

        Skips embedding layers (lm_head, embed_tokens) and layer norms —
        these are never LoRA targets and computing their norms is wasted work
        for 70B+ models.
        """
        skip_suffixes = ("embed_tokens.weight", "lm_head.weight",
                         "layernorm.weight", "layer_norm.weight",
                         "norm.weight", "ln_f.weight")
        layers = {}
        for name, param in self._model.named_parameters():
            if any(name.endswith(s) for s in skip_suffixes):
                continue
            layers[name] = {
                "shape": list(param.shape),
                "dtype": str(param.dtype),
                "requires_grad": param.requires_grad,
                "norm": _chunked_norm(param.data),
            }
        return layers

    def save_checkpoint(self, path: Path, cycle: int) -> None:
        """Save model checkpoint for a given cycle."""
        save_path = path / f"cycle_{cycle}"
        save_path.mkdir(parents=True, exist_ok=True)
        # Check disk space — warn if less than 1GB free (small models may still fit).
        try:
            free = shutil.disk_usage(save_path).free
            if free < 1024 * 1024 * 1024:
                logger.warning(f"Low disk space ({free // 1024 // 1024}MB free) — checkpoint save may fail")
        except OSError:
            pass
        self._model.save_pretrained(save_path)
        self._tokenizer.save_pretrained(save_path)

    def load_from_checkpoint(self, checkpoint_path: str) -> None:
        """Reload model weights from a saved checkpoint (for early stopping revert)."""
        from transformers import AutoModelForCausalLM
        logger.info(f"Reloading model from checkpoint: {checkpoint_path}")
        trust = bool(getattr(self.config, "allow_remote_code", False))
        target_dtype = getattr(torch, self.config.dtype)
        from transformers import __version__ as _tf_version
        dtype_key = "dtype" if int(_tf_version.split(".")[0]) >= 5 else "torch_dtype"
        kwargs = {
            dtype_key: target_dtype,
            "device_map": self.config.device_map,
            "trust_remote_code": trust,
        }
        if self.config.quantization_config:
            kwargs["quantization_config"] = self._build_quant_config()
        self._model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **kwargs)
        self._model.eval()
