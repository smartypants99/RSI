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
                 quantization_config: dict | None = None,
                 enable_lora: bool = False,
                 max_lora_rank: int = 64,
                 max_num_seqs: int = 0,
                 enforce_eager: bool = False,
                 coresident_training_enabled: bool = False,
                 coresident_vllm_mem_frac: float = 0.42,
                 enable_chunked_prefill: bool = True):
        self.model_path = model_path
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.allow_remote_code = bool(allow_remote_code)
        self.quantization_config = quantization_config
        self.enable_lora = bool(enable_lora)
        self.max_lora_rank = int(max_lora_rank)
        # 0 = use vLLM's built-in default. Positive = pin engine to that
        # concurrent-sequence cap. Task #10: 32 amortizes decode overhead
        # once propose/solve fan-out shrinks (k=3, tasks=12).
        self.max_num_seqs = int(max_num_seqs)
        # Task #19: coresident-training knobs. When coresident is ON we
        # initialize vLLM with a clipped gpu_memory_utilization (gemini
        # consult: ~0.42 on 48GB A6000 for 32B-4bit to leave room for HF)
        # and with enforce_eager=True to avoid the 1-2GB CUDA-graph static
        # buffer that otherwise OOMs the HF backward pass. When coresident
        # is OFF these knobs are ignored for the mem fraction (the caller-
        # provided gpu_memory_utilization is used verbatim) and enforce_eager
        # applies only if the caller set it explicitly.
        self.enforce_eager = bool(enforce_eager)
        self.coresident_training_enabled = bool(coresident_training_enabled)
        self.coresident_vllm_mem_frac = float(coresident_vllm_mem_frac)
        # Task #18 step 2: chunked prefill. Interleaves prefill of long
        # prompts with decode across the 120-prompt solve batch so prefill
        # spikes don't stall the decode pipeline.
        self.enable_chunked_prefill = bool(enable_chunked_prefill)
        if self.coresident_training_enabled:
            # Override the effective VRAM fraction and force eager mode.
            self.gpu_memory_utilization = self.coresident_vllm_mem_frac
            self.enforce_eager = True
        # Track whether vLLM has been put to sleep (KV cache freed, weights
        # retained). When True, HF training can load a second base copy on
        # top; on exit_training we wake_up the engine.
        self._vllm_sleeping: bool = False
        self._llm = None
        self._tokenizer = None
        self._sampling_params_cls = None
        # HF model/tokenizer for training phases
        self._hf_model = None
        self._hf_tokenizer = None
        # Track the current model path (changes after checkpoint save)
        self._current_model_path = model_path
        # Active PEFT adapter (used on vLLM inference when set).
        self._lora_adapter_path: str | None = None
        self._lora_adapter_id: int = 0
        self._lora_adapter_name: str = "rsi_adapter"
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
        llm_kwargs = dict(
            model=self._current_model_path,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=self.allow_remote_code,
            disable_log_stats=True,
            # Explicit: propose/solve prompts share a long system-prompt prefix
            # ("<think>\n\n</think>\n\n" + chat template + task intro). With
            # prefix caching the attn KV for that shared prefix is computed once
            # per cycle and reused across all N candidate generations — a large
            # steady-state win. vLLM defaults this on in recent versions, but we
            # set it explicitly so the optimization survives version drift.
            enable_prefix_caching=True,
        )
        if self.max_num_seqs > 0:
            llm_kwargs["max_num_seqs"] = self.max_num_seqs
        if self.enforce_eager:
            # Task #19: disable CUDA-graph capture. Required when the engine
            # must coexist with HF training on a 48GB GPU (gemini consult:
            # the static graph buffer is the hidden 1-2GB that OOMs the
            # backward pass). Costs ~10-15% inference throughput.
            llm_kwargs["enforce_eager"] = True
        if self.enable_chunked_prefill:
            # Task #18 step 2: chunked prefill. Essential for the 120-prompt
            # solve batch with a long shared system prefix — without this,
            # prefill waves stall decode for multiple seconds per batch.
            llm_kwargs["enable_chunked_prefill"] = True
        if self.coresident_training_enabled:
            # Task #19: ask vLLM to support sleep_mode so we can drop KV cache
            # during the HF training span without destroying weights. Newer
            # vLLM exposes this as `enable_sleep_mode=True` on LLM(); older
            # versions auto-support sleep() on any engine. We set the kwarg
            # defensively; the TypeError retry below drops it if rejected.
            llm_kwargs["enable_sleep_mode"] = True
        # vLLM-side bitsandbytes 4-bit. Without this a 32B model can't
        # fit inference on a 48 GB GPU. The `load_format='bitsandbytes'`
        # tells vLLM to read pre-quantized .safetensors shards (as
        # published by unsloth's `-bnb-4bit` repos); for a raw fp16 model
        # vLLM would quantize on-the-fly but take much longer to start.
        qc = self.quantization_config
        if qc and qc.get("load_in_4bit"):
            llm_kwargs["quantization"] = "bitsandbytes"
            llm_kwargs["load_format"] = "bitsandbytes"
        elif qc and qc.get("load_in_8bit"):
            llm_kwargs["quantization"] = "bitsandbytes"
            llm_kwargs["load_format"] = "bitsandbytes"
        if self.enable_lora:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = self.max_lora_rank
        # enable_prefix_caching has been a vLLM kwarg since 0.3.x, but guard
        # against older/forked vLLM that rejects the kwarg. On TypeError we
        # retry without it; vLLM will still default-enable it where supported.
        try:
            self._llm = LLM(**llm_kwargs)
        except TypeError as _e:
            msg = str(_e)
            recovered = False
            for kw in (
                "enable_prefix_caching", "enable_lora", "max_lora_rank",
                "max_num_seqs", "enforce_eager", "enable_sleep_mode",
                "enable_chunked_prefill",
            ):
                if kw in msg and kw in llm_kwargs:
                    logger.warning(f"vLLM LLM() rejected {kw} — retrying without it")
                    llm_kwargs.pop(kw, None)
                    recovered = True
            if recovered:
                self._llm = LLM(**llm_kwargs)
            else:
                raise
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
            self._sampling_params_cls = None
            # Aggressive cleanup — run-6 saw 42 GiB PyTorch-allocated
            # during training when the HF model is only 15 GiB, suggesting
            # vLLM's KV cache / CUDA graph pool wasn't fully released.
            # Trigger multiple rounds of GC + ipc_collect + synchronize
            # so the allocator returns memory to the OS before HF loads.
            for _ in range(3):
                gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
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
                 temperature: float = 0.7, top_p: float = 0.9,
                 stop: list[str] | None = None) -> str:
        results = self.generate_batch([prompt], max_new_tokens, temperature, top_p, stop=stop)
        return results[0] if results else ""

    def generate_batch(self, prompts: list[str], max_new_tokens: int = 2048,
                       temperature: float = 0.7, top_p: float = 0.9,
                       stop: list[str] | None = None) -> list[str]:
        if not prompts:
            return []

        if self._llm is not None:
            # vLLM path (fast)
            greedy = temperature <= 0
            sp_kwargs = dict(
                max_tokens=max_new_tokens,
                temperature=0.0 if greedy else temperature,
                top_p=1.0 if greedy else top_p,
            )
            if stop:
                sp_kwargs["stop"] = list(stop)
            params = self._sampling_params_cls(**sp_kwargs)
            gen_kwargs = {}
            lora_req = self._build_lora_request()
            if lora_req is not None:
                gen_kwargs["lora_request"] = lora_req
            outputs = self._llm.generate(prompts, params, **gen_kwargs)
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
            # Left-padding: generated tokens start at input_ids.shape[1] for
            # every sample. Using attention_mask.sum() (non-pad length) as the
            # slice start was wrong — for padded samples it landed inside the
            # padding, so the decoded "response" silently included the prompt.
            input_len = inputs["input_ids"].shape[1]
            responses = []
            for output in outputs:
                responses.append(self._hf_tokenizer.decode(
                    output[input_len:], skip_special_tokens=True))
            del inputs, outputs
            return responses
        finally:
            self._hf_tokenizer.padding_side = original_side

    def _build_lora_request(self):
        """Construct a vLLM LoRARequest for the active adapter, if any.

        Returns None when no adapter is registered or LoRARequest import fails
        (older vLLM without LoRA support) — caller falls back to plain generate.
        """
        if not self._lora_adapter_path or not self.enable_lora:
            return None
        try:
            from vllm.lora.request import LoRARequest
        except ImportError:
            return None
        return LoRARequest(
            self._lora_adapter_name,
            self._lora_adapter_id,
            self._lora_adapter_path,
        )

    def set_lora_adapter(self, adapter_path: str | None) -> None:
        """Register a PEFT adapter directory for subsequent vLLM generate calls.

        Passing None clears the adapter (fall back to base-only inference).
        Increments the int id each time so vLLM's cache invalidates when the
        adapter directory contents change cycle-to-cycle.
        """
        if adapter_path is None:
            self._lora_adapter_path = None
            return
        p = Path(adapter_path)
        required = ("adapter_model.safetensors", "adapter_config.json")
        if not p.is_dir() or not all((p / f).exists() for f in required):
            logger.warning(
                f"set_lora_adapter: {adapter_path} missing PEFT files "
                f"({required}); skipping registration."
            )
            self._lora_adapter_path = None
            return
        self._lora_adapter_path = str(p)
        self._lora_adapter_id += 1
        logger.info(
            f"Registered LoRA adapter id={self._lora_adapter_id} at {p}"
        )

    # ---- Training swap interface ----

    def _vllm_sleep(self) -> bool:
        """Put vLLM to sleep (KV cache freed, weights retained). vLLM >= 0.6.

        Returns True on success. On any exception (older vLLM, driver quirk,
        AMD ROCm variant), returns False and leaves the engine running —
        caller should fall back to the destroy-and-reload path.

        NOTE: sleep level 1 drops KV blocks AND activations but keeps the
        loaded weights. Level 2 also offloads weights to CPU. We use level 1
        because the coresident path on 48GB A6000 assumes weights stay on
        GPU (HF training wants its OWN copy; offloading vLLM's weights only
        helps when memory is even tighter, e.g. FP8 KV + 70B model).
        """
        if self._llm is None:
            return False
        try:
            self._llm.sleep(level=1)
            self._vllm_sleeping = True
            logger.info("vLLM slept (KV cache released, weights retained)")
            return True
        except TypeError:
            # Older vLLM: sleep() may not accept `level` kwarg.
            try:
                self._llm.sleep()
                self._vllm_sleeping = True
                logger.info("vLLM slept (level kwarg not supported; using default)")
                return True
            except Exception as e:
                logger.warning(f"vllm.sleep() failed ({type(e).__name__}: {e}); falling back to reload path")
                return False
        except Exception as e:
            logger.warning(f"vllm.sleep() failed ({type(e).__name__}: {e}); falling back to reload path")
            return False

    def _vllm_wake(self) -> bool:
        """Wake vLLM after sleep. Returns True on success."""
        if self._llm is None or not self._vllm_sleeping:
            return False
        try:
            self._llm.wake_up()
            self._vllm_sleeping = False
            logger.info("vLLM woke up (KV cache reinitialized)")
            return True
        except Exception as e:
            logger.warning(f"vllm.wake_up() failed ({type(e).__name__}: {e}); engine state uncertain")
            # Attempt recovery: destroy and reload cleanly.
            self._vllm_sleeping = False
            try:
                self._unload_vllm()
                self._load_vllm()
                return True
            except Exception as e2:
                logger.error(f"vLLM wake recovery via reload also failed: {type(e2).__name__}: {e2}")
                return False

    def swap_to_hf_for_training(self) -> None:
        """Unload vLLM, load HF model for LoRA training.

        Task #19: when coresident_training_enabled is set AND vLLM supports
        sleep_mode, we SUSPEND vLLM instead of destroying it. The engine
        retains its weight allocation; only the KV cache is released. On
        exit_training we wake it back up — saving the 4-6 min/cycle
        unload+reload cost. Falls back to the legacy destroy-and-reload
        path on any failure.
        """
        if self.coresident_training_enabled and self._llm is not None:
            if self._vllm_sleep():
                # Sleep succeeded — vLLM weights still on GPU. Load HF on
                # top (duplicate 4-bit copy, ~18GB for 32B; budget set via
                # coresident_vllm_mem_frac).
                self._load_hf()
                return
            # Sleep failed — fall through to legacy path.
            logger.info("coresident sleep failed; using legacy unload/reload swap")
        self._unload_vllm()
        self._load_hf()

    def swap_to_vllm_after_training(self, checkpoint_path: str | None = None,
                                    adapter_path: str | None = None) -> None:
        """Unload HF, reload vLLM with merged weights. Falls back to HF on failure.

        Validates ``_current_model_path`` before loading. A local directory
        missing config.json (or marked incomplete) means a prior cycle's
        training failed without writing weights; loading it would raise
        "Can't load the configuration" and kill every subsequent cycle.
        When that happens, fall back to the original repo-id / base model
        so the loop keeps making progress.
        """
        if checkpoint_path:
            self._current_model_path = checkpoint_path
            self.config.model_path = checkpoint_path
        self._unload_hf()
        # Task #19: if the engine is merely sleeping (coresident path took
        # effect on enter_training), waking it up restores inference without
        # paying the full ~3-5 min reload cost. On bnb-4bit where the base
        # weights don't change, we don't need to reload from disk at all —
        # just register the new adapter via set_lora_adapter on the woken
        # engine. Gracefully falls through to the destroy-and-reload path
        # if wake fails.
        if self._vllm_sleeping and self._llm is not None:
            if self._vllm_wake():
                if adapter_path is not None:
                    self.enable_lora = True
                    self.set_lora_adapter(adapter_path)
                logger.info(
                    "coresident swap: woke vLLM + registered adapter "
                    "(reload skipped, saved ~3-5 min)"
                )
                return
            logger.info(
                "coresident wake failed; reverting to destroy-and-reload path"
            )
        # If vLLM is already loaded (e.g. resume path called before training),
        # unload it first so _load_vllm doesn't leak the prior engine's GPU mem.
        self._unload_vllm()

        if not _check_vllm() or not torch.cuda.is_available():
            logger.warning("vLLM or CUDA not available after training. Falling back to HF.")
            self._load_hf()
            return

        # Guard against stale/incomplete local checkpoints. HF repo ids have
        # exactly one slash and no path separator beyond that — treat those
        # as valid. Local dirs must contain config.json and not be marked
        # .incomplete.
        from pathlib import Path as _P
        p = _P(self._current_model_path)
        is_local_dir = p.exists() and p.is_dir()
        if is_local_dir:
            has_config = (p / "config.json").exists()
            is_incomplete = (p / ".incomplete").exists()
            if not has_config or is_incomplete:
                reason = "missing config.json" if not has_config else "marked incomplete"
                logger.warning(
                    f"Stale checkpoint at {p} ({reason}). Falling back to "
                    f"base model {self.model_path} for this swap."
                )
                self._current_model_path = self.model_path
                self.config.model_path = self.model_path

        # If the caller provided an adapter dir, enable LoRA on this reload and
        # register it so _build_lora_request returns it on generate(). Adapter
        # persistence matters most on bnb-4bit bases where merge_lora no-ops —
        # without this, every cycle evaluates the untrained base.
        if adapter_path is not None:
            self.enable_lora = True
            self.set_lora_adapter(adapter_path)

        try:
            self._load_vllm()
        except Exception as e:
            logger.warning(f"vLLM reload failed ({type(e).__name__}: {e}). Falling back to HF.")
            self._llm = None
            # If loading from checkpoint path failed, try the base model as a
            # last resort so the loop doesn't cascade-fail on a bad path.
            if self._current_model_path != self.model_path:
                logger.info(f"Retrying HF load from base model: {self.model_path}")
                self._current_model_path = self.model_path
                self.config.model_path = self.model_path
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
        """Save model checkpoint (only works in HF mode).

        QLoRA-4bit note: bnb-quantized HF models raise NotImplementedError
        on save_pretrained() because packed 4bit weights can't round-trip
        through safetensors shard save. Since merge_lora also skips 4bit
        (packed bytes ≠ dense delta), the checkpoint would be identical to
        the base model anyway — no point writing it, and a failed partial
        save leaves a corrupt dir that vLLM reload chokes on. Skip entirely
        when the HF model is quantized; vLLM reloads from original base.
        """
        if self._hf_model is None:
            logger.warning("Cannot save checkpoint — HF model not loaded")
            return
        # Detect bnb quantized base via hf_quantizer or model config.
        is_4bit = bool(
            getattr(self._hf_model, "is_loaded_in_4bit", False)
            or getattr(self._hf_model, "is_quantized", False)
            or (self.quantization_config and self.quantization_config.get("load_in_4bit"))
            or (self.quantization_config and self.quantization_config.get("load_in_8bit"))
        )
        if is_4bit:
            logger.info(
                "save_checkpoint: skipping for bnb-quantized base "
                "(save_pretrained would raise NotImplementedError; merge_lora "
                "already skipped so checkpoint == base). vLLM will reload base."
            )
            # Drop a marker dir so the swap_to_vllm_after_training guard
            # sees .incomplete and falls back to the base model path rather
            # than trying to load a nonexistent checkpoint.
            save_path = path / f"cycle_{cycle}"
            try:
                save_path.mkdir(parents=True, exist_ok=True)
                (save_path / ".incomplete").touch()
            except OSError:
                pass
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
