"""Custom LoRA trainer — built specifically for recursive self-improvement.

Key innovations over standard LoRA:
- Weakness-adaptive rank: weaker layers get higher rank (more capacity to learn)
- Proper lifecycle: LoRA layers are removed/reinjected between cycles
- Gradient scaling based on weakness severity
- Training loss weighted by sample confidence
"""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ..utils.config import TrainerConfig
from ..utils.model_loader import ModelLoader
from ..utils.structured_logs import emit as _emit_structured_log, is_enabled as _obs_enabled
from ..generator.data_generator import TrainingSample, PreferencePair

logger = logging.getLogger(__name__)

# HuggingFace Conv1D (used by GPT-2) — weight is transposed vs nn.Linear.
# Import it if available; otherwise LoRA only targets nn.Linear.
try:
    from transformers.pytorch_utils import Conv1D as _HFConv1D
except ImportError:
    _HFConv1D = None

def _is_linear_like(module):
    """Check if a module is nn.Linear or HuggingFace Conv1D."""
    if isinstance(module, nn.Linear):
        return True
    if _HFConv1D is not None and isinstance(module, _HFConv1D):
        return True
    return False

def _get_features(module):
    """Get (in_features, out_features) from a linear-like module."""
    if isinstance(module, nn.Linear):
        return module.in_features, module.out_features
    # HF Conv1D stores weight as (in_features, out_features) — transposed
    return module.weight.shape[0], module.weight.shape[1]


class LoRALayer(nn.Module):
    """Modernized LoRA layer composing four 2023-2024 improvements over Hu 2021.

    Stacks all four orthogonal techniques on top of the original LoRA:

    - **rsLoRA** (Kalajdzievski 2023): scaling = alpha/sqrt(rank) stabilizes
      gradient magnitude across ranks so high-rank LoRA actually helps.
    - **PiSSA init** (Meng et al. 2024): initialize A,B from the top-r SVD
      components of W and subtract them from the frozen weight — LoRA starts
      on the principal subspace, ~2-3× faster convergence.
    - **DoRA** (Liu et al. 2024): decompose W into magnitude (per-input
      scalar) and direction (LoRA-adapted). Same rank, materially better
      adaptation quality because magnitude and direction train independently.
    - **Weakness-adaptive rank + gradient scaling** (RSI-specific): weaker
      layers get higher rank and a backward-pass gradient multiplier so
      they catch up across cycles.

    When use_dora=True, the forward becomes

        V           = W_frozen + scaling · (B @ A)    (adapted direction, full)
        magnitude   : trainable R^in (one scalar per input feature)
        W_effective = magnitude ⊙ V / ‖V‖_c           (column-normalized)
        y           = W_effective · x

    ‖V‖_c is computed under no_grad (standard DoRA trick from §3.1) so
    backward only has to flow through magnitude, A, and B. At init the
    magnitude is set to ‖W_frozen‖_c so W_effective ≡ W_frozen exactly
    (B is zero for kaiming init; for PiSSA the residual math still holds).
    """

    def __init__(
        self,
        original_layer: nn.Module,
        rank: int,
        alpha: int,
        dropout: float = 0.0,
        weakness_scale: float = 1.0,
        *,
        use_rslora: bool = True,
        init_method: str = "kaiming",
        use_dora: bool = False,
    ):
        super().__init__()
        self.original = original_layer
        self.rank = rank
        self.alpha = alpha
        # rsLoRA (Kalajdzievski 2023): scaling = alpha / sqrt(rank) keeps
        # gradient magnitude stable across ranks, unlike classic alpha/rank
        # which collapses as rank grows.
        if use_rslora:
            self.scaling = alpha / math.sqrt(max(rank, 1))
        else:
            self.scaling = alpha / max(rank, 1)
        self.use_rslora = use_rslora
        self.init_method = init_method
        self.use_dora = bool(use_dora)
        # Auto-disable DoRA when base is bitsandbytes-4bit or 8bit: DoRA's
        # magnitude scale needs the dense [out, in] weight to compute
        # ‖W + scaling·BA‖_c, but bnb quantized layers store .weight as a
        # packed Params4bit/Int8Params (1D-ish flat shape). Dequantizing
        # every forward pass would defeat quantization's VRAM savings, and
        # plain LoRA on quantized base is the canonical QLoRA recipe anyway.
        #
        # Detection is ATTRIBUTE-based, not class-name-based: various wrappers
        # (Qwen2's quant modules, transformers' Linear4bit subclasses, vLLM-
        # integrated paths, etc.) have different class names but all expose
        # `weight.quant_state` (bnb-4bit) or `weight.CB` / `weight.SCB`
        # (bnb-8bit). Class-name checks miss these and we saw the exact same
        # packed-weight crash on DeepSeek-R1-Distill-Qwen-32B-bnb-4bit.
        w = getattr(original_layer, "weight", None)
        self._base_is_4bit = (
            w is not None and (
                hasattr(w, "quant_state")
                or type(w).__name__ in ("Params4bit",)
            )
        )
        self._base_is_8bit = (
            w is not None and (
                hasattr(w, "CB") or hasattr(w, "SCB")
                or type(w).__name__ in ("Int8Params",)
            )
        )
        # Backstop: also check class name for environments where bnb's private
        # attrs were renamed.
        cls_name = type(original_layer).__name__
        if cls_name in ("Linear4bit", "LinearFP4", "LinearNF4"):
            self._base_is_4bit = True
        if cls_name == "Linear8bitLt":
            self._base_is_8bit = True
        if self.use_dora and (self._base_is_4bit or self._base_is_8bit):
            self.use_dora = False
        self.weakness_scale = weakness_scale
        # Track if original is Conv1D (transposed weight) for merge
        self._is_conv1d = _HFConv1D is not None and isinstance(original_layer, _HFConv1D)

        in_features, out_features = _get_features(original_layer)
        device = original_layer.weight.device
        dtype = original_layer.weight.dtype

        # Keep LoRA parameters in float32 for numerical stability during training.
        # LoRA params are tiny (rank × dim) so the 2x VRAM cost vs bfloat16 is
        # negligible compared to the base model. This avoids precision loss in
        # gradient accumulation and high-rank matmuls.
        if init_method == "pissa":
            # PiSSA (Meng et al. 2024): init A,B from top-r SVD components of W,
            # and subtract those from original so the layer's initial function is
            # unchanged. LoRA then trains the principal subspace directly.
            import time
            t0 = time.time()
            W = original_layer.weight.data
            # For Conv1D, weight is (in, out); SVD expects (out, in)-style
            # orientation that lets A=(r, in), B=(out, r).
            W_mat = W.T if self._is_conv1d else W
            W_f32 = W_mat.to(torch.float32)
            U, S, Vh = torch.linalg.svd(W_f32, full_matrices=False)
            r = min(rank, S.shape[0])
            U_r = U[:, :r]
            S_r = S[:r]
            Vh_r = Vh[:r, :]
            # PiSSA's invariant is scaling·(B @ A) == top_r(W) at init — NOT
            # B @ A == top_r. With rsLoRA the output scaling is alpha/sqrt(rank)
            # (typically 4 for alpha=16, rank=16), so leaving it out of the init
            # makes the layer compute W + (scaling-1)·top_r at step 0 — a
            # substantial weight corruption that every cycle had to unlearn.
            # Divide the SVD singular values by `scaling` so B @ A = top_r/scaling
            # and scaling·(B @ A) = top_r as the math requires.
            sqrt_S = (S_r / self.scaling).clamp_min(0.0).sqrt()
            lora_A_init = torch.zeros(rank, in_features, device=device, dtype=torch.float32)
            lora_B_init = torch.zeros(out_features, rank, device=device, dtype=torch.float32)
            lora_A_init[:r] = (sqrt_S.unsqueeze(1) * Vh_r).to(device=device)
            lora_B_init[:, :r] = (U_r * sqrt_S.unsqueeze(0)).to(device=device)
            # Subtract captured components from original weight.
            residual = W_f32 - (U_r * S_r) @ Vh_r
            residual = residual.to(dtype=dtype)
            if self._is_conv1d:
                residual = residual.T
            original_layer.weight.data.copy_(residual)
            self.lora_A = nn.Parameter(lora_A_init)
            self.lora_B = nn.Parameter(lora_B_init)
            elapsed = time.time() - t0
            logger.info(
                f"PiSSA init: ({out_features}x{in_features}) rank={rank} took {elapsed:.2f}s"
            )
        else:
            lora_A_init = torch.zeros(rank, in_features, device=device, dtype=torch.float32)
            nn.init.kaiming_uniform_(lora_A_init, a=math.sqrt(5))
            self.lora_A = nn.Parameter(lora_A_init)
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=device, dtype=torch.float32))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # DoRA magnitude — trainable per-input scalar initialized so the layer's
        # initial function matches W_frozen. After PiSSA the frozen weight is
        # the RESIDUAL (W - top-r components), so we compute magnitude from
        # whatever is currently in original.weight — the math
        #   W_effective = magnitude ⊙ (W_frozen + scaling·BA) / ‖...‖_c
        # stays correct because PiSSA's init gives scaling·BA = top-r, so
        # W_frozen + scaling·BA = W (the original full weight) at step 0.
        if self.use_dora:
            with torch.no_grad():
                # At init, V = W_frozen + scaling·BA. For kaiming, B=0 so V=W_frozen.
                # For PiSSA, V equals the pre-residual W (by construction). Compute
                # magnitude to match whichever case applies.
                W_cur = self.original.weight.float()
                # For kaiming: B=0 → V=W_cur. For PiSSA: add back the top-r so
                # V equals the original pre-residual weight.
                if init_method == "pissa":
                    BA_init = (self.lora_B.float() @ self.lora_A.float())
                    if self._is_conv1d:
                        V_init = W_cur + self.scaling * BA_init.T
                    else:
                        V_init = W_cur + self.scaling * BA_init
                else:
                    V_init = W_cur
                if self._is_conv1d:
                    init_mag = V_init.norm(dim=1)  # (in,) — rows of stored Conv1D
                else:
                    init_mag = V_init.norm(dim=0)  # (in,) — columns of Linear
                init_mag = init_mag.clamp_min(1e-8)
            self.magnitude = nn.Parameter(init_mag)
        else:
            self.register_parameter("magnitude", None)

        # Freeze original weights
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

        # Gradient scaling: amplify gradients for weaker layers so they learn faster.
        # This is how weakness_scale works — NOT in the forward pass (which would
        # create a mismatch at merge time) but in the backward pass.
        # Skip for scales < 1.05 — a <5% gradient boost isn't worth the overhead
        # of a hook on every backward pass (2x with gradient checkpointing).
        if weakness_scale >= 1.05:
            self.lora_A.register_hook(lambda grad, s=weakness_scale: grad * s)
            self.lora_B.register_hook(lambda grad, s=weakness_scale: grad * s)
            # DoRA: magnitude is also trainable and part of the adapter. Without
            # the hook, A/B get the 1.5× boost but magnitude stays at 1×, so
            # the adapter learns asymmetrically and the weakness-scale mechanism
            # is only partially applied.
            if self.use_dora and self.magnitude is not None:
                self.magnitude.register_hook(lambda grad, s=weakness_scale: grad * s)

    def _dora_scale_factor(self) -> torch.Tensor:
        """Compute (magnitude / ‖V‖_c) — the per-input DoRA gate.

        Returns a (in_features,) tensor. ‖V‖_c is computed under no_grad so
        gradient only flows through the trainable magnitude parameter; A and B
        still receive gradient via their path through the forward matmul.
        """
        with torch.no_grad():
            W0 = self.original.weight.float()
            BA = self.lora_B @ self.lora_A  # (out, in) float32
            if self._is_conv1d:
                # Conv1D stored as (in, out); effective V is (out, in) = V_stored.T
                V_stored = W0 + self.scaling * BA.T
                V_norm = V_stored.norm(dim=1).clamp_min(1e-8)  # (in,)
            else:
                V = W0 + self.scaling * BA  # (out, in)
                V_norm = V.norm(dim=0).clamp_min(1e-8)  # (in,)
        return self.magnitude / V_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_dora:
            # DoRA (Liu 2024): rewrite y = W_effective x as
            #   y = (W_frozen + scaling·BA) · (magnitude / ‖V‖_c ⊙ x)
            # Efficient because the per-input scale absorbs the magnitude/norm
            # modulation without materializing the full effective weight.
            scale = self._dora_scale_factor().to(x.dtype)
            x_mod = x * scale  # (..., in)
            original_output = self.original(x_mod)
            dropped = self.lora_dropout(x_mod)
            low_rank = dropped @ self.lora_A.to(dropped.dtype).T
            lora_output = low_rank.float() @ self.lora_B.T
            return original_output + lora_output.to(original_output.dtype) * self.scaling

        # Plain LoRA path (unchanged, with rsLoRA scaling baked into self.scaling).
        original_output = self.original(x)
        # Apply dropout in the ORIGINAL dtype (bfloat16) to avoid allocating a
        # full float32 copy of x. Dropout is applied on the INPUT (standard LoRA),
        # not between A and B (which would zero rank-dim units, far more aggressive).
        dropped = self.lora_dropout(x)
        # First matmul with A reduces (batch, seq, in_features) → (batch, seq, rank).
        # Upcast x to float32 only for the matmul — but that's expensive for large x.
        # Instead, cast A down to bf16 for the first matmul. The `.to()` on A is cheap
        # because A is (rank, in_features) which is small. THEN upcast the small
        # (batch, seq, rank) result to float32 for the second matmul with B.
        # Total extra alloc: (rank, in_features) bf16 copy of A ≈ 1MB for rank=64, dim=8192
        # vs 512MB for (4, 4096, 8192) if we upcasted x instead.
        low_rank = dropped @ self.lora_A.to(dropped.dtype).T  # (batch, seq, rank) in bf16
        lora_output = low_rank.float() @ self.lora_B.T  # (batch, seq, out_features) in f32
        # Only use scaling (alpha/rank) here — NOT weakness_scale.
        return original_output + lora_output.to(original_output.dtype) * self.scaling


class TrainingDataset(Dataset):
    """Dataset wrapping verified training samples with confidence weighting."""

    def __init__(self, samples: list[TrainingSample], tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._encoded = []
        self._encode_all(samples)

    def _encode_all(self, samples: list[TrainingSample]):
        # Curriculum: sort easy→hard (fewer reasoning steps = easier)
        samples = sorted(samples, key=lambda s: len(s.reasoning_chain))
        for sample in samples:
            formatted = sample.to_training_format()

            # Encode prompt and completion SEPARATELY, then concatenate IDs.
            # This avoids BPE tokenization mismatches: tokenizing "prompt\n\ncompletion"
            # as one string can produce different token boundaries at the join point
            # than tokenizing "prompt\n\n" alone, making the old prompt_len unreliable.
            prompt_text = formatted["prompt"] + "\n\n"
            completion_text = formatted["completion"]
            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
            completion_ids = self.tokenizer(completion_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]

            # Some tokenizers add EOS with add_special_tokens=True. Strip it from
            # the prompt — a mid-sequence EOS teaches the model to ignore the stop
            # signal, defeating the purpose of the EOS we add after completion.
            eos_id = self.tokenizer.eos_token_id
            if eos_id is not None and len(prompt_ids) > 1 and prompt_ids[-1].item() == eos_id:
                prompt_ids = prompt_ids[:-1]

            # Append EOS so the model learns to stop generating after the conclusion.
            # Without this, fine-tuning teaches the model to produce text that never
            # terminates, since it never sees a stop signal during training.
            if eos_id is not None:
                completion_ids = torch.cat([completion_ids, torch.tensor([eos_id], dtype=completion_ids.dtype)])

            prompt_len = len(prompt_ids)
            combined = torch.cat([prompt_ids, completion_ids])

            # Truncate to max_length, preserving EOS at the end.
            # Without this, long samples lose the stop signal we appended,
            # teaching the model to generate indefinitely for hard problems.
            # Truncate to max_length-1 then re-append EOS so the last content
            # token isn't overwritten — the model learns a clean content→EOS
            # transition instead of a corrupted label.
            if len(combined) > self.max_length:
                if eos_id is not None:
                    combined = torch.cat([combined[:self.max_length - 1],
                                          torch.tensor([eos_id], dtype=combined.dtype)])
                else:
                    combined = combined[:self.max_length]

            # Skip if prompt fills the entire sequence (no completion tokens to train on).
            # When EOS is present, need at least 1 content token + EOS = 2 beyond prompt.
            # When EOS is absent, need at least 1 content token.
            min_completion = 2 if eos_id is not None else 1
            if prompt_len + min_completion > len(combined):
                continue

            # Task #20 throughput: DO NOT pre-pad to max_length here. The
            # dataset emits un-padded per-sample tensors and the collate_fn
            # (`_dynamic_pad_collate`) pads each batch to the longest sample
            # in that batch. On mixed 200-1000 token pools this cuts attention
            # FLOPs ~2-4× vs the old all-pad-to-1024 behavior. Truncation
            # above already preserved EOS at max_length, so dynamic padding
            # is a pure compute win.
            actual_len = len(combined)

            labels = combined.clone()
            labels[:prompt_len] = -100

            attention_mask = torch.ones(actual_len, dtype=torch.long)

            entry = {
                "input_ids": combined,
                "attention_mask": attention_mask,
                "labels": labels,
            }
            # Confidence of 0.0 means unset — use 1.0 as default so it doesn't
            # zero out the loss. Verified samples always have confidence > 0.
            weight = sample.confidence if sample.confidence > 0 else 1.0
            # Amplify weight by weakness severity so the trainer focuses harder
            # on samples generated for severe weaknesses. severity is in [0,1];
            # 0.0 means unset (pre-rebuild samples) so leaves weight unchanged.
            severity = getattr(sample, "severity_at_generation", 0.0) or 0.0
            weight = weight * (1.0 + severity)
            # Self-consistency: samples where multiple independent generations
            # agreed on the final answer are more trustworthy. Default 1.0 means
            # the check wasn't run. Values in (0, 1] downweight uncertain samples.
            consistency = getattr(sample, "consistency_score", 1.0)
            if consistency > 0.0:
                weight = weight * consistency
            entry["sample_weight"] = torch.tensor(weight, dtype=torch.float32)
            # Per-sample Brier score over its own [C:...] markers, compared
            # against the sample's ground-truth correctness. NaN -> -1 marker
            # for "no confidences to penalize"; the trainer reads this tensor
            # and skips the aux loss when the flag is set but no data exists.
            confs = [c for c in (sample.per_step_confidence or [])
                     if isinstance(c, (int, float)) and 0.0 <= c <= 1.0]
            if confs:
                gt = 1.0 if getattr(sample, "ground_truth_verified", False) else 0.0
                brier = sum((c - gt) ** 2 for c in confs) / len(confs)
            else:
                brier = -1.0  # sentinel: no calibration data for this sample
            entry["calibration_brier"] = torch.tensor(brier, dtype=torch.float32)
            self._encoded.append(entry)

    def __len__(self):
        return len(self._encoded)

    def __getitem__(self, idx):
        return self._encoded[idx]


class PreferenceDataset(Dataset):
    """Dataset wrapping preference pairs for DPO training.

    Each item encodes the shared prompt once and both responses, yielding
    (input_ids, labels, attention_mask) for chosen and rejected separately.
    Labels mask the prompt so only response tokens contribute to logp.
    """

    def __init__(self, pairs: list[PreferencePair], tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._encoded: list[dict] = []
        self._encode_all(pairs)

    def _encode_side(self, prompt_text: str, completion_text: str) -> Optional[dict]:
        """Encode one side of a preference pair, returning None if it doesn't fit."""
        prompt_ids = self.tokenizer(
            prompt_text, add_special_tokens=True, return_tensors="pt"
        )["input_ids"][0]
        completion_ids = self.tokenizer(
            completion_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0]

        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None and len(prompt_ids) > 1 and prompt_ids[-1].item() == eos_id:
            prompt_ids = prompt_ids[:-1]
        if eos_id is not None:
            completion_ids = torch.cat(
                [completion_ids, torch.tensor([eos_id], dtype=completion_ids.dtype)]
            )

        prompt_len = len(prompt_ids)
        combined = torch.cat([prompt_ids, completion_ids])

        if len(combined) > self.max_length:
            if eos_id is not None:
                combined = torch.cat(
                    [combined[: self.max_length - 1],
                     torch.tensor([eos_id], dtype=combined.dtype)]
                )
            else:
                combined = combined[: self.max_length]

        min_completion = 2 if eos_id is not None else 1
        if prompt_len + min_completion > len(combined):
            return None

        # Task #20 throughput: emit un-padded tensors; dynamic collate pads
        # per-batch to the longest sequence.
        labels = combined.clone()
        labels[:prompt_len] = -100
        attention_mask = torch.ones(len(combined), dtype=torch.long)
        return {
            "input_ids": combined,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _encode_all(self, pairs: list[PreferencePair]):
        for pair in pairs:
            prompt_text = pair.prompt + "\n\n"
            chosen = self._encode_side(prompt_text, pair.chosen_response)
            rejected = self._encode_side(prompt_text, pair.rejected_response)
            if chosen is None or rejected is None:
                continue
            weight = pair.weight if pair.weight > 0 else 1.0
            severity = pair.severity_at_generation or 0.0
            weight = weight * (1.0 + severity)
            entry = {
                "chosen_input_ids": chosen["input_ids"],
                "chosen_attention_mask": chosen["attention_mask"],
                "chosen_labels": chosen["labels"],
                "rejected_input_ids": rejected["input_ids"],
                "rejected_attention_mask": rejected["attention_mask"],
                "rejected_labels": rejected["labels"],
                "sample_weight": torch.tensor(weight, dtype=torch.float32),
            }
            self._encoded.append(entry)

    def __len__(self):
        return len(self._encoded)

    def __getitem__(self, idx):
        return self._encoded[idx]


def _pad_right(tensors: list[torch.Tensor], pad_value: int) -> torch.Tensor:
    """Right-pad a list of 1-D tensors to the max length in the list."""
    max_len = max(t.shape[0] for t in tensors)
    out = torch.full((len(tensors), max_len), pad_value, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        out[i, : t.shape[0]] = t
    return out


def make_dynamic_pad_collate(pad_token_id: int):
    """Collate fn for SFT: dynamic per-batch right-padding.

    Keeps the old batch-dict shape (input_ids/attention_mask/labels +
    sample_weight + calibration_brier) so the trainer inner loop is
    unchanged — only the sequence dimension is now batch-local.
    """
    def _collate(items: list[dict]) -> dict:
        input_ids = _pad_right([it["input_ids"] for it in items], pad_token_id)
        attention_mask = _pad_right([it["attention_mask"] for it in items], 0)
        labels = _pad_right([it["labels"] for it in items], -100)
        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        # Optional per-sample auxiliary tensors. Stack if present on every item.
        if all("sample_weight" in it for it in items):
            out["sample_weight"] = torch.stack([it["sample_weight"] for it in items])
        if all("calibration_brier" in it for it in items):
            out["calibration_brier"] = torch.stack([it["calibration_brier"] for it in items])
        return out
    return _collate


def make_dynamic_pad_collate_dpo(pad_token_id: int):
    """Collate fn for DPO: pad chosen and rejected sides independently,
    each to the longest sequence of its own side within the batch.
    """
    def _collate(items: list[dict]) -> dict:
        out = {
            "chosen_input_ids": _pad_right(
                [it["chosen_input_ids"] for it in items], pad_token_id
            ),
            "chosen_attention_mask": _pad_right(
                [it["chosen_attention_mask"] for it in items], 0
            ),
            "chosen_labels": _pad_right(
                [it["chosen_labels"] for it in items], -100
            ),
            "rejected_input_ids": _pad_right(
                [it["rejected_input_ids"] for it in items], pad_token_id
            ),
            "rejected_attention_mask": _pad_right(
                [it["rejected_attention_mask"] for it in items], 0
            ),
            "rejected_labels": _pad_right(
                [it["rejected_labels"] for it in items], -100
            ),
            "sample_weight": torch.stack([it["sample_weight"] for it in items]),
        }
        return out
    return _collate


@dataclass
class TrainingMetrics:
    """Metrics from a training run."""
    cycle: int
    avg_loss: float  # mean loss across all batches (unweighted)
    final_loss: float
    steps: int
    samples_used: int
    samples_rejected: int
    learning_rate: float
    lora_layers_injected: int = 0
    avg_rank: float = 0.0
    undertrained: bool = False
    # DPO-specific (0.0 when SFT-only).
    training_mode: str = "sft"
    dpo_pairs_used: int = 0
    avg_reward_margin: float = 0.0  # mean (logp_chosen - logp_rejected) in final pass
    # GRPO-specific (0.0 when not GRPO).
    grpo_prompts_used: int = 0
    grpo_group_size: int = 0
    avg_reward: float = 0.0        # mean reward across all rollouts
    avg_reward_std: float = 0.0    # mean within-group reward std (advantage denom)
    rollout_refreshes: int = 0
    # Metacognitive calibration (metacog_calib).
    # calibration_ece: Expected Calibration Error over per-step [C:x] markers
    #   found in training data this cycle. NaN when no markers were present.
    # calibration_brier: mean Brier score over the same population.
    # calibration_samples: number of (step, correctness) pairs contributing.
    calibration_ece: float = float("nan")
    calibration_brier: float = float("nan")
    calibration_samples: int = 0
    # Optional per-step training loss trajectory. Empty list unless the trainer
    # was constructed with collect_loss_trajectory=True. Keeps memory small when
    # the observability path is off.
    loss_trajectory: list[float] = field(default_factory=list)


def _compute_calibration(
    samples: list["TrainingSample"], num_bins: int = 10
) -> tuple[float, float, int]:
    """Compute (ECE, mean Brier, #pairs) from per-step confidences vs. correctness.

    Each reasoning step's confidence is compared against a per-step correctness
    proxy: the sample's ``ground_truth_verified`` flag (broadcast to all steps).
    Steps with unset confidence (-1.0) are skipped. Returns (nan, nan, 0) when
    no markers are available — the system simply didn't emit any, and we don't
    want to fabricate a calibration score.
    """
    confs: list[float] = []
    correct: list[float] = []
    for s in samples:
        gt = 1.0 if getattr(s, "ground_truth_verified", False) else 0.0
        for c in getattr(s, "per_step_confidence", []) or []:
            if c is None or c < 0.0 or c > 1.0:
                continue
            confs.append(float(c))
            correct.append(gt)
    n = len(confs)
    if n == 0:
        return float("nan"), float("nan"), 0
    brier = sum((c - y) ** 2 for c, y in zip(confs, correct)) / n
    # Bucket into num_bins equal-width bins over [0, 1]; ECE = sum_k (|B_k|/n) * |acc_k - conf_k|
    bins_conf: list[float] = [0.0] * num_bins
    bins_acc: list[float] = [0.0] * num_bins
    bins_cnt: list[int] = [0] * num_bins
    for c, y in zip(confs, correct):
        idx = min(int(c * num_bins), num_bins - 1)
        bins_conf[idx] += c
        bins_acc[idx] += y
        bins_cnt[idx] += 1
    ece = 0.0
    for k in range(num_bins):
        if bins_cnt[k] == 0:
            continue
        mean_conf = bins_conf[k] / bins_cnt[k]
        mean_acc = bins_acc[k] / bins_cnt[k]
        ece += (bins_cnt[k] / n) * abs(mean_acc - mean_conf)
    return ece, brier, n


class _EarlyStop(Exception):
    """Internal signal: loss dropped below early_stop_loss mid-training."""


def _filter_any_fail_when_clean_enough(
    samples: list,
    clean_floor: int,
) -> tuple[list, int]:
    """Drop samples carrying a verifier 'any_fail' warning when the clean pool
    is large enough on its own. Returns (filtered_samples, n_dropped).

    When ``clean_floor <= 0`` the filter is disabled. When either the total
    pool size OR the clean-only subset size is below the floor, the filter
    is also disabled — we'd rather train on a mixed-quality pool than starve
    the cycle. This is the floor-protected filter asked for in task #14.
    """
    if clean_floor <= 0 or not samples:
        return list(samples), 0
    clean = [s for s in samples
             if "any_fail" not in (getattr(s, "verdict_warnings", ()) or ())]
    total = len(samples)
    if total < clean_floor or len(clean) < clean_floor:
        return list(samples), 0
    dropped = total - len(clean)
    return clean, dropped


def _plan_step_budget(
    total_batches: int,
    base_accum: int,
    max_steps_per_cycle: int,
    min_steps_per_cycle: int,
) -> tuple[int, int, bool]:
    """Plan (effective_accum, total_steps, skip) for a training cycle.

    Pure function, extracted from `_train_inner` so the adaptive-step logic
    is unit-testable without spinning up a torch model. Scales grad_accum up
    (never down) so that total_steps <= max_steps_per_cycle. If even with
    accum = total_batches we fall below min_steps_per_cycle, returns skip=True.
    """
    base_accum = max(1, base_accum)
    max_steps = max(1, max_steps_per_cycle)
    min_steps = max(1, min_steps_per_cycle)
    if total_batches <= 0:
        return base_accum, 0, True
    need_accum = max(1, (total_batches + max_steps - 1) // max_steps)
    effective_accum = max(base_accum, need_accum)
    total_steps = max(
        1,
        total_batches // effective_accum
        + (1 if total_batches % effective_accum != 0 else 0),
    )
    skip = total_steps < min_steps
    return effective_accum, total_steps, skip


class CustomLoRATrainer:
    """Custom LoRA trainer with proper lifecycle management.

    Critical: LoRA layers are stripped and reinjected between cycles.
    This prevents the LoRA-on-LoRA wrapping bug.
    """

    def __init__(
        self,
        config: TrainerConfig,
        model_loader: ModelLoader,
        reward_fn: Optional[Callable[[str, str, "TrainingSample"], float]] = None,
    ):
        self.config = config
        self.model_loader = model_loader
        self._lora_layers: dict[str, LoRALayer] = {}
        self._original_layers: dict[str, nn.Module] = {}  # Linear or Conv1D
        # Pluggable reward function for GRPO. Signature:
        #   reward_fn(prompt: str, completion: str, sample: TrainingSample) -> float
        # `sample` is the originating TrainingSample (gives access to canonical
        # answer, domain, check_type, etc.). A later PRM (prm_train) can supply
        # a process-level reward via this same hook.
        # Default: canonical-answer grade if available, else 0.0.
        self._reward_fn = reward_fn
        # Observability hook: when True, per-batch unweighted losses are
        # accumulated into `_loss_trajectory` and surfaced via TrainingMetrics.
        # Off by default to avoid growing lists in long runs.
        self._collect_loss_trajectory: bool = False
        self._loss_trajectory: list[float] = []
        # Structured-observability: orchestrator wires an OrchestratorConfig
        # stand-in here (.output_dir, .structured_observability_enabled,
        # .structured_log_training_steps). When None, emit is a no-op.
        self._obs_cfg = None

    def set_observability_config(self, obs_cfg) -> None:
        """Wire an OrchestratorConfig (or any object with the expected
        attributes) so the optimizer-step loop can emit training_steps
        records. Safe to pass None (disables)."""
        self._obs_cfg = obs_cfg

    def _emit_training_step_log(
        self,
        *,
        cycle: int,
        step_idx: int,
        loss_unweighted: float,
        loss_weighted: float,
        sample_weight: Optional[float],
        verdict_warnings: Optional[tuple[str, ...]],
        lora_params: list,
        lr_A: Optional[float],
        lr_B: Optional[float],
        clip_fraction: Optional[float],
        time_ms: float,
        sample_idx_in_batch: Optional[int] = None,
    ) -> None:
        """Append one training_steps.jsonl record. Never raises.

        Called AFTER optimizer.step() so post-step B statistics reflect the
        update that just landed. Grad norms are read from .grad BEFORE
        zero_grad() by keeping this call sequenced correctly at the call
        site; when grads have already been zeroed the values come back 0.0
        (acceptable — the emitter can't reconstruct vanished state).
        """
        try:
            if not _obs_enabled(self._obs_cfg, "training_steps"):
                return
            # Split lora_params into A, B, magnitude (DoRA) groups by name hint.
            grad_sq_A = 0.0
            grad_sq_B = 0.0
            grad_sq_mag = 0.0
            grad_sq_total = 0.0
            post_B_max = 0.0
            post_B_sum = 0.0
            post_B_count = 0
            for p in lora_params:
                g = getattr(p, "grad", None)
                name = getattr(p, "_lora_role", "")  # best-effort tag
                if g is not None:
                    v = float(g.detach().float().pow(2).sum().item())
                    grad_sq_total += v
                    if "A" in name or name == "":
                        # Fallback: bucket by tensor shape. LoRA A has
                        # (rank, in_features); B has (out_features, rank).
                        # Heuristic — rank is typically the smaller dim.
                        if p.ndim >= 2 and p.shape[0] <= p.shape[1]:
                            grad_sq_A += v
                        elif p.ndim >= 2:
                            grad_sq_B += v
                            # Track post-step B statistics.
                            b_abs = p.detach().float().abs()
                            post_B_max = max(post_B_max, float(b_abs.max().item()))
                            post_B_sum += float(b_abs.mean().item())
                            post_B_count += 1
                        else:
                            grad_sq_mag += v
            record = {
                "cycle": int(cycle),
                "step_idx": int(step_idx),
                "sample_idx_in_batch": sample_idx_in_batch,
                "loss_unweighted": float(loss_unweighted),
                "loss_weighted": float(loss_weighted),
                "sample_weight": (
                    float(sample_weight) if sample_weight is not None else None
                ),
                "verdict_warnings": (
                    list(verdict_warnings) if verdict_warnings else []
                ),
                "grad_norm_lora_A": math.sqrt(grad_sq_A),
                "grad_norm_lora_B": math.sqrt(grad_sq_B),
                "grad_norm_magnitude": math.sqrt(grad_sq_mag),
                "grad_norm_total": math.sqrt(grad_sq_total),
                "lr_A": lr_A,
                "lr_B": lr_B,
                "post_step_B_max_abs": post_B_max,
                "post_step_B_mean_abs": (
                    post_B_sum / post_B_count if post_B_count else 0.0
                ),
                "clip_fraction": clip_fraction,
                "time_ms": float(time_ms),
            }
            _emit_structured_log("training_steps", record, self._obs_cfg)
        except Exception as _e:  # pragma: no cover — defensive
            logger.debug(
                "training_steps emit failed (%s): %s",
                type(_e).__name__, _e,
            )

    def set_collect_loss_trajectory(self, enabled: bool) -> None:
        """Toggle per-step loss collection. Must be set before `train()`."""
        self._collect_loss_trajectory = bool(enabled)

    def set_reward_fn(
        self, reward_fn: Callable[[str, str, "TrainingSample"], float],
    ) -> None:
        """Swap in a new reward function (e.g. a trained PRM from prm_train)."""
        self._reward_fn = reward_fn

    def inject_lora(self, weak_layers: Optional[dict[str, float]] = None) -> None:
        """Inject LoRA layers, removing any existing ones first."""
        # CRITICAL: Remove any existing LoRA layers before injecting new ones
        self.strip_lora()

        model = self.model_loader.model
        weak_layers = weak_layers or {}
        ranks = []

        for name, module in list(model.named_modules()):
            if not _is_linear_like(module):
                continue
            # Already a LoRA layer — skip (safety check)
            if isinstance(module, LoRALayer):
                continue
            # Skip nested .original modules inside LoRA layers — if strip_lora
            # failed partway, these are the original Linear layers stored inside
            # un-stripped LoRALayers. Wrapping them would create LoRA-on-LoRA.
            if ".original" in name:
                continue

            module_type = name.split(".")[-1]
            if module_type not in self.config.target_modules:
                continue

            # Calculate weakness-adaptive rank.
            # layer_health keys from diagnostics include ".weight"/".bias" suffix
            # (from named_parameters), but module names from named_modules don't.
            # Try both forms. Default to 1.0 (healthy) — unknown layers should get
            # base rank, not boosted rank. The old 0.5 default wasted LoRA capacity
            # on undiagnosed layers, especially when activation analysis is disabled.
            health = weak_layers.get(name, weak_layers.get(f"{name}.weight", 1.0))
            weakness = 1.0 - health
            adjusted_rank = int(self.config.lora_rank * (1.0 + weakness * self.config.weakness_rank_scale))
            adjusted_rank = max(self.config.min_rank, min(adjusted_rank, self.config.max_rank))
            ranks.append(adjusted_rank)

            # Store original for later restoration
            self._original_layers[name] = module

            # Scale alpha proportionally with rank so that scaling = alpha/rank
            # stays constant. Without this, weak layers (high rank) get tiny scaling
            # that counteracts the extra capacity, while strong layers (low rank) get
            # amplified scaling — the opposite of what we want.
            adjusted_alpha = int(self.config.lora_alpha * adjusted_rank / max(self.config.lora_rank, 1))

            lora_layer = LoRALayer(
                original_layer=module,
                rank=adjusted_rank,
                alpha=adjusted_alpha,
                dropout=self.config.lora_dropout,
                weakness_scale=1.0 + weakness,
                use_rslora=self.config.use_rslora,
                init_method=self.config.init_method,
                use_dora=self.config.use_dora,
            )

            # Replace in model
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], lora_layer)

            self._lora_layers[name] = lora_layer

        avg_rank = sum(ranks) / len(ranks) if ranks else 0
        logger.info(f"Injected {len(self._lora_layers)} LoRA layers, avg rank: {avg_rank:.0f}")

    def strip_lora(self) -> None:
        """Remove all LoRA layers and restore originals.

        IMPORTANT: Does NOT unfreeze original weights. The base model weights
        must stay frozen between cycles — only LoRA parameters should be trained.
        Unfreezing here would cause the next cycle to train base weights directly.
        """
        if not self._lora_layers:
            return

        model = self.model_loader.model
        for name, original in self._original_layers.items():
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], original)
            # Keep original weights FROZEN — only LoRA should be trainable

        logger.info(f"Stripped {len(self._lora_layers)} LoRA layers")
        self._lora_layers.clear()
        self._original_layers.clear()

    def train(
        self,
        verified_samples: list[TrainingSample],
        cycle: int,
        preference_pairs: Optional[list[PreferencePair]] = None,
    ) -> TrainingMetrics:
        """Train on verified samples and/or preference pairs.

        Routes based on ``self.config.training_mode``:
          - "sft"   → SFT on ``verified_samples`` (ignores ``preference_pairs``)
          - "dpo"   → DPO on ``preference_pairs`` (ignores ``verified_samples``)
          - "mixed" → alternates SFT and DPO batches when both sources present;
                      falls back to whichever is available if one is empty.

        OOM recovery: if torch.cuda.OutOfMemoryError occurs during training,
        halves the batch size and retries once. If the retry also OOMs, raises.
        """
        mode = self.config.training_mode
        pairs = preference_pairs or []
        # Reset loss trajectory accumulator at the start of every train() call
        # so metrics reflect just this cycle.
        self._loss_trajectory = []

        # Warmup-cycle epoch cap (task #14). For early cycles, temporarily
        # override num_epochs to num_epochs_warmup. The reference is still
        # being calibrated — smaller per-cycle updates reduce the chance
        # of locking in a lucky cycle-1 eval as the permanent best.
        warmup_cycles = int(getattr(self.config, "num_epochs_warmup_cycles", 0))
        warmup_epochs = int(getattr(self.config, "num_epochs_warmup", 1))
        _orig_num_epochs = self.config.num_epochs
        _epochs_overridden = False
        if warmup_cycles > 0 and cycle <= warmup_cycles:
            effective = min(self.config.num_epochs, warmup_epochs)
            if effective != self.config.num_epochs:
                self.config.num_epochs = effective
                _epochs_overridden = True
                logger.info(
                    f"  Warmup-cycle epoch cap: cycle {cycle} <= "
                    f"{warmup_cycles}, num_epochs {_orig_num_epochs} → {effective}"
                )
        try:
            return self._train_dispatch(
                verified_samples, cycle, pairs, mode, preference_pairs,
            )
        finally:
            if _epochs_overridden:
                self.config.num_epochs = _orig_num_epochs

    def _train_dispatch(
        self,
        verified_samples: list[TrainingSample],
        cycle: int,
        pairs: list,
        mode: str,
        preference_pairs: Optional[list[PreferencePair]],
    ) -> TrainingMetrics:
        """Routes to the SFT/DPO/GRPO/mixed path. Extracted from train() so
        the warmup-cycle epoch override + sample-quality filter in train()
        don't balloon. Behavior unchanged from the pre-refactor inline body.
        """

        # Sample-quality clean-floor filter (task #14). Reads the floor from
        # the generator config (where the other sample-quality knobs live)
        # with a safe default of 0 (disabled) if the attribute is missing.
        clean_floor = 0
        try:
            gen_cfg = getattr(self.model_loader, "_gen_config_hint", None)
            if gen_cfg is None:
                # Fallback: system-level config may not be reachable from here;
                # plumbed-in attribute on self takes precedence.
                clean_floor = int(getattr(self, "_sample_quality_min_clean_floor", 0))
            else:
                clean_floor = int(getattr(gen_cfg, "sample_quality_min_clean_floor", 0))
        except Exception:
            clean_floor = 0
        if clean_floor > 0 and verified_samples:
            filtered, dropped = _filter_any_fail_when_clean_enough(
                verified_samples, clean_floor,
            )
            if dropped > 0:
                logger.info(
                    f"  Sample-quality filter: dropped {dropped}/{len(verified_samples)} "
                    f"'any_fail' samples (clean pool {len(filtered)} >= floor {clean_floor})"
                )
                verified_samples = filtered

        if mode == "grpo":
            if not verified_samples:
                logger.warning("training_mode=grpo but no samples provided — skipping training")
                return TrainingMetrics(
                    cycle=cycle, avg_loss=0, final_loss=0, steps=0,
                    samples_used=0, samples_rejected=0,
                    learning_rate=self.config.learning_rate, training_mode=mode,
                )
            return self._train_grpo(verified_samples, cycle)

        if mode == "dpo":
            if not pairs:
                logger.warning("training_mode=dpo but no preference_pairs provided — skipping training")
                return TrainingMetrics(
                    cycle=cycle, avg_loss=0, final_loss=0, steps=0,
                    samples_used=0, samples_rejected=0,
                    learning_rate=self.config.learning_rate, training_mode=mode,
                )
            return self._train_dpo(pairs, cycle)

        if mode == "mixed" and pairs and verified_samples:
            return self._train_mixed(verified_samples, pairs, cycle)

        if mode == "mixed" and pairs and not verified_samples:
            return self._train_dpo(pairs, cycle)

        if not verified_samples:
            return TrainingMetrics(cycle=cycle, avg_loss=0, final_loss=0,
                                  steps=0, samples_used=0, samples_rejected=0,
                                  learning_rate=self.config.learning_rate,
                                  training_mode=mode)

        model = self.model_loader.model
        tokenizer = self.model_loader.tokenizer

        # Use train_max_seq_length (default 1024) to cap padding — model
        # max_seq_length (4096) is for generation and far exceeds typical
        # property-verified code sample length (~500 toks). Padding to
        # 4096 wasted ~16x attention compute per step. Clamped to the
        # model cap so a misconfig never over-runs available positions.
        _train_cap = min(
            getattr(self.config, "train_max_seq_length",
                    self.model_loader.config.max_seq_length),
            self.model_loader.config.max_seq_length,
        )
        dataset = TrainingDataset(
            verified_samples,
            tokenizer,
            max_length=_train_cap,
        )
        samples_rejected = len(verified_samples) - len(dataset)
        if samples_rejected > 0:
            logger.info(f"  Skipped {samples_rejected} samples (prompt too long for sequence length)")

        batch_size = self.config.batch_size
        # Use the LR as configured. Previously applied a silent √cycle decay
        # here, which collided with the meta LR bandit's proposals — the
        # bandit would set config.learning_rate = X, then training would
        # silently apply X/√cycle, so cycle 2's bandit-set LR became
        # 0.7× its proposal. Meta decisions are no longer silently
        # overridden; users wanting per-cycle LR decay can configure it
        # through the meta controller or pass --learning-rate explicitly.
        cycle_lr = self.config.learning_rate
        return self._train_inner(dataset, model, tokenizer, cycle, batch_size,
                                 samples_rejected, verified_samples, cycle_lr,
                                 retry_on_oom=True)

    def _train_inner(
        self,
        dataset: 'TrainingDataset',
        model,
        tokenizer,
        cycle: int,
        batch_size: int,
        samples_rejected: int,
        verified_samples: list[TrainingSample],
        cycle_lr: float,
        retry_on_oom: bool = True,
    ) -> TrainingMetrics:
        """Inner training loop, separated to allow OOM retry with smaller batch."""
        pad_id = (tokenizer.pad_token_id
                  if tokenizer.pad_token_id is not None else 0)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=self.config.num_epochs > 1,  # Curriculum order for single epoch; shuffle for multi-epoch to avoid order overfitting
            drop_last=False,
            collate_fn=make_dynamic_pad_collate(pad_id),
        )

        # Only optimize LoRA parameters
        optimizer = self._build_optimizer(model, cycle_lr)
        if optimizer is None:
            logger.warning("No trainable parameters found")
            return TrainingMetrics(cycle=cycle, avg_loss=0, final_loss=0,
                                  steps=0, samples_used=len(verified_samples),
                                  samples_rejected=0, learning_rate=self.config.learning_rate)
        # Flatten optimizer param groups — used below for gradient clipping.
        # The LoRA+ refactor moved optimizer construction into _build_optimizer,
        # so the trainable param list has to be recovered from the optimizer.
        lora_params = [p for g in optimizer.param_groups for p in g["params"]]

        # Install GradientNormTracker hook on optimizer.step. Exposes this
        # cycle's gradient-health summary to meta_meta. Safe no-op when torch
        # is unavailable; always uninstalled in the finally block.
        try:
            from .stability import GradientNormTracker
            _grad_tracker = GradientNormTracker()
            _grad_tracker.begin_cycle(cycle, lora_params=lora_params)
            _grad_tracker.install_on_optimizer(optimizer)
        except Exception as _e:
            logger.debug("GradientNormTracker install failed (%s): %s",
                         type(_e).__name__, _e)
            _grad_tracker = None

        total_batches = len(dataloader) * self.config.num_epochs
        # Adaptive grad-accum: the configured grad_accum assumes larger datasets.
        # With 9 samples and num_epochs=5, grad_accum=1 yields ~25 optimizer steps
        # and reliably memorizes (cycle-3 regression). Scale grad_accum up so
        # actual steps ≤ max_steps_per_cycle. Honor the user's configured value
        # as a floor — never reduce it.
        base_accum = self.config.gradient_accumulation_steps
        effective_accum, total_steps, skip_cycle = _plan_step_budget(
            total_batches,
            base_accum,
            self.config.max_steps_per_cycle,
            self.config.min_steps_per_cycle,
        )
        if effective_accum != base_accum:
            logger.info(
                f"  Adaptive grad_accum: {base_accum} → {effective_accum}"
                f" (total_batches={total_batches}, cap={self.config.max_steps_per_cycle})"
            )
        if skip_cycle:
            logger.warning(
                f"  Skipping cycle: expected_steps={total_steps}"
                f" < min_steps_per_cycle={self.config.min_steps_per_cycle}"
                f" (total_batches={total_batches}, accum={effective_accum})"
            )
            return TrainingMetrics(
                cycle=cycle, avg_loss=0, final_loss=0, steps=0,
                samples_used=len(dataset), samples_rejected=samples_rejected,
                learning_rate=cycle_lr,
            )
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        scheduler = self._build_scheduler(optimizer, warmup_steps, total_steps)

        model.train()
        # Disable KV cache during training (incompatible with gradient computation)
        model.config.use_cache = False
        # Clear VRAM from prior phases (diagnostics: 1600+ inferences + activation
        # capture; generation: 100s of inferences). Without this, the CUDA allocator
        # may be fragmented, preventing gradient checkpointing from allocating the
        # contiguous blocks it needs for recomputation.
        torch.cuda.empty_cache()
        # Enable gradient checkpointing to save VRAM on A6000s (gated —
        # for 32B-4bit + short-seq LoRA there's ~25GB of unused VRAM and
        # GC is pure slowdown; operator can set use_gradient_checkpointing=False).
        if getattr(self.config, "use_gradient_checkpointing", True) \
                and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            # QLoRA gotcha: 4-bit frozen base + gradient checkpointing severs
            # autograd at the embedding (requires_grad=False), so gradient
            # never reaches LoRA — B stays at zero init across every step.
            # enable_input_require_grads reconnects the graph. No-op on
            # non-quantized bases.
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

        # Pre-training loss probe: if the very first batch's forward loss is
        # already below the skip threshold, the model has effectively memorized
        # this training distribution. The loss-based early-stop triggers AFTER
        # step 1, but by then the single damaging update has already been
        # applied — which is exactly the failure mode observed in cycle 4
        # (step 1 loss 0.0547, held-out went to 0.000). This probe catches that
        # case BEFORE any gradient is applied.
        skip_loss = getattr(self.config, "skip_if_initial_loss_below", 0.0)
        if skip_loss > 0.0:
            try:
                first_batch = next(iter(dataloader))
                # Resolve device here — the cached `device` local below is set
                # only after this probe runs.
                probe_device = self.model_loader.device
                probe_batch = {
                    k: v.to(probe_device) for k, v in first_batch.items()
                    if k not in ("sample_weight", "calibration_brier")
                }
                with torch.no_grad():
                    probe_out = model(**probe_batch)
                    probe_loss = float(probe_out.loss.item())
                del probe_out, probe_batch, first_batch
                if probe_loss < skip_loss:
                    logger.warning(
                        f"  Pre-training loss probe: {probe_loss:.4f} < "
                        f"skip_if_initial_loss_below={skip_loss:.3f}. "
                        f"Model has already memorized this distribution; "
                        f"skipping training to avoid further corruption."
                    )
                    # Clean up gradient checkpointing + cache before returning
                    if hasattr(model, "gradient_checkpointing_disable"):
                        model.gradient_checkpointing_disable()
                    model.config.use_cache = True
                    model.eval()
                    torch.cuda.empty_cache()
                    return TrainingMetrics(
                        cycle=cycle, avg_loss=probe_loss, final_loss=probe_loss,
                        steps=0, samples_used=len(dataset), samples_rejected=samples_rejected,
                        learning_rate=cycle_lr, lora_layers_injected=len(self._lora_layers),
                        avg_rank=(sum(l.rank for l in self._lora_layers.values())
                                  / max(1, len(self._lora_layers))),
                    )
            except StopIteration:
                pass  # empty dataloader — the main loop will handle it

        total_loss = 0.0
        batch_count = 0  # for averaging total_loss
        last_loss = 0.0
        step_count = 0
        accum_count = 0  # tracks accumulation across epoch boundaries
        device = self.model_loader.device  # cache — avoid next(params).device per batch

        try:
            for epoch in range(self.config.num_epochs):
                for batch in dataloader:
                    # Extract sample weights WITHOUT mutating the batch dict
                    # (pop would remove it permanently, breaking epoch 2+)
                    sample_weights = batch.get("sample_weight")
                    if sample_weights is not None:
                        sample_weights = sample_weights.to(device)
                    cal_brier = batch.get("calibration_brier")
                    if cal_brier is not None:
                        cal_brier = cal_brier.to(device)
                    # Strip non-model tensors so HF forward doesn't see them.
                    model_batch = {
                        k: v.to(device) for k, v in batch.items()
                        if k not in ("sample_weight", "calibration_brier")
                    }

                    # Compute per-sample weighted loss instead of scaling batch mean.
                    # HF's outputs.loss averages over all tokens in the batch, making it
                    # impossible to weight individual samples. Compute unreduced loss instead.
                    outputs = model(**model_batch)
                    if sample_weights is not None and model_batch.get("labels") is not None:
                        # Manual cross-entropy with per-sample weighting.
                        # Avoid .contiguous() on logits — it would allocate ~2GB for
                        # (batch×seq×vocab) in float32. reshape() handles non-contiguous
                        # tensors by copying only when needed (usually not for slice views).
                        logits = outputs.logits[:, :-1, :]
                        labels = model_batch["labels"][:, 1:]
                        batch_sz, seq_len = labels.shape
                        # Per-token loss: (batch, seq_len)
                        per_token = F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                            ignore_index=-100, reduction="none",
                        ).view(batch_sz, seq_len)
                        # Per-sample mean, then weight.
                        # F.cross_entropy with ignore_index=-100 already zeros ignored
                        # positions, so we only need valid_tokens for the denominator.
                        valid_tokens = (labels != -100).sum(dim=1).clamp(min=1).float()
                        per_sample_loss = per_token.sum(dim=1) / valid_tokens
                        effective_weights = sample_weights
                        if (self.config.enable_calibration_loss
                                and cal_brier is not None
                                and self.config.calibration_loss_weight > 0):
                            # Upweight samples the model's confidences missed —
                            # Brier in [0,1] (sentinel -1 means "no markers"; mask to 0).
                            # Factor = 1 + lambda * brier, so λ=0.1 gives at most +10%.
                            brier_clean = torch.where(
                                cal_brier >= 0,
                                cal_brier,
                                torch.zeros_like(cal_brier),
                            )
                            cal_factor = 1.0 + self.config.calibration_loss_weight * brier_clean
                            effective_weights = sample_weights * cal_factor
                        loss = (per_sample_loss * effective_weights).mean()
                        # Track unweighted loss for metrics — weighted loss varies with
                        # confidence distribution, making cross-cycle comparison unreliable.
                        unweighted_loss = per_sample_loss.mean().item()
                        # Free intermediate tensors immediately to reduce peak memory
                        del logits, labels, per_token, valid_tokens, per_sample_loss
                    else:
                        loss = outputs.loss
                        unweighted_loss = loss.item()

                    # Pre-backward early stop: if the forward-pass loss is
                    # already below threshold, DO NOT do backward+step.
                    # Critical: cycle 5 showed that once step 1 fires on a
                    # near-memorized distribution, the damage lands even if
                    # loss post-step is low. Checking BEFORE backward prevents
                    # that single-step corruption. Also gates when grad_accum>1:
                    # refuse to accumulate more grads that will eventually be
                    # applied.
                    #
                    # H9 (hot_spots): single-batch loss can be noisily low
                    # (e.g. a short sample with high prior-mass tokens) and
                    # fire the guard before ANY optimizer step, leaving the
                    # cycle with steps=0. Evidence: cycles 4/7/8 showed that
                    # failure mode in 8 cycles of data. Require a minimum
                    # patience of 2 × effective_accum forward passes before
                    # the guard may fire — that's one full accumulation group
                    # finalized into an optimizer step, plus a second group
                    # worth of evidence. Preserves cycle-5's protection (the
                    # memorization run would still trip it after ~1 full group)
                    # while tolerating single-batch noise.
                    min_patience_batches = 2 * max(1, effective_accum)
                    if (unweighted_loss < self.config.early_stop_loss
                            and batch_count + 1 >= min_patience_batches):
                        logger.warning(
                            f"  Early stop (pre-backward): loss {unweighted_loss:.4f}"
                            f" < early_stop_loss {self.config.early_stop_loss}"
                            f" at batch {batch_count + 1}"
                            f" (step_count={step_count}, accum={accum_count},"
                            f" patience={min_patience_batches})"
                        )
                        # If there are pending accumulated grads from earlier
                        # batches, flush them before exiting — but only when
                        # we've already fired at least one full-group step.
                        # Otherwise drop the partial grads cleanly.
                        if accum_count > 0 and step_count == 0:
                            optimizer.zero_grad()
                        raise _EarlyStop()

                    loss = loss / effective_accum
                    loss.backward()
                    accum_count += 1

                    last_loss = unweighted_loss
                    if self._collect_loss_trajectory:
                        self._loss_trajectory.append(float(unweighted_loss))
                    if accum_count % effective_accum == 0:
                        import time as _time
                        _t0 = _time.perf_counter()
                        _pre_norm = torch.nn.utils.clip_grad_norm_(
                            lora_params, self.config.max_grad_norm,
                        )
                        _clip_frac = (
                            1.0 if float(_pre_norm) > self.config.max_grad_norm
                            else 0.0
                        )
                        optimizer.step()
                        scheduler.step()
                        # Emit BEFORE zero_grad so grad-norm stats are available.
                        _lr_A = _lr_B = None
                        for g in optimizer.param_groups:
                            tag = g.get("name", "")
                            if "A" in tag and _lr_A is None:
                                _lr_A = g.get("lr")
                            elif "B" in tag and _lr_B is None:
                                _lr_B = g.get("lr")
                        if _lr_A is None and optimizer.param_groups:
                            _lr_A = optimizer.param_groups[0].get("lr")
                        _elapsed_ms = (_time.perf_counter() - _t0) * 1000.0
                        self._emit_training_step_log(
                            cycle=cycle,
                            step_idx=step_count,
                            loss_unweighted=float(unweighted_loss),
                            loss_weighted=float(loss.item()) * float(effective_accum),
                            sample_weight=None,
                            verdict_warnings=None,
                            lora_params=lora_params,
                            lr_A=_lr_A, lr_B=_lr_B,
                            clip_fraction=_clip_frac,
                            time_ms=_elapsed_ms,
                        )
                        optimizer.zero_grad()
                        step_count += 1

                    total_loss += unweighted_loss
                    batch_count += 1

            # Flush any remaining accumulated gradients. last_loss already holds
            # the most recent batch's unweighted loss from the loop above.
            if accum_count % effective_accum != 0:
                import time as _time
                _t0 = _time.perf_counter()
                _pre_norm = torch.nn.utils.clip_grad_norm_(
                    lora_params, self.config.max_grad_norm,
                )
                _clip_frac = (
                    1.0 if float(_pre_norm) > self.config.max_grad_norm
                    else 0.0
                )
                optimizer.step()
                scheduler.step()
                _lr_A = optimizer.param_groups[0].get("lr") if optimizer.param_groups else None
                _elapsed_ms = (_time.perf_counter() - _t0) * 1000.0
                self._emit_training_step_log(
                    cycle=cycle,
                    step_idx=step_count,
                    loss_unweighted=float(last_loss),
                    loss_weighted=float(last_loss),
                    sample_weight=None,
                    verdict_warnings=None,
                    lora_params=lora_params,
                    lr_A=_lr_A, lr_B=None,
                    clip_fraction=_clip_frac,
                    time_ms=_elapsed_ms,
                )
                optimizer.zero_grad()
                step_count += 1
        except _EarlyStop:
            # Loss-based early stop. Any partially-accumulated gradients are
            # deliberately dropped — applying them would undercut the very
            # regularization we are enforcing.
            pass
        except torch.cuda.OutOfMemoryError:
            # OOM during training — clean up, then retry with halved batch if allowed.
            logger.warning(f"OOM during training (batch_size={batch_size})")
            del optimizer, scheduler
            if hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
            model.config.use_cache = True
            model.eval()
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            if retry_on_oom and batch_size > 1:
                new_bs = max(1, batch_size // 2)
                logger.warning(f"  Retrying with batch_size={new_bs}")
                return self._train_inner(
                    dataset, model, tokenizer, cycle, new_bs,
                    samples_rejected, verified_samples, cycle_lr,
                    retry_on_oom=False,
                )
            raise
        finally:
            # Cleanup: free optimizer/scheduler VRAM, restore inference mode.
            # In a finally block so model state is restored even if training OOMs —
            # without this, a caught exception leaves the model in train() mode with
            # use_cache=False, wasting VRAM on the next diagnostic phase.
            if _grad_tracker is not None:
                try:
                    _grad_tracker.uninstall_on_optimizer(optimizer)
                    _grad_summary = _grad_tracker.end_cycle(lora_params=lora_params)
                    self._last_grad_summary = _grad_summary
                except Exception as _e:
                    logger.debug("GradientNormTracker teardown failed: %s", _e)
            try:
                del optimizer, scheduler
            except UnboundLocalError:
                pass  # already deleted in OOM handler
            if hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
            model.config.use_cache = True  # re-enable KV cache for inference
            model.eval()
            torch.cuda.empty_cache()

        avg_loss = total_loss / max(batch_count, 1)
        ranks = [l.rank for l in self._lora_layers.values()]
        ece, brier, cal_n = _compute_calibration(verified_samples)
        # Overfit detector: low final loss on few samples indicates memorization,
        # which matches the cycle-3 regression pattern (loss 0.044 on 9 samples).
        if last_loss < 0.1 and len(dataset) < 20 and step_count > 0:
            logger.warning(
                f"  Overfit suspected: final_loss={last_loss:.4f} < 0.1"
                f" on samples_used={len(dataset)} < 20."
                f" Training likely memorized; consider revert."
            )
        return TrainingMetrics(
            cycle=cycle,
            avg_loss=avg_loss,
            final_loss=last_loss,
            steps=step_count,
            samples_used=len(dataset),
            samples_rejected=samples_rejected,
            learning_rate=cycle_lr,
            lora_layers_injected=len(self._lora_layers),
            avg_rank=sum(ranks) / len(ranks) if ranks else 0,
            calibration_ece=ece,
            calibration_brier=brier,
            calibration_samples=cal_n,
            loss_trajectory=list(self._loss_trajectory),
        )

    # ------------------------------------------------------------------
    # DPO: preference-pair training with LoRA-zeroed reference pass
    # ------------------------------------------------------------------

    @staticmethod
    def _sum_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Sum of token log-probs for each sequence, masking labels == -100.

        Returns (batch,) tensor. Uses the standard next-token-prediction shift:
        logits[:, :-1] predict labels[:, 1:].
        """
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        mask = (shift_labels != -100)
        # Replace ignore-index with 0 so gather doesn't blow up; masked out below.
        safe_labels = shift_labels.masked_fill(~mask, 0)
        logp = F.log_softmax(shift_logits.float(), dim=-1)
        token_logp = logp.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        token_logp = token_logp * mask.float()
        return token_logp.sum(dim=-1)

    def _forward_logprobs(self, model, input_ids, attention_mask, labels,
                          *, use_reference: bool) -> torch.Tensor:
        """Forward pass returning per-sequence summed log-probs.

        When ``use_reference`` is True, temporarily zeroes every LoRA A matrix so
        the LoRA contribution collapses (A=0 → B@A=0), making the model numerically
        equivalent to the frozen base — our reference policy. This avoids loading a
        second copy of the base model (huge VRAM win on A6000) at the cost of one
        extra forward pass per batch.

        The reference pass runs under ``torch.no_grad()`` AND with LoRA params
        swapped to zero buffers (not in-place mutated) so autograd state of the
        training pass is untouched.
        """
        if use_reference and self._lora_layers:
            saved_A = {}
            for name, layer in self._lora_layers.items():
                saved_A[name] = layer.lora_A.data
                layer.lora_A.data = torch.zeros_like(layer.lora_A.data)
            try:
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logp = self._sum_logprobs(outputs.logits, labels)
            finally:
                for name, layer in self._lora_layers.items():
                    layer.lora_A.data = saved_A[name]
            return logp.detach()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        return self._sum_logprobs(outputs.logits, labels)

    def _train_dpo(self, pairs: list[PreferencePair], cycle: int) -> TrainingMetrics:
        """DPO training loop. Doubled forward-pass memory (policy+reference) is
        mitigated by: (1) reference shares the same weights (LoRA zeroed), (2)
        gradient checkpointing on policy, (3) no grads on reference.

        A6000 memory budget (48GB): with 7B model in bf16 (~14GB) + grad
        checkpointing + LoRA params, batch_size=1 pairs (=2 forward seqs) at
        4096 seq_len fits with ~8GB headroom. batch_size=2 pairs (=4 forward
        seqs) risks OOM — the OOM-retry path halves and retries.
        """
        model = self.model_loader.model
        tokenizer = self.model_loader.tokenizer

        # See SFT path: cap padding at train_max_seq_length to avoid paying
        # 4096-token attention for ~500-token samples. Clamped to model cap.
        _train_cap = min(
            getattr(self.config, "train_max_seq_length",
                    self.model_loader.config.max_seq_length),
            self.model_loader.config.max_seq_length,
        )
        dataset = PreferenceDataset(
            pairs, tokenizer, max_length=_train_cap,
        )
        rejected_count = len(pairs) - len(dataset)
        if rejected_count > 0:
            logger.info(f"  Skipped {rejected_count} preference pairs (too long to fit)")
        if len(dataset) == 0:
            return TrainingMetrics(
                cycle=cycle, avg_loss=0, final_loss=0, steps=0,
                samples_used=0, samples_rejected=rejected_count,
                learning_rate=self.config.learning_rate,
                training_mode="dpo",
            )

        # DPO doubles activation memory vs SFT — err on smaller batch.
        batch_size = max(1, self.config.batch_size // 2)
        # Use configured LR as-is. The SFT path removed silent √cycle decay
        # because it silently overrode meta-LR-bandit proposals — DPO/GRPO had
        # the same bug; honor the configured LR so meta decisions stick.
        cycle_lr = self.config.learning_rate
        return self._train_dpo_inner(
            dataset, model, cycle, batch_size, rejected_count, cycle_lr,
            retry_on_oom=True,
        )

    def _train_dpo_inner(
        self, dataset, model, cycle, batch_size, rejected_count, cycle_lr,
        retry_on_oom: bool = True,
    ) -> TrainingMetrics:
        tokenizer = self.model_loader.tokenizer
        pad_id = (tokenizer.pad_token_id
                  if tokenizer.pad_token_id is not None else 0)
        dataloader = DataLoader(
            dataset, batch_size=batch_size,
            shuffle=self.config.num_epochs > 1, drop_last=False,
            collate_fn=make_dynamic_pad_collate_dpo(pad_id),
        )
        optimizer = self._build_optimizer(model, cycle_lr)
        if optimizer is None:
            logger.warning("No trainable parameters for DPO")
            return TrainingMetrics(
                cycle=cycle, avg_loss=0, final_loss=0, steps=0,
                samples_used=len(dataset), samples_rejected=0,
                learning_rate=cycle_lr, training_mode="dpo",
            )
        lora_params = [p for g in optimizer.param_groups for p in g["params"]]
        total_batches = len(dataloader) * self.config.num_epochs
        # Adaptive step-budget — same contract as SFT: scale grad_accum UP to
        # keep total_steps <= max_steps_per_cycle, and skip the cycle when even
        # with accum=total_batches we fall below min_steps_per_cycle.
        base_accum = self.config.gradient_accumulation_steps
        effective_accum, total_steps, skip_cycle = _plan_step_budget(
            total_batches,
            base_accum,
            self.config.max_steps_per_cycle,
            self.config.min_steps_per_cycle,
        )
        if effective_accum != base_accum:
            logger.info(
                f"  DPO adaptive grad_accum: {base_accum} → {effective_accum}"
                f" (total_batches={total_batches}, cap={self.config.max_steps_per_cycle})"
            )
        if skip_cycle:
            logger.warning(
                f"  Skipping DPO cycle: expected_steps={total_steps}"
                f" < min_steps_per_cycle={self.config.min_steps_per_cycle}"
            )
            try:
                del optimizer
            except UnboundLocalError:
                pass
            return TrainingMetrics(
                cycle=cycle, avg_loss=0, final_loss=0, steps=0,
                samples_used=len(dataset), samples_rejected=rejected_count,
                learning_rate=cycle_lr, training_mode="dpo",
            )
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        scheduler = self._build_scheduler(optimizer, warmup_steps, total_steps)

        model.train()
        model.config.use_cache = False
        torch.cuda.empty_cache()
        if getattr(self.config, "use_gradient_checkpointing", True) \
                and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            # QLoRA gotcha: 4-bit frozen base + gradient checkpointing severs
            # autograd at the embedding (requires_grad=False), so gradient
            # never reaches LoRA — B stays at zero init across every step.
            # enable_input_require_grads reconnects the graph. No-op on
            # non-quantized bases.
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

        beta = self.config.dpo_beta
        total_loss = 0.0
        last_loss = 0.0
        batch_count = 0
        step_count = 0
        accum_count = 0
        reward_margin_sum = 0.0
        reward_margin_count = 0
        device = self.model_loader.device

        try:
            for epoch in range(self.config.num_epochs):
                for batch in dataloader:
                    weights = batch["sample_weight"].to(device)
                    chosen_ids = batch["chosen_input_ids"].to(device)
                    chosen_mask = batch["chosen_attention_mask"].to(device)
                    chosen_labels = batch["chosen_labels"].to(device)
                    rejected_ids = batch["rejected_input_ids"].to(device)
                    rejected_mask = batch["rejected_attention_mask"].to(device)
                    rejected_labels = batch["rejected_labels"].to(device)

                    # Policy logps (LoRA active, grads on).
                    pi_chosen = self._forward_logprobs(
                        model, chosen_ids, chosen_mask, chosen_labels,
                        use_reference=False,
                    )
                    pi_rejected = self._forward_logprobs(
                        model, rejected_ids, rejected_mask, rejected_labels,
                        use_reference=False,
                    )
                    # Reference logps (LoRA zeroed, no grads).
                    ref_chosen = self._forward_logprobs(
                        model, chosen_ids, chosen_mask, chosen_labels,
                        use_reference=True,
                    )
                    ref_rejected = self._forward_logprobs(
                        model, rejected_ids, rejected_mask, rejected_labels,
                        use_reference=True,
                    )

                    logits = beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected))
                    # Per-sample DPO loss, weighted, then averaged.
                    per_sample_loss = -F.logsigmoid(logits)
                    loss = (per_sample_loss * weights).mean()
                    unweighted_loss = per_sample_loss.mean().item()

                    margin = (pi_chosen - pi_rejected).detach().mean().item()
                    reward_margin_sum += margin
                    reward_margin_count += 1

                    loss = loss / effective_accum
                    loss.backward()
                    accum_count += 1
                    last_loss = unweighted_loss

                    if accum_count % effective_accum == 0:
                        torch.nn.utils.clip_grad_norm_(lora_params, self.config.max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        step_count += 1

                    total_loss += unweighted_loss
                    batch_count += 1

            if accum_count % effective_accum != 0:
                torch.nn.utils.clip_grad_norm_(lora_params, self.config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step_count += 1
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"OOM during DPO training (batch_size={batch_size})")
            del optimizer, scheduler
            if hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
            model.config.use_cache = True
            model.eval()
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            if retry_on_oom and batch_size > 1:
                new_bs = max(1, batch_size // 2)
                logger.warning(f"  Retrying DPO with batch_size={new_bs}")
                return self._train_dpo_inner(
                    dataset, model, cycle, new_bs, rejected_count, cycle_lr,
                    retry_on_oom=False,
                )
            raise
        finally:
            try:
                del optimizer, scheduler
            except UnboundLocalError:
                pass
            if hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
            model.config.use_cache = True
            model.eval()
            torch.cuda.empty_cache()

        avg_loss = total_loss / max(batch_count, 1)
        avg_margin = reward_margin_sum / max(reward_margin_count, 1)
        ranks = [l.rank for l in self._lora_layers.values()]
        return TrainingMetrics(
            cycle=cycle, avg_loss=avg_loss, final_loss=last_loss,
            steps=step_count, samples_used=len(dataset),
            samples_rejected=rejected_count, learning_rate=cycle_lr,
            lora_layers_injected=len(self._lora_layers),
            avg_rank=sum(ranks) / len(ranks) if ranks else 0,
            training_mode="dpo", dpo_pairs_used=len(dataset),
            avg_reward_margin=avg_margin,
        )

    def _train_mixed(
        self, verified_samples: list[TrainingSample],
        pairs: list[PreferencePair], cycle: int,
    ) -> TrainingMetrics:
        """Mixed mode: run SFT then DPO sequentially. Simpler and more stable
        than per-batch alternation, which would require a single fused optimizer
        and two dataloaders in lockstep — messy with heterogeneous batch shapes.
        """
        sft_metrics = self._train_sft_entry(verified_samples, cycle)
        dpo_metrics = self._train_dpo(pairs, cycle)
        return TrainingMetrics(
            cycle=cycle,
            avg_loss=(sft_metrics.avg_loss + dpo_metrics.avg_loss) / 2.0,
            final_loss=dpo_metrics.final_loss,
            steps=sft_metrics.steps + dpo_metrics.steps,
            samples_used=sft_metrics.samples_used + dpo_metrics.samples_used,
            samples_rejected=sft_metrics.samples_rejected + dpo_metrics.samples_rejected,
            learning_rate=dpo_metrics.learning_rate,
            lora_layers_injected=len(self._lora_layers),
            avg_rank=dpo_metrics.avg_rank or sft_metrics.avg_rank,
            training_mode="mixed",
            dpo_pairs_used=dpo_metrics.dpo_pairs_used,
            avg_reward_margin=dpo_metrics.avg_reward_margin,
        )

    def _train_sft_entry(
        self, verified_samples: list[TrainingSample], cycle: int,
    ) -> TrainingMetrics:
        """Direct entry to the SFT path — used by mixed mode to bypass routing."""
        if not verified_samples:
            return TrainingMetrics(
                cycle=cycle, avg_loss=0, final_loss=0, steps=0,
                samples_used=0, samples_rejected=0,
                learning_rate=self.config.learning_rate, training_mode="sft",
            )
        model = self.model_loader.model
        tokenizer = self.model_loader.tokenizer
        # Match the main SFT path: cap padding at train_max_seq_length so mixed
        # mode doesn't pay 4096-token attention for ~500-token samples.
        _train_cap = min(
            getattr(self.config, "train_max_seq_length",
                    self.model_loader.config.max_seq_length),
            self.model_loader.config.max_seq_length,
        )
        dataset = TrainingDataset(
            verified_samples, tokenizer, max_length=_train_cap,
        )
        samples_rejected = len(verified_samples) - len(dataset)
        batch_size = self.config.batch_size
        # Configured LR honored as-is — silent √cycle decay was removed for the
        # same reason as the main SFT path (meta-LR-bandit must not be silently
        # overridden).
        cycle_lr = self.config.learning_rate
        return self._train_inner(
            dataset, model, tokenizer, cycle, batch_size,
            samples_rejected, verified_samples, cycle_lr, retry_on_oom=True,
        )

    # ------------------------------------------------------------------
    # GRPO: Group Relative Policy Optimization (DeepSeek 2024)
    # ------------------------------------------------------------------
    #
    # For each prompt, sample G completions from the current policy π, score
    # each with a reward function, normalize rewards within the group to get
    # advantages, and take a PPO-clipped policy-gradient step using π_old
    # (the policy at rollout time) as the importance-sampling base.
    #
    # Memory footprint on a 48GB A6000 with an 8B bf16 model:
    #   base weights       ~16 GB
    #   grad checkpointing ~ 2 GB activations per fwd
    #   LoRA params/grads  ~ 0.5 GB
    #   rollout cache      ~ G * max_new_tokens * 8B  (CPU-side, negligible VRAM)
    #   per-step forward   1 policy pass + 1 reference pass (no-grad, LoRA zeroed)
    # Recommended: grpo_group_size=8, batch_size=1 prompt per step, grad-accum=16,
    # max_new_tokens<=512. OOM-retry halves per-step prompt count (down to 1) then
    # halves G (down to 2) before giving up.

    def _default_reward(self, prompt: str, completion: str, sample: "TrainingSample") -> float:
        """Default GRPO reward: canonical-answer grade if available, else 0.

        Grade maps to {0.0, 1.0}. If the sample carries no canonical answer (no
        expected_answer / no ground_truth_check_type), we return 0 — GRPO then
        relies entirely on within-group normalization, which with all-zero rewards
        collapses to a zero advantage (no update). Callers who want dense reward
        without canonical answers must inject a reward_fn via set_reward_fn.
        """
        expected = getattr(sample, "expected_answer", "") or ""
        if not expected:
            return 0.0
        # Build an ephemeral "sample" that mirrors the rollout, so the verifier's
        # grader dispatches on the real completion rather than the cached response.
        try:
            from ..verifier.verifier import grade_against_canonical as _grade
            from ..generator.data_generator import TrainingSample as _TS
            probe = _TS(
                prompt=sample.prompt,
                response=completion,
                domain=sample.domain,
                ground_truth_check_type=getattr(sample, "ground_truth_check_type", "") or "",
            )
            ok, _ = _grade(probe, expected, sample.domain or "")
            return 1.0 if ok else 0.0
        except Exception as e:
            logger.debug(f"default reward grading failed: {e}")
            return 0.0

    def _score_rollout(self, prompt: str, completion: str, sample: "TrainingSample") -> float:
        fn = self._reward_fn if self._reward_fn is not None else self._default_reward
        try:
            r = float(fn(prompt, completion, sample))
        except Exception as e:
            logger.warning(f"reward_fn raised ({type(e).__name__}: {e}); returning 0")
            return 0.0
        if not math.isfinite(r):
            return 0.0
        return r

    @torch.no_grad()
    def _sample_rollouts(
        self, prompts: list[str], samples: list["TrainingSample"], G: int,
    ) -> list[list[dict]]:
        """Generate G completions per prompt using model.generate.

        Returns a list of length len(prompts), each a list of G dicts with:
          {"completion_text", "input_ids", "completion_ids", "reward"}

        Runs in eval mode with gradient checkpointing disabled (both incompatible
        with generate()); the caller is responsible for restoring train state.
        """
        model = self.model_loader.model
        tokenizer = self.model_loader.tokenizer
        device = self.model_loader.device

        was_training = model.training
        had_cache = model.config.use_cache
        gc_was_on = getattr(model, "is_gradient_checkpointing", False) or False
        if gc_was_on and hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        model.eval()
        model.config.use_cache = True

        max_new = self.config.grpo_max_new_tokens
        temperature = self.config.grpo_rollout_temperature
        top_p = self.config.grpo_rollout_top_p
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

        out: list[list[dict]] = []
        try:
            for prompt, sample in zip(prompts, samples):
                enc = tokenizer(
                    prompt + "\n\n", add_special_tokens=True, return_tensors="pt",
                ).to(device)
                input_ids = enc["input_ids"]
                attn = enc["attention_mask"]
                prompt_len = input_ids.shape[1]
                # Replicate G times in the batch dim for one generate() call.
                batch_input = input_ids.expand(G, -1).contiguous()
                batch_attn = attn.expand(G, -1).contiguous()
                gen = model.generate(
                    input_ids=batch_input,
                    attention_mask=batch_attn,
                    max_new_tokens=max_new,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                )
                group: list[dict] = []
                for g in range(G):
                    full = gen[g]
                    completion_ids = full[prompt_len:]
                    # Trim trailing pads
                    if pad_id is not None:
                        nonpad = (completion_ids != pad_id).nonzero(as_tuple=False)
                        if nonpad.numel() > 0:
                            completion_ids = completion_ids[: int(nonpad[-1].item()) + 1]
                        else:
                            completion_ids = completion_ids[:0]
                    completion_text = tokenizer.decode(
                        completion_ids, skip_special_tokens=True,
                    )
                    reward = self._score_rollout(prompt, completion_text, sample)
                    group.append({
                        "completion_text": completion_text,
                        "input_ids": input_ids[0].detach().cpu(),
                        "completion_ids": completion_ids.detach().cpu(),
                        "reward": reward,
                    })
                out.append(group)
        finally:
            if gc_was_on and hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            model.config.use_cache = had_cache
            if was_training:
                model.train()
        return out

    def _grpo_build_batch(self, rollouts_for_prompt: list[dict]) -> Optional[dict]:
        """Pack G rollouts for one prompt into a padded batch tensor.

        Returns dict with input_ids/attention_mask/labels/advantages, all on device,
        or None if the group has degenerate rewards (std ≈ 0 → zero advantages).
        """
        device = self.model_loader.device
        tokenizer = self.model_loader.tokenizer
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        max_len_cap = self.model_loader.config.max_seq_length

        rewards = torch.tensor([r["reward"] for r in rollouts_for_prompt], dtype=torch.float32)
        std = float(rewards.std(unbiased=False).item())
        mean = float(rewards.mean().item())
        # If all rewards equal, advantage is zero and this group produces no signal.
        if std < 1e-6:
            return None
        advantages = (rewards - mean) / (std + 1e-6)

        seqs = []
        labels_list = []
        attns = []
        for r in rollouts_for_prompt:
            prompt_ids = r["input_ids"]
            comp_ids = r["completion_ids"]
            prompt_len = prompt_ids.shape[0]
            combined = torch.cat([prompt_ids, comp_ids])
            if combined.shape[0] > max_len_cap:
                combined = combined[:max_len_cap]
            labels = combined.clone()
            labels[:prompt_len] = -100
            seqs.append(combined)
            labels_list.append(labels)
            attns.append(torch.ones_like(combined))

        max_len = max(s.shape[0] for s in seqs)
        def _pad(t, val):
            if t.shape[0] == max_len:
                return t
            pad = torch.full((max_len - t.shape[0],), val, dtype=t.dtype)
            return torch.cat([t, pad])

        input_ids = torch.stack([_pad(s, pad_id) for s in seqs]).to(device)
        labels = torch.stack([_pad(l, -100) for l in labels_list]).to(device)
        attention_mask = torch.stack([_pad(a, 0) for a in attns]).to(device)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "advantages": advantages.to(device),
            "reward_mean": mean,
            "reward_std": std,
        }

    @staticmethod
    def _per_token_logprobs(logits: torch.Tensor, labels: torch.Tensor):
        """Returns (per_token_logp, mask): both (batch, seq-1).

        per_token_logp has zeros at masked positions; mask is bool.
        """
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        mask = (shift_labels != -100)
        safe = shift_labels.masked_fill(~mask, 0)
        logp = F.log_softmax(shift_logits.float(), dim=-1)
        tok_lp = logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
        tok_lp = tok_lp * mask.float()
        return tok_lp, mask

    def _train_grpo(
        self, verified_samples: list["TrainingSample"], cycle: int,
    ) -> TrainingMetrics:
        model = self.model_loader.model
        if not any(p.requires_grad for p in model.parameters()):
            logger.warning("No trainable parameters for GRPO (inject_lora first)")
            return TrainingMetrics(
                cycle=cycle, avg_loss=0, final_loss=0, steps=0,
                samples_used=0, samples_rejected=0,
                learning_rate=self.config.learning_rate, training_mode="grpo",
            )

        G = self.config.grpo_group_size
        clip_eps = self.config.grpo_clip_eps
        refresh = self.config.grpo_rollout_refresh_steps
        # Use configured LR — see _train_dpo note on removed √cycle decay.
        cycle_lr = self.config.learning_rate

        # Build prompt list from verified samples. We reuse samples as the prompt
        # source and preserve them for canonical-grading during reward scoring.
        prompts = [s.prompt for s in verified_samples]

        optimizer = self._build_optimizer(model, cycle_lr)
        if optimizer is None:
            logger.warning("No trainable parameters for GRPO (inject_lora first)")
            return TrainingMetrics(
                cycle=cycle, avg_loss=0, final_loss=0, steps=0,
                samples_used=0, samples_rejected=0,
                learning_rate=cycle_lr, training_mode="grpo",
            )
        lora_params = [p for g in optimizer.param_groups for p in g["params"]]
        # Heuristic step count: one optimizer step per (grad_accum) prompts, per epoch.
        accum = self.config.gradient_accumulation_steps
        prompts_per_epoch = len(prompts)
        total_steps = max(
            1, (prompts_per_epoch * self.config.num_epochs) // max(accum, 1),
        )
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        scheduler = self._build_scheduler(optimizer, warmup_steps, total_steps)

        total_loss = 0.0
        last_loss = 0.0
        step_count = 0
        accum_count = 0
        batch_count = 0
        reward_sum = 0.0
        reward_count = 0
        group_std_sum = 0.0
        group_std_count = 0
        rollout_refreshes = 0

        # Rollout cache: prompt_idx -> {"rollouts": list[dict], "pi_old_logp": list[Tensor]}
        # pi_old_logp is the *detached* sum-log-prob of each completion under the
        # policy at rollout time — used as the PPO importance-sampling base.
        cache: dict[int, dict] = {}

        def _refresh_rollouts(prompt_idx: int):
            nonlocal rollout_refreshes
            grp = self._sample_rollouts(
                [prompts[prompt_idx]], [verified_samples[prompt_idx]], G,
            )[0]
            # Compute π_old log-probs right now, under current (but about-to-be-detached) policy.
            batch = self._grpo_build_batch(grp)
            pi_old = None
            if batch is not None:
                model.eval()
                with torch.no_grad():
                    out = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                    tok_lp, mask = self._per_token_logprobs(out.logits, batch["labels"])
                    # Sum per sequence. Keep per-token for ratio later.
                    pi_old = tok_lp.detach()
                    pi_old_mask = mask.detach()
                model.train()
            cache[prompt_idx] = {
                "rollouts": grp,
                "batch": batch,
                "pi_old_tok_lp": pi_old,
                "pi_old_mask": pi_old_mask if batch is not None else None,
            }
            rollout_refreshes += 1

        model.train()
        model.config.use_cache = False
        torch.cuda.empty_cache()
        if getattr(self.config, "use_gradient_checkpointing", True) \
                and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            # QLoRA gotcha: 4-bit frozen base + gradient checkpointing severs
            # autograd at the embedding (requires_grad=False), so gradient
            # never reaches LoRA — B stays at zero init across every step.
            # enable_input_require_grads reconnects the graph. No-op on
            # non-quantized bases.
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

        try:
            global_step = 0
            for epoch in range(self.config.num_epochs):
                order = list(range(len(prompts)))
                if self.config.num_epochs > 1:
                    import random as _r
                    _r.shuffle(order)
                for pidx in order:
                    # (Re)sample rollouts if cache stale. "Stale" = never sampled,
                    # or the last refresh was more than `refresh` optimizer steps ago.
                    entry = cache.get(pidx)
                    if entry is None or (global_step > 0 and global_step % refresh == 0 and entry.get("step_at_rollout", -1) != global_step):
                        # Refresh must happen outside the train() forward path — done above
                        # under torch.no_grad() in _refresh_rollouts.
                        _refresh_rollouts(pidx)
                        cache[pidx]["step_at_rollout"] = global_step
                        entry = cache[pidx]

                    batch = entry["batch"]
                    if batch is None:
                        # Degenerate group (all same reward) — skip; advantages zero.
                        continue

                    # Track reward stats for metrics.
                    reward_sum += batch["reward_mean"] * G
                    reward_count += G
                    group_std_sum += batch["reward_std"]
                    group_std_count += 1

                    # Policy forward WITH grads (LoRA active).
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                    tok_lp, mask = self._per_token_logprobs(outputs.logits, batch["labels"])
                    pi_old_tok = entry["pi_old_tok_lp"]
                    pi_old_mask = entry["pi_old_mask"]
                    # Shapes may differ if refresh happened with different max_len
                    # for this group vs current forward — align to min length.
                    min_len = min(tok_lp.shape[1], pi_old_tok.shape[1])
                    tok_lp_c = tok_lp[:, :min_len]
                    pi_old_c = pi_old_tok[:, :min_len]
                    mask_c = mask[:, :min_len].float()

                    # PPO ratio per token; clipped surrogate per token, then
                    # masked-average per sequence, then advantage-weighted mean.
                    ratio = torch.exp(tok_lp_c - pi_old_c)
                    adv = batch["advantages"].unsqueeze(-1)  # (G, 1)
                    unclipped = ratio * adv
                    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
                    per_tok_loss = -torch.min(unclipped, clipped)
                    # Masked mean per sequence, then mean over group.
                    seq_tok_count = mask_c.sum(dim=1).clamp(min=1.0)
                    per_seq_loss = (per_tok_loss * mask_c).sum(dim=1) / seq_tok_count
                    loss = per_seq_loss.mean()

                    unweighted_loss = float(loss.detach().item())
                    loss = loss / accum
                    loss.backward()
                    accum_count += 1
                    last_loss = unweighted_loss
                    total_loss += unweighted_loss
                    batch_count += 1

                    del outputs, tok_lp, ratio, unclipped, clipped, per_tok_loss

                    if accum_count % accum == 0:
                        torch.nn.utils.clip_grad_norm_(lora_params, self.config.max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        step_count += 1
                        global_step += 1

            if accum_count % accum != 0:
                torch.nn.utils.clip_grad_norm_(lora_params, self.config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step_count += 1
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"OOM during GRPO (G={G})")
            import gc
            try:
                del optimizer, scheduler
            except UnboundLocalError:
                pass
            if hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
            model.config.use_cache = True
            model.eval()
            gc.collect()
            torch.cuda.empty_cache()
            # Retry with halved G (min 2). We don't recurse more than once.
            if G > 2:
                new_G = max(2, G // 2)
                logger.warning(f"  Retrying GRPO with grpo_group_size={new_G}")
                old_G = self.config.grpo_group_size
                self.config.grpo_group_size = new_G
                try:
                    return self._train_grpo(verified_samples, cycle)
                finally:
                    self.config.grpo_group_size = old_G
            raise
        finally:
            try:
                del optimizer, scheduler
            except UnboundLocalError:
                pass
            if hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
            model.config.use_cache = True
            model.eval()
            torch.cuda.empty_cache()

        avg_loss = total_loss / max(batch_count, 1)
        avg_reward = reward_sum / max(reward_count, 1)
        avg_std = group_std_sum / max(group_std_count, 1)
        ranks = [l.rank for l in self._lora_layers.values()]
        return TrainingMetrics(
            cycle=cycle,
            avg_loss=avg_loss,
            final_loss=last_loss,
            steps=step_count,
            samples_used=len(verified_samples),
            samples_rejected=0,
            learning_rate=cycle_lr,
            lora_layers_injected=len(self._lora_layers),
            avg_rank=sum(ranks) / len(ranks) if ranks else 0,
            training_mode="grpo",
            grpo_prompts_used=len(prompts),
            grpo_group_size=G,
            avg_reward=avg_reward,
            avg_reward_std=avg_std,
            rollout_refreshes=rollout_refreshes,
        )

    def _build_optimizer(self, model, cycle_lr: float):
        """Build AdamW over trainable params, splitting LoRA+ groups if enabled.

        LoRA+ (Hayou et al. 2024): B starts at zero, A near identity — B must
        train faster. Two param groups with lr_B = lr_A * lora_plus_ratio.
        If DoRA is active, the ``magnitude`` parameter joins the A (slow) group.
        Any other trainable params fall into the A group as a safe default.
        """
        a_params: list[nn.Parameter] = []
        b_params: list[nn.Parameter] = []
        other: list[nn.Parameter] = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if ".lora_A" in name:
                a_params.append(p)
            elif ".lora_B" in name:
                b_params.append(p)
            elif name.endswith(".magnitude") or ".magnitude" in name:
                # DoRA magnitude — canonical LoRA+ choice: slow (A) group.
                a_params.append(p)
            else:
                other.append(p)

        if not (a_params or b_params or other):
            return None

        wd = self.config.weight_decay
        if getattr(self.config, "use_lora_plus", False) and a_params and b_params:
            ratio = float(self.config.lora_plus_ratio)
            groups = [
                {"params": a_params + other, "lr": cycle_lr},
                {"params": b_params, "lr": cycle_lr * ratio},
            ]
            return torch.optim.AdamW(groups, weight_decay=wd)

        all_params = a_params + b_params + other
        return torch.optim.AdamW(all_params, lr=cycle_lr, weight_decay=wd)

    def _build_scheduler(self, optimizer, warmup_steps: int, total_steps: int):
        """Cosine schedule with warmup.

        For very small datasets (total_steps <= 2), cosine decay is
        counterproductive — the LR drops to near-zero before meaningful
        training happens. Fall back to constant LR in those cases.
        """
        from torch.optim.lr_scheduler import LambdaLR

        if total_steps <= 2:
            # Too few steps for cosine to make sense — use constant LR
            return LambdaLR(optimizer, lambda _: 1.0)

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return current_step / max(warmup_steps, 1)
            progress = (current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda)

    def save_lora_weights(self, path: Path, cycle: int) -> Path | None:
        """Save only the LoRA weights.

        Writes the native `lora_weights.pt` (used by `load_lora_weights` to
        resume training in-process) AND a PEFT-compatible adapter directory
        (`adapter_model.safetensors` + `adapter_config.json`) so vLLM can
        load this adapter at inference time via LoRARequest.

        Returns the PEFT adapter directory path on success, else None.
        """
        if not self._lora_layers:
            return None
        save_path = path / f"lora_cycle_{cycle}"
        save_path.mkdir(parents=True, exist_ok=True)
        # Check disk space before writing — 100MB headroom for LoRA weights.
        try:
            free = shutil.disk_usage(save_path).free
            if free < 100 * 1024 * 1024:
                logger.error(f"Disk nearly full ({free // 1024 // 1024}MB free) — skipping LoRA weight save")
                return None
        except OSError as e:
            logger.warning(f"Could not check disk space: {e}")
        state_dict = {}
        for name, layer in self._lora_layers.items():
            # Save in bfloat16 to halve disk usage — LoRA params are trained in
            # float32 for precision, but the precision loss from bf16 storage is
            # negligible for checkpoint/resume purposes (merge reloads exactly).
            state_dict[f"{name}.lora_A"] = layer.lora_A.data.cpu().to(torch.bfloat16)
            state_dict[f"{name}.lora_B"] = layer.lora_B.data.cpu().to(torch.bfloat16)
            state_dict[f"{name}.rank"] = torch.tensor(layer.rank)
            state_dict[f"{name}.weakness_scale"] = torch.tensor(layer.weakness_scale)
        torch.save(state_dict, save_path / "lora_weights.pt")

        # Also emit PEFT-format adapter so vLLM --enable-lora can load it.
        try:
            return self._save_peft_adapter(save_path)
        except Exception as e:
            logger.warning(f"PEFT adapter export failed ({type(e).__name__}: {e}); "
                           f"native lora_weights.pt still saved")
            return None

    def _save_peft_adapter(self, save_path: Path) -> Path:
        """Write PEFT-compatible adapter_model.safetensors + adapter_config.json.

        PEFT key convention: `base_model.model.<module_path>.lora_A.weight`
        and `.lora_B.weight`. vLLM 0.19 loads this via LoRARequest.

        NOTE: weakness_scale, rsLoRA, DoRA, per-layer weakness-adaptive rank,
        and PiSSA residual-subtraction are native-LoRA-only features that
        PEFT doesn't model. The exported adapter preserves the raw A/B
        matrices — enough for vLLM to apply `scaling·(B@A)` — but the
        base weights at inference must match the base weights used during
        training. DoRA/PiSSA bases differ from the original (residual or
        magnitude-adjusted), so this export is ONLY safe for plain LoRA on
        a bnb-4bit base (the QLoRA persistence path).
        """
        import json

        try:
            from safetensors.torch import save_file
        except ImportError as e:
            raise RuntimeError("safetensors not installed — can't write PEFT adapter") from e

        # Collect per-layer ranks to compute one global r for the config.
        # vLLM expects a single rank; per-layer rank variation maps to the
        # `rank_pattern` field (PEFT 0.6+).
        ranks: dict[str, int] = {}
        target_modules_set: set[str] = set()
        peft_state: dict[str, torch.Tensor] = {}
        for name, layer in self._lora_layers.items():
            key_A = f"base_model.model.{name}.lora_A.weight"
            key_B = f"base_model.model.{name}.lora_B.weight"
            # vLLM expects bfloat16 adapter weights (same dtype as base compute).
            peft_state[key_A] = layer.lora_A.data.cpu().to(torch.bfloat16).contiguous()
            peft_state[key_B] = layer.lora_B.data.cpu().to(torch.bfloat16).contiguous()
            ranks[name] = int(layer.rank)
            target_modules_set.add(name.split(".")[-1])

        save_file(peft_state, str(save_path / "adapter_model.safetensors"))

        # Rank config: pick majority rank as the default `r`, keep the rest
        # in rank_pattern (PEFT reads these per-module).
        from collections import Counter
        rank_counts = Counter(ranks.values())
        default_r = rank_counts.most_common(1)[0][0] if rank_counts else int(self.config.lora_rank)
        rank_pattern = {n: r for n, r in ranks.items() if r != default_r}

        base_model_path = ""
        try:
            base_model_path = str(getattr(self.model_loader.config, "model_path", "") or "")
        except Exception:
            pass

        adapter_config = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": default_r,
            "lora_alpha": int(self.config.lora_alpha),
            "lora_dropout": float(self.config.lora_dropout),
            "bias": "none",
            "target_modules": sorted(target_modules_set),
            "rank_pattern": rank_pattern,
            "alpha_pattern": {},
            "fan_in_fan_out": False,
            "inference_mode": True,
            "base_model_name_or_path": base_model_path,
            "use_rslora": bool(getattr(self.config, "use_rslora", False)),
        }
        with open(save_path / "adapter_config.json", "w") as f:
            json.dump(adapter_config, f, indent=2)
        logger.info(
            f"Wrote PEFT adapter at {save_path} "
            f"(r={default_r}, {len(self._lora_layers)} layers, "
            f"targets={sorted(target_modules_set)})"
        )
        return save_path

    def load_lora_weights(self, path: Path | str) -> None:
        """Load saved LoRA weights and inject them into the model.

        Uses the saved rank/weakness_scale directly rather than recomputing from
        health values, ensuring A/B weight shapes match exactly.
        """
        path = Path(path)
        weights_path = path / "lora_weights.pt"
        if not weights_path.exists():
            raise FileNotFoundError(f"No LoRA weights at {weights_path}")

        try:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        except Exception as e:
            raise RuntimeError(f"Corrupt checkpoint at {weights_path}: {e}") from e
        if not state_dict:
            raise RuntimeError(f"Empty checkpoint at {weights_path}")
        self.strip_lora()

        model = self.model_loader.model
        device = self.model_loader.device

        # Extract layer info from saved state and inject LoRA with exact saved ranks
        layer_names = {k.rsplit(".lora_A", 1)[0] for k in state_dict if k.endswith(".lora_A")}

        for name in layer_names:
            saved_A = state_dict[f"{name}.lora_A"]
            saved_B = state_dict[f"{name}.lora_B"]
            saved_rank = state_dict[f"{name}.rank"].item()
            saved_ws = state_dict.get(f"{name}.weakness_scale", torch.tensor(1.0)).item()

            # Find the original module in the model
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            module = getattr(parent, parts[-1])

            if not _is_linear_like(module):
                continue

            self._original_layers[name] = module
            adjusted_alpha = int(self.config.lora_alpha * saved_rank / max(self.config.lora_rank, 1))

            lora_layer = LoRALayer(
                original_layer=module,
                rank=int(saved_rank),
                alpha=adjusted_alpha,
                dropout=self.config.lora_dropout,
                weakness_scale=saved_ws,
                use_rslora=self.config.use_rslora,
                init_method=self.config.init_method,
                use_dora=self.config.use_dora,
            )
            # Overwrite random init with saved weights. Saved weights are bfloat16
            # (halved disk usage) but LoRA params are float32 for training precision.
            # copy_() auto-casts, but the loaded values start from bf16-quantized
            # values — this is acceptable for checkpoint/resume since the precision
            # loss is negligible compared to one training step's weight change.
            lora_layer.lora_A.data.copy_(saved_A.to(device).float())
            lora_layer.lora_B.data.copy_(saved_B.to(device).float())

            setattr(parent, parts[-1], lora_layer)
            self._lora_layers[name] = lora_layer

        logger.info(f"Loaded LoRA weights from {path} ({len(self._lora_layers)} layers)")

    def merge_lora(self) -> bool:
        """Merge LoRA weights into base model, then strip LoRA layers.

        Returns True if merge succeeded normally, False if undertrained (>50% of
        LoRA layers had zero B matrix, indicating no meaningful gradients).

        IMPORTANT: Only use lora_scaling (alpha/rank) for the merge, NOT weakness_scale.
        weakness_scale amplifies gradients during training — it already shaped the learned
        lora_A/lora_B weights. Applying it again at merge time would double-count the
        weakness correction, over-writing base weights more than intended each cycle.
        """
        skipped = 0
        quant_skipped = 0  # separate tally: bnb base skipped (not a gradient signal)
        zero_B = 0  # separate tally: B actually stayed at init (real signal)
        # QLoRA 4bit/8bit guard: `orig.data += delta` on a Params4bit/Int8Params
        # is a shape mismatch (packed bytes ≠ dense [out, in]). The canonical
        # QLoRA recipe keeps LoRA as a SEPARATE adapter rather than merging.
        # Detect bnb-quantized base via the SAME attribute check LoRALayer
        # uses (`_base_is_4bit` / `_base_is_8bit`) and skip in-place merge.
        # Training remains applied in-memory on the HF model; adapter can be
        # saved separately for vLLM --enable-lora reload in a later pass.
        for name, lora_layer in self._lora_layers.items():
            with torch.no_grad():
                is_dora = getattr(lora_layer, "use_dora", False)
                base_quant = bool(
                    getattr(lora_layer, "_base_is_4bit", False)
                    or getattr(lora_layer, "_base_is_8bit", False)
                )
                # Additional safety: detect Params4bit/Int8Params by attribute
                # even if the flag didn't fire, so we never do the bad += on a
                # packed buffer.
                orig = lora_layer.original.weight
                if not base_quant:
                    base_quant = (
                        hasattr(orig, "quant_state")
                        or hasattr(orig, "CB")
                        or hasattr(orig, "SCB")
                        or type(orig).__name__ in ("Params4bit", "Int8Params")
                    )
                if base_quant:
                    # Don't merge. LoRA stays in-memory on HF model; stripped
                    # only at the end (next cycle rebuilds fresh LoRA).
                    quant_skipped += 1
                    # Still probe B so the undertrained diagnostic works on QLoRA:
                    # on bnb-4bit we never merge, but we still want to know whether
                    # the adapter actually moved off its zero init.
                    if not is_dora and float(lora_layer.lora_B.detach().abs().max()) < 1e-6:
                        zero_B += 1
                    continue
                # For plain LoRA: skip if B ≈ 0 (no gradient signal).
                # For DoRA: even with B=0, magnitude may have moved — still merge.
                if not is_dora and float(lora_layer.lora_B.detach().abs().max()) < 1e-6:
                    skipped += 1
                    zero_B += 1
                    continue
                # Cast LoRA params to target dtype BEFORE the matmul to avoid
                # allocating a full-sized (out_features × in_features) float32 matrix.
                # For 8192×8192 layers that's 1GB float32 vs 512MB bfloat16.
                A = lora_layer.lora_A.to(device=orig.device, dtype=torch.float32)
                B = lora_layer.lora_B.to(device=orig.device, dtype=torch.float32)

                if is_dora:
                    # DoRA merge: W_new = magnitude · (W_frozen + scaling·BA) / ‖…‖_c
                    #
                    # This is the canonical DoRA merge identity — after this
                    # overwrite, forward(x) through a plain Linear with W_new
                    # produces the same output as the DoRA forward did.
                    W0 = orig.data.to(torch.float32)
                    BA = B @ A  # (out, in) float32
                    if lora_layer._is_conv1d:
                        V_stored = W0 + lora_layer.scaling * BA.T
                        V_norm = V_stored.norm(dim=1).clamp_min(1e-8)  # (in,)
                        mag = lora_layer.magnitude.detach().to(
                            device=orig.device, dtype=torch.float32
                        )
                        # Scale each row by magnitude_i / V_norm_i. V_stored is
                        # (in, out); row i is indexed by in-feature i.
                        W_new = V_stored * (mag / V_norm).unsqueeze(1)
                    else:
                        V = W0 + lora_layer.scaling * BA  # (out, in)
                        V_norm = V.norm(dim=0).clamp_min(1e-8)  # (in,)
                        mag = lora_layer.magnitude.detach().to(
                            device=orig.device, dtype=torch.float32
                        )
                        # Scale each column by magnitude_j / V_norm_j.
                        W_new = V * (mag / V_norm)
                    orig.data.copy_(W_new.to(orig.dtype))
                    # Detect "no useful gradient" in DoRA: magnitude hasn't moved
                    # AND B is essentially zero.
                    if (float(lora_layer.lora_B.detach().abs().max()) < 1e-6
                            and float((mag - V_norm).abs().max()) < 1e-6):
                        skipped += 1
                else:
                    delta = (B @ A) * lora_layer.scaling
                    # Conv1D stores weight as (in, out) — transposed from Linear's (out, in)
                    if lora_layer._is_conv1d:
                        delta = delta.T
                    orig.data += delta.to(orig.dtype)
        total = len(self._lora_layers)
        if quant_skipped:
            logger.info(
                f"  Skipped {quant_skipped}/{total} LoRA merges (bnb-quantized base — "
                f"adapter stays separate for vLLM --enable-lora)"
            )
        if skipped:
            logger.info(f"  Skipped {skipped}/{total} zero-contribution LoRA layers during merge")
        # Real undertrained signal: B stayed at zero init. On QLoRA this is the
        # only way to tell if gradient actually reached the adapter, since the
        # merge is always skipped on bnb.
        if total and zero_B / total > 0.5:
            logger.warning(
                f"  Undertrained: {zero_B}/{total} LoRA layers had zero B matrix. "
                f"Training likely did not produce meaningful gradients this cycle."
            )
            self._last_merge_undertrained = True
        else:
            if zero_B:
                logger.info(f"  {zero_B}/{total} LoRA layers had zero B (below undertrained threshold)")
            self._last_merge_undertrained = False

        # After merging, strip LoRA so next cycle starts clean
        self.strip_lora()
        # Reclaim VRAM from freed LoRA parameters before post-training eval
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        return not self._last_merge_undertrained
