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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ..utils.config import TrainerConfig
from ..utils.model_loader import ModelLoader
from ..generator.data_generator import TrainingSample

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
    """Custom LoRA layer with weakness-adaptive rank and gradient scaling.

    Unlike standard LoRA:
    - rank is adjusted per-layer based on diagnostic weakness scores
    - weakness_scale amplifies gradients for layers that need more correction
    """

    def __init__(
        self,
        original_layer: nn.Module,
        rank: int,
        alpha: int,
        dropout: float = 0.0,
        weakness_scale: float = 1.0,
    ):
        super().__init__()
        self.original = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / max(rank, 1)
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
        lora_A_init = torch.zeros(rank, in_features, device=device, dtype=torch.float32)
        nn.init.kaiming_uniform_(lora_A_init, a=math.sqrt(5))
        self.lora_A = nn.Parameter(lora_A_init)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=device, dtype=torch.float32))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

            # Pad to max_length
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            actual_len = len(combined)
            if actual_len < self.max_length:
                padding = torch.full((self.max_length - actual_len,), pad_id, dtype=combined.dtype)
                combined = torch.cat([combined, padding])

            labels = combined.clone()
            labels[:prompt_len] = -100
            # Also mask padding tokens in labels
            if actual_len < self.max_length:
                labels[actual_len:] = -100

            attention_mask = torch.zeros(self.max_length, dtype=torch.long)
            attention_mask[:actual_len] = 1

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
            self._encoded.append(entry)

    def __len__(self):
        return len(self._encoded)

    def __getitem__(self, idx):
        return self._encoded[idx]


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


class CustomLoRATrainer:
    """Custom LoRA trainer with proper lifecycle management.

    Critical: LoRA layers are stripped and reinjected between cycles.
    This prevents the LoRA-on-LoRA wrapping bug.
    """

    def __init__(self, config: TrainerConfig, model_loader: ModelLoader):
        self.config = config
        self.model_loader = model_loader
        self._lora_layers: dict[str, LoRALayer] = {}
        self._original_layers: dict[str, nn.Module] = {}  # Linear or Conv1D

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
    ) -> TrainingMetrics:
        """Train on verified samples only.

        OOM recovery: if torch.cuda.OutOfMemoryError occurs during training,
        halves the batch size and retries once. If the retry also OOMs, raises.
        """
        if not verified_samples:
            return TrainingMetrics(cycle=cycle, avg_loss=0, final_loss=0,
                                  steps=0, samples_used=0, samples_rejected=0,
                                  learning_rate=self.config.learning_rate)

        model = self.model_loader.model
        tokenizer = self.model_loader.tokenizer

        dataset = TrainingDataset(
            verified_samples,
            tokenizer,
            max_length=self.model_loader.config.max_seq_length,
        )
        samples_rejected = len(verified_samples) - len(dataset)
        if samples_rejected > 0:
            logger.info(f"  Skipped {samples_rejected} samples (prompt too long for sequence length)")

        batch_size = self.config.batch_size
        # Cross-cycle LR decay: early cycles do coarse correction (higher LR),
        # later cycles do refinement (lower LR). Decay by sqrt to avoid
        # decaying too aggressively — cycle 1 gets full LR, cycle 100 gets 1/10.
        cycle_lr = self.config.learning_rate / math.sqrt(max(cycle, 1))
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
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=self.config.num_epochs > 1,  # Curriculum order for single epoch; shuffle for multi-epoch to avoid order overfitting
            drop_last=False,
        )

        # Only optimize LoRA parameters
        lora_params = [p for p in model.parameters() if p.requires_grad]
        if not lora_params:
            logger.warning("No trainable parameters found")
            return TrainingMetrics(cycle=cycle, avg_loss=0, final_loss=0,
                                  steps=0, samples_used=len(verified_samples),
                                  samples_rejected=0, learning_rate=self.config.learning_rate)

        optimizer = torch.optim.AdamW(
            lora_params,
            lr=cycle_lr,
            weight_decay=self.config.weight_decay,
        )

        total_batches = len(dataloader) * self.config.num_epochs
        # +1 for potential flush of remaining accumulated gradients
        total_steps = max(1, total_batches // self.config.gradient_accumulation_steps
                         + (1 if total_batches % self.config.gradient_accumulation_steps != 0 else 0))
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
        # Enable gradient checkpointing to save VRAM on A6000s
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

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
                    model_batch = {k: v.to(device) for k, v in batch.items() if k != "sample_weight"}

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
                        loss = (per_sample_loss * sample_weights).mean()
                        # Track unweighted loss for metrics — weighted loss varies with
                        # confidence distribution, making cross-cycle comparison unreliable.
                        unweighted_loss = per_sample_loss.mean().item()
                        # Free intermediate tensors immediately to reduce peak memory
                        del logits, labels, per_token, valid_tokens, per_sample_loss
                    else:
                        loss = outputs.loss
                        unweighted_loss = loss.item()

                    loss = loss / self.config.gradient_accumulation_steps
                    loss.backward()
                    accum_count += 1

                    last_loss = unweighted_loss
                    if accum_count % self.config.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(lora_params, self.config.max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        step_count += 1

                    total_loss += unweighted_loss
                    batch_count += 1

            # Flush any remaining accumulated gradients. last_loss already holds
            # the most recent batch's unweighted loss from the loop above.
            if accum_count % self.config.gradient_accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(lora_params, self.config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step_count += 1
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
        )

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

    def save_lora_weights(self, path: Path, cycle: int) -> None:
        """Save only the LoRA weights."""
        if not self._lora_layers:
            return
        save_path = path / f"lora_cycle_{cycle}"
        save_path.mkdir(parents=True, exist_ok=True)
        # Check disk space before writing — 100MB headroom for LoRA weights.
        try:
            free = shutil.disk_usage(save_path).free
            if free < 100 * 1024 * 1024:
                logger.error(f"Disk nearly full ({free // 1024 // 1024}MB free) — skipping LoRA weight save")
                return
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
        for name, lora_layer in self._lora_layers.items():
            with torch.no_grad():
                # Skip layers where B is still all zeros (initialized to zero and
                # received no gradient). The merge would be a no-op but allocates
                # a full (out_features × in_features) matrix for the matmul result.
                # Use a small-magnitude threshold instead of `.any()` — after
                # gradient checkpointing + hooks, B can accumulate numerical noise
                # (~1e-10) without meaningful gradient signal, yet `.any()` returns
                # True and the merge allocates a full (out × in) matrix for a
                # near-zero delta.
                if float(lora_layer.lora_B.detach().abs().max()) < 1e-6:
                    skipped += 1
                    continue
                orig = lora_layer.original.weight
                # Cast LoRA params to target dtype BEFORE the matmul to avoid
                # allocating a full-sized (out_features × in_features) float32 matrix.
                # For 8192×8192 layers that's 1GB float32 vs 512MB bfloat16.
                A = lora_layer.lora_A.to(device=orig.device, dtype=orig.dtype)
                B = lora_layer.lora_B.to(device=orig.device, dtype=orig.dtype)
                delta = (B @ A) * lora_layer.scaling
                # Conv1D stores weight as (in, out) — transposed from Linear's (out, in)
                if lora_layer._is_conv1d:
                    delta = delta.T
                orig.data += delta
        total = len(self._lora_layers)
        if skipped:
            logger.info(f"  Skipped {skipped}/{total} zero-contribution LoRA layers during merge")
        if total and skipped / total > 0.5:
            logger.warning(
                f"  Undertrained: {skipped}/{total} LoRA layers had zero B matrix. "
                f"Training likely did not produce meaningful gradients this cycle."
            )
            self._last_merge_undertrained = True
        else:
            self._last_merge_undertrained = False

        # After merging, strip LoRA so next cycle starts clean
        self.strip_lora()
        # Reclaim VRAM from freed LoRA parameters before post-training eval
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        return not self._last_merge_undertrained
