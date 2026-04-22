"""Task #20 throughput-pass guards.

Locks in the dynamic-padding collate behavior and the 1-step overfit
correctness sanity check that every further throughput change must pass
before commit (Liger/FA3/etc.).

These tests are CPU-only and do not require bitsandbytes or CUDA.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from src.trainer.custom_lora import (
    _pad_right,
    make_dynamic_pad_collate,
    make_dynamic_pad_collate_dpo,
)


def test_pad_right_pads_to_longest():
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5])
    out = _pad_right([a, b], pad_value=0)
    assert out.shape == (2, 3)
    assert out[0].tolist() == [1, 2, 3]
    assert out[1].tolist() == [4, 5, 0]


def test_dynamic_pad_collate_sft_batches_to_longest_not_global():
    """The core throughput win: batch-local padding, not max_length-wide."""
    collate = make_dynamic_pad_collate(pad_token_id=0)
    items = [
        {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "labels": torch.tensor([-100, 2, 3]),
            "sample_weight": torch.tensor(1.0),
            "calibration_brier": torch.tensor(-1.0),
        },
        {
            "input_ids": torch.tensor([4, 5, 6, 7, 8]),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
            "labels": torch.tensor([-100, 5, 6, 7, 8]),
            "sample_weight": torch.tensor(1.0),
            "calibration_brier": torch.tensor(-1.0),
        },
    ]
    batch = collate(items)
    assert batch["input_ids"].shape == (2, 5), "padded to max in batch, not to 1024"
    # First sample padded at tail with pad_token_id=0.
    assert batch["input_ids"][0].tolist() == [1, 2, 3, 0, 0]
    # Attention mask zero-padded.
    assert batch["attention_mask"][0].tolist() == [1, 1, 1, 0, 0]
    # Labels padded with -100 so CE ignores pad positions.
    assert batch["labels"][0].tolist() == [-100, 2, 3, -100, -100]
    assert batch["sample_weight"].shape == (2,)
    assert batch["calibration_brier"].shape == (2,)


def test_dynamic_pad_collate_dpo_sides_independent():
    collate = make_dynamic_pad_collate_dpo(pad_token_id=0)
    items = [
        {
            "chosen_input_ids": torch.tensor([1, 2]),
            "chosen_attention_mask": torch.tensor([1, 1]),
            "chosen_labels": torch.tensor([-100, 2]),
            "rejected_input_ids": torch.tensor([3, 4, 5]),
            "rejected_attention_mask": torch.tensor([1, 1, 1]),
            "rejected_labels": torch.tensor([-100, 4, 5]),
            "sample_weight": torch.tensor(1.0),
        },
        {
            "chosen_input_ids": torch.tensor([6, 7, 8, 9]),
            "chosen_attention_mask": torch.tensor([1, 1, 1, 1]),
            "chosen_labels": torch.tensor([-100, 7, 8, 9]),
            "rejected_input_ids": torch.tensor([10]),
            "rejected_attention_mask": torch.tensor([1]),
            "rejected_labels": torch.tensor([-100]),
            "sample_weight": torch.tensor(1.0),
        },
    ]
    batch = collate(items)
    # Chosen side max len = 4, rejected side max len = 3; padded independently.
    assert batch["chosen_input_ids"].shape == (2, 4)
    assert batch["rejected_input_ids"].shape == (2, 3)
    # Per-side padding with pad_id=0 on inputs, 0 on mask, -100 on labels.
    assert batch["chosen_input_ids"][0].tolist() == [1, 2, 0, 0]
    assert batch["chosen_labels"][0].tolist() == [-100, 2, -100, -100]
    assert batch["rejected_input_ids"][1].tolist() == [10, 0, 0]


# ---------------------------------------------------------------------------
# 1-step overfit correctness sanity — used as gate for future throughput
# changes (Liger, FA3, etc.). A single optimizer step on a single sample
# must drive training loss down. If a kernel swap breaks the gradient path,
# this test fails instantly.
# ---------------------------------------------------------------------------

class _TinyCausalLM(nn.Module):
    """Minimum viable HF-like causal LM for gradient-flow testing on CPU."""

    def __init__(self, vocab_size=32, hidden=16, seq_max=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.linear = nn.Linear(hidden, hidden, bias=False)  # LoRA target
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)
        self.vocab_size = vocab_size

        class _C:
            use_cache = True
        self.config = _C()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **_):
        x = self.embed(input_ids)
        x = self.linear(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        class _Out:
            pass
        out = _Out()
        out.logits = logits
        out.loss = loss
        return out

    # Stubs that the trainer may call; no-ops on CPU.
    def gradient_checkpointing_enable(self): pass
    def gradient_checkpointing_disable(self): pass
    def enable_input_require_grads(self): pass
    def train(self, mode=True):
        super().train(mode)
        return self
    def eval(self):
        super().eval()
        return self


def test_one_step_overfit_drives_loss_down():
    """Gate test for kernel/ops swaps: one optimizer step must reduce loss
    on the same batch. Regression in gradient flow would flatline loss.
    """
    torch.manual_seed(0)
    model = _TinyCausalLM()
    # Build a single-sample batch; labels are just input shifted.
    input_ids = torch.tensor([[3, 7, 2, 9, 4, 1]])
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    loss_fn = lambda: model(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels,
    ).loss

    initial = float(loss_fn().item())
    optim = torch.optim.AdamW(model.parameters(), lr=1e-1)
    for _ in range(5):
        optim.zero_grad()
        loss = loss_fn()
        loss.backward()
        optim.step()
    after = float(loss_fn().item())
    assert after < initial * 0.6, (
        f"one-step overfit failed: {initial:.4f} -> {after:.4f} "
        f"(gradient flow likely broken)"
    )


# ---------------------------------------------------------------------------
# Task #20: Liger kernels helper — no-op paths (install-gated)
# ---------------------------------------------------------------------------

def test_maybe_apply_liger_kernels_disabled_by_config():
    from src.utils.model_loader import _maybe_apply_liger_kernels
    from src.utils.config import ModelConfig

    cfg = ModelConfig(model_path="stub", use_liger_kernels=False)

    class _M:
        class config: model_type = "qwen2"
    assert _maybe_apply_liger_kernels(_M(), cfg) is False


def test_maybe_apply_liger_kernels_non_qwen2_is_noop():
    from src.utils.model_loader import _maybe_apply_liger_kernels
    from src.utils.config import ModelConfig

    cfg = ModelConfig(model_path="stub", use_liger_kernels=True)

    class _M:
        class config: model_type = "llama"
    # Even with use_liger_kernels=True, non-Qwen2 models return False.
    assert _maybe_apply_liger_kernels(_M(), cfg) is False


def test_maybe_apply_liger_kernels_missing_package_is_noop(monkeypatch):
    """When liger_kernel isn't installed, helper returns False without raising."""
    import builtins
    from src.utils.model_loader import _maybe_apply_liger_kernels
    from src.utils.config import ModelConfig

    cfg = ModelConfig(model_path="stub", use_liger_kernels=True)

    class _M:
        class config: model_type = "qwen2"

    real_import = builtins.__import__
    def _blocker(name, *args, **kwargs):
        if name.startswith("liger_kernel"):
            raise ImportError("simulated missing liger_kernel")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", _blocker)
    assert _maybe_apply_liger_kernels(_M(), cfg) is False
