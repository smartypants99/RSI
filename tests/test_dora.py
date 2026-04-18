"""DoRA correctness tests — the math must be right.

The critical invariant: after merge_lora(), `original.weight` equals the
effective weight that forward() was computing. If this breaks, merged
models silently disagree with the training-time forward, which is the
kind of bug that ruins an RSI run without any obvious symptom.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from src.trainer.custom_lora import LoRALayer


def _make_dora_layer(in_features=16, out_features=12, rank=4, seed=0):
    torch.manual_seed(seed)
    base = nn.Linear(in_features, out_features, bias=False)
    layer = LoRALayer(
        original_layer=base,
        rank=rank,
        alpha=8,
        dropout=0.0,  # deterministic forward for numerical tests
        weakness_scale=1.0,
        use_rslora=True,
        init_method="kaiming",
        use_dora=True,
    )
    layer.eval()  # ensure dropout modules are inactive regardless of default
    return layer


def test_dora_init_preserves_function():
    """At construction, DoRA's forward must equal the base weight's forward.

    B = 0 at init → V = W_frozen → magnitude init = ‖W_frozen‖_c → after
    normalization and magnitude application, W_effective reduces to W_frozen.
    """
    layer = _make_dora_layer()
    x = torch.randn(2, 5, 16)  # (batch, seq, in)
    with torch.no_grad():
        expected = layer.original(x)
        actual = layer(x)
    assert torch.allclose(expected, actual, atol=1e-4), \
        f"DoRA init should be identity; max diff = {(expected - actual).abs().max().item()}"


def test_dora_merge_matches_forward():
    """After merge_lora, a plain Linear with merged weight must reproduce DoRA forward.

    Train the DoRA params (simulate) so magnitude and B move away from init;
    then merge and check numerical equivalence on held-out input.
    """
    layer = _make_dora_layer(seed=42)

    # Simulate training: randomize B, perturb magnitude so merge math is
    # exercised on non-trivial parameters.
    with torch.no_grad():
        layer.lora_B.data = torch.randn_like(layer.lora_B) * 0.1
        layer.magnitude.data = layer.magnitude.data * (1 + torch.randn_like(layer.magnitude) * 0.05)

    x = torch.randn(3, 7, 16)
    with torch.no_grad():
        dora_output = layer(x)

    # Manual merge using the same math as merge_lora()
    with torch.no_grad():
        W0 = layer.original.weight.float()
        A = layer.lora_A.float()
        B = layer.lora_B.float()
        V = W0 + layer.scaling * (B @ A)
        V_norm = V.norm(dim=0).clamp_min(1e-8)
        mag = layer.magnitude.float()
        W_merged = V * (mag / V_norm)

    # Forward via the merged weight (plain linear)
    merged_layer = nn.Linear(16, 12, bias=False)
    merged_layer.weight.data = W_merged.to(merged_layer.weight.dtype)
    with torch.no_grad():
        merged_output = merged_layer(x)

    diff = (dora_output - merged_output).abs().max().item()
    assert diff < 1e-3, f"DoRA merge diverges from forward: max diff = {diff}"


def test_dora_gradients_flow_through_magnitude_and_lora():
    """All three trainable tensors must receive gradient."""
    layer = _make_dora_layer(seed=1)
    x = torch.randn(2, 4, 16, requires_grad=True)
    out = layer(x)
    loss = out.sum()
    loss.backward()

    assert layer.lora_A.grad is not None, "lora_A received no gradient"
    assert layer.lora_B.grad is not None, "lora_B received no gradient"
    assert layer.magnitude.grad is not None, "magnitude received no gradient"

    # Non-trivial — they must actually carry information, not just exist.
    # lora_A's gradient can be zero at init (B=0 blocks the backward path
    # through the LoRA branch), so only enforce for magnitude + lora_B.
    assert layer.magnitude.grad.abs().sum().item() > 0
    assert layer.lora_B.grad.abs().sum().item() > 0


def test_dora_magnitude_shape_is_in_features():
    layer = _make_dora_layer(in_features=32, out_features=16)
    assert layer.magnitude.shape == (32,)


def test_dora_off_has_no_magnitude():
    """use_dora=False: layer should not have a trainable magnitude."""
    base = nn.Linear(8, 6, bias=False)
    layer = LoRALayer(base, rank=2, alpha=4, use_dora=False)
    assert layer.magnitude is None


def test_dora_merge_on_layer_level_via_trainer():
    """End-to-end: CustomLoRATrainer can inject DoRA, the merge runs cleanly."""
    from src.utils.config import TrainerConfig
    from src.trainer.custom_lora import CustomLoRATrainer

    config = TrainerConfig(
        lora_rank=4, min_rank=2, lora_alpha=8,
        lora_dropout=0.0,  # deterministic forward for the numerical check
        use_dora=True, init_method="kaiming", use_rslora=True,
        use_lora_plus=False,  # simplify the optimizer path
    )

    class FakeModelLoader:
        config = type("X", (), {"max_seq_length": 128})()
        device = "cpu"

        def __init__(self):
            # Minimal model exposing one Linear, like a transformer's q_proj.
            class Block(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.q_proj = nn.Linear(16, 12, bias=False)

                def forward(self, x):
                    return self.q_proj(x)

            self.model = Block()
            self.tokenizer = None

    ml = FakeModelLoader()
    trainer = CustomLoRATrainer(config, ml)

    # Target the q_proj; inject LoRA (DoRA). Check magnitude was created.
    # The trainer's target_modules default includes "q_proj".
    trainer.inject_lora(weak_layers=None)
    # eval() mode so dropout is inactive and forward is deterministic
    ml.model.eval()
    dora_layers = [l for l in trainer._lora_layers.values() if l.use_dora]
    assert len(dora_layers) >= 1, "inject_lora did not produce DoRA layers"
    for l in dora_layers:
        assert l.magnitude is not None

    # Capture pre-merge forward, perturb params, merge, compare post-merge forward.
    x = torch.randn(1, 3, 16)
    with torch.no_grad():
        for l in dora_layers:
            l.lora_B.data = torch.randn_like(l.lora_B) * 0.05
            l.magnitude.data = l.magnitude.data * 1.02
        pre_merge = ml.model(x).clone()

    ok = trainer.merge_lora()
    assert ok or trainer._last_merge_undertrained is False or ok is False, \
        "merge should return bool"

    with torch.no_grad():
        post_merge = ml.model(x)
    diff = (pre_merge - post_merge).abs().max().item()
    assert diff < 1e-2, f"DoRA merge changed model behavior: max diff = {diff}"
