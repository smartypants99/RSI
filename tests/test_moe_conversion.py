"""Tests for src/trainer/moe_conversion.py — post-hoc dense→sparse MoE upcycling."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.trainer.moe_conversion import (
    MoEConversionConfig,
    SparseMoELayer,
    convert_dense_ffn_to_moe,
    convert_model_ffn_to_moe,
)


# ---- tiny FFN ---------------------------------------------------------------


class _SwiGLUFFN(nn.Module):
    def __init__(self, hidden: int = 8, intermediate: int = 16):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)


class _BareFFN(nn.Module):
    def __init__(self, hidden: int = 8, intermediate: int = 16):
        super().__init__()
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)


class _Block(nn.Module):
    def __init__(self, hidden: int = 8, intermediate: int = 16):
        super().__init__()
        self.mlp = _SwiGLUFFN(hidden, intermediate)


class _Skel(nn.Module):
    def __init__(self, num_layers: int = 3, hidden: int = 8, intermediate: int = 16):
        super().__init__()
        class _Inner(nn.Module):
            pass
        inner = _Inner()
        inner.layers = nn.ModuleList(_Block(hidden, intermediate) for _ in range(num_layers))
        self.model = inner


# ---- config -----------------------------------------------------------------


def test_config_validates():
    MoEConversionConfig()
    with pytest.raises(ValueError):
        MoEConversionConfig(num_experts=1)
    with pytest.raises(ValueError):
        MoEConversionConfig(top_k=0)
    with pytest.raises(ValueError):
        MoEConversionConfig(top_k=99)
    with pytest.raises(ValueError):
        MoEConversionConfig(num_experts=4, shared_experts=4)
    with pytest.raises(ValueError):
        MoEConversionConfig(init_method="nope")


# ---- conversion primitives --------------------------------------------------


@pytest.mark.parametrize("init", ["copy_perturb", "slice", "clustering"])
def test_convert_dense_ffn_to_moe_produces_runnable_layer(init):
    cfg = MoEConversionConfig(num_experts=4, top_k=2, shared_experts=1, init_method=init)
    ffn = _SwiGLUFFN(hidden=8, intermediate=16)
    moe = convert_dense_ffn_to_moe(ffn, cfg, hidden_size=8)

    assert isinstance(moe, SparseMoELayer)
    assert len(moe.experts) == 4
    x = torch.randn(2, 5, 8)
    y = moe(x)
    assert y.shape == x.shape
    # aux loss is recorded and non-negative
    assert moe.last_aux_loss.item() >= 0.0


def test_convert_dense_ffn_bare_no_gate():
    cfg = MoEConversionConfig(num_experts=3, top_k=1, shared_experts=0, init_method="slice")
    ffn = _BareFFN(hidden=8, intermediate=12)
    moe = convert_dense_ffn_to_moe(ffn, cfg, hidden_size=8)
    y = moe(torch.randn(3, 8))
    assert y.shape == (3, 8)
    # no gate should still produce non-zero outputs (ReLU path)
    assert not torch.allclose(y, torch.zeros_like(y))


def test_convert_ffn_rejects_unknown_module():
    cfg = MoEConversionConfig()
    with pytest.raises(AttributeError):
        convert_dense_ffn_to_moe(nn.Linear(4, 4), cfg, hidden_size=4)


# ---- model-level conversion -------------------------------------------------


def test_convert_model_ffn_to_moe_replaces_every_layer():
    cfg = MoEConversionConfig(num_experts=2, top_k=1, shared_experts=0, init_method="slice")
    model = _Skel(num_layers=3)
    converted = convert_model_ffn_to_moe(model, cfg, hidden_size=8, ffn_attr="mlp")
    # teacher untouched
    for layer in model.model.layers:
        assert isinstance(layer.mlp, _SwiGLUFFN)
    # student: every layer has an MoE
    for layer in converted.model.layers:
        assert isinstance(layer.mlp, SparseMoELayer)


def test_moe_layer_top_k_respected():
    cfg = MoEConversionConfig(num_experts=4, top_k=2, shared_experts=0, init_method="slice")
    ffn = _SwiGLUFFN(hidden=8, intermediate=16)
    moe = convert_dense_ffn_to_moe(ffn, cfg, hidden_size=8)
    # top_k=2 means the forward pass sums contributions from 2 experts per token.
    y = moe(torch.randn(4, 8))
    assert y.shape == (4, 8)


def test_shared_experts_always_active():
    # With shared=num_experts-1 and top_k=1 over the single routed expert,
    # all experts should contribute — a shared-off ablation should differ.
    cfg_shared = MoEConversionConfig(num_experts=3, top_k=1, shared_experts=2, init_method="copy_perturb", copy_perturb_std=0.0)
    cfg_plain = MoEConversionConfig(num_experts=3, top_k=1, shared_experts=0, init_method="copy_perturb", copy_perturb_std=0.0)
    ffn = _SwiGLUFFN(hidden=8, intermediate=16)
    torch.manual_seed(0)
    moe_a = convert_dense_ffn_to_moe(ffn, cfg_shared, hidden_size=8)
    torch.manual_seed(0)
    moe_b = convert_dense_ffn_to_moe(ffn, cfg_plain, hidden_size=8)
    x = torch.randn(2, 8)
    ya = moe_a(x)
    yb = moe_b(x)
    # Different architectures (shared vs routed) must produce different outputs.
    assert not torch.allclose(ya, yb)
