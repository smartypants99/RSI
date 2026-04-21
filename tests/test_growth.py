"""Tests for src/trainer/growth.py — upward distillation / weight growth.

Uses a tiny fake CausalLM skeleton (no transformers / HF model load) so the
suite stays fast and CPU-only.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.trainer.growth import (
    GrowthConfig,
    GrowthResult,
    build_student_layers,
    distill_step,
    grow_and_distill,
    grow_model,
    layer_mapping,
    plan_target_layers,
)


# ---- fake skeleton -----------------------------------------------------------


class FakeBlock(nn.Module):
    """Pretends to be a transformer block with an ``o_proj`` and ``down_proj``."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.o_proj = nn.Linear(dim, dim)
        self.down_proj = nn.Linear(dim, dim)

    def forward(self, x):
        return x + self.down_proj(self.o_proj(self.norm(x)))


class FakeInner(nn.Module):
    def __init__(self, dim: int, num_layers: int, vocab: int):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList(FakeBlock(dim) for _ in range(num_layers))


class FakeCausalLM(nn.Module):
    """Minimal Llama-ish skeleton: ``self.model.layers`` is the stack."""

    def __init__(self, dim: int = 8, num_layers: int = 4, vocab: int = 32):
        super().__init__()
        self.model = FakeInner(dim, num_layers, vocab)
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        self.config = SimpleNamespace(num_hidden_layers=num_layers, hidden_size=dim)

    def forward(self, input_ids):
        x = self.model.embed(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        logits = self.lm_head(x)
        return SimpleNamespace(logits=logits)


# ---- pure helpers ------------------------------------------------------------


def test_plan_target_layers_15x():
    assert plan_target_layers(4, 1.5) == 6
    assert plan_target_layers(10, 1.5) == 15
    # always strictly larger than teacher
    assert plan_target_layers(2, 1.01) == 3


def test_layer_mapping_interleaves():
    mapping = layer_mapping(4, 6)
    assert len(mapping) == 6
    assert min(mapping) == 0 and max(mapping) == 3
    # every teacher layer is used at least once
    assert set(mapping) == {0, 1, 2, 3}


def test_layer_mapping_invalid():
    with pytest.raises(ValueError):
        layer_mapping(4, 4)
    with pytest.raises(ValueError):
        layer_mapping(4, 3)


def test_growth_config_validation():
    with pytest.raises(ValueError):
        GrowthConfig(growth_factor=1.0)
    with pytest.raises(ValueError):
        GrowthConfig(init_method="xavier")
    with pytest.raises(ValueError):
        GrowthConfig(duplicate_noise_std=-0.1)
    # valid case
    GrowthConfig()


# ---- build_student_layers ----------------------------------------------------


def test_build_student_layers_duplicate_no_noise_preserves_weights():
    teacher_layers = [FakeBlock(4), FakeBlock(4)]
    cfg = GrowthConfig(
        growth_factor=1.5,
        init_method="duplicate",
        duplicate_noise_std=0.0,
    )
    new_stack = build_student_layers(teacher_layers, 3, cfg)
    assert len(new_stack) == 3
    # New layers must be independent copies, not aliases
    for src, dup in zip([0, 0, 1], new_stack):
        teacher = teacher_layers[src]
        assert dup is not teacher
        assert torch.equal(dup.o_proj.weight, teacher.o_proj.weight)
        # mutating the copy must not leak
        with torch.no_grad():
            dup.o_proj.weight.add_(1.0)
        assert not torch.equal(dup.o_proj.weight, teacher.o_proj.weight)


def test_build_student_layers_duplicate_with_noise_diverges():
    torch.manual_seed(0)
    teacher_layers = [FakeBlock(4)]
    cfg = GrowthConfig(
        growth_factor=2.0,
        init_method="duplicate",
        duplicate_noise_std=0.01,
    )
    new_stack = build_student_layers(teacher_layers, 2, cfg)
    for new in new_stack:
        assert not torch.equal(new.o_proj.weight, teacher_layers[0].o_proj.weight)
        diff = (new.o_proj.weight - teacher_layers[0].o_proj.weight).abs().mean()
        # noise is small — within ~3*std of expected magnitude
        assert 0.0 < diff.item() < 0.05


def test_build_student_layers_identity_zeros_only_inserted_duplicates():
    teacher_layers = [FakeBlock(4)]
    cfg = GrowthConfig(
        growth_factor=2.0,
        init_method="identity",
    )
    new_stack = build_student_layers(teacher_layers, 2, cfg)
    # First slot: original weights kept (function-preserving)
    assert torch.equal(new_stack[0].o_proj.weight, teacher_layers[0].o_proj.weight)
    # Second slot: duplicate insertion — residual projections zeroed
    assert torch.count_nonzero(new_stack[1].o_proj.weight) == 0
    assert torch.count_nonzero(new_stack[1].down_proj.weight) == 0
    # LayerNorm weights kept on both
    assert torch.count_nonzero(new_stack[1].norm.weight) > 0


# ---- grow_model --------------------------------------------------------------


def test_grow_model_expands_layer_stack_and_updates_config():
    torch.manual_seed(0)
    teacher = FakeCausalLM(dim=8, num_layers=4, vocab=16)
    cfg = GrowthConfig(growth_factor=1.5, duplicate_noise_std=0.0)

    student = grow_model(teacher, cfg)

    assert len(student.model.layers) == 6
    assert len(teacher.model.layers) == 4, "teacher must not be mutated"
    assert student.config.num_hidden_layers == 6
    # student params share no storage with teacher
    for s_p, t_p in zip(student.parameters(), teacher.parameters()):
        assert s_p.data_ptr() != t_p.data_ptr() or s_p.numel() == 0


def test_grow_model_identity_init_is_function_preserving():
    """With identity-init + zero noise, the student's forward at t=0 should
    match the teacher's on a fresh input (the extra layers are no-ops)."""
    torch.manual_seed(0)
    teacher = FakeCausalLM(dim=8, num_layers=4, vocab=16)
    cfg = GrowthConfig(growth_factor=1.5, init_method="identity")
    student = grow_model(teacher, cfg)

    input_ids = torch.randint(0, 16, (2, 5))
    with torch.no_grad():
        t_logits = teacher(input_ids).logits
        s_logits = student(input_ids).logits
    # identity init: residual contribution of new layers is zero; outputs
    # should match up to floating noise from extra (now-zero) projection
    # additions. Accept tight tolerance.
    assert torch.allclose(t_logits, s_logits, atol=1e-5)


# ---- distill_step ------------------------------------------------------------


def test_distill_step_runs_and_gradients_flow():
    torch.manual_seed(0)
    teacher = FakeCausalLM(dim=8, num_layers=4, vocab=16)
    cfg = GrowthConfig(growth_factor=1.5, duplicate_noise_std=0.0)
    student = grow_model(teacher, cfg)

    input_ids = torch.randint(0, 16, (2, 5))
    labels = torch.randint(0, 16, (2, 5))
    loss = distill_step(student, teacher, input_ids, labels, cfg)
    assert loss.ndim == 0
    loss.backward()
    # Some student param must have accumulated a gradient
    grad_norms = [p.grad.norm().item() for p in student.parameters() if p.grad is not None]
    assert any(g > 0 for g in grad_norms)
    # Teacher must remain gradient-free
    for p in teacher.parameters():
        assert p.grad is None


# ---- grow_and_distill --------------------------------------------------------


def _make_batches(n: int = 2, bs: int = 2, seqlen: int = 5, vocab: int = 16):
    for _ in range(n):
        yield {
            "input_ids": torch.randint(0, vocab, (bs, seqlen)),
            "labels": torch.randint(0, vocab, (bs, seqlen)),
        }


def test_grow_and_distill_happy_path_returns_student():
    torch.manual_seed(0)
    teacher = FakeCausalLM(dim=8, num_layers=4, vocab=16)
    cfg = GrowthConfig(growth_factor=1.5, duplicate_noise_std=0.0, distill_epochs=1)

    # Held-out eval: report a better score for whichever model has more layers.
    def heldout(m):
        return 0.5 + 0.01 * len(m.model.layers)

    model, result = grow_and_distill(
        teacher, list(_make_batches()), cfg, heldout_eval=heldout
    )
    assert result.grew is True
    assert result.teacher_layers == 4
    assert result.target_layers == 6
    assert model is not teacher
    assert len(model.model.layers) == 6


def test_grow_and_distill_aborts_on_regression():
    torch.manual_seed(0)
    teacher = FakeCausalLM(dim=8, num_layers=4, vocab=16)
    cfg = GrowthConfig(
        growth_factor=1.5, duplicate_noise_std=0.0, distill_epochs=1,
        abort_if_worse_by=0.01,
    )
    # Held-out deliberately regresses on the student.
    call = {"n": 0}

    def heldout(_m):
        call["n"] += 1
        return 0.80 if call["n"] == 1 else 0.70  # teacher then student

    model, result = grow_and_distill(
        teacher, list(_make_batches()), cfg, heldout_eval=heldout
    )
    assert result.grew is False
    assert result.abort_reason is not None
    assert model is teacher  # caller gets the original back
