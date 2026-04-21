"""Tests for TDQ disk-streaming path in growth.

Exercises the forward-looking escape hatch: when a grown model would
exceed the VRAM budget, growth serializes to a HF dir and compresses to
.tdq rather than instantiating in VRAM. We mock both save and compress
so these tests stay CPU-only and don't touch the external TDQ tooling.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.trainer.growth import (
    GrowthConfig,
    StreamGrowthResult,
    _decide_backend,
    _detect_vram_budget_bytes,
    _estimate_grown_bytes,
    grow_and_stream,
)


class _Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.o_proj = nn.Linear(dim, dim)
        self.down_proj = nn.Linear(dim, dim)

    def forward(self, x):
        return x + self.down_proj(self.o_proj(x))


class _Inner(nn.Module):
    def __init__(self, dim, n_layers):
        super().__init__()
        self.embed = nn.Embedding(8, dim)
        self.layers = nn.ModuleList(_Block(dim) for _ in range(n_layers))


class _Fake(nn.Module):
    def __init__(self, dim=4, n_layers=4):
        super().__init__()
        self.model = _Inner(dim, n_layers)
        self.lm_head = nn.Linear(dim, 8, bias=False)
        self.config = SimpleNamespace(num_hidden_layers=n_layers)


# ---- storage_backend validation ---------------------------------------------


def test_growth_config_storage_backend_valid():
    GrowthConfig(storage_backend="vram")
    GrowthConfig(storage_backend="tdq_stream")
    GrowthConfig(storage_backend="auto")


def test_growth_config_storage_backend_invalid():
    with pytest.raises(ValueError):
        GrowthConfig(storage_backend="onnx")
    with pytest.raises(ValueError):
        GrowthConfig(vram_safety_margin_gb=-1)
    with pytest.raises(ValueError):
        GrowthConfig(vram_budget_gb=0)


# ---- estimate + budget ------------------------------------------------------


def test_estimate_grown_bytes_scales_with_growth_factor():
    teacher = _Fake(dim=8, n_layers=4)
    cfg_small = GrowthConfig(growth_factor=1.5)
    cfg_big = GrowthConfig(growth_factor=3.0)
    assert _estimate_grown_bytes(teacher, cfg_big) > _estimate_grown_bytes(
        teacher, cfg_small
    )


def test_detect_vram_budget_prefers_explicit_gb():
    cfg = GrowthConfig(vram_budget_gb=10.0, vram_safety_margin_gb=2.0)
    budget = _detect_vram_budget_bytes(cfg)
    assert budget == int(8 * (1024 ** 3))


# ---- backend resolution -----------------------------------------------------


def test_decide_backend_explicit_passthrough():
    teacher = _Fake()
    assert _decide_backend(teacher, GrowthConfig(storage_backend="vram")) == "vram"
    assert _decide_backend(
        teacher, GrowthConfig(storage_backend="tdq_stream")
    ) == "tdq_stream"


def test_decide_backend_auto_under_budget_picks_vram():
    teacher = _Fake()
    cfg = GrowthConfig(storage_backend="auto", vram_budget_gb=100.0,
                       vram_safety_margin_gb=0.0)
    assert _decide_backend(teacher, cfg) == "vram"


def test_decide_backend_auto_over_budget_picks_tdq_stream():
    teacher = _Fake(dim=256, n_layers=8)
    # Set a tiny budget so any grown model exceeds it.
    cfg = GrowthConfig(storage_backend="auto", vram_budget_gb=0.00001,
                       vram_safety_margin_gb=0.0)
    assert _decide_backend(teacher, cfg) == "tdq_stream"


# ---- grow_and_stream mocked round-trip --------------------------------------


def test_grow_and_stream_vram_backend_is_noop(tmp_path):
    teacher = _Fake()
    cfg = GrowthConfig(storage_backend="vram")
    result = grow_and_stream(teacher, cfg, output_dir=tmp_path)
    assert isinstance(result, StreamGrowthResult)
    assert result.grew is False
    assert result.backend == "vram"
    assert result.tdq_path is None


def test_grow_and_stream_tdq_path_saves_and_compresses(tmp_path):
    teacher = _Fake(dim=4, n_layers=4)
    cfg = GrowthConfig(storage_backend="tdq_stream", duplicate_noise_std=0.0)

    saved = {}

    def fake_save(model, tokenizer, hf_dir):
        hf_dir = Path(hf_dir)
        hf_dir.mkdir(parents=True, exist_ok=True)
        (hf_dir / "config.json").write_text("{}")
        saved["layers"] = len(model.model.layers)
        saved["hf_dir"] = hf_dir
        return hf_dir

    def fake_compress(hf_dir, tdq_path, config="A"):
        tdq_path = Path(tdq_path)
        tdq_path.write_bytes(b"TDQ\x00fake")
        saved["tdq_path"] = tdq_path
        saved["config"] = config
        return tdq_path

    result = grow_and_stream(
        teacher, cfg, output_dir=tmp_path,
        tokenizer=None, save_fn=fake_save, compress_fn=fake_compress,
    )

    assert result.grew is True
    assert result.backend == "tdq_stream"
    assert result.target_layers == 6  # 4 * 1.5
    assert saved["layers"] == 6, "grown model should have target layers"
    assert Path(result.tdq_path).exists()
    assert saved["config"] == "A"
    # Teacher was not mutated
    assert len(teacher.model.layers) == 4


def test_grow_and_stream_auto_falls_back_to_vram_when_budget_fits(tmp_path):
    teacher = _Fake()
    cfg = GrowthConfig(storage_backend="auto", vram_budget_gb=100.0,
                       vram_safety_margin_gb=0.0)
    result = grow_and_stream(teacher, cfg, output_dir=tmp_path)
    assert result.grew is False
    assert result.backend == "vram"
    assert "within budget" in (result.abort_reason or "")


def test_tdq_bridge_compress_requires_output_file(tmp_path):
    from src.utils.tdq_bridge import compress_model_dir_to_tdq

    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    out = tmp_path / "x.tdq"

    def broken_compressor(model_id, output_path, config="A"):
        return  # writes nothing

    with pytest.raises(RuntimeError, match="output file missing"):
        compress_model_dir_to_tdq(hf_dir, out, compressor=broken_compressor)


def test_tdq_bridge_compress_calls_compressor(tmp_path):
    from src.utils.tdq_bridge import compress_model_dir_to_tdq

    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    out = tmp_path / "x.tdq"
    calls = {}

    def ok(model_id, output_path, config="A"):
        calls["model_id"] = model_id
        calls["output_path"] = output_path
        calls["config"] = config
        Path(output_path).write_bytes(b"ok")

    result = compress_model_dir_to_tdq(hf_dir, out, config="B", compressor=ok)
    assert result == out
    assert calls["config"] == "B"
    assert calls["model_id"] == str(hf_dir)
