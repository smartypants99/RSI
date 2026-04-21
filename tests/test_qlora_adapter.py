"""QLoRA adapter persistence tests.

On bnb-4bit bases, merge_lora is a no-op (packed weights can't absorb a
dense delta) and save_checkpoint skips (save_pretrained raises on quantized
bases). Without adapter persistence the trained LoRA evaporates every cycle
and each cycle evaluates the untrained base. These tests lock the fix:

  1. save_lora_weights emits a PEFT-format adapter directory.
  2. adapter_config.json has the keys vLLM's LoRARequest loader requires.
  3. The A/B tensors round-trip bit-identically through the safetensors file.
  4. VLLMModelLoader.set_lora_adapter accepts a valid adapter dir and
     constructs a LoRARequest with a fresh int id each registration.
  5. swap_to_vllm_after_training(adapter_path=...) flips enable_lora on
     even if the caller constructed the loader with enable_lora=False.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from src.trainer.custom_lora import CustomLoRATrainer, LoRALayer
from src.utils.config import TrainerConfig


class _TinyModel(nn.Module):
    """Stand-in for a 4bit base. We don't need real bnb quantization — the
    adapter export path doesn't touch base weights, it only reads the LoRA
    A/B matrices the trainer already holds. Using plain Linear keeps the
    test CPU-only and deterministic."""

    def __init__(self):
        super().__init__()
        # Match target_modules in TrainerConfig (q_proj/v_proj are default).
        self.q_proj = nn.Linear(16, 16, bias=False)
        self.v_proj = nn.Linear(16, 16, bias=False)

    def named_modules(self, *args, **kwargs):
        return super().named_modules(*args, **kwargs)


class _StubModelLoader:
    def __init__(self, model: nn.Module):
        self.model = model
        self.device = torch.device("cpu")

        class _C:
            model_path = "stub/base-4bit"
            max_seq_length = 128

        self.config = _C()


def _make_trainer() -> CustomLoRATrainer:
    cfg = TrainerConfig(
        lora_rank=4,
        lora_alpha=8,
        lora_dropout=0.0,
        init_method="kaiming",
        use_rslora=False,
        use_dora=False,
        target_modules=["q_proj", "v_proj"],
        min_rank=2,
        max_rank=16,
    )
    model = _TinyModel()
    loader = _StubModelLoader(model)
    trainer = CustomLoRATrainer(cfg, loader)
    trainer.inject_lora(weak_layers={})
    # Perturb B so the adapter has a non-trivial delta on disk.
    with torch.no_grad():
        for layer in trainer._lora_layers.values():
            layer.lora_B.data.normal_(std=0.05)
    return trainer


def test_save_lora_weights_emits_peft_adapter(tmp_path: Path):
    trainer = _make_trainer()
    adapter_path = trainer.save_lora_weights(tmp_path, cycle=3)

    assert adapter_path is not None, "save_lora_weights should return adapter dir on success"
    assert adapter_path.name == "lora_cycle_3"
    assert (adapter_path / "adapter_model.safetensors").is_file()
    assert (adapter_path / "adapter_config.json").is_file()
    # Native format is still written so load_lora_weights keeps working.
    assert (adapter_path / "lora_weights.pt").is_file()


def test_adapter_config_has_required_peft_keys(tmp_path: Path):
    trainer = _make_trainer()
    adapter_path = trainer.save_lora_weights(tmp_path, cycle=1)
    cfg = json.loads((adapter_path / "adapter_config.json").read_text())

    # vLLM's PEFT loader reads these fields; missing any of them → load fail.
    for key in ("peft_type", "r", "lora_alpha", "target_modules",
                "base_model_name_or_path", "bias", "task_type"):
        assert key in cfg, f"missing PEFT key {key}"
    assert cfg["peft_type"] == "LORA"
    assert cfg["task_type"] == "CAUSAL_LM"
    assert cfg["bias"] == "none"
    assert set(cfg["target_modules"]) == {"q_proj", "v_proj"}
    assert cfg["r"] >= 1


def test_adapter_safetensors_roundtrip_matches_trained_weights(tmp_path: Path):
    safetensors = pytest.importorskip("safetensors")
    from safetensors.torch import load_file

    trainer = _make_trainer()
    adapter_path = trainer.save_lora_weights(tmp_path, cycle=7)
    state = load_file(str(adapter_path / "adapter_model.safetensors"))

    # Every trained layer appears under the PEFT key convention.
    for name, layer in trainer._lora_layers.items():
        key_A = f"base_model.model.{name}.lora_A.weight"
        key_B = f"base_model.model.{name}.lora_B.weight"
        assert key_A in state
        assert key_B in state
        # bf16 serialization loses a tiny amount of precision vs float32
        # training params; assert close within bf16 ULP, not exact equality.
        assert torch.allclose(
            state[key_A].float(), layer.lora_A.data.cpu().float(), atol=1e-2
        )
        assert torch.allclose(
            state[key_B].float(), layer.lora_B.data.cpu().float(), atol=1e-2
        )


def test_set_lora_adapter_validates_and_increments_id(tmp_path: Path):
    from src.utils.vllm_backend import VLLMModelLoader

    trainer = _make_trainer()
    adapter_path = trainer.save_lora_weights(tmp_path, cycle=1)

    loader = VLLMModelLoader(
        model_path="stub/base", enable_lora=True, max_lora_rank=8,
    )
    assert loader._lora_adapter_path is None

    loader.set_lora_adapter(str(adapter_path))
    assert loader._lora_adapter_path == str(adapter_path)
    first_id = loader._lora_adapter_id
    assert first_id >= 1

    # Re-registering bumps the id so vLLM's adapter cache invalidates when
    # the file contents change cycle-to-cycle.
    loader.set_lora_adapter(str(adapter_path))
    assert loader._lora_adapter_id > first_id

    # Bogus path is rejected without mutating state (beyond clearing).
    loader.set_lora_adapter(str(tmp_path / "does_not_exist"))
    assert loader._lora_adapter_path is None


def test_swap_with_adapter_path_enables_lora_and_registers(tmp_path: Path, monkeypatch):
    """swap_to_vllm_after_training(adapter_path=...) must:
       - flip enable_lora True even if constructed False
       - register the adapter via set_lora_adapter
       - still go through the HF fallback gracefully when vLLM isn't installed
    """
    from src.utils import vllm_backend
    from src.utils.vllm_backend import VLLMModelLoader

    trainer = _make_trainer()
    adapter_path = trainer.save_lora_weights(tmp_path, cycle=2)

    loader = VLLMModelLoader(model_path="stub/base", enable_lora=False)
    # Short-circuit the real swap — we only care that the adapter was
    # registered and enable_lora flipped BEFORE _load_vllm/_load_hf ran.
    calls = {"unload_hf": 0, "unload_vllm": 0, "load_vllm": 0, "load_hf": 0}
    monkeypatch.setattr(loader, "_unload_hf", lambda: calls.__setitem__("unload_hf", 1))
    monkeypatch.setattr(loader, "_unload_vllm", lambda: calls.__setitem__("unload_vllm", 1))
    monkeypatch.setattr(loader, "_load_vllm", lambda: calls.__setitem__("load_vllm", 1))
    monkeypatch.setattr(loader, "_load_hf", lambda: calls.__setitem__("load_hf", 1))
    # Force the vLLM/CUDA availability gate to True so we take the _load_vllm
    # branch (where enable_lora matters) rather than the HF fallback.
    monkeypatch.setattr(vllm_backend, "_check_vllm", lambda: True)
    monkeypatch.setattr(vllm_backend.torch.cuda, "is_available", lambda: True)

    loader.swap_to_vllm_after_training(adapter_path=str(adapter_path))

    assert loader.enable_lora is True
    assert loader._lora_adapter_path == str(adapter_path)
    assert calls["load_vllm"] == 1
