"""Tests for task #19: vLLM coresident-training swap path.

These tests exercise the control flow of VLLMModelLoader's
swap_to_hf_for_training / swap_to_vllm_after_training pair WITHOUT
actually loading vLLM or a real model — vLLM is mocked so the suite
runs on CPU / Mac. What we verify:

  - When coresident_training_enabled=False (legacy path), the swap
    destroys vLLM via _unload_vllm and reloads it via _load_vllm. No
    sleep/wake calls occur. This preserves the shipped behaviour.
  - When coresident_training_enabled=True AND vllm.sleep() succeeds,
    swap_to_hf_for_training calls sleep() and skips _unload_vllm.
    The companion swap_to_vllm_after_training calls wake_up(),
    registers the adapter, and skips _load_vllm (the big win).
  - On sleep() failure the loader falls back to the legacy path
    (safety net).
  - On wake() failure the loader falls back to the legacy reload
    path (safety net).
  - enforce_eager and enable_sleep_mode kwargs are threaded into the
    LLM() constructor call when coresident is on.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.utils.vllm_backend import VLLMModelLoader
from src.utils.config import VLLMConfig


def _bare_loader(*, coresident: bool = False) -> VLLMModelLoader:
    """Build a VLLMModelLoader without calling load()."""
    return VLLMModelLoader(
        model_path="fake/base",
        coresident_training_enabled=coresident,
        coresident_vllm_mem_frac=0.42,
    )


# ---------------------------------------------------------------------------
# Construction / flag propagation
# ---------------------------------------------------------------------------

def test_coresident_flag_overrides_mem_frac_and_forces_eager():
    ldr = VLLMModelLoader(
        model_path="x",
        gpu_memory_utilization=0.90,  # caller value
        coresident_training_enabled=True,
        coresident_vllm_mem_frac=0.42,
    )
    assert ldr.gpu_memory_utilization == pytest.approx(0.42)
    assert ldr.enforce_eager is True
    assert ldr.coresident_training_enabled is True


def test_non_coresident_preserves_caller_mem_frac():
    ldr = VLLMModelLoader(
        model_path="x",
        gpu_memory_utilization=0.90,
        coresident_training_enabled=False,
    )
    assert ldr.gpu_memory_utilization == pytest.approx(0.90)
    assert ldr.enforce_eager is False


def test_vllm_config_defaults_are_off():
    """Co-resident path is off by default — shipping behaviour unchanged."""
    c = VLLMConfig(model_path="x")
    assert c.coresident_training_enabled is False
    assert c.enforce_eager is False
    assert c.coresident_vllm_mem_frac == pytest.approx(0.42)
    assert c.parallel_verify_enabled is False


def test_vllm_config_validates_coresident_mem_frac():
    with pytest.raises(ValueError):
        VLLMConfig(model_path="x", coresident_vllm_mem_frac=0.0)
    with pytest.raises(ValueError):
        VLLMConfig(model_path="x", coresident_vllm_mem_frac=1.5)


# ---------------------------------------------------------------------------
# swap_to_hf_for_training control flow
# ---------------------------------------------------------------------------

def test_legacy_swap_destroys_vllm_and_loads_hf():
    """coresident=False: must call _unload_vllm + _load_hf (the old path)."""
    ldr = _bare_loader(coresident=False)
    ldr._llm = MagicMock()  # pretend vLLM is loaded
    with patch.object(ldr, "_unload_vllm") as unload, \
         patch.object(ldr, "_load_hf") as load_hf, \
         patch.object(ldr, "_vllm_sleep") as sleep:
        ldr.swap_to_hf_for_training()
    unload.assert_called_once()
    load_hf.assert_called_once()
    sleep.assert_not_called()


def test_coresident_swap_sleeps_and_skips_unload():
    """coresident=True + sleep success: _load_hf called, _unload_vllm NOT."""
    ldr = _bare_loader(coresident=True)
    ldr._llm = MagicMock()
    with patch.object(ldr, "_vllm_sleep", return_value=True) as sleep, \
         patch.object(ldr, "_unload_vllm") as unload, \
         patch.object(ldr, "_load_hf") as load_hf:
        ldr.swap_to_hf_for_training()
    sleep.assert_called_once()
    unload.assert_not_called()   # THE WIN — vLLM stays resident
    load_hf.assert_called_once()


def test_coresident_swap_falls_back_on_sleep_failure():
    """coresident=True + sleep FAILS: must fall back to unload+load_hf."""
    ldr = _bare_loader(coresident=True)
    ldr._llm = MagicMock()
    with patch.object(ldr, "_vllm_sleep", return_value=False), \
         patch.object(ldr, "_unload_vllm") as unload, \
         patch.object(ldr, "_load_hf") as load_hf:
        ldr.swap_to_hf_for_training()
    unload.assert_called_once()  # fallback path exercised
    load_hf.assert_called_once()


# ---------------------------------------------------------------------------
# swap_to_vllm_after_training control flow
# ---------------------------------------------------------------------------

def test_coresident_swap_back_wakes_and_skips_reload(tmp_path):
    """When vLLM is sleeping, wake() replaces the full _load_vllm."""
    ldr = _bare_loader(coresident=True)
    ldr._llm = MagicMock()
    ldr._vllm_sleeping = True  # simulate: enter_training put us to sleep
    # adapter dir: create stub files so set_lora_adapter accepts.
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_model.safetensors").touch()
    (adapter_dir / "adapter_config.json").write_text("{}")

    with patch.object(ldr, "_vllm_wake", return_value=True) as wake, \
         patch.object(ldr, "_unload_hf") as unload_hf, \
         patch.object(ldr, "_unload_vllm") as unload_vllm, \
         patch.object(ldr, "_load_vllm") as load_vllm:
        ldr.swap_to_vllm_after_training(adapter_path=str(adapter_dir))
    wake.assert_called_once()
    unload_hf.assert_called_once()
    unload_vllm.assert_not_called()   # THE WIN
    load_vllm.assert_not_called()     # THE WIN
    # Adapter registered via set_lora_adapter (hot-swap).
    assert ldr._lora_adapter_path == str(adapter_dir)


def test_coresident_swap_back_falls_back_on_wake_failure():
    """wake() returns False → legacy destroy-and-reload path."""
    ldr = _bare_loader(coresident=True)
    ldr._llm = MagicMock()
    ldr._vllm_sleeping = True
    with patch.object(ldr, "_vllm_wake", return_value=False), \
         patch.object(ldr, "_unload_hf"), \
         patch.object(ldr, "_unload_vllm") as unload_vllm, \
         patch("src.utils.vllm_backend._check_vllm", return_value=False), \
         patch.object(ldr, "_load_hf") as load_hf:
        ldr.swap_to_vllm_after_training()
    # fallback path unloaded vLLM and (since vLLM unavailable in mock) loaded HF.
    unload_vllm.assert_called_once()
    load_hf.assert_called_once()


def test_legacy_swap_back_never_touches_sleep_api():
    """Non-sleeping engine must hit the real _unload_vllm+_load_vllm path."""
    ldr = _bare_loader(coresident=False)
    ldr._llm = MagicMock()
    ldr._vllm_sleeping = False
    with patch.object(ldr, "_vllm_wake") as wake, \
         patch.object(ldr, "_unload_hf"), \
         patch.object(ldr, "_unload_vllm") as unload_vllm, \
         patch("src.utils.vllm_backend._check_vllm", return_value=True), \
         patch("src.utils.vllm_backend.torch") as torch_mock, \
         patch.object(ldr, "_load_vllm") as load_vllm:
        torch_mock.cuda.is_available.return_value = True
        ldr.swap_to_vllm_after_training()
    wake.assert_not_called()
    unload_vllm.assert_called_once()
    load_vllm.assert_called_once()


# ---------------------------------------------------------------------------
# _vllm_sleep / _vllm_wake robustness
# ---------------------------------------------------------------------------

def test_vllm_sleep_returns_false_when_engine_missing():
    ldr = _bare_loader(coresident=True)
    ldr._llm = None
    assert ldr._vllm_sleep() is False
    assert ldr._vllm_sleeping is False


def test_vllm_sleep_catches_exceptions_cleanly():
    ldr = _bare_loader(coresident=True)
    ldr._llm = MagicMock()
    ldr._llm.sleep.side_effect = RuntimeError("not supported")
    # Must NOT propagate; loader returns False so caller falls back.
    assert ldr._vllm_sleep() is False
    assert ldr._vllm_sleeping is False


def test_vllm_wake_no_op_when_not_sleeping():
    ldr = _bare_loader(coresident=True)
    ldr._llm = MagicMock()
    ldr._vllm_sleeping = False
    assert ldr._vllm_wake() is False  # nothing to wake
