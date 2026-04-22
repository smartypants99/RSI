"""AWQ + speculative-decoding plumbing (task #22 followup).

Tests that VLLMConfig and VLLMModelLoader route quantization_scheme and
speculative kwargs correctly to vLLM's LLM() constructor, without
requiring GPU or the real vLLM package.

No GPU, no real model. vLLM's LLM is monkeypatched to capture kwargs.
"""
from __future__ import annotations

import pytest

from src.utils.config import VLLMConfig


# ---------------------------------------------------------------------------
# VLLMConfig validation
# ---------------------------------------------------------------------------


def test_vllm_config_defaults_auto_no_speculative():
    c = VLLMConfig()
    assert c.quantization_scheme == "auto"
    assert c.speculative_draft_model is None
    assert c.num_speculative_tokens == 0


def test_vllm_config_rejects_bad_scheme():
    with pytest.raises(ValueError, match="quantization_scheme"):
        VLLMConfig(quantization_scheme="int8-fake")


def test_vllm_config_rejects_speculative_without_draft():
    with pytest.raises(ValueError, match="speculative_draft_model"):
        VLLMConfig(num_speculative_tokens=5, speculative_draft_model=None)


def test_vllm_config_rejects_speculative_with_bnb():
    """Speculative + bnb is broken in vLLM — config must refuse."""
    with pytest.raises(ValueError, match="incompatible with quantization_scheme='bnb'"):
        VLLMConfig(
            quantization_scheme="bnb",
            speculative_draft_model="Qwen/Qwen2.5-0.5B-Instruct",
            num_speculative_tokens=5,
        )


def test_vllm_config_accepts_awq_with_speculative():
    c = VLLMConfig(
        quantization_scheme="awq",
        speculative_draft_model="Qwen/Qwen2.5-0.5B-Instruct",
        num_speculative_tokens=5,
    )
    assert c.quantization_scheme == "awq"
    assert c.num_speculative_tokens == 5


# ---------------------------------------------------------------------------
# VLLMModelLoader kwarg plumbing (monkeypatch vllm.LLM)
# ---------------------------------------------------------------------------


class _FakeLLM:
    """Captures kwargs passed to vllm.LLM() so tests can assert routing."""
    last_kwargs: dict | None = None

    def __init__(self, **kwargs):
        _FakeLLM.last_kwargs = kwargs

    def get_tokenizer(self):
        class _T:
            pad_token = "[PAD]"
            eos_token = "[EOS]"
        return _T()


@pytest.fixture
def patched_vllm(monkeypatch):
    """Inject a fake vllm module so VLLMModelLoader._load_vllm runs without GPU."""
    import sys
    import types
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = _FakeLLM
    fake_vllm.SamplingParams = type("SamplingParams", (), {})
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    # Force _check_vllm cache to recompute and see the fake
    import src.utils.vllm_backend as vb
    monkeypatch.setattr(vb, "_VLLM_AVAILABLE", True)
    _FakeLLM.last_kwargs = None
    yield


def test_awq_scheme_routes_quantization_awq_no_load_format(patched_vllm):
    from src.utils.vllm_backend import VLLMModelLoader
    loader = VLLMModelLoader(
        model_path="fake/awq-model",
        quantization_scheme="awq",
    )
    loader._load_vllm()
    kw = _FakeLLM.last_kwargs
    assert kw is not None
    assert kw.get("quantization") == "awq"
    assert "load_format" not in kw  # AWQ must NOT use bitsandbytes load_format


def test_bnb_scheme_routes_bitsandbytes(patched_vllm):
    from src.utils.vllm_backend import VLLMModelLoader
    loader = VLLMModelLoader(
        model_path="fake/bnb-model",
        quantization_scheme="bnb",
    )
    loader._load_vllm()
    kw = _FakeLLM.last_kwargs
    assert kw.get("quantization") == "bitsandbytes"
    assert kw.get("load_format") == "bitsandbytes"


def test_auto_scheme_with_load_in_4bit_routes_bnb(patched_vllm):
    """Back-compat: legacy quantization_config={'load_in_4bit': True} still works."""
    from src.utils.vllm_backend import VLLMModelLoader
    loader = VLLMModelLoader(
        model_path="fake/legacy",
        quantization_config={"load_in_4bit": True},
        # quantization_scheme defaults to "auto"
    )
    loader._load_vllm()
    kw = _FakeLLM.last_kwargs
    assert kw.get("quantization") == "bitsandbytes"
    assert kw.get("load_format") == "bitsandbytes"


def test_speculative_kwargs_passed_when_set(patched_vllm):
    from src.utils.vllm_backend import VLLMModelLoader
    loader = VLLMModelLoader(
        model_path="fake/awq-model",
        quantization_scheme="awq",
        speculative_draft_model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        num_speculative_tokens=5,
    )
    loader._load_vllm()
    kw = _FakeLLM.last_kwargs
    assert kw.get("speculative_model") == "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    assert kw.get("num_speculative_tokens") == 5


def test_speculative_kwargs_absent_when_unset(patched_vllm):
    from src.utils.vllm_backend import VLLMModelLoader
    loader = VLLMModelLoader(
        model_path="fake/awq-model",
        quantization_scheme="awq",
    )
    loader._load_vllm()
    kw = _FakeLLM.last_kwargs
    assert "speculative_model" not in kw
    assert "num_speculative_tokens" not in kw
