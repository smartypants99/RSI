"""Shared test fixtures for timedilate.

Centralizes the shared `sys.modules["vllm"]` MagicMock so every test
file sees the same object, and resets it between tests to avoid
order-dependent flakes from leaked side_effect / call_args state.
"""
import sys
from unittest.mock import MagicMock

import pytest


def _install_vllm_mock() -> MagicMock:
    existing = sys.modules.get("vllm")
    if isinstance(existing, MagicMock):
        return existing
    mock = MagicMock()
    sys.modules["vllm"] = mock
    return mock


_vllm_mock = _install_vllm_mock()


def _make_default_output(text: str = "ok"):
    out = MagicMock()
    inner = MagicMock()
    inner.text = text
    inner.token_ids = [0, 0, 0]
    out.outputs = [inner]
    out.prompt_token_ids = [0, 0]
    return out


@pytest.fixture(autouse=True)
def _reset_vllm_mock_between_tests():
    """Reset the shared vllm MagicMock before every test.

    Clears side_effect, return_value, and call history on LLM and
    SamplingParams so tests never inherit state from a prior test.
    """
    _vllm_mock.reset_mock(return_value=True, side_effect=True)
    _vllm_mock.LLM.reset_mock(return_value=True, side_effect=True)
    _vllm_mock.SamplingParams.reset_mock(return_value=True, side_effect=True)
    _vllm_mock.LLM.side_effect = None
    _vllm_mock.SamplingParams.side_effect = None

    llm_instance = _vllm_mock.LLM.return_value
    llm_instance.reset_mock(return_value=True, side_effect=True)
    llm_instance.generate.reset_mock(return_value=True, side_effect=True)
    llm_instance.generate.side_effect = None
    llm_instance.generate.return_value = [_make_default_output("ok")]

    yield


@pytest.fixture
def vllm_mock():
    """Direct access to the shared vllm MagicMock."""
    return _vllm_mock


@pytest.fixture
def make_vllm_output():
    """Factory for vllm RequestOutput-shaped mocks."""
    def _factory(text: str = "ok", token_ids=None, prompt_token_ids=None):
        out = MagicMock()
        inner = MagicMock()
        inner.text = text
        inner.token_ids = token_ids if token_ids is not None else [0] * 3
        out.outputs = [inner]
        out.prompt_token_ids = prompt_token_ids if prompt_token_ids is not None else [0] * 2
        return out
    return _factory


@pytest.fixture
def sample_prompt():
    return "Write a Python function that reverses a string."
