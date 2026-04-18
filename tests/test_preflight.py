"""Preflight tests — the safety net that catches misconfig before GPU time."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.config import SystemConfig, ModelConfig
from src.utils.preflight import (
    run_preflight, CheckResult, PreflightReport,
    _check_python_version, _check_import, _check_model_path,
    _check_config_coherence,
)


def test_python_version_passes():
    r = _check_python_version()
    assert r.ok, r.message


def test_import_check_detects_missing():
    r = _check_import("this_module_definitely_does_not_exist_xyz", "testing")
    assert not r.ok
    assert "pip install" in r.fix_hint


def test_import_check_finds_existing():
    r = _check_import("json", "testing")
    assert r.ok


def test_model_path_rejects_empty():
    r = _check_model_path("")
    assert not r.ok
    assert "empty" in r.message.lower()


def test_model_path_rejects_nonexistent_local():
    r = _check_model_path("/nonexistent/path/that/should/not/exist")
    assert not r.ok


def test_model_path_accepts_hf_repo_id():
    r = _check_model_path("Qwen/Qwen3-8B")
    assert r.ok
    assert "HF repo id" in r.message


def test_model_path_rejects_missing_config_json(tmp_path):
    d = tmp_path / "fake-model"
    d.mkdir()
    r = _check_model_path(str(d))
    assert not r.ok
    assert "config.json" in r.message


def test_model_path_accepts_proper_dir(tmp_path):
    d = tmp_path / "fake-model"
    d.mkdir()
    (d / "config.json").write_text("{}")
    r = _check_model_path(str(d))
    assert r.ok


def test_config_coherence_catches_mutually_exclusive_quant():
    cfg = SystemConfig()
    cfg.model = ModelConfig(
        model_path="test",
        quantization_config={"load_in_8bit": True, "load_in_4bit": True},
    )
    results = _check_config_coherence(cfg)
    bad = [r for r in results if r.name == "quant_exclusive"]
    assert bad and not bad[0].ok


def test_config_coherence_catches_step_count_mismatch():
    cfg = SystemConfig()
    cfg.generator.min_reasoning_steps = 1
    cfg.verifier.min_chain_steps = 5  # verifier would reject every sample
    results = _check_config_coherence(cfg)
    bad = [r for r in results if r.name == "step_compat"]
    assert bad and not bad[0].ok


def test_report_ok_requires_zero_errors():
    rep = PreflightReport()
    rep.checks.append(CheckResult("ok_one", True, "fine"))
    rep.checks.append(CheckResult("warn", False, "meh", severity="warning"))
    assert rep.ok, "warnings should not block run"
    rep.checks.append(CheckResult("err", False, "bad"))
    assert not rep.ok, "an error must block"


def test_full_preflight_on_default_config(tmp_path):
    """Full preflight runs against a default config without crashing.

    CUDA errors are expected on dev machines — we pass require_cuda=False so
    the report still returns cleanly. What we care about is: no check raises.
    """
    cfg = SystemConfig()
    cfg.model = ModelConfig(model_path="Qwen/Qwen3-8B")
    cfg.orchestrator.output_dir = tmp_path / "out"
    cfg.orchestrator.log_dir = tmp_path / "out" / "logs"
    report = run_preflight(cfg, require_cuda=False)
    assert isinstance(report, PreflightReport)
    # Should have run at least the core checks
    names = {c.name for c in report.checks}
    assert "python_version" in names
    assert "torch" in names
    assert "model_path" in names
    assert "output_dir_writable" in names
