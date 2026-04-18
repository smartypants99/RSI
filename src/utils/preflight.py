"""Preflight checks — validate every prerequisite BEFORE GPU work starts.

Philosophy: fail fast, fail loud, fail with a fix. A 3-hour training run
that dies at hour 2 because bitsandbytes wasn't installed is the kind of
pain this module prevents. Every check returns (ok, message) and the
preflight runner aggregates them into a PreflightReport with a clear
pass/fail verdict and remediation hints for anything that failed.

Usage:
    from src.utils.preflight import run_preflight
    report = run_preflight(config)
    if not report.ok:
        report.print_and_exit()  # prints actionable errors and exits 2
"""

from __future__ import annotations

import importlib
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    name: str
    ok: bool
    message: str
    fix_hint: str = ""
    severity: str = "error"  # "error" blocks run; "warning" is informational


@dataclass
class PreflightReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(c.severity == "error" and not c.ok for c in self.checks)

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if c.severity == "warning" and not c.ok]

    @property
    def errors(self) -> list[CheckResult]:
        return [c for c in self.checks if c.severity == "error" and not c.ok]

    def print_summary(self, stream=None) -> None:
        import io
        stream = stream or sys.stderr
        print("=" * 60, file=stream)
        print("PREFLIGHT CHECK", file=stream)
        print("=" * 60, file=stream)
        for c in self.checks:
            icon = "✓" if c.ok else ("✗" if c.severity == "error" else "⚠")
            print(f"  {icon} {c.name}: {c.message}", file=stream)
            if not c.ok and c.fix_hint:
                print(f"    → fix: {c.fix_hint}", file=stream)
        print("-" * 60, file=stream)
        if self.ok:
            print(f"PASS — {len(self.checks) - len(self.warnings)} checks ok, "
                  f"{len(self.warnings)} warnings", file=stream)
        else:
            print(f"FAIL — {len(self.errors)} error(s), {len(self.warnings)} warning(s)",
                  file=stream)
        print("=" * 60, file=stream)

    def print_and_exit(self) -> None:
        self.print_summary()
        if not self.ok:
            sys.exit(2)


# ── individual checks ──────────────────────────────────────────────────


def _check_python_version() -> CheckResult:
    """Require Python 3.9+ (future annotations used throughout)."""
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 9):
        return CheckResult(
            "python_version", False,
            f"Python {v.major}.{v.minor} detected; need >= 3.9",
            "Install Python 3.10+ via pyenv or conda",
        )
    return CheckResult("python_version", True, f"Python {v.major}.{v.minor}.{v.micro}")


def _check_import(module_name: str, required_for: str,
                  severity: str = "error") -> CheckResult:
    try:
        importlib.import_module(module_name)
        return CheckResult(f"import_{module_name}", True, f"{module_name} available")
    except ImportError as e:
        return CheckResult(
            f"import_{module_name}", False,
            f"{module_name} not installed — required for {required_for}",
            f"pip install {module_name}",
            severity=severity,
        )


def _check_torch_cuda(require_cuda: bool = True) -> list[CheckResult]:
    """Verify torch + CUDA + enough VRAM. Real GPU constraints, not guesses."""
    results = []
    try:
        import torch
        results.append(CheckResult("torch", True, f"torch {torch.__version__}"))
    except ImportError:
        return [CheckResult(
            "torch", False, "torch not installed",
            "pip install 'torch>=2.0'",
        )]

    if not torch.cuda.is_available():
        severity = "error" if require_cuda else "warning"
        return results + [CheckResult(
            "cuda", False,
            "CUDA not available; training will be infeasibly slow on CPU",
            "Install CUDA-enabled torch: pip install torch --index-url "
            "https://download.pytorch.org/whl/cu121",
            severity=severity,
        )]

    try:
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / (1024 ** 3)
        results.append(CheckResult(
            "cuda", True,
            f"{name}, {total_gb:.0f}GB total VRAM, "
            f"capability {props.major}.{props.minor}",
        ))

        # A6000 = 48GB, A100 = 40/80, 4090 = 24. Below 16GB, 8B training is
        # not realistic even with 4-bit quantization.
        if total_gb < 16:
            results.append(CheckResult(
                "vram_sufficient", False,
                f"Only {total_gb:.0f}GB VRAM; 8B model training needs 24GB+",
                "Use --load-in-4bit or switch to a smaller model",
                severity="warning",
            ))

        # Free VRAM check — catches "old training process still loaded"
        free, _ = torch.cuda.mem_get_info()
        free_gb = free / (1024 ** 3)
        if free_gb < total_gb * 0.8:
            results.append(CheckResult(
                "vram_free", False,
                f"Only {free_gb:.1f}GB/{total_gb:.0f}GB VRAM free — another "
                f"process is holding memory",
                "nvidia-smi | grep python  # find and kill stale processes",
                severity="warning",
            ))
        else:
            results.append(CheckResult(
                "vram_free", True, f"{free_gb:.1f}GB/{total_gb:.0f}GB VRAM free",
            ))
    except Exception as e:
        results.append(CheckResult(
            "cuda_info", False, f"Could not query CUDA device: {e}",
            "Check nvidia-smi works and CUDA drivers are current",
            severity="warning",
        ))

    return results


def _check_model_path(path: str) -> CheckResult:
    """Validate model path is either local-accessible or a HF-style repo id."""
    if not path:
        return CheckResult(
            "model_path", False, "model path is empty",
            "Pass --model <path_or_hf_id>",
        )
    # Local path
    p = Path(path)
    if p.exists():
        if not p.is_dir():
            return CheckResult(
                "model_path", False,
                f"{path} exists but is not a directory",
                "Point --model at the model directory containing config.json",
            )
        if not (p / "config.json").exists():
            return CheckResult(
                "model_path", False,
                f"{path}/config.json not found — not a HF model directory",
                "Ensure the path contains config.json, tokenizer files, and weights",
            )
        return CheckResult("model_path", True, f"local model at {path}")

    # HF repo id — loose check (org/name shape)
    if "/" in path and len(path.split("/")) == 2:
        return CheckResult(
            "model_path", True,
            f"HF repo id {path} (will be downloaded if not cached)",
        )
    return CheckResult(
        "model_path", False,
        f"{path} does not exist locally and doesn't look like 'org/name' HF id",
        "Use a local path or a HF repo id like 'Qwen/Qwen3-8B'",
    )


def _check_output_dir(path: Path) -> list[CheckResult]:
    """Output dir must be creatable + have enough free space for checkpoints."""
    results = []
    try:
        path.mkdir(parents=True, exist_ok=True)
        # Writability
        test_file = path / ".preflight_write_test"
        try:
            test_file.write_text("ok")
            test_file.unlink()
            results.append(CheckResult("output_dir_writable", True, f"{path} writable"))
        except OSError as e:
            results.append(CheckResult(
                "output_dir_writable", False, f"cannot write to {path}: {e}",
                "Pick an --output-dir you have write permission for",
            ))
            return results
    except OSError as e:
        results.append(CheckResult(
            "output_dir_create", False, f"cannot create {path}: {e}",
            "Pick a writable --output-dir path",
        ))
        return results

    # Disk space. Each 8B checkpoint = ~16GB; keeping 2 + best = ~50GB headroom.
    try:
        usage = shutil.disk_usage(path)
        free_gb = usage.free / (1024 ** 3)
        if free_gb < 20:
            results.append(CheckResult(
                "disk_space", False,
                f"Only {free_gb:.1f}GB free at {path} — 8B checkpoints are ~16GB each",
                "Free disk space or pick a roomier --output-dir",
                severity="warning" if free_gb >= 5 else "error",
            ))
        else:
            results.append(CheckResult(
                "disk_space", True, f"{free_gb:.0f}GB free at {path}",
            ))
    except OSError:
        # Non-fatal; some filesystems don't report
        pass
    return results


def _check_config_coherence(config) -> list[CheckResult]:
    """Catch config combinations that will fail at runtime."""
    results = []
    q = getattr(config.model, "quantization_config", None) or {}

    # 8bit + 4bit are mutually exclusive — validators in main.py already block
    # this, but double-check in case the config was built programmatically.
    if q.get("load_in_8bit") and q.get("load_in_4bit"):
        results.append(CheckResult(
            "quant_exclusive", False,
            "both load_in_8bit and load_in_4bit set",
            "Pass only one of --load-in-8bit / --load-in-4bit",
        ))
    else:
        results.append(CheckResult("quant_exclusive", True, "quantization config ok"))

    # Quantization requires bitsandbytes
    if q.get("load_in_8bit") or q.get("load_in_4bit"):
        bnb = _check_import("bitsandbytes", "8bit/4bit quantization")
        if not bnb.ok:
            bnb.fix_hint = "pip install bitsandbytes"
            results.append(bnb)

    # vLLM mode needs vLLM
    if getattr(config, "use_vllm", False):
        vllm_check = _check_import("vllm", "fast inference", severity="error")
        results.append(vllm_check)

        vcfg = getattr(config, "vllm", None)
        if vcfg is None:
            results.append(CheckResult(
                "vllm_config", False,
                "use_vllm=True but config.vllm is None",
                "Build SystemConfig.vllm from VLLMConfig",
            ))
        else:
            gpu_frac = getattr(vcfg, "gpu_memory_utilization", 0.90)
            if gpu_frac > 0.95:
                results.append(CheckResult(
                    "vllm_gpu_frac", False,
                    f"gpu_memory_utilization={gpu_frac} leaves no headroom for the "
                    f"HF swap during training",
                    "Use --gpu-memory-utilization 0.85 or lower when training alongside",
                    severity="warning",
                ))

    # Clamp invariants
    d = config.diagnostics
    if d.min_questions_per_domain > d.max_questions_per_domain:
        results.append(CheckResult(
            "diagnostics_clamp", False,
            f"min_questions_per_domain ({d.min_questions_per_domain}) > "
            f"max_questions_per_domain ({d.max_questions_per_domain})",
            "Adjust DiagnosticsConfig so min <= max",
        ))

    # Verifier/generator compatibility
    if config.generator.min_reasoning_steps < config.verifier.min_chain_steps:
        results.append(CheckResult(
            "step_compat", False,
            f"generator.min_reasoning_steps ({config.generator.min_reasoning_steps}) < "
            f"verifier.min_chain_steps ({config.verifier.min_chain_steps}) — verifier "
            f"will reject every sample",
            "Raise --gen-min-steps or lower verifier min_chain_steps",
        ))

    # Trainer coherence
    t = config.trainer
    if t.lora_rank < t.min_rank:
        results.append(CheckResult(
            "lora_rank_vs_min", False,
            f"lora_rank ({t.lora_rank}) < min_rank ({t.min_rank})",
            "Raise --lora-rank or lower TrainerConfig.min_rank",
        ))

    # GRPO requires reference model (LoRA-zeroed); incompatible with 4bit because
    # quantized base can't be cleanly zeroed out for the reference pass
    mode = getattr(t, "training_mode", "sft")
    if mode in ("dpo", "grpo", "mixed") and q.get("load_in_4bit"):
        results.append(CheckResult(
            "rl_vs_4bit", False,
            f"training_mode={mode} needs clean reference forward; 4-bit quantization "
            f"breaks LoRA-zeroing",
            "Drop --load-in-4bit or use --training-mode sft",
            severity="warning",
        ))

    return results


# ── orchestrator ────────────────────────────────────────────────────────


def run_preflight(config, *, require_cuda: bool = True) -> PreflightReport:
    """Run every check against `config` and return a PreflightReport.

    Set require_cuda=False when validating a config on a dev machine (tests,
    CI). Otherwise CUDA-not-available is an error, because 8B CPU training
    is not feasible.
    """
    report = PreflightReport()

    report.checks.append(_check_python_version())
    report.checks.append(_check_import("torch", "everything"))
    report.checks.append(_check_import("transformers", "model loading"))
    report.checks.append(_check_import("sympy", "math verification"))

    # CUDA/VRAM family
    report.checks.extend(_check_torch_cuda(require_cuda=require_cuda))

    # Paths
    report.checks.append(_check_model_path(config.model.model_path))
    report.checks.extend(_check_output_dir(Path(config.orchestrator.output_dir)))

    # Cross-field config coherence
    report.checks.extend(_check_config_coherence(config))

    return report
