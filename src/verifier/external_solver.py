"""Shared interface for external formal verifiers (Z3 SMT, Lean4, simulators).

Each backend implements ExternalSolver. SolverResult is the normalized return
shape — backends map native solver outcomes into {sat, unsat, unknown, timeout,
error} and optionally return a variable assignment (model) when sat.

Coordination note: this file is touched by verifier-z3, verifier-lean, and
verifier-sim (see TEAM_RSI_FOOM_PROTOCOL.md §File ownership). Any additive
change is fine; breaking the Protocol requires SendMessage to both peers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass
class SolverResult:
    """Normalized result from an external solver.

    status values:
      - "sat"      : constraints satisfiable; `model` holds witness assignment
      - "unsat"    : constraints unsatisfiable (disproved)
      - "unknown"  : solver could not decide within resources
      - "timeout"  : wall-clock limit reached
      - "error"    : malformed input, parse failure, solver crash — see detail
    """
    status: str
    model: Optional[dict[str, Any]] = None
    detail: str = ""
    elapsed_s: float = 0.0
    backend: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in ("sat", "unsat")


_VALID_STATUSES = frozenset({"sat", "unsat", "unknown", "timeout", "error"})


def make_result(
    status: str,
    *,
    model: Optional[dict[str, Any]] = None,
    detail: str = "",
    elapsed_s: float = 0.0,
    backend: str = "",
    meta: Optional[dict[str, Any]] = None,
) -> SolverResult:
    if status not in _VALID_STATUSES:
        raise ValueError(f"status must be in {sorted(_VALID_STATUSES)}, got {status!r}")
    return SolverResult(
        status=status,
        model=model,
        detail=detail,
        elapsed_s=elapsed_s,
        backend=backend,
        meta=meta or {},
    )


@runtime_checkable
class ExternalSolver(Protocol):
    """Any formal solver (Z3, Lean, physics simulator) conforms to this."""

    name: str

    def check(self, problem: str, timeout_s: float = 5.0) -> SolverResult:
        """Decide a problem description. Must not raise on solver failure —
        return SolverResult(status="error", detail=...) instead."""
        ...


# ═════════════════════════════════════════════════════════════════════════
# Shared subprocess helper — used by lean/z3/sim backends to shell out to
# an external CLI with uniform timeout, scrubbed env, ephemeral cwd, and
# missing-binary detection. Added for task #3 after SendMessage coordination
# with verifier-z3 and verifier-sim. Additive — does not break the
# ExternalSolver protocol above.
# ═════════════════════════════════════════════════════════════════════════

import logging as _logging   # noqa: E402  (module-scope aliases after dataclasses)
import os as _os             # noqa: E402
import shutil as _shutil     # noqa: E402
import subprocess as _subprocess  # noqa: E402
import tempfile as _tempfile # noqa: E402
import time as _time         # noqa: E402

_logger = _logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 10.0
_MAX_TIMEOUT_S = 60.0


@dataclass
class SubprocessOutcome:
    """Raw subprocess result, backend-agnostic. Backends translate this into
    the normalized SolverResult above (e.g. Lean returncode==0 → status='unsat'
    for a theorem being proved, i.e. 'no counterexample')."""
    returncode: int
    stdout: str
    stderr: str
    elapsed_s: float
    timed_out: bool = False
    missing_binary: bool = False

    @property
    def ok(self) -> bool:
        return (
            not self.timed_out
            and not self.missing_binary
            and self.returncode == 0
        )

    def tail(self, limit: int = 500) -> str:
        blob = self.stderr or self.stdout or ""
        return blob[-limit:]


def _scrubbed_env(extra: Optional[dict] = None) -> dict:
    keep = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "USER",
            "ELAN_HOME", "LEAN_PATH", "LAKE_HOME")
    env = {k: _os.environ[k] for k in keep if k in _os.environ}
    if extra:
        env.update(extra)
    return env


def run_subprocess_verifier(
    cmd: list[str],
    input_file: Optional[str] = None,
    stdin: Optional[str] = None,
    timeout: float = _DEFAULT_TIMEOUT_S,
    extra_env: Optional[dict] = None,
) -> SubprocessOutcome:
    """Invoke `cmd` with timeout + scrubbed env + ephemeral cwd.

    If `input_file` is given, it's appended to argv (e.g. `lean foo.lean`).
    If `stdin` is given, it's fed to the child's stdin.
    Never raises on tool absence — check `.missing_binary` and warn.
    """
    if not cmd:
        return SubprocessOutcome(-1, "", "empty cmd", 0.0, missing_binary=True)
    timeout = max(0.1, min(float(timeout), _MAX_TIMEOUT_S))

    if _shutil.which(cmd[0]) is None:
        return SubprocessOutcome(
            returncode=-1, stdout="", stderr=f"binary not found: {cmd[0]}",
            elapsed_s=0.0, missing_binary=True,
        )

    full_cmd = list(cmd)
    if input_file is not None:
        full_cmd.append(input_file)

    t0 = _time.monotonic()
    with _tempfile.TemporaryDirectory(prefix="extsolver_") as cwd:
        try:
            proc = _subprocess.run(
                full_cmd,
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                env=_scrubbed_env(extra_env),
                check=False,
            )
        except _subprocess.TimeoutExpired as e:
            return SubprocessOutcome(
                returncode=-1,
                stdout=(e.stdout.decode("utf-8", "replace")
                        if isinstance(e.stdout, (bytes, bytearray))
                        else (e.stdout or "")),
                stderr=(e.stderr.decode("utf-8", "replace")
                        if isinstance(e.stderr, (bytes, bytearray))
                        else (e.stderr or f"timeout after {timeout}s")),
                elapsed_s=_time.monotonic() - t0,
                timed_out=True,
            )
        except FileNotFoundError as e:
            return SubprocessOutcome(-1, "", str(e),
                                     _time.monotonic() - t0, missing_binary=True)
        except OSError as e:
            return SubprocessOutcome(-1, "", f"OSError: {e}",
                                     _time.monotonic() - t0)

    return SubprocessOutcome(
        returncode=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        elapsed_s=_time.monotonic() - t0,
    )


def binary_available(binary: str) -> bool:
    """Cheap install check for backend __init__."""
    return _shutil.which(binary) is not None


__all__ = [
    "SolverResult", "make_result", "ExternalSolver",
    "SubprocessOutcome", "run_subprocess_verifier", "binary_available",
]
