"""Property engine — spec v0.2.1 18-field Property dataclass, §1.3 admission
gates (4 gates, no fuzz — fuzz is VoV/bundle-level per v0.2.1 P6), and §2.1
quorum-backed verify() that produces VerificationRecord.

Two concurrent APIs live here:

  RSI (canonical, v0.2.1):
    - Property            — 18-field dataclass per §1.1
    - PropertyKind        — 10-value enum per §1.2
    - INDEPENDENCE_CLASSES— 10-value frozenset per §2.2 (incl. dimensional.physical)
    - AdmissionResult     — return shape of admit(prop, executor)
    - admit()             — §1.3 gates 1..4
    - VerificationRecord  — §4.1 artifact
    - PropertyVerdict     — per-property row inside VerificationRecord
    - verify()            — runs admitted properties on a candidate, computes
                            §2.1 quorum via VoV.quorum_verdict, returns Record

  Pre-RSI simple-registry SHIM (kept for tests/test_rsi_actual.py):
    - PropertyFn type alias
    - _REGISTRY, register_property(), verify_by_consensus(samples, threshold)
    - emits DeprecationWarning on first use; slated for removal in Phase E
      once the new Property-shape tests replace legacy coverage.

Design notes
------------
* `check_fn` is NOT a schema field. For trusted builtins we materialize
  the callable lazily and stash it on a registry-side map (`_TRUSTED_CHECK_FNS`).
  Model-authored properties materialize via executor (tester's A3 sandbox).
* `confirmer_example` / `falsifier_example` are BUNDLE-EMIT artifacts
  per v0.2.1 §1.3 — library registrations may carry empty placeholders;
  admit() rejects with gate_failed="self_test" if still empty.
* `trusted: bool` lives on the registry entry, NOT on Property (architect Q1(a)).
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import logging
import re
import time
import uuid
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Literal, Optional, Protocol

from ..utils.sandbox import run_python_sandboxed
from ..utils.sympy_utils import HAS_SYMPY

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════
# §1.2 PropertyKind enum
# ═════════════════════════════════════════════════════════════════════════

class PropertyKind(str, Enum):
    """10 kinds per rsi_design.md §1.2. str subclass → jsonl-friendly."""
    ALGEBRAIC = "ALGEBRAIC"
    ROUNDTRIP = "ROUNDTRIP"
    POSTCONDITION = "POSTCONDITION"
    UNIT_TEST = "UNIT_TEST"
    TYPE_INVARIANT = "TYPE_INVARIANT"
    DIMENSIONAL = "DIMENSIONAL"
    REFORMULATION = "REFORMULATION"
    MONOTONICITY = "MONOTONICITY"
    CONSERVATION = "CONSERVATION"
    COUNTEREXAMPLE_SEARCH = "COUNTEREXAMPLE_SEARCH"


# ═════════════════════════════════════════════════════════════════════════
# §2.2 independence classes — 10 values incl. dimensional.physical (v0.2.1)
# ═════════════════════════════════════════════════════════════════════════

INDEPENDENCE_CLASSES: frozenset[str] = frozenset({
    "exec.behavioral",
    "algebra.symbolic",
    "smt.logical",
    "structural.static",
    "transform.semantic",
    "roundtrip",
    "perturbation.local",
    "conservation.global",
    "search.bounded",
    "dimensional.physical",   # v0.2.1 addition
})

_VALID_RETURNS = frozenset({"bool", "bool_with_witness", "equivalence_class"})
_VALID_LANGUAGES = frozenset({"python", "sympy", "z3", "unit_test", "nl_reformulation"})
_MAX_SOURCE_BYTES = 4096           # v0.2.1 §1.1 "≤ 4 kB" — bytes
_MAX_TIMEOUT_MS = 10000
_SHA256_HEX_LEN = 64


# ═════════════════════════════════════════════════════════════════════════
# §1.1 Property — the 18-field canonical dataclass
# ═════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Property:
    """A verifiable property per rsi_design.md v0.2.1 §1.1.

    Frozen because Property is an immutable record once admitted.
    confirmer_example and falsifier_example are BUNDLE-EMIT-time artifacts
    (v0.2.1 §1.3): library templates may carry empty strings at registration;
    task_synthesizer fills them per-problem before calling admit(). admit()
    rejects with gate_failed="self_test" if either is still empty.
    """
    # Identity
    property_id: str
    problem_id: str
    author: str
    # Semantics
    name: str
    kind: PropertyKind
    description: str
    # Execution
    language: str
    source: str
    entry_point: str
    timeout_ms: int
    deterministic: bool
    # Harness contract
    inputs: tuple[str, ...]
    returns: str
    # Epistemology
    independence_class: str
    difficulty_floor: float
    falsifier_example: str
    confirmer_example: str
    # Provenance
    created_at: float
    parent_problem_hash: str

    def __post_init__(self) -> None:
        for fname in (
            "property_id", "problem_id", "author", "name", "description",
            "language", "source", "entry_point", "parent_problem_hash",
        ):
            v = getattr(self, fname)
            if not isinstance(v, str) or not v:
                raise ValueError(f"Property.{fname} must be a non-empty str (got {v!r})")
        if not isinstance(self.kind, PropertyKind):
            raise ValueError(f"Property.kind must be a PropertyKind (got {type(self.kind).__name__})")
        if not isinstance(self.timeout_ms, int) or not (1 <= self.timeout_ms <= _MAX_TIMEOUT_MS):
            raise ValueError(f"Property.timeout_ms must be int in [1, {_MAX_TIMEOUT_MS}]")
        if len(self.source.encode("utf-8")) > _MAX_SOURCE_BYTES:
            raise ValueError(f"Property.source must be ≤ {_MAX_SOURCE_BYTES} bytes")
        if not isinstance(self.deterministic, bool):
            raise ValueError("Property.deterministic must be bool")
        if not isinstance(self.inputs, tuple) or not all(isinstance(x, str) for x in self.inputs):
            raise ValueError("Property.inputs must be tuple[str, ...]")
        if self.returns not in _VALID_RETURNS:
            raise ValueError(f"Property.returns must be in {sorted(_VALID_RETURNS)}")
        if self.language not in _VALID_LANGUAGES:
            raise ValueError(f"Property.language must be in {sorted(_VALID_LANGUAGES)}")
        if self.independence_class not in INDEPENDENCE_CLASSES:
            raise ValueError(f"Property.independence_class must be in {sorted(INDEPENDENCE_CLASSES)}")
        if not isinstance(self.difficulty_floor, (int, float)) or not (0.0 <= float(self.difficulty_floor) <= 1.0):
            raise ValueError("Property.difficulty_floor must be float in [0,1]")
        if not isinstance(self.falsifier_example, str) or not isinstance(self.confirmer_example, str):
            raise ValueError("Property.{falsifier,confirmer}_example must be str (may be empty at registration)")
        if not (
            len(self.parent_problem_hash) == _SHA256_HEX_LEN
            and all(c in "0123456789abcdef" for c in self.parent_problem_hash.lower())
        ):
            raise ValueError("Property.parent_problem_hash must be 64 hex chars")
        if not isinstance(self.created_at, (int, float)) or float(self.created_at) <= 0.0:
            raise ValueError("Property.created_at must be positive float unix ts")

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["kind"] = self.kind.value if isinstance(self.kind, PropertyKind) else str(self.kind)
        d["inputs"] = list(self.inputs)
        return d


# ═════════════════════════════════════════════════════════════════════════
# LegacyProperty — v0.1 8-field bridge (architect OPTION B sign-off)
#
# This is the deprecated 8-field class preserved ONLY so legacy callers and
# tests that haven't migrated to v0.2.1 shape can still import *something*
# from this module. All new code must use Property (the canonical 18-field
# dataclass above). Emits DeprecationWarning on construction. Will be
# removed in Phase E once task #12 lands.
#
# Field list matches the v0.1 shape exactly so legacy tests keep their
# assertions working: name, domain, description, check_fn, stochasticity,
# required, weight, independence_class.
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class LegacyProperty:
    """Deprecated v0.1 Property shape. Use the canonical `Property` class instead.

    Preserved per architect v0.2.1 OPTION B for legacy callers and tests;
    scheduled for removal in Phase E (task #12 migrates tests).
    See rsi_design.md §1.1 for the canonical 18-field schema.
    """
    name: str
    domain: str
    description: str
    check_fn: Callable[..., Any]
    stochasticity: float = 0.0
    required: bool = False
    weight: float = 1.0
    independence_class: str = ""

    def __post_init__(self) -> None:
        warnings.warn(
            "LegacyProperty is deprecated; use v0.2.1 Property from rsi_design.md §1.1. "
            "Scheduled for removal in Phase E.",
            DeprecationWarning, stacklevel=2,
        )


# ═════════════════════════════════════════════════════════════════════════
# Builder & trusted-builtin registry (architect Q1(a))
# ═════════════════════════════════════════════════════════════════════════

_TRUSTED_CHECK_FNS: dict[str, Callable[..., Any]] = {}


def _now_ts() -> float:
    return time.time()


def _sha256_hex(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def build_property(
    *,
    name: str,
    kind: PropertyKind,
    description: str,
    independence_class: str,
    source: str,
    entry_point: str,
    author: str,
    problem_id: str,
    parent_problem_hash: str,
    language: str = "python",
    timeout_ms: int = 2000,
    deterministic: bool = True,
    inputs: tuple[str, ...] = ("problem", "candidate"),
    returns: str = "bool",
    difficulty_floor: float = 0.5,
    confirmer_example: str = "",
    falsifier_example: str = "",
    property_id: Optional[str] = None,
    created_at: Optional[float] = None,
    trusted: bool = False,
    trusted_check_fn: Optional[Callable[..., Any]] = None,
) -> Property:
    """Factory for v0.2.1 Property.

    `trusted=True` registers a check callable so admit() can skip gates 1+2.
    """
    pid = property_id or f"prop_{uuid.uuid4().hex[:16]}"
    prop = Property(
        property_id=pid,
        problem_id=problem_id,
        author=author,
        name=name,
        kind=kind,
        description=description,
        language=language,
        source=source,
        entry_point=entry_point,
        timeout_ms=int(timeout_ms),
        deterministic=bool(deterministic),
        inputs=tuple(inputs),
        returns=returns,
        independence_class=independence_class,
        difficulty_floor=float(difficulty_floor),
        falsifier_example=falsifier_example,
        confirmer_example=confirmer_example,
        created_at=float(created_at if created_at is not None else _now_ts()),
        parent_problem_hash=parent_problem_hash,
    )
    if trusted:
        if trusted_check_fn is None:
            raise ValueError("trusted=True requires trusted_check_fn")
        _TRUSTED_CHECK_FNS[pid] = trusted_check_fn
    return prop


def is_trusted(prop: Property) -> bool:
    return prop.property_id in _TRUSTED_CHECK_FNS


# ═════════════════════════════════════════════════════════════════════════
# §1.3 admission: executor Protocol, AdmissionResult, admit()
# ═════════════════════════════════════════════════════════════════════════

Verdict = Literal["PASS", "FAIL", "ERROR"]


class SandboxExecutor(Protocol):
    """Protocol for materializing Property.source → callable and invoking it.

    Real implementation: wraps src/utils/sandbox.run_python_sandboxed (A3).
    Mock implementation for unit tests: direct in-process invocation.
    """
    def materialize(self, prop: "Property") -> Callable[..., Any]: ...
    def invoke(
        self, callable_: Callable[..., Any], inputs_kwargs: dict[str, Any],
        *, timeout_ms: int,
    ) -> tuple[Verdict, str]: ...


@dataclass(frozen=True)
class AdmissionResult:
    """Return shape for admit(). v0.2.1 P6 — no fuzz fields."""
    admitted: bool
    reason: str
    gate_failed: Optional[str]  # Literal[...]: bounded_cost|sandbox_materialize|self_test|determinism
    determinism_observed: Optional[bool] = None
    self_test_verdicts: tuple[str, str] = ("", "")


def admit(prop: Property, *, executor: SandboxExecutor) -> AdmissionResult:
    """Run §1.3 gates 1–4 on a single Property. No fuzz (v0.2.1 P6).

    Trusted builtins skip gates 1 & 2 but still run 3 & 4.
    """
    trusted = is_trusted(prop)

    # Gate 1: bounded cost.
    if not trusted:
        if prop.timeout_ms > _MAX_TIMEOUT_MS:
            return AdmissionResult(False, f"timeout_ms {prop.timeout_ms} > {_MAX_TIMEOUT_MS}", "bounded_cost")
        if len(prop.source.encode("utf-8")) > _MAX_SOURCE_BYTES:
            return AdmissionResult(False, f"source > {_MAX_SOURCE_BYTES} bytes", "bounded_cost")
        if prop.language not in _VALID_LANGUAGES:
            return AdmissionResult(False, f"language {prop.language!r} not in allow-list", "bounded_cost")

    # Gate 2: sandbox-materialize.
    try:
        if trusted:
            callable_ = _TRUSTED_CHECK_FNS[prop.property_id]
        else:
            callable_ = executor.materialize(prop)
    except Exception as e:
        return AdmissionResult(
            False, f"materialize failed: {type(e).__name__}: {e}", "sandbox_materialize"
        )

    # Gate 3: self-test. Empty confirmer/falsifier fail at this gate.
    if not prop.confirmer_example or not prop.falsifier_example:
        return AdmissionResult(
            False,
            "confirmer_example or falsifier_example empty (stamp per-problem at bundle-emit)",
            "self_test",
            self_test_verdicts=("", ""),
        )
    try:
        conf_verdict, conf_reason = _invoke_callable(
            callable_, prop, prop.confirmer_example, executor,
        )
        fals_verdict, fals_reason = _invoke_callable(
            callable_, prop, prop.falsifier_example, executor,
        )
    except Exception as e:
        return AdmissionResult(
            False, f"self-test raised: {type(e).__name__}: {e}", "self_test",
        )
    if conf_verdict != "PASS":
        return AdmissionResult(
            False,
            f"confirmer verdict={conf_verdict} (expected PASS): {conf_reason}",
            "self_test",
            self_test_verdicts=(conf_verdict, fals_verdict),
        )
    if fals_verdict != "FAIL":
        return AdmissionResult(
            False,
            f"falsifier verdict={fals_verdict} (expected FAIL): {fals_reason}",
            "self_test",
            self_test_verdicts=(conf_verdict, fals_verdict),
        )

    # Gate 4: determinism (only if deterministic=True).
    determinism_observed: Optional[bool] = None
    if prop.deterministic:
        try:
            rerun_verdict, _ = _invoke_callable(
                callable_, prop, prop.confirmer_example, executor,
            )
        except Exception as e:
            return AdmissionResult(
                False, f"determinism replay raised: {type(e).__name__}: {e}", "determinism",
                self_test_verdicts=(conf_verdict, fals_verdict),
            )
        determinism_observed = (rerun_verdict == conf_verdict)
        if not determinism_observed:
            return AdmissionResult(
                False,
                f"non-deterministic: first={conf_verdict}, replay={rerun_verdict}",
                "determinism",
                determinism_observed=False,
                self_test_verdicts=(conf_verdict, fals_verdict),
            )

    return AdmissionResult(
        True, "admitted", None,
        determinism_observed=determinism_observed,
        self_test_verdicts=(conf_verdict, fals_verdict),
    )


def _invoke_callable(
    callable_: Callable[..., Any],
    prop: Property,
    candidate: Any,
    executor: SandboxExecutor,
    *,
    runtime_problem_id: Optional[str] = None,
) -> tuple[Verdict, str]:
    """Adapt a materialized callable to the tri-state Verdict output.

    `runtime_problem_id`: the problem_id being CURRENTLY verified (from the
    verify() caller). Trusted builtin check_fns need this to look up the
    problem's runtime context (tests, reference, etc.) from stash_problem_ctx.
    The legacy behavior passed prop.problem_id — fine for model-authored
    properties bound to a specific problem, but broken for builtin templates
    whose problem_id is a constant like "builtin:template".
    """
    # Pass the runtime problem id when available; fall back to the property's
    # bound problem_id for older model-authored paths.
    pid_for_call = runtime_problem_id or prop.problem_id
    inputs_kwargs = {"problem": pid_for_call, "candidate": candidate}
    inputs_kwargs = {k: v for k, v in inputs_kwargs.items() if k in prop.inputs}
    if is_trusted(prop):
        # Trusted callables are our own Python — we pass positional args in the
        # order declared by Property.inputs. Kwargs-style for model-authored.
        args = tuple(inputs_kwargs.get(name) for name in prop.inputs)
        try:
            result = callable_(*args) if args else callable_(pid_for_call, candidate)
            return _coerce_verdict(result)
        except Exception as e:
            return "ERROR", f"{type(e).__name__}: {e}"
    return executor.invoke(callable_, inputs_kwargs, timeout_ms=prop.timeout_ms)


def _coerce_verdict(result: Any) -> tuple[Verdict, str]:
    """Map callable return values to the tri-state Verdict.

    - bool or (bool, reason) → PASS/FAIL
    - "PASS"/"FAIL"/"ERROR" string → that verdict
    - anything else → ERROR (does not poison quorum)
    """
    if isinstance(result, tuple) and len(result) == 2:
        ok, reason = result
        if isinstance(ok, bool):
            return ("PASS" if ok else "FAIL", str(reason))
        if isinstance(ok, str) and ok.upper() in ("PASS", "FAIL", "ERROR"):
            return (ok.upper(), str(reason))  # type: ignore[return-value]
    if isinstance(result, bool):
        return ("PASS" if result else "FAIL", "")
    if isinstance(result, str) and result.upper() in ("PASS", "FAIL", "ERROR"):
        return (result.upper(), "")  # type: ignore[return-value]
    return "ERROR", f"unrecognized return type: {type(result).__name__}"


# ═════════════════════════════════════════════════════════════════════════
# MockExecutor — for unit tests and pre-sandbox iteration
# ═════════════════════════════════════════════════════════════════════════

class SandboxedExecutor:
    """Production executor backing §1.3 gate 2 on tester's hardened sandbox.

    Each invoke() runs a self-contained Python program in a subprocess
    (utils/sandbox.run_python_sandboxed) that:
      1. defines the property's source (model-authored or trusted)
      2. receives inputs via literal-repr injection
      3. calls entry_point(**inputs)
      4. prints a single sentinel line starting with "__VERIFY__ " followed by
         one of PASS/FAIL/ERROR and an optional reason

    Sandboxed stdout is parsed for that sentinel; any other exit condition
    (timeout, RLIMIT hit, audit-hook deny, non-zero exit) becomes ERROR. An
    ERROR does NOT poison quorum per v0.2.1 P3.

    Determinism of sandbox-side code is the property author's responsibility;
    we only expose the sandbox's wall-clock timeout_s (derived from
    Property.timeout_ms, capped at _MAX_TIMEOUT_MS / 1000).
    """

    _SENTINEL = "__VERIFY__"

    def __init__(self, memory_mb: int = 256):
        self.memory_mb = int(memory_mb)

    def materialize(self, prop: "Property") -> Callable[..., Any]:
        # No real materialization here — each invoke compiles the program
        # from scratch so the sandbox sees a self-contained source. We return
        # the property itself as the "callable" handle; invoke() reads
        # prop.source / prop.entry_point off it.
        return prop  # type: ignore[return-value]

    def invoke(
        self,
        callable_: Any,  # here this is the Property handle from materialize()
        inputs_kwargs: dict[str, Any],
        *,
        timeout_ms: int,
    ) -> tuple[Verdict, str]:
        prop = callable_ if isinstance(callable_, Property) else None
        if prop is None:
            return "ERROR", "SandboxedExecutor: materialize() handle was not a Property"

        # Literal-repr inputs. repr() on primitive candidate/problem values is
        # safe; if caller passes something non-repr-able (e.g. a complex
        # object), we'll get a SyntaxError when the sandbox compiles, which
        # becomes ERROR (logged via stderr tail).
        try:
            kwargs_literal = ", ".join(
                f"{k}={repr(v)}" for k, v in inputs_kwargs.items()
            )
        except Exception as e:
            return "ERROR", f"input repr failed: {type(e).__name__}: {e}"

        # Compose the sandbox program. The property's source should define
        # entry_point. We wrap the call in a try/except so uncaught exceptions
        # emit ERROR rather than a stack trace that looks like a crash.
        program = (
            f"{prop.source}\n"
            f"try:\n"
            f"    _result = {prop.entry_point}({kwargs_literal})\n"
            f"except BaseException as _e:\n"
            f"    print(f'{self._SENTINEL} ERROR {{type(_e).__name__}}: {{_e}}')\n"
            f"else:\n"
            f"    if isinstance(_result, tuple) and len(_result) == 2:\n"
            f"        _ok, _reason = _result\n"
            f"        _v = 'PASS' if bool(_ok) else 'FAIL'\n"
            f"        print(f'{self._SENTINEL} {{_v}} {{str(_reason)[:200]}}')\n"
            f"    elif isinstance(_result, bool):\n"
            f"        print(f'{self._SENTINEL} {{\"PASS\" if _result else \"FAIL\"}} ')\n"
            f"    elif isinstance(_result, str) and _result.upper() in ('PASS','FAIL','ERROR'):\n"
            f"        print(f'{self._SENTINEL} {{_result.upper()}} ')\n"
            f"    else:\n"
            f"        print(f'{self._SENTINEL} ERROR non-tri-state return: {{type(_result).__name__}}')\n"
        )

        timeout_s = max(1, min(_MAX_TIMEOUT_MS // 1000, (timeout_ms + 999) // 1000))
        ok, tail = run_python_sandboxed(program, timeout_s=timeout_s, memory_mb=self.memory_mb)

        if not ok:
            # Subprocess died (timeout, RLIMIT, audit deny, syntax error).
            return "ERROR", f"sandbox: {tail[:200]}"

        # Parse last occurrence of the sentinel line from stdout.
        for line in reversed((tail or "").splitlines()):
            line = line.strip()
            if line.startswith(self._SENTINEL + " "):
                body = line[len(self._SENTINEL) + 1:]
                parts = body.split(" ", 1)
                verdict_tok = parts[0].upper() if parts else ""
                reason = parts[1] if len(parts) > 1 else ""
                if verdict_tok in ("PASS", "FAIL", "ERROR"):
                    return verdict_tok, reason  # type: ignore[return-value]
        return "ERROR", f"no verdict sentinel in stdout: {tail[-200:]}"


class MockExecutor:
    """Executor that runs Python source in-process. NOT FOR PRODUCTION."""

    def materialize(self, prop: "Property") -> Callable[..., Any]:
        ns: dict[str, Any] = {}
        exec(compile(prop.source, f"<prop:{prop.property_id}>", "exec"), ns)  # noqa: S102
        fn = ns.get(prop.entry_point)
        if not callable(fn):
            raise ValueError(f"entry_point {prop.entry_point!r} not found or not callable")
        return fn

    def invoke(
        self, callable_: Callable[..., Any], inputs_kwargs: dict[str, Any],
        *, timeout_ms: int,
    ) -> tuple[Verdict, str]:
        try:
            result = callable_(**inputs_kwargs) if inputs_kwargs else callable_()
            return _coerce_verdict(result)
        except Exception as e:
            return "ERROR", f"{type(e).__name__}: {e}"


# ═════════════════════════════════════════════════════════════════════════
# §2.1 / §4.1 VerificationRecord + verify()
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class PropertyVerdict:
    """One row of a VerificationRecord. Tri-state per v0.2.1 P3."""
    property_id: str
    verdict: str  # "PASS" | "FAIL" | "ERROR"
    independence_class: str
    author: str
    name: str
    reason: str = ""
    witness: Any = None
    duration_ms: int = 0


@dataclass
class VerificationRecord:
    """§4.1 artifact — written to outputs/verifications/<sid>.jsonl."""
    record_id: str
    problem_id: str
    candidate_hash: str
    per_property: list[PropertyVerdict]
    accepted: bool
    quorum_n: int
    pass_count: int
    fail_count: int
    error_count: int
    distinct_classes: tuple[str, ...]
    reject_reason: str
    created_at: float
    quorum_distinct_classes_required: int = 3
    adversarial: bool = False


class CalibrationView(Protocol):
    """Read-only view that integrator's CalibrationLedger offers to verify()."""
    def suspended_classes(self) -> set[str]: ...


def verify(
    *,
    problem_id: str,
    candidate: Any,
    admitted_properties: list[Property],
    executor: SandboxExecutor,
    calibration: Optional[CalibrationView] = None,
    quorum_distinct_classes_required: int = 3,
    min_properties: int = 3,
) -> VerificationRecord:
    """Run admitted properties on a candidate; compute §2.1 quorum.

    Classes in calibration.suspended_classes() are dropped from the quorum.
    When calibration is None (unit tests / early ticks), all classes count.
    """
    cand_hash = _sha256_hex(repr(candidate))
    record_id = f"ver_{uuid.uuid4().hex[:16]}"
    t0 = time.time()
    suspended = calibration.suspended_classes() if calibration is not None else set()

    verdicts: list[PropertyVerdict] = []
    for prop in admitted_properties:
        if prop.independence_class in suspended:
            continue
        try:
            callable_ = _TRUSTED_CHECK_FNS.get(prop.property_id) or executor.materialize(prop)
        except Exception as e:
            verdicts.append(PropertyVerdict(
                property_id=prop.property_id, verdict="ERROR",
                independence_class=prop.independence_class,
                author=prop.author, name=prop.name,
                reason=f"materialize: {type(e).__name__}: {e}",
            ))
            continue
        v_start = time.time()
        verdict, reason = _invoke_callable(
            callable_, prop, candidate, executor,
            runtime_problem_id=problem_id,
        )
        verdicts.append(PropertyVerdict(
            property_id=prop.property_id, verdict=verdict,
            independence_class=prop.independence_class,
            author=prop.author, name=prop.name,
            reason=reason,
            duration_ms=int((time.time() - v_start) * 1000),
        ))

    pass_count = sum(1 for v in verdicts if v.verdict == "PASS")
    fail_count = sum(1 for v in verdicts if v.verdict == "FAIL")
    error_count = sum(1 for v in verdicts if v.verdict == "ERROR")
    distinct_classes = tuple(sorted({v.independence_class for v in verdicts if v.verdict == "PASS"}))

    # Duplicate-author rule (§2.1 rule 4): exists to prevent a single model
    # run from "voting" for a candidate via 3 properties it authored. Trusted
    # builtins aren't model-authored — they're a shared immutable library —
    # so their shared author string is not a conflict of interest. Only
    # apply the rule to model-authored pass authors.
    pass_authors = [
        v.author for v in verdicts
        if v.verdict == "PASS" and not str(v.author).startswith("builtin:")
    ]
    dup_author = len(pass_authors) != len(set(pass_authors))

    reject_reason = ""
    if len(verdicts) < min_properties:
        accepted = False
        reject_reason = f"n={len(verdicts)} < min_properties={min_properties}"
    elif fail_count > 0:
        accepted = False
        reject_reason = f"{fail_count} property FAIL (any-FAIL veto §2.1)"
    elif len(distinct_classes) < quorum_distinct_classes_required:
        accepted = False
        reject_reason = (
            f"distinct_classes={len(distinct_classes)} < required={quorum_distinct_classes_required}"
        )
    elif dup_author:
        accepted = False
        reject_reason = "two PASSing properties share author run_id (§2.1 rule 4)"
    else:
        accepted = True

    return VerificationRecord(
        record_id=record_id,
        problem_id=problem_id,
        candidate_hash=cand_hash,
        per_property=verdicts,
        accepted=accepted,
        quorum_n=len(verdicts),
        pass_count=pass_count,
        fail_count=fail_count,
        error_count=error_count,
        distinct_classes=distinct_classes,
        reject_reason=reject_reason,
        created_at=t0,
        quorum_distinct_classes_required=quorum_distinct_classes_required,
        adversarial=False,
    )


def property_to_payload(prop: Property) -> dict[str, Any]:
    """Helper for integrator's PropertyRegistry.append_property payload field."""
    return prop.to_dict()


def write_admitted_bundle(
    bundle: list[Property],
    vov_report: Any,
    *,
    regs: Any,
) -> list[str]:
    """Persist a §1.4-passing bundle to the PropertyRegistry.

    Spec v0.2.1 §1.3 / §1.4 ordering: a property admitted at §1.3 is still a
    candidate. Only bundles passing §1.4 (VoV.verify_properties_trustworthy)
    may be persisted. Callsite:

        vov_report = verify_properties_trustworthy(task_id, ref, bundle, ctx, domain)
        if vov_report.passed:
            record_ids = write_admitted_bundle(bundle, vov_report, regs=regs)
            # problem_registry write happens in task_synthesizer
        else:
            pass  # registry stays clean

    Fails closed: if vov_report.passed is False, raises ValueError rather than
    corrupting the registry. Returns the list of property_ids persisted, in
    bundle order, so the caller can reference them from VerificationLog
    entries.
    """
    if vov_report is None or not getattr(vov_report, "passed", False):
        raise ValueError(
            "write_admitted_bundle: vov_report.passed must be True; "
            "run VoV.verify_properties_trustworthy first and only call this on PASS."
        )
    if not bundle:
        return []
    property_registry = getattr(regs, "property_registry", None)
    if property_registry is None:
        raise ValueError(
            "write_admitted_bundle: regs must expose property_registry (RSIRegistries)."
        )
    persisted: list[str] = []
    for prop in bundle:
        # PropertyRegistry.append_property accepts the Property dataclass directly;
        # the integrator's store serializes via __dataclass_fields__.
        property_registry.append_property(prop, bundle_passed_vov=True)
        persisted.append(prop.property_id)
    return persisted


# ═════════════════════════════════════════════════════════════════════════
# PRE-RSI COMPATIBILITY SHIM — v0.1 simple-registry API
# ═════════════════════════════════════════════════════════════════════════
#
# Kept per architect v0.2.1 P10 for tests/test_rsi_actual.py until Phase E.
# Emits DeprecationWarning on first use; removal tracked in task #12.
# ═════════════════════════════════════════════════════════════════════════

PropertyFn = Callable[[Any], bool]
_REGISTRY: dict[str, PropertyFn] = {}
_SHIM_WARNED = False


def _shim_warn() -> None:
    global _SHIM_WARNED
    if not _SHIM_WARNED:
        warnings.warn(
            "property_engine simple-registry API is a pre-RSI compat shim; "
            "remove in Phase E once v0.2.1 Property migration (task #12) lands.",
            DeprecationWarning, stacklevel=3,
        )
        _SHIM_WARNED = True


def sample_has_properties(sample: Any) -> bool:
    """Pre-RSI compat helper used by verifier.py's verify_batch hook.

    Returns True if the sample carries a legacy `properties` / `property_ids`
    attribute. The new v0.2.1 path wires properties via Verifier.verify(...,
    properties=...) keyword (§5.1); this helper supports only the legacy
    attribute-discovery path.
    """
    for attr in ("properties", "property_ids"):
        if getattr(sample, attr, None):
            return True
    meta = getattr(sample, "verifier_meta", None)
    if isinstance(meta, dict) and (meta.get("properties") or meta.get("property_ids")):
        return True
    return False


@dataclass
class _LegacyVerdict:
    """Minimal duck-compatible verdict returned by the compat hook."""
    passed: bool
    confidence: float
    per_property_results: list = field(default_factory=list)
    disagreement_flags: list = field(default_factory=list)
    notes: str = ""

    def summary(self) -> str:
        return f"passed={self.passed} conf={self.confidence:.2f}"


def verify_sample_by_properties(sample: Any) -> _LegacyVerdict:
    """Pre-RSI compat helper used by verifier.py's verify_batch hook.

    The legacy v0.1 path expected a VerdictWithEvidence. The v0.2.1 path uses
    verify(..., properties=..., executor=...) with a VerificationRecord return.
    This compat helper returns a _LegacyVerdict(passed=True, confidence=1.0)
    stub — effectively a no-op pass — so Verifier.verify_batch does not
    short-circuit any sample that declares legacy `properties`. Downstream
    callers should migrate to Verifier.verify(..., properties=...) directly.
    """
    return _LegacyVerdict(passed=True, confidence=1.0, notes="legacy compat stub")


def register_property(name: str, fn: PropertyFn) -> None:
    """Legacy v0.1 register. Kept for test_rsi_actual.py."""
    _shim_warn()
    _REGISTRY[name] = fn


def verify_by_consensus(
    samples: list,
    threshold: float = 0.7,
    *,
    properties: Optional[dict[str, PropertyFn]] = None,
) -> list:
    """Legacy v0.1 filter-by-pass-rate. Kept for test_rsi_actual.py."""
    _shim_warn()
    registry = properties if properties is not None else _REGISTRY
    if not registry:
        return list(samples)
    out: list = []
    for s in samples:
        passes = 0
        for fn in registry.values():
            try:
                if fn(s):
                    passes += 1
            except Exception as e:
                logger.debug("shim property raised %s: %s", type(e).__name__, e)
        if passes / len(registry) >= threshold:
            out.append(s)
    return out


# ═════════════════════════════════════════════════════════════════════════
# Builtin catalog — 13 trusted templates, confirmer/falsifier empty.
# task_synthesizer rebinds per-problem via build_property.
# ═════════════════════════════════════════════════════════════════════════

_BUILTIN: dict[str, Property] = {}
_BUILTIN_PROBLEM_ID = "builtin:template"
_BUILTIN_PROBLEM_HASH = _sha256_hex("builtin:template")


def _register_builtin(
    *,
    name: str,
    kind: PropertyKind,
    description: str,
    independence_class: str,
    check_fn: Callable[..., Any],
    deterministic: bool = True,
) -> Property:
    prop = build_property(
        name=name,
        kind=kind,
        description=description,
        independence_class=independence_class,
        language="python",
        source=f"# builtin template: {name}",
        entry_point=f"_b_{name}",
        timeout_ms=2000,
        deterministic=deterministic,
        inputs=("problem", "candidate"),
        returns="bool",
        difficulty_floor=0.5,
        confirmer_example="",
        falsifier_example="",
        author="builtin:property_engine",
        problem_id=_BUILTIN_PROBLEM_ID,
        parent_problem_hash=_BUILTIN_PROBLEM_HASH,
        trusted=True,
        trusted_check_fn=check_fn,
    )
    _BUILTIN[name] = prop
    return prop


def get_property(name: str) -> Optional[Property]:
    return _BUILTIN.get(name)


def builtin_properties(independence_class: Optional[str] = None) -> list[Property]:
    if independence_class is None:
        return list(_BUILTIN.values())
    return [p for p in _BUILTIN.values() if p.independence_class == independence_class]


# ═════════════════════════════════════════════════════════════════════════
# Builtin check callables — signature (problem, candidate) → (bool, reason).
# Stubbed to the minimum needed for admission; real problem context is
# stamped by task_synthesizer via build_property rebind.
# ═════════════════════════════════════════════════════════════════════════

_RE_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
_RE_DEFCLASS = re.compile(r"((?:def|class)\s+\w+.*)", re.DOTALL)
_CODE_TIMEOUT = 5
_CODE_MEM_MB = 256


def _is_valid_python(src: str) -> bool:
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False


def _extract_code(solution: Any) -> Optional[str]:
    if solution is None:
        return None
    if isinstance(solution, str):
        texts = [solution]
    else:
        texts = []
        resp = getattr(solution, "response", None)
        if resp:
            texts.append(resp)
        chain = getattr(solution, "reasoning_chain", None) or []
        if chain:
            last = getattr(chain[-1], "content", None)
            if last:
                texts.append(last)
    for t in texts:
        if not t:
            continue
        m = _RE_FENCE.search(t)
        if m and _is_valid_python(m.group(1).strip()):
            return m.group(1).strip()
        m = _RE_DEFCLASS.search(t)
        if m and _is_valid_python(m.group(1)):
            return m.group(1)
        if _is_valid_python(t):
            return t
    return None


# Per-problem context registry. task_synthesizer stashes (tests, reference,
# entry_point, etc.) for each ProposedProblem BEFORE verify() runs, and
# trusted builtin check_fns look up their context by the runtime problem_id
# that verify() threads through _invoke_callable. This replaces the stubs
# that always returned True — which silently collapsed §2.1 quorum to
# "everything passes, quorum accepted, garbage into training data".
_PROBLEM_CTX: dict[str, dict] = {}


def stash_problem_ctx(problem_id: str, ctx: dict) -> None:
    """Register per-problem runtime context for builtin check_fns.

    Called by task_synthesizer (or any other proposer) before verify() so
    the trusted builtins have the specific data they need — unit tests for
    passes_provided_tests, reference solution for passes_generated_edge_cases,
    equations for substitute_back, etc.

    Context keys (all optional):
      tests          list[str]  — assert statements like "assert solve(5)==10"
      entry_point    str        — function name, e.g. "solve"
      reference      str        — reference implementation source code
      expected_type  str        — type name the return value should match
      equations      list[str]  — for math problems, equations to check
      bounds         tuple      — (lo, hi) plausibility range for numerics
      empty_input    str        — repr of an empty-container input for the fn
    """
    _PROBLEM_CTX[problem_id] = dict(ctx or {})


def get_problem_ctx(problem_id: str) -> dict:
    return _PROBLEM_CTX.get(problem_id, {})


def clear_problem_ctx(problem_id: Optional[str] = None) -> None:
    """Drop one or all cached problem contexts. Called between cycles to
    prevent unbounded growth across long runs."""
    if problem_id is None:
        _PROBLEM_CTX.clear()
    else:
        _PROBLEM_CTX.pop(problem_id, None)


def _b_executes(problem, candidate):
    code = _extract_code(candidate)
    if not code:
        return False, "no extractable Python code"
    ok, detail = run_python_sandboxed(code, _CODE_TIMEOUT, _CODE_MEM_MB)
    return (True, "") if ok else (False, f"sandbox: {detail[:160]}")


def _b_passes_provided_tests(problem, candidate):
    """Run the problem's assert-statement tests against candidate code.

    Context keys used: `tests` (list[str] of assert statements).
    If no tests are registered, returns ERROR (not silent PASS) so the
    caller knows the property needs per-problem ctx to discriminate.
    """
    ctx = get_problem_ctx(problem)
    tests = ctx.get("tests") or []
    if not tests:
        return "ERROR", "no tests in problem ctx"
    code = _extract_code(candidate)
    if not code:
        return False, "no extractable Python code"
    test_block = "\n".join(
        t if t.strip().startswith(("assert ", "assert(")) else f"assert ({t})"
        for t in tests if t.strip()
    )
    if not test_block:
        return "ERROR", "no non-empty tests"
    full = code + "\n\n# --- provided tests ---\n" + test_block + "\n"
    ok, detail = run_python_sandboxed(full, _CODE_TIMEOUT, _CODE_MEM_MB)
    return (True, "") if ok else (False, f"failed: {detail[:160]}")


def _b_passes_generated_edge_cases(problem, candidate):
    """Differential test: candidate vs reference on a few random inputs.

    Context keys: `reference` (source of a function named `entry_point`),
    `entry_point` (function name), `edge_inputs` (list of repr-inputs to try).
    Probe is bounded (≤5 inputs, 3s total) so verification stays fast.
    """
    ctx = get_problem_ctx(problem)
    reference = ctx.get("reference") or ""
    entry = ctx.get("entry_point") or "solve"
    edges_raw = ctx.get("edge_inputs") or []
    if not reference or not edges_raw:
        return "ERROR", "no reference or edge_inputs in ctx"
    code = _extract_code(candidate)
    if not code:
        return False, "no extractable Python code"
    # Parse edge inputs in the HOST via ast.literal_eval (no eval() needed in
    # the sandbox, which blocks eval anyway). Only well-formed Python
    # literals survive; malformed strings are skipped.
    import ast as _ast
    edges: list = []
    for raw in edges_raw[:5]:
        try:
            edges.append(_ast.literal_eval(raw) if isinstance(raw, str) else raw)
        except (ValueError, SyntaxError):
            continue
    if not edges:
        return "ERROR", "no parseable edge inputs"
    # Redefine entry twice would collide; rename via function alias.
    harness = (
        f"{reference}\n"
        f"_ref = {entry}\n"
        f"{code}\n"
        f"_cand = {entry}\n"
        f"_edges = {edges!r}\n"
        "_fail = None\n"
        "for _arg in _edges:\n"
        "    try:\n"
        "        _r = _ref(_arg); _c = _cand(_arg)\n"
        "    except Exception as _e:\n"
        "        _fail = f'raised on {_arg!r}: {type(_e).__name__}'\n"
        "        break\n"
        "    if _r != _c:\n"
        "        _fail = f'differs on {_arg!r}: ref={_r!r} cand={_c!r}'\n"
        "        break\n"
        "print('FAIL:', _fail) if _fail else print('OK')\n"
    )
    ok, detail = run_python_sandboxed(harness, _CODE_TIMEOUT, _CODE_MEM_MB)
    if not ok:
        return False, f"harness crashed: {detail[:160]}"
    if "FAIL:" in (detail or ""):
        return False, detail[detail.index("FAIL:"):][:160]
    return True, ""


def _b_output_type_matches_signature(problem, candidate):
    """Check candidate returns the expected type on a sample input.

    Context keys: `entry_point`, `expected_type` (Python type name), `sample_input`.
    """
    ctx = get_problem_ctx(problem)
    entry = ctx.get("entry_point") or "solve"
    expected_type = (ctx.get("expected_type") or "").strip()
    sample = ctx.get("sample_input")
    if not expected_type or sample is None:
        return "ERROR", "no expected_type or sample_input in ctx"
    code = _extract_code(candidate)
    if not code:
        return False, "no extractable Python code"
    harness = (
        f"{code}\n"
        f"_r = {entry}({sample!r})\n"
        f"import sys\n"
        f"_expected = {expected_type!r}\n"
        "_actual = type(_r).__name__\n"
        "print(('OK' if _actual == _expected else f'FAIL: expected {_expected}, got {_actual}'))\n"
    )
    ok, detail = run_python_sandboxed(harness, _CODE_TIMEOUT, _CODE_MEM_MB)
    if not ok:
        return False, f"harness crashed: {detail[:160]}"
    if "FAIL:" in (detail or ""):
        return False, detail[detail.index("FAIL:"):][:160]
    return True, ""


def _b_no_exceptions_on_empty_input(problem, candidate):
    """Candidate tolerates an empty input without raising.

    Context keys: `entry_point`, `empty_input` (repr of the empty container).
    """
    ctx = get_problem_ctx(problem)
    entry = ctx.get("entry_point") or "solve"
    empty = ctx.get("empty_input")
    if empty is None:
        return "ERROR", "no empty_input in ctx"
    code = _extract_code(candidate)
    if not code:
        return False, "no extractable Python code"
    harness = (
        f"{code}\n"
        f"try:\n"
        f"    {entry}({empty!r}); print('OK')\n"
        f"except Exception as _e:\n"
        f"    print('FAIL:', type(_e).__name__, str(_e)[:80])\n"
    )
    ok, detail = run_python_sandboxed(harness, _CODE_TIMEOUT, _CODE_MEM_MB)
    if not ok:
        return False, f"harness crashed: {detail[:160]}"
    if "FAIL:" in (detail or ""):
        return False, detail[detail.index("FAIL:"):][:160]
    return True, ""


def _b_idempotent_where_applicable(problem, candidate):
    """f(f(x)) == f(x) on a sample input, if the problem declares idempotence."""
    ctx = get_problem_ctx(problem)
    if not ctx.get("idempotent"):
        return "ERROR", "problem does not declare idempotence"
    entry = ctx.get("entry_point") or "solve"
    sample = ctx.get("sample_input")
    if sample is None:
        return "ERROR", "no sample_input for idempotence check"
    code = _extract_code(candidate)
    if not code:
        return False, "no extractable Python code"
    harness = (
        f"{code}\n"
        f"_once = {entry}({sample!r}); _twice = {entry}(_once)\n"
        "print('OK' if _once == _twice else f'FAIL: f(x)={_once!r} f(f(x))={_twice!r}')\n"
    )
    ok, detail = run_python_sandboxed(harness, _CODE_TIMEOUT, _CODE_MEM_MB)
    if not ok:
        return False, f"harness crashed: {detail[:160]}"
    if "FAIL:" in (detail or ""):
        return False, detail[detail.index("FAIL:"):][:160]
    return True, ""


def _b_substitute_back(problem, candidate):
    """Plug candidate answer into the problem's equations via sympy."""
    ctx = get_problem_ctx(problem)
    equations = ctx.get("equations") or []
    var = ctx.get("variable") or "x"
    if not equations:
        return "ERROR", "no equations in ctx"
    # candidate is a numeric or expression string
    cand_str = candidate if isinstance(candidate, str) else getattr(candidate, "response", str(candidate))
    try:
        from sympy import symbols, sympify, simplify
    except ImportError:
        return "ERROR", "sympy unavailable"
    try:
        sym = symbols(var)
        val = sympify(cand_str.strip())
    except Exception as e:
        return False, f"can't parse candidate as number/expr: {e}"
    for eq in equations:
        if "=" in eq:
            lhs, rhs = eq.split("=", 1)
        else:
            lhs, rhs = eq, "0"
        try:
            diff = simplify(sympify(lhs).subs(sym, val) - sympify(rhs).subs(sym, val))
            if diff != 0:
                return False, f"equation fails: {eq} at {var}={val}"
        except Exception as e:
            return "ERROR", f"simplify raised on {eq}: {e}"
    return True, ""


def _b_dimensional_consistency(problem, candidate):
    ctx = get_problem_ctx(problem)
    expected_units = ctx.get("expected_units") or ""
    if not expected_units:
        return "ERROR", "no expected_units in ctx"
    text = candidate if isinstance(candidate, str) else str(candidate)
    # Minimal check: expected unit token appears in the candidate text.
    ok = expected_units.lower() in text.lower()
    return (True, "") if ok else (False, f"missing unit: expected '{expected_units}'")


def _b_alternative_derivation_agrees(problem, candidate):
    """Candidate matches an independently-derived reference numerically."""
    ctx = get_problem_ctx(problem)
    alt = ctx.get("alternative_answer")
    if alt is None:
        return "ERROR", "no alternative_answer in ctx"
    cand = candidate if isinstance(candidate, str) else getattr(candidate, "response", str(candidate))
    try:
        from sympy import sympify, simplify
        diff = simplify(sympify(str(cand).strip()) - sympify(str(alt).strip()))
        return (True, "") if diff == 0 else (False, f"diff = {diff}")
    except Exception as e:
        return "ERROR", f"sympy: {e}"


def _b_numerical_plausibility(problem, candidate):
    ctx = get_problem_ctx(problem)
    bounds = ctx.get("bounds")
    if not bounds or len(bounds) != 2:
        return "ERROR", "no bounds in ctx"
    lo, hi = bounds
    cand = candidate if isinstance(candidate, str) else str(candidate)
    import re as _re
    m = _re.search(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", cand)
    if not m:
        return False, "no number in candidate"
    try:
        v = float(m.group(0))
    except ValueError:
        return False, "unparseable number"
    return (True, "") if lo <= v <= hi else (False, f"{v} outside [{lo}, {hi}]")


def _b_contrapositive_holds(problem, candidate):
    ctx = get_problem_ctx(problem)
    antecedent = (ctx.get("antecedent") or "").lower()
    consequent = (ctx.get("consequent") or "").lower()
    if not antecedent or not consequent:
        return "ERROR", "no antecedent/consequent in ctx"
    text = (candidate if isinstance(candidate, str) else str(candidate)).lower()
    ok = (antecedent in text) and (consequent in text)
    return (True, "") if ok else (False, "missing antecedent or consequent reference")


def _b_premise_reformulation_preserves_conclusion(problem, candidate):
    ctx = get_problem_ctx(problem)
    conclusion = (ctx.get("conclusion") or "").lower()
    if not conclusion:
        return "ERROR", "no conclusion in ctx"
    text = (candidate if isinstance(candidate, str) else str(candidate)).lower()
    return (True, "") if conclusion in text else (False, "conclusion not present")


def _b_trivial_case_correct(problem, candidate):
    """Candidate gives the correct answer on a declared trivial case."""
    ctx = get_problem_ctx(problem)
    trivial = ctx.get("trivial_case")  # {"input": ..., "expected": ...}
    if not trivial or "input" not in trivial or "expected" not in trivial:
        return "ERROR", "no trivial_case in ctx"
    entry = ctx.get("entry_point") or "solve"
    code = _extract_code(candidate)
    if not code:
        return False, "no extractable Python code"
    inp = trivial["input"]
    exp = trivial["expected"]
    harness = (
        f"{code}\n"
        f"_r = {entry}({inp!r})\n"
        f"print('OK' if _r == {exp!r} else f'FAIL: got {{_r!r}}, expected {{{exp!r}!r}}')\n"
    )
    ok, detail = run_python_sandboxed(harness, _CODE_TIMEOUT, _CODE_MEM_MB)
    if not ok:
        return False, f"harness crashed: {detail[:160]}"
    if "FAIL:" in (detail or ""):
        return False, detail[detail.index("FAIL:"):][:160]
    return True, ""


# Register 13 builtins under v0.2.1-canonical independence_class values.
_register_builtin(
    name="executes", kind=PropertyKind.UNIT_TEST,
    description="Code executes to completion in the sandbox without error.",
    independence_class="exec.behavioral",
    check_fn=_b_executes,
)
_register_builtin(
    name="passes_provided_tests", kind=PropertyKind.UNIT_TEST,
    description="Code passes every test supplied in the problem context.",
    independence_class="exec.behavioral",
    check_fn=_b_passes_provided_tests,
)
_register_builtin(
    name="passes_generated_edge_cases", kind=PropertyKind.COUNTEREXAMPLE_SEARCH,
    description="Code agrees with a reference impl on generated edge cases.",
    independence_class="search.bounded",
    check_fn=_b_passes_generated_edge_cases,
    deterministic=False,
)
_register_builtin(
    name="output_type_matches_signature", kind=PropertyKind.TYPE_INVARIANT,
    description="Function returns the declared type on a sample input.",
    independence_class="structural.static",
    check_fn=_b_output_type_matches_signature,
)
_register_builtin(
    name="no_exceptions_on_empty_input", kind=PropertyKind.UNIT_TEST,
    description="Function tolerates an empty-container input without raising.",
    independence_class="exec.behavioral",
    check_fn=_b_no_exceptions_on_empty_input,
)
_register_builtin(
    name="idempotent_where_applicable", kind=PropertyKind.ROUNDTRIP,
    description="f(f(x)) == f(x) when the problem declares idempotence.",
    independence_class="roundtrip",
    check_fn=_b_idempotent_where_applicable,
)
_register_builtin(
    name="substitute_back", kind=PropertyKind.ALGEBRAIC,
    description="Substituting the answer back into the equations satisfies them.",
    independence_class="algebra.symbolic",
    check_fn=_b_substitute_back,
)
_register_builtin(
    name="dimensional_consistency", kind=PropertyKind.DIMENSIONAL,
    description="Answer carries the expected units.",
    independence_class="dimensional.physical",    # v0.2.1 10th class
    check_fn=_b_dimensional_consistency,
)
_register_builtin(
    name="alternative_derivation_agrees", kind=PropertyKind.ALGEBRAIC,
    description="Answer matches an independently-derived reference answer.",
    independence_class="algebra.symbolic",
    check_fn=_b_alternative_derivation_agrees,
)
_register_builtin(
    name="numerical_plausibility", kind=PropertyKind.POSTCONDITION,
    description="Answer falls within a plausible numeric range.",
    independence_class="search.bounded",
    check_fn=_b_numerical_plausibility,
)
_register_builtin(
    name="contrapositive_holds", kind=PropertyKind.REFORMULATION,
    description="Solution references both antecedent and consequent of the implication.",
    independence_class="smt.logical",              # v0.2.1 remap
    check_fn=_b_contrapositive_holds,
)
_register_builtin(
    name="premise_reformulation_preserves_conclusion", kind=PropertyKind.REFORMULATION,
    description="Conclusion does not depend on surface wording of the original premise.",
    independence_class="transform.semantic",
    check_fn=_b_premise_reformulation_preserves_conclusion,
)
_register_builtin(
    name="trivial_case_correct", kind=PropertyKind.UNIT_TEST,
    description="Solution gives the correct answer on a declared trivial case.",
    independence_class="exec.behavioral",
    check_fn=_b_trivial_case_correct,
)


__all__ = [
    # v0.2.1 canonical
    "Property", "PropertyKind", "INDEPENDENCE_CLASSES", "LegacyProperty",
    "build_property", "is_trusted",
    "admit", "AdmissionResult", "SandboxExecutor", "SandboxedExecutor", "MockExecutor", "Verdict",
    "verify", "VerificationRecord", "PropertyVerdict", "CalibrationView",
    "property_to_payload", "write_admitted_bundle",
    "builtin_properties", "get_property",
    "stash_problem_ctx", "get_problem_ctx", "clear_problem_ctx",
    # legacy shim (to remove in Phase E)
    "PropertyFn", "register_property", "verify_by_consensus",
]
