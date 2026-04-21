"""Simulator backend — physics/chemistry verification via numerical ODE
integration, conservation-law checking, and dimensional analysis.

Task #5 (verifier-sim). Adds a new independence class `simulation.numerical`
and a new PropertyKind SIMULATION. Candidate answers on physics-domain
problems are scored against a ground-truth numerical simulation.

The backend is a collection of pure check functions plus thin
`build_property` factories. Each check follows the trusted-builtin
signature `(problem, candidate) -> (bool, reason) | ("ERROR", reason)`
so property_engine.admit/verify treat them like any other trusted builtin.

Gracefully skips when scipy or sympy.physics are unavailable: every check
returns ("ERROR", "...") which does NOT poison quorum per v0.2.1 P3. A
helper `simulator_available()` lets task_synthesizer decide whether to
bother registering these properties.

Context keys (via property_engine.stash_problem_ctx) read by these checks:

  ode               dict:
                      "rhs":  str, body of f(t, y, *params) returning dy/dt
                              (y is a list/ndarray). Executed via exec() in
                              a restricted namespace — this is TRUSTED
                              library code, not sandboxed (scipy needs to
                              call the callable many times per step and the
                              subprocess overhead would dominate). Only
                              invoked when the problem author supplies the
                              rhs, so the trust boundary is the synthesizer.
                      "y0":   list[float]   initial state
                      "t":    tuple(t0, tf) integration bounds
                      "params": tuple       extra args to rhs
                      "target_index": int   which y[i] the candidate predicts
                      "tolerance": float    relative tolerance, default 1e-2
  conservation      dict:
                      "quantity": "energy" | "momentum" | callable-as-string
                      "initial_state": dict of named scalars
                      "final_state_from_candidate": str — python expression
                          evaluating to the quantity using `cand` (the
                          parsed candidate numeric answer)
                      "tolerance": float
  dimensions        dict:
                      "expected": str — sympy.physics.units expression,
                                        e.g. "meter/second" or "kg*m/s**2"
                      "candidate_expr": str — how to read the candidate's
                                              value+unit, default just parse
                                              the candidate text.

All three are independent: a physics problem may register any subset.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Optional

from .property_engine import (
    PropertyKind,
    Property,
    build_property,
    get_problem_ctx,
)

logger = logging.getLogger(__name__)

try:
    import scipy.integrate as _scipy_integrate  # type: ignore
    _HAS_SCIPY = True
except ImportError:
    _scipy_integrate = None  # type: ignore
    _HAS_SCIPY = False

try:
    from sympy.physics import units as _sp_units  # type: ignore
    from sympy import sympify as _sympify, simplify as _simplify
    _HAS_SYMPY_PHYSICS = True
except ImportError:
    _sp_units = None  # type: ignore
    _sympify = None  # type: ignore
    _simplify = None  # type: ignore
    _HAS_SYMPY_PHYSICS = False


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
_DEFAULT_REL_TOL = 1e-2
_SIM_TIMEOUT_MS = 5000


def simulator_available() -> bool:
    """True iff scipy.integrate is importable. sympy.physics is optional
    per-check (only dimensional_consistency_sim needs it)."""
    return _HAS_SCIPY


def _parse_candidate_number(candidate: Any) -> Optional[float]:
    text = candidate if isinstance(candidate, str) else getattr(candidate, "response", str(candidate))
    m = _NUM_RE.search(text or "")
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _compile_rhs(rhs_src: str) -> Optional[Callable]:
    """Compile an ODE rhs body into f(t, y, *params). Restricted namespace
    — only math + numpy are available. Returns None if compilation fails."""
    import math
    try:
        import numpy as _np
    except ImportError:
        _np = None
    ns: dict[str, Any] = {"math": math, "np": _np, "__builtins__": {
        "abs": abs, "min": min, "max": max, "pow": pow, "sum": sum,
        "len": len, "range": range, "float": float, "int": int,
        "tuple": tuple, "list": list,
    }}
    try:
        body = rhs_src.strip()
        if not body.lstrip().startswith("def "):
            # Allow a bare expression/list return — wrap it.
            indented = "\n".join("    " + line for line in body.splitlines())
            src = "def _rhs(t, y, *params):\n" + indented
        else:
            src = body + "\n_rhs = " + (body.split("(", 1)[0].split()[1])
        exec(compile(src, "<ode-rhs>", "exec"), ns)  # noqa: S102
        fn = ns.get("_rhs")
        return fn if callable(fn) else None
    except Exception as e:
        logger.debug("rhs compile failed: %s", e)
        return None


def check_ode_match(problem: Any, candidate: Any) -> tuple:
    """Integrate the problem's ODE and require candidate within rel tol."""
    if not _HAS_SCIPY:
        return "ERROR", "scipy.integrate unavailable"
    ctx = get_problem_ctx(problem)
    ode = ctx.get("ode")
    if not isinstance(ode, dict):
        return "ERROR", "no ode spec in problem ctx"
    rhs_src = ode.get("rhs")
    y0 = ode.get("y0")
    tspan = ode.get("t")
    params = tuple(ode.get("params") or ())
    idx = int(ode.get("target_index", 0))
    tol = float(ode.get("tolerance", _DEFAULT_REL_TOL))
    if not (rhs_src and y0 and tspan):
        return "ERROR", "ode missing rhs/y0/t"
    fn = _compile_rhs(rhs_src)
    if fn is None:
        return "ERROR", "could not compile rhs"
    try:
        sol = _scipy_integrate.solve_ivp(
            fn, tspan, y0, args=params, rtol=1e-8, atol=1e-10,
            dense_output=False, t_eval=[tspan[1]],
        )
    except Exception as e:
        return "ERROR", f"solve_ivp raised: {type(e).__name__}: {e}"
    if not sol.success:
        return "ERROR", f"integrator failed: {sol.message}"
    try:
        truth = float(sol.y[idx][-1])
    except (IndexError, ValueError) as e:
        return "ERROR", f"bad target_index: {e}"
    cand = _parse_candidate_number(candidate)
    if cand is None:
        return False, "no numeric answer in candidate"
    denom = max(abs(truth), 1e-12)
    rel_err = abs(cand - truth) / denom
    if rel_err <= tol:
        return True, f"rel_err={rel_err:.3e} ≤ {tol}"
    return False, f"rel_err={rel_err:.3e} > {tol} (truth={truth:.6g}, cand={cand:.6g})"


def check_conservation(problem: Any, candidate: Any) -> tuple:
    """Check that the candidate's answer preserves a conserved quantity.

    Evaluates initial quantity from `initial_state` and final quantity via
    `final_state_from_candidate` (an expression using `cand`). Passes iff
    |final - initial| / (|initial|+eps) <= tolerance.
    """
    ctx = get_problem_ctx(problem)
    cons = ctx.get("conservation")
    if not isinstance(cons, dict):
        return "ERROR", "no conservation spec in ctx"
    init_state = cons.get("initial_state") or {}
    final_expr = cons.get("final_state_from_candidate")
    tol = float(cons.get("tolerance", _DEFAULT_REL_TOL))
    quantity = cons.get("quantity", "energy")
    if not final_expr:
        return "ERROR", "no final_state_from_candidate expr"
    cand = _parse_candidate_number(candidate)
    if cand is None:
        return False, "no numeric answer in candidate"

    # Standard conserved-quantity recipes.
    import math
    def energy_kinetic(state: dict) -> float:
        m = float(state.get("m", 1.0))
        v = float(state.get("v", 0.0))
        return 0.5 * m * v * v
    def momentum(state: dict) -> float:
        return float(state.get("m", 1.0)) * float(state.get("v", 0.0))
    def total_energy(state: dict) -> float:
        m = float(state.get("m", 1.0))
        v = float(state.get("v", 0.0))
        h = float(state.get("h", 0.0))
        g = float(state.get("g", 9.81))
        return 0.5 * m * v * v + m * g * h

    recipes = {
        "energy": total_energy,
        "kinetic_energy": energy_kinetic,
        "momentum": momentum,
    }
    q_init_fn = recipes.get(str(quantity))
    if q_init_fn is None:
        return "ERROR", f"unknown conserved quantity: {quantity}"
    try:
        q_init = q_init_fn(init_state)
    except Exception as e:
        return "ERROR", f"initial-state eval: {type(e).__name__}: {e}"

    try:
        q_final = float(eval(  # noqa: S307 - trusted library, expr from synthesizer
            final_expr,
            {"__builtins__": {}, "math": math, "abs": abs, "min": min, "max": max},
            {"cand": cand, **{k: float(v) for k, v in init_state.items() if isinstance(v, (int, float))}},
        ))
    except Exception as e:
        return "ERROR", f"final-state eval: {type(e).__name__}: {e}"

    denom = max(abs(q_init), 1e-12)
    rel_err = abs(q_final - q_init) / denom
    if rel_err <= tol:
        return True, f"{quantity} conserved: rel_err={rel_err:.3e}"
    return False, f"{quantity} NOT conserved: init={q_init:.6g} final={q_final:.6g} rel_err={rel_err:.3e}"


def check_dimensional_consistency_sim(problem: Any, candidate: Any) -> tuple:
    """Use sympy.physics.units to check candidate's units match expected.

    Stricter than property_engine._b_dimensional_consistency which just does
    a substring match — this one builds the full unit expression and checks
    dimensional equivalence.
    """
    if not _HAS_SYMPY_PHYSICS:
        return "ERROR", "sympy.physics.units unavailable"
    ctx = get_problem_ctx(problem)
    dims = ctx.get("dimensions")
    if not isinstance(dims, dict):
        return "ERROR", "no dimensions spec in ctx"
    expected = dims.get("expected")
    if not expected:
        return "ERROR", "no expected unit expression"
    text = candidate if isinstance(candidate, str) else str(candidate)
    unit_match = re.search(
        r"([A-Za-z]+(?:\s*[*/]\s*[A-Za-z0-9*+/^()\s]+)?)",
        text.split(None, 1)[-1] if text else "",
    )
    cand_unit_str = (dims.get("candidate_unit") or (unit_match.group(1) if unit_match else "")).strip()
    if not cand_unit_str:
        return False, "no unit token in candidate"
    try:
        ns = {name: getattr(_sp_units, name) for name in dir(_sp_units) if not name.startswith("_")}
        exp_u = eval(str(expected), {"__builtins__": {}}, ns)  # noqa: S307 - sympy units only
        cand_u = eval(cand_unit_str.replace("^", "**"), {"__builtins__": {}}, ns)  # noqa: S307
    except Exception as e:
        return False, f"could not parse units: {type(e).__name__}: {e}"
    try:
        ratio = _simplify(exp_u / cand_u)
        if ratio.is_number:
            return True, f"units match (ratio={ratio})"
        return False, f"dimensional mismatch: expected {expected}, got {cand_unit_str}"
    except Exception as e:
        return "ERROR", f"simplify raised: {type(e).__name__}: {e}"


# ═════════════════════════════════════════════════════════════════════════
# Factories — task_synthesizer calls these per-problem, stamps confirmer/
# falsifier examples, then admits.
# ═════════════════════════════════════════════════════════════════════════

def make_ode_match_property(
    *, problem_id: str, parent_problem_hash: str,
    confirmer_example: str = "", falsifier_example: str = "",
) -> Property:
    return build_property(
        name="ode_numerical_match",
        kind=PropertyKind.SIMULATION,
        description="Candidate answer agrees with scipy ODE integration within tolerance.",
        independence_class="simulation.numerical",
        language="python",
        source="# trusted: simulator_backend.check_ode_match",
        entry_point="check_ode_match",
        timeout_ms=_SIM_TIMEOUT_MS,
        deterministic=True,
        inputs=("problem", "candidate"),
        returns="bool",
        difficulty_floor=0.5,
        confirmer_example=confirmer_example,
        falsifier_example=falsifier_example,
        author="builtin:simulator_backend",
        problem_id=problem_id,
        parent_problem_hash=parent_problem_hash,
        trusted=True,
        trusted_check_fn=check_ode_match,
    )


def make_conservation_property(
    *, problem_id: str, parent_problem_hash: str,
    confirmer_example: str = "", falsifier_example: str = "",
) -> Property:
    return build_property(
        name="conservation_law",
        kind=PropertyKind.CONSERVATION,
        description="Candidate answer preserves the declared conserved quantity.",
        independence_class="conservation.global",
        language="python",
        source="# trusted: simulator_backend.check_conservation",
        entry_point="check_conservation",
        timeout_ms=_SIM_TIMEOUT_MS,
        deterministic=True,
        inputs=("problem", "candidate"),
        returns="bool",
        difficulty_floor=0.5,
        confirmer_example=confirmer_example,
        falsifier_example=falsifier_example,
        author="builtin:simulator_backend",
        problem_id=problem_id,
        parent_problem_hash=parent_problem_hash,
        trusted=True,
        trusted_check_fn=check_conservation,
    )


def make_dimensional_sim_property(
    *, problem_id: str, parent_problem_hash: str,
    confirmer_example: str = "", falsifier_example: str = "",
) -> Property:
    return build_property(
        name="dimensional_consistency_sim",
        kind=PropertyKind.DIMENSIONAL,
        description="Candidate units equal expected units under sympy.physics.units.",
        independence_class="dimensional.physical",
        language="python",
        source="# trusted: simulator_backend.check_dimensional_consistency_sim",
        entry_point="check_dimensional_consistency_sim",
        timeout_ms=_SIM_TIMEOUT_MS,
        deterministic=True,
        inputs=("problem", "candidate"),
        returns="bool",
        difficulty_floor=0.5,
        confirmer_example=confirmer_example,
        falsifier_example=falsifier_example,
        author="builtin:simulator_backend",
        problem_id=problem_id,
        parent_problem_hash=parent_problem_hash,
        trusted=True,
        trusted_check_fn=check_dimensional_consistency_sim,
    )


__all__ = [
    "simulator_available",
    "check_ode_match",
    "check_conservation",
    "check_dimensional_consistency_sim",
    "make_ode_match_property",
    "make_conservation_property",
    "make_dimensional_sim_property",
]
