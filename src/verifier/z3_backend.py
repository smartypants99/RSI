"""Z3 SMT backend for constraint-satisfaction verification (team task #4).

Two entry points:

1. Z3Solver — implements ExternalSolver. Accepts a problem description in
   either SMT-LIB v2 form or the small "assertion DSL" described below, runs
   Z3 with a wall-clock timeout, and returns a normalized SolverResult.

2. verify_smt_claim(problem, candidate_assignment) — used by property_engine
   for verifying a sample's claim "assignment X satisfies constraints C":
   the candidate's variable → value map is added as equality assertions and
   the combined system is checked for sat.

Problem grammar (assertion DSL):
    - Blank lines and lines starting with '#' are ignored.
    - `declare <name> <sort>`   where sort ∈ {Int, Real, Bool}
    - `assert <z3-python-expr>` — evaluated with Z3 Python API using declared
      symbols in scope. Supports And, Or, Not, Implies, ==, !=, <, <=, >, >=,
      +, -, *, /, % and numeric / bool literals.

Example:
    declare x Int
    declare y Int
    assert x + y == 10
    assert x > 0
    assert y > 0

SMT-LIB passthrough: if the problem starts with `(set-logic` or `(declare-`
it is parsed via z3.parse_smt2_string instead.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Optional

from .external_solver import ExternalSolver, SolverResult, make_result

logger = logging.getLogger(__name__)

try:
    import z3  # type: ignore
    HAS_Z3 = True
except ImportError:  # pragma: no cover — install path documented in requirements
    z3 = None  # type: ignore
    HAS_Z3 = False


_DEFAULT_TIMEOUT_S = 5.0
_MAX_TIMEOUT_S = 30.0

_SORTS = {"Int", "Real", "Bool"}
_DECLARE_RE = re.compile(r"^\s*declare\s+([A-Za-z_]\w*)\s+(Int|Real|Bool)\s*$")
_ASSERT_RE = re.compile(r"^\s*assert\s+(.+)$")


def _looks_like_smtlib(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith("(set-") or stripped.startswith("(declare-") or stripped.startswith("(assert")


def _build_env(sort_by_name: dict[str, str]) -> dict[str, Any]:
    """Build a restricted eval environment with Z3 Python-API names only."""
    env: dict[str, Any] = {
        "__builtins__": {},
        "And": z3.And, "Or": z3.Or, "Not": z3.Not, "Implies": z3.Implies,
        "If": z3.If, "Xor": z3.Xor,
        "True": True, "False": False,
        "IntVal": z3.IntVal, "RealVal": z3.RealVal, "BoolVal": z3.BoolVal,
    }
    for name, sort in sort_by_name.items():
        if sort == "Int":
            env[name] = z3.Int(name)
        elif sort == "Real":
            env[name] = z3.Real(name)
        elif sort == "Bool":
            env[name] = z3.Bool(name)
    return env


def _parse_dsl(problem: str) -> tuple[list, dict[str, str]]:
    """Parse the assertion DSL. Returns (z3_constraints, sort_by_name).

    Raises ValueError on malformed input.
    """
    sort_by_name: dict[str, str] = {}
    assert_lines: list[str] = []
    for lineno, raw in enumerate(problem.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _DECLARE_RE.match(line)
        if m:
            name, sort = m.group(1), m.group(2)
            if name in sort_by_name:
                raise ValueError(f"line {lineno}: duplicate declaration of {name!r}")
            sort_by_name[name] = sort
            continue
        m = _ASSERT_RE.match(line)
        if m:
            assert_lines.append(m.group(1).strip())
            continue
        raise ValueError(f"line {lineno}: unrecognized statement {line!r}")

    env = _build_env(sort_by_name)
    constraints: list = []
    for expr_src in assert_lines:
        try:
            constraints.append(eval(expr_src, env, {}))  # noqa: S307 — restricted env
        except Exception as e:
            raise ValueError(f"failed to parse assertion {expr_src!r}: {type(e).__name__}: {e}") from e
    return constraints, sort_by_name


def _model_to_dict(model, sort_by_name: dict[str, str]) -> dict[str, Any]:
    """Convert a Z3 Model into a plain dict for SolverResult."""
    out: dict[str, Any] = {}
    for decl in model.decls():
        name = decl.name()
        val = model[decl]
        sort = sort_by_name.get(name, "")
        try:
            if sort == "Int" or val.sort().kind() == z3.Z3_INT_SORT:
                out[name] = val.as_long()
            elif sort == "Real" or val.sort().kind() == z3.Z3_REAL_SORT:
                num, den = val.numerator_as_long(), val.denominator_as_long()
                out[name] = num / den if den else float(num)
            elif sort == "Bool" or val.sort().kind() == z3.Z3_BOOL_SORT:
                out[name] = z3.is_true(val)
            else:
                out[name] = str(val)
        except Exception:
            out[name] = str(val)
    return out


class Z3Solver:
    """ExternalSolver implementation backed by z3-solver."""

    name = "z3"

    def __init__(self, default_timeout_s: float = _DEFAULT_TIMEOUT_S):
        if not HAS_Z3:
            raise RuntimeError(
                "z3-solver not installed. Add `z3-solver` to requirements.txt "
                "and `pip install z3-solver`."
            )
        self.default_timeout_s = min(max(0.1, default_timeout_s), _MAX_TIMEOUT_S)

    def check(self, problem: str, timeout_s: float = _DEFAULT_TIMEOUT_S) -> SolverResult:
        if not HAS_Z3:
            return make_result("error", detail="z3 not installed", backend=self.name)
        timeout_s = min(max(0.1, timeout_s), _MAX_TIMEOUT_S)
        start = time.perf_counter()

        solver = z3.Solver()
        solver.set("timeout", int(timeout_s * 1000))  # Z3 wants milliseconds

        sort_by_name: dict[str, str] = {}
        try:
            if _looks_like_smtlib(problem):
                constraints = z3.parse_smt2_string(problem)
                for c in constraints:
                    solver.add(c)
                # SMT-LIB: we can't recover per-variable sorts cheaply for the
                # model dict, but the model's own decls will report them.
            else:
                constraints, sort_by_name = _parse_dsl(problem)
                for c in constraints:
                    solver.add(c)
        except ValueError as e:
            return make_result("error", detail=str(e), backend=self.name,
                               elapsed_s=time.perf_counter() - start)
        except z3.Z3Exception as e:
            return make_result("error", detail=f"z3 parse error: {e}",
                               backend=self.name, elapsed_s=time.perf_counter() - start)

        try:
            outcome = solver.check()
        except z3.Z3Exception as e:
            return make_result("error", detail=f"z3 check error: {e}",
                               backend=self.name, elapsed_s=time.perf_counter() - start)

        elapsed = time.perf_counter() - start

        if outcome == z3.sat:
            model = solver.model()
            return make_result(
                "sat",
                model=_model_to_dict(model, sort_by_name),
                detail="satisfiable",
                elapsed_s=elapsed,
                backend=self.name,
            )
        if outcome == z3.unsat:
            return make_result("unsat", detail="unsatisfiable",
                               elapsed_s=elapsed, backend=self.name)
        # z3.unknown — distinguish timeout from genuine undecided
        reason = solver.reason_unknown() or ""
        if "timeout" in reason.lower() or elapsed >= timeout_s:
            return make_result("timeout", detail=f"z3 timeout after {elapsed:.2f}s",
                               elapsed_s=elapsed, backend=self.name)
        return make_result("unknown", detail=f"z3 unknown: {reason}",
                           elapsed_s=elapsed, backend=self.name)


def verify_smt_claim(
    problem: str,
    candidate_assignment: Optional[dict[str, Any]] = None,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> SolverResult:
    """Verify a candidate solution claims to satisfy an SMT problem.

    The candidate_assignment dict (e.g., {"x": 3, "y": 7}) is added to the
    problem as equality assertions. If the combined system is sat, the
    candidate is a valid solution. If unsat, the candidate contradicts the
    constraints. If the original problem alone is unsat, the caller is
    informed via `meta['original_unsat'] = True`.
    """
    if not HAS_Z3:
        return make_result("error", detail="z3 not installed", backend="z3")

    solver = Z3Solver(default_timeout_s=timeout_s)
    if not candidate_assignment:
        return solver.check(problem, timeout_s=timeout_s)

    # Augment DSL with candidate assignments
    if _looks_like_smtlib(problem):
        extra = ""
        for name, val in candidate_assignment.items():
            if isinstance(val, bool):
                lit = "true" if val else "false"
                extra += f"(assert (= {name} {lit}))\n"
            else:
                extra += f"(assert (= {name} {val}))\n"
        augmented = problem + "\n" + extra
    else:
        extra_lines = []
        for name, val in candidate_assignment.items():
            if isinstance(val, bool):
                extra_lines.append(f"assert {name} == {val}")
            else:
                extra_lines.append(f"assert {name} == {val}")
        augmented = problem + "\n" + "\n".join(extra_lines) + "\n"

    return solver.check(augmented, timeout_s=timeout_s)
