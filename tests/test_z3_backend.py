"""Unit tests for src/verifier/z3_backend.py (team task #4)."""

from __future__ import annotations

import pytest

z3_mod = pytest.importorskip("z3", reason="z3-solver not installed")

from src.verifier.external_solver import ExternalSolver, SolverResult
from src.verifier.z3_backend import Z3Solver, verify_smt_claim, HAS_Z3


def test_has_z3():
    assert HAS_Z3


def test_conforms_to_external_solver_protocol():
    solver = Z3Solver()
    assert isinstance(solver, ExternalSolver)
    assert solver.name == "z3"


def test_boolean_sat():
    problem = """
    declare p Bool
    declare q Bool
    assert Or(p, q)
    assert Not(And(p, q))
    """
    result = Z3Solver().check(problem)
    assert result.status == "sat"
    assert result.model is not None
    assert set(result.model.keys()) == {"p", "q"}
    assert result.model["p"] != result.model["q"]


def test_simple_integer_arithmetic_sat():
    problem = """
    declare x Int
    declare y Int
    assert x + y == 10
    assert x > 0
    assert y > 0
    """
    result = Z3Solver().check(problem)
    assert result.status == "sat"
    assert result.model is not None
    x, y = result.model["x"], result.model["y"]
    assert x + y == 10 and x > 0 and y > 0


def test_unsat_case():
    problem = """
    declare x Int
    assert x > 5
    assert x < 3
    """
    result = Z3Solver().check(problem)
    assert result.status == "unsat"
    assert result.model is None


def test_unsat_boolean():
    problem = """
    declare p Bool
    assert p
    assert Not(p)
    """
    result = Z3Solver().check(problem)
    assert result.status == "unsat"


def test_real_arithmetic():
    problem = """
    declare r Real
    assert r * r == 2
    assert r > 0
    """
    result = Z3Solver().check(problem)
    # sqrt(2) is irrational — Z3 returns sat with an algebraic number
    assert result.status in ("sat", "unknown")


def test_malformed_input_returns_error():
    result = Z3Solver().check("not a valid statement")
    assert result.status == "error"
    assert "unrecognized" in result.detail.lower() or "parse" in result.detail.lower()


def test_bad_assertion_expression_returns_error():
    problem = """
    declare x Int
    assert x @@ 5
    """
    result = Z3Solver().check(problem)
    assert result.status == "error"


def test_duplicate_declaration_returns_error():
    problem = """
    declare x Int
    declare x Int
    assert x == 1
    """
    result = Z3Solver().check(problem)
    assert result.status == "error"
    assert "duplicate" in result.detail.lower()


def test_smtlib_passthrough():
    problem = """
    (declare-const x Int)
    (declare-const y Int)
    (assert (= (+ x y) 10))
    (assert (> x 0))
    (assert (> y 0))
    """
    result = Z3Solver().check(problem)
    assert result.status == "sat"


def test_verify_smt_claim_valid_assignment():
    problem = """
    declare x Int
    declare y Int
    assert x + y == 10
    assert x > 0
    assert y > 0
    """
    result = verify_smt_claim(problem, {"x": 3, "y": 7})
    assert result.status == "sat"
    assert result.model is not None


def test_verify_smt_claim_invalid_assignment():
    problem = """
    declare x Int
    declare y Int
    assert x + y == 10
    assert x > 0
    assert y > 0
    """
    result = verify_smt_claim(problem, {"x": 4, "y": 5})  # sum=9 ≠ 10
    assert result.status == "unsat"


def test_verify_smt_claim_no_assignment_falls_through():
    problem = "declare x Int\nassert x == 42"
    result = verify_smt_claim(problem, None)
    assert result.status == "sat"
    assert result.model["x"] == 42


def test_timeout_respected():
    # A simple problem should finish well within 5s.
    problem = """
    declare x Int
    assert x == 1
    """
    result = Z3Solver().check(problem, timeout_s=5.0)
    assert result.status == "sat"
    assert result.elapsed_s < 5.0


def test_backend_name_stamped_on_result():
    result = Z3Solver().check("declare x Int\nassert x == 1")
    assert result.backend == "z3"


def test_ok_property():
    sat_result = Z3Solver().check("declare x Int\nassert x == 1")
    assert sat_result.ok
    unsat_result = Z3Solver().check("declare x Int\nassert x == 1\nassert x == 2")
    assert unsat_result.ok
    err_result = Z3Solver().check("garbage")
    assert not err_result.ok
