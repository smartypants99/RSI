"""Tests for src/verifier/simulator_backend.py — task #5 (verifier-sim).

Covers:
  * ODE numerical match (pass) + mismatch (fail)
  * Conservation preserved (pass) + violation detected (fail)
  * Dimensional mismatch rejection
  * Graceful skip when scipy missing
"""
from __future__ import annotations

import importlib
import sys

import pytest

from src.verifier import simulator_backend as sim
from src.verifier.property_engine import (
    INDEPENDENCE_CLASSES,
    PropertyKind,
    stash_problem_ctx,
    clear_problem_ctx,
)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    clear_problem_ctx()


def test_independence_class_registered():
    assert "simulation.numerical" in INDEPENDENCE_CLASSES


def test_simulation_kind_registered():
    assert PropertyKind.SIMULATION.value == "SIMULATION"


# --- ODE -----------------------------------------------------------------

@pytest.mark.skipif(not sim.simulator_available(), reason="scipy missing")
def test_ode_match_pass():
    """dy/dt = -y, y(0)=1, t=[0,1] → y(1) = 1/e ≈ 0.3679. Candidate 0.368 passes."""
    stash_problem_ctx("p1", {
        "ode": {
            "rhs": "return [-y[0]]",
            "y0": [1.0],
            "t": (0.0, 1.0),
            "target_index": 0,
            "tolerance": 1e-2,
        },
    })
    verdict, reason = sim.check_ode_match("p1", "the answer is 0.368")
    assert verdict is True, reason


@pytest.mark.skipif(not sim.simulator_available(), reason="scipy missing")
def test_ode_match_fail():
    stash_problem_ctx("p2", {
        "ode": {
            "rhs": "return [-y[0]]",
            "y0": [1.0],
            "t": (0.0, 1.0),
            "target_index": 0,
            "tolerance": 1e-3,
        },
    })
    verdict, reason = sim.check_ode_match("p2", "answer: 0.5")
    assert verdict is False
    assert "rel_err" in reason


@pytest.mark.skipif(not sim.simulator_available(), reason="scipy missing")
def test_ode_match_projectile():
    """Projectile: dv/dt = -g, dh/dt = v. y0=[v=10, h=0], t=1 → v=10-9.81=0.19, h=10-4.905=5.095."""
    stash_problem_ctx("proj", {
        "ode": {
            "rhs": "return [-9.81, y[0]]",
            "y0": [10.0, 0.0],
            "t": (0.0, 1.0),
            "target_index": 1,
            "tolerance": 1e-2,
        },
    })
    verdict, reason = sim.check_ode_match("proj", "height is 5.095 m")
    assert verdict is True, reason


def test_ode_missing_ctx_errors():
    stash_problem_ctx("p3", {})
    v, r = sim.check_ode_match("p3", "0.5")
    assert v == "ERROR"


# --- Conservation --------------------------------------------------------

def test_conservation_energy_preserved():
    """Free-fall energy conservation. Initial: m=1, v=0, h=10, g=9.81 → E=98.1.
    Candidate answer v at h=0: v = sqrt(2gh) ≈ 14.007. Final KE = 0.5*1*v^2 = 98.1."""
    stash_problem_ctx("fall", {
        "conservation": {
            "quantity": "energy",
            "initial_state": {"m": 1.0, "v": 0.0, "h": 10.0, "g": 9.81},
            "final_state_from_candidate": "0.5 * m * cand * cand",
            "tolerance": 1e-2,
        },
    })
    verdict, reason = sim.check_conservation("fall", "v = 14.007 m/s")
    assert verdict is True, reason


def test_conservation_violated():
    stash_problem_ctx("fall2", {
        "conservation": {
            "quantity": "energy",
            "initial_state": {"m": 1.0, "v": 0.0, "h": 10.0, "g": 9.81},
            "final_state_from_candidate": "0.5 * m * cand * cand",
            "tolerance": 1e-2,
        },
    })
    verdict, reason = sim.check_conservation("fall2", "v = 5.0 m/s")
    assert verdict is False
    assert "NOT conserved" in reason


def test_conservation_missing_ctx_errors():
    stash_problem_ctx("p4", {})
    v, r = sim.check_conservation("p4", "1.0")
    assert v == "ERROR"


# --- Dimensional ---------------------------------------------------------

def test_dimensional_mismatch_rejected():
    pytest.importorskip("sympy.physics.units")
    stash_problem_ctx("dim1", {
        "dimensions": {
            "expected": "meter/second",
            "candidate_unit": "kilogram",
        },
    })
    verdict, reason = sim.check_dimensional_consistency_sim("dim1", "5 kilogram")
    assert verdict is False
    assert "mismatch" in reason.lower() or "could not" in reason.lower()


def test_dimensional_match_passes():
    pytest.importorskip("sympy.physics.units")
    stash_problem_ctx("dim2", {
        "dimensions": {
            "expected": "meter/second",
            "candidate_unit": "meter/second",
        },
    })
    verdict, reason = sim.check_dimensional_consistency_sim("dim2", "3.2 m/s")
    assert verdict is True, reason


# --- Factories / graceful skip ------------------------------------------

def test_factories_build_valid_properties():
    import hashlib
    h = hashlib.sha256(b"x").hexdigest()
    p = sim.make_ode_match_property(
        problem_id="pid1", parent_problem_hash=h,
        confirmer_example="c", falsifier_example="f",
    )
    assert p.kind == PropertyKind.SIMULATION
    assert p.independence_class == "simulation.numerical"

    p2 = sim.make_conservation_property(
        problem_id="pid1", parent_problem_hash=h,
        confirmer_example="c", falsifier_example="f",
    )
    assert p2.kind == PropertyKind.CONSERVATION
    assert p2.independence_class == "conservation.global"


def test_graceful_skip_when_scipy_missing(monkeypatch):
    monkeypatch.setattr(sim, "_HAS_SCIPY", False)
    stash_problem_ctx("pskip", {"ode": {"rhs": "return [0]", "y0": [1], "t": (0, 1)}})
    v, r = sim.check_ode_match("pskip", "1.0")
    assert v == "ERROR"
    assert "scipy" in r.lower()


def test_graceful_skip_when_sympy_physics_missing(monkeypatch):
    monkeypatch.setattr(sim, "_HAS_SYMPY_PHYSICS", False)
    stash_problem_ctx("pskip2", {"dimensions": {"expected": "meter"}})
    v, r = sim.check_dimensional_consistency_sim("pskip2", "1 m")
    assert v == "ERROR"
