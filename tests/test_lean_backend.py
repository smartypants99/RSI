"""Tests for src/verifier/lean_backend.py.

These tests exercise import, graceful-skip on missing lean, source
extraction, and the property_engine builtin registration. They do NOT
require a real `lean` binary — CI doesn't have Lean installed, and the
backend's whole point is to degrade gracefully.
"""

from __future__ import annotations

import pytest

from src.verifier import lean_backend
from src.verifier import property_engine as pe


def test_lean_backend_registers_builtin():
    assert "lean_proof_valid" in pe._BUILTIN
    prop = pe._BUILTIN["lean_proof_valid"]
    assert prop.independence_class == "proof.formal"
    assert prop.kind == pe.PropertyKind.LEAN_PROOF


def test_propertykind_has_lean_proof():
    assert pe.PropertyKind.LEAN_PROOF.value == "LEAN_PROOF"


def test_proof_formal_in_independence_classes():
    assert "proof.formal" in pe.INDEPENDENCE_CLASSES


def test_lean_available_returns_bool():
    # Should not raise; value depends on host.
    assert isinstance(lean_backend.lean_available(), bool)


def test_b_lean_proof_valid_graceful_skip_when_missing(monkeypatch):
    """If lean is not installed, the builtin returns ERROR (quorum skip)."""
    monkeypatch.setattr(lean_backend, "lean_available", lambda: False)
    verdict, reason = lean_backend._b_lean_proof_valid(
        "problem_x", "theorem t : True := trivial"
    )
    assert verdict == "ERROR"
    assert "lean4 not installed" in reason


def test_b_lean_proof_valid_empty_source_is_failure(monkeypatch):
    """Empty candidate — FAIL (not ERROR). lean missing → ERROR wins first."""
    monkeypatch.setattr(lean_backend, "lean_available", lambda: True)
    # Avoid actually invoking lean on empty source by patching check_lean_proof
    from src.verifier.external_solver import SubprocessOutcome
    monkeypatch.setattr(
        lean_backend, "check_lean_proof",
        lambda *a, **kw: SubprocessOutcome(-1, "", "empty", 0.0),
    )
    # _extract_lean_source returns "" from empty strings → (False, "no extractable ...")
    verdict, reason = lean_backend._b_lean_proof_valid("p", "")
    assert verdict is False
    assert "no extractable" in reason


def test_extract_lean_source_from_fence():
    text = "Here is my proof:\n```lean\ntheorem t : 1 + 1 = 2 := rfl\n```\nDone."
    extracted = lean_backend._extract_lean_source(text)
    assert extracted is not None
    assert "theorem t" in extracted
    assert "```" not in extracted


def test_extract_lean_source_bare():
    text = "theorem t : True := trivial"
    assert lean_backend._extract_lean_source(text) == text


def test_wrap_if_bare_adds_theorem_header():
    # No `theorem` keyword, goal provided → wrapper kicks in.
    body = "rfl"
    wrapped = lean_backend._wrap_if_bare(body, goal="1 + 1 = 2")
    assert "theorem candidate" in wrapped
    assert ": 1 + 1 = 2" in wrapped
    assert "rfl" in wrapped


def test_wrap_if_bare_passes_through_full_decl():
    full = "theorem foo : True := trivial"
    assert lean_backend._wrap_if_bare(full, goal="anything") == full


def test_wrap_if_bare_no_goal_returns_source_unchanged():
    body = "some raw proof term"
    assert lean_backend._wrap_if_bare(body, goal=None) == body


@pytest.mark.skipif(
    not lean_backend.lean_available(),
    reason="lean4 binary not installed on host",
)
def test_check_lean_proof_trivial_theorem():
    """Real-lean smoke test. Skipped on hosts without `lean`."""
    out = lean_backend.check_lean_proof("theorem t : True := trivial")
    assert out.ok, f"expected ok, got rc={out.returncode} err={out.tail()}"


@pytest.mark.skipif(
    not lean_backend.lean_available(),
    reason="lean4 binary not installed on host",
)
def test_check_lean_proof_bad_proof_rejected():
    out = lean_backend.check_lean_proof("theorem t : False := trivial")
    assert not out.ok
