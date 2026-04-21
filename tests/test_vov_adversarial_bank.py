"""Tests for the VoV adversarial bank.

The bank accumulates candidates that cleared §1.4 quorum but correlated
with a post-training regression. Future VoV audits run admitted properties
against every bank entry; properties that accept a bank entry are rejected
as toothless. These tests cover:

  * Bank persistence (JSONL round-trip)
  * LRU eviction at cap (500 by default)
  * Bank presence causing VoV to reject otherwise-strong-looking properties
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest

from src.verifier.vov_adversarial_bank import (
    AdversarialBank,
    BankEntry,
    get_default_bank,
    set_default_bank,
)
from src.verifier.verifier_of_verifiers import verify_properties_trustworthy


@dataclass
class _StubProperty:
    name: str
    check_fn: Callable[[Any, Any], Any]
    stochasticity: float = 0.0
    required: bool = False


def test_bank_persists_across_instances(tmp_path: Path):
    """Entries written to disk are reloaded by a fresh AdversarialBank."""
    path = tmp_path / "bank.jsonl"
    b1 = AdversarialBank(path, max_entries=10)
    b1.append(
        problem_id="p1", candidate="def f(x): return x",
        domain="code", problem_ctx={"tests": ["assert f(1)==1"]},
        cycle=3, reason="test",
    )
    b1.append(
        problem_id="p2", candidate="garbage",
        domain="code", cycle=4, reason="test",
    )
    assert len(b1) == 2

    # Fresh instance over the same path — must reload exactly what was written.
    b2 = AdversarialBank(path, max_entries=10)
    assert len(b2) == 2
    pids = sorted(e.problem_id for e in b2.entries())
    assert pids == ["p1", "p2"]
    # JSONL sanity — one non-empty line per entry.
    with path.open() as fh:
        lines = [l for l in fh.read().splitlines() if l.strip()]
    assert len(lines) == 2
    # Each line is valid JSON and carries the canonical keys.
    obj = json.loads(lines[0])
    for key in ("bank_id", "problem_id", "candidate", "domain", "added_at"):
        assert key in obj


def test_bank_lru_evicts_oldest_triggered(tmp_path: Path):
    """When over capacity, the least-recently-triggered entries roll out."""
    path = tmp_path / "bank.jsonl"
    cap = 3
    bank = AdversarialBank(path, max_entries=cap)
    # Add entries with ascending added_at (time.time advances between appends).
    ids = []
    for i in range(5):
        e = bank.append(
            problem_id=f"p{i}", candidate=f"cand{i}",
            domain="code", cycle=i, reason="fill",
        )
        ids.append(e.bank_id)
        time.sleep(0.002)  # ensure monotonic added_at

    assert len(bank) == cap
    # The first two inserted must have been evicted (oldest last_triggered_at).
    remaining = {e.problem_id for e in bank.entries()}
    assert remaining == {"p2", "p3", "p4"}

    # mark_triggered on p2 refreshes its last_triggered_at; adding another
    # should evict p3 (now oldest), not p2.
    p2 = [e for e in bank.entries() if e.problem_id == "p2"][0]
    time.sleep(0.002)
    bank.mark_triggered(p2.bank_id)
    time.sleep(0.002)
    bank.append(
        problem_id="p5", candidate="cand5",
        domain="code", cycle=5, reason="fill",
    )
    remaining = {e.problem_id for e in bank.entries()}
    assert "p2" in remaining, "triggered entry must survive LRU"
    assert "p3" not in remaining, "oldest untouched entry must be evicted"


def test_bank_entry_causes_vov_to_reject_property(tmp_path: Path):
    """A property that ACCEPTS a bank entry must be rejected by VoV even if
    it catches every synthetic corruption."""
    path = tmp_path / "bank.jsonl"
    bank = AdversarialBank(path, max_entries=50)
    # Seed: an obviously-broken candidate string we'll want future properties
    # to reject. The "strong" property below accepts it by mistake, which is
    # exactly the toothless pattern the bank exists to catch.
    poisoned_candidate = "def f(x): return 0  # always wrong"
    bank.append(
        problem_id="bank_seed", candidate=poisoned_candidate,
        domain="code", cycle=1, reason="seeded for test",
    )

    # A property that looks strong on corruptions (rejects them all) but ALSO
    # accepts the bank entry. Check: returns True iff the source contains
    # "def f". The corruptions in _corrupt_code mangle that in several ways,
    # so it still earns a nonzero kill rate; critically it says yes to the
    # poisoned candidate.
    def accepts_bank(sol, ctx):
        if not isinstance(sol, str):
            return False, "not str"
        # Always accept anything that looks like code — including the banked
        # poisoned candidate. Also accepts the reference.
        return "def " in sol, ""

    prop = _StubProperty(name="loose", check_fn=accepts_bank)
    report = verify_properties_trustworthy(
        task_id="t_bank",
        reference_solution="def f(x):\n    return x + 1\n",
        properties=[prop],
        problem_ctx={},
        domain="code",
        adversarial_bank=bank,
    )
    assert not report.passed, f"expected fail, got: {report.reason}"
    # The property-level trust record must carry the bank-rejection reason.
    ptrust = report.properties[0]
    assert ptrust.rejected_reason is not None
    assert "adversarial-bank" in ptrust.rejected_reason.lower()


def test_empty_bank_is_noop_for_vov():
    """An empty bank must not change the verdict of existing properties."""
    def behavior_check(sol, ctx):
        if not isinstance(sol, str):
            return (False, "not code")
        ns: dict = {}
        try:
            exec(sol, ns)  # noqa: S102
            f = ns.get("f")
            if f is None:
                return (False, "no f")
            return (f(5) == 6 and f(0) == 1 and f(10) == 11, "")
        except Exception as e:
            return (False, str(e))

    empty_bank = AdversarialBank(Path("/tmp/_vov_bank_empty_shouldnotexist.jsonl"))
    empty_bank._entries = []  # force empty regardless of stale disk state
    strong = _StubProperty(name="strong", check_fn=behavior_check)
    report = verify_properties_trustworthy(
        task_id="t_empty_bank",
        reference_solution="def f(x):\n    return x + 1\n",
        properties=[strong],
        problem_ctx={},
        domain="code",
        adversarial_bank=empty_bank,
    )
    assert report.passed, report.reason


def test_default_bank_singleton_override(tmp_path: Path):
    """set_default_bank replaces the module-level bank for tests."""
    path = tmp_path / "override.jsonl"
    custom = AdversarialBank(path, max_entries=5)
    set_default_bank(custom)
    try:
        assert get_default_bank() is custom
    finally:
        set_default_bank(None)
