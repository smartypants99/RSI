"""Unit tests for src.orchestrator.self_edit — validation, policy, and orchestration.

No GPU, no real model. The orchestration test uses pure-function stubs for
`model_propose` and `smoke_eval` and runs the git-worktree path against the
actual repo on disk.
"""
from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path

import pytest

from src.orchestrator import self_edit as se


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_should_run_meta_cycle():
    assert not se.should_run_meta_cycle(0, 8)
    assert not se.should_run_meta_cycle(7, 8)
    assert se.should_run_meta_cycle(8, 8)
    assert se.should_run_meta_cycle(16, 8)
    assert not se.should_run_meta_cycle(5, 0)   # disabled
    assert not se.should_run_meta_cycle(5, -1)  # disabled


def test_extract_patch_block_happy():
    raw = "noise\n<PATCH>\n--- a/x\n+++ b/x\n@@\n-a\n+b\n</PATCH>\ntrailing"
    body = se.extract_patch_block(raw)
    assert body and body.startswith("--- a/x")
    assert "trailing" not in body


def test_extract_patch_block_missing():
    assert se.extract_patch_block("no patch here") is None
    assert se.extract_patch_block("") is None


def test_extract_target_files():
    diff = "--- a/src/generator/foo.py\n+++ b/src/generator/foo.py\n@@\n-x\n+y\n"
    assert se.extract_target_files(diff) == ["src/generator/foo.py"]


def test_count_diff_hunks():
    diff = "--- a/x\n+++ b/x\n@@\n-a\n+b\n+c\n"
    added, removed = se.count_diff_hunks(diff)
    assert added == 2 and removed == 1


# ---------------------------------------------------------------------------
# Validation policy
# ---------------------------------------------------------------------------


def _good_diff():
    return (
        "--- a/src/generator/foo.py\n"
        "+++ b/src/generator/foo.py\n"
        "@@ -1 +1 @@\n"
        "-x = 1\n"
        "+x = 2\n"
    )


def test_validate_allows_generator_path():
    r = se.validate_patch(_good_diff())
    assert r.ok, r.reasons


def test_validate_rejects_loop_py():
    diff = _good_diff().replace("src/generator/foo.py", "src/orchestrator/loop.py")
    r = se.validate_patch(diff)
    assert not r.ok
    assert any("deny-list" in reason for reason in r.reasons)


def test_validate_rejects_config_py():
    diff = _good_diff().replace("src/generator/foo.py", "src/utils/config.py")
    r = se.validate_patch(diff)
    assert not r.ok


def test_validate_rejects_trainer():
    diff = _good_diff().replace("src/generator/foo.py", "src/trainer/custom_lora.py")
    r = se.validate_patch(diff)
    assert not r.ok


def test_validate_rejects_self_edit_itself():
    diff = _good_diff().replace("src/generator/foo.py", "src/orchestrator/self_edit.py")
    r = se.validate_patch(diff)
    assert not r.ok


def test_validate_rejects_safety_module():
    diff = _good_diff().replace("src/generator/foo.py", "src/safety/review.py")
    r = se.validate_patch(diff)
    assert not r.ok


def test_validate_rejects_tests():
    diff = _good_diff().replace("src/generator/foo.py", "tests/test_smoke.py")
    r = se.validate_patch(diff)
    assert not r.ok


def test_validate_rejects_too_large():
    body = "".join(f"+line{i}\n" for i in range(50))
    diff = f"--- a/src/generator/foo.py\n+++ b/src/generator/foo.py\n@@\n{body}"
    r = se.validate_patch(diff, max_diff_lines=40)
    assert not r.ok
    assert any("too large" in reason for reason in r.reasons)


def test_validate_rejects_unknown_path():
    diff = _good_diff().replace("src/generator/foo.py", "src/diagnostics/engine.py")
    r = se.validate_patch(diff)
    assert not r.ok
    assert any("allow-list" in reason for reason in r.reasons)


def test_validate_rejects_empty():
    assert not se.validate_patch("").ok
    assert not se.validate_patch("   ").ok


def test_validate_rejects_binary():
    diff = _good_diff() + "\x00binary"
    r = se.validate_patch(diff)
    assert not r.ok


def test_validate_rejects_shallow_path():
    diff = _good_diff().replace("src/generator/foo.py", "src/top.py")
    r = se.validate_patch(diff)
    assert not r.ok


# ---------------------------------------------------------------------------
# History persistence
# ---------------------------------------------------------------------------


def test_history_roundtrip(tmp_path: Path):
    p = tmp_path / "h.jsonl"
    se.append_history(p, {"attempt_id": "a1", "decision": "merged"})
    se.append_history(p, {"attempt_id": "a2", "decision": "rejected_validation"})
    recs = se.load_history(p)
    assert len(recs) == 2
    assert recs[0]["decision"] == "merged"
    assert recs[1]["attempt_id"] == "a2"


# ---------------------------------------------------------------------------
# Full meta-cycle orchestration — rejected paths
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_meta_cycle_no_patch_block_rejected(tmp_path: Path):
    cfg = se.SelfEditConfig(history_path=tmp_path / "h.jsonl")
    out = se.run_self_edit_meta_cycle(
        cycle=8,
        repo_root=_repo_root(),
        candidate_path="src/generator/__init__.py",
        delta_history=[0.01, 0.02],
        model_propose=lambda prompt: "no patch block here",
        smoke_eval=lambda p: 0.0,
        config=cfg,
    )
    assert out.decision == "rejected_validation"
    assert any("PATCH" in r for r in out.reasons)
    # History was written.
    assert (tmp_path / "h.jsonl").exists()
    recs = se.load_history(tmp_path / "h.jsonl")
    assert len(recs) == 1 and recs[0]["decision"] == "rejected_validation"


def test_meta_cycle_deny_list_violation_rejected(tmp_path: Path):
    cfg = se.SelfEditConfig(history_path=tmp_path / "h.jsonl")
    bad_diff = (
        "--- a/src/orchestrator/loop.py\n"
        "+++ b/src/orchestrator/loop.py\n"
        "@@ -1 +1 @@\n"
        "-x\n"
        "+y\n"
    )
    raw = f"<PATCH>\n{bad_diff}</PATCH>"
    out = se.run_self_edit_meta_cycle(
        cycle=8,
        repo_root=_repo_root(),
        candidate_path="src/generator/__init__.py",
        delta_history=[],
        model_propose=lambda prompt: raw,
        smoke_eval=lambda p: 1.0,
        config=cfg,
    )
    assert out.decision == "rejected_validation"
    assert any("deny-list" in r for r in out.reasons)


def test_meta_cycle_safety_missing_fails_closed(tmp_path: Path, monkeypatch):
    """If src.safety.review is unavailable, must fail closed (never merge)."""
    # Force _safety_review's import to raise by patching the helper directly.
    monkeypatch.setattr(
        se, "_safety_review",
        lambda diff, allow: (False, ["safety module unavailable: forced"])
    )
    cfg = se.SelfEditConfig(history_path=tmp_path / "h.jsonl")
    good = _good_diff_text()
    out = se.run_self_edit_meta_cycle(
        cycle=8,
        repo_root=_repo_root(),
        candidate_path="src/generator/__init__.py",
        delta_history=[],
        model_propose=lambda prompt: f"<PATCH>\n{good}</PATCH>",
        smoke_eval=lambda p: 1.0,
        config=cfg,
    )
    assert out.decision == "rejected_safety"


def _good_diff_text() -> str:
    # Target a file that actually exists in the repo so `git apply --check`
    # can at least parse headers — but we won't reach apply in the tests that
    # use this (they stop at safety or validation).
    return (
        "--- a/src/generator/__init__.py\n"
        "+++ b/src/generator/__init__.py\n"
        "@@ -1 +1,2 @@\n"
        " \n"
        "+# touched\n"
    )


def test_allow_list_for_safety_translation():
    got = se._allow_list_for_safety(["src/generator/*.py", "src/verifier/*.py"])
    assert got == ["src/generator/", "src/verifier/"]


def test_safety_review_integration_rejects_deny_path():
    """End-to-end: real src.safety.review should reject loop.py."""
    bad = (
        "--- a/src/orchestrator/loop.py\n"
        "+++ b/src/orchestrator/loop.py\n"
        "@@ -1 +1 @@\n"
        "-x\n"
        "+y\n"
    )
    ok, reasons = se._safety_review(bad, se.DEFAULT_ALLOW_LIST)
    assert not ok
    assert reasons  # must explain why


def test_meta_cycle_candidate_missing(tmp_path: Path):
    cfg = se.SelfEditConfig(history_path=tmp_path / "h.jsonl")
    out = se.run_self_edit_meta_cycle(
        cycle=8,
        repo_root=_repo_root(),
        candidate_path="src/generator/does_not_exist_zzz.py",
        delta_history=[],
        model_propose=lambda prompt: "",
        smoke_eval=lambda p: 0.0,
        config=cfg,
    )
    assert out.decision == "error"
    assert any("failed to read candidate" in r for r in out.reasons)
