"""Worktree sandbox tests."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.safety.worktree_sandbox import worktree_sandbox


REPO_ROOT = Path(__file__).resolve().parent.parent


def _has_git() -> bool:
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_git(), reason="git not available")
def test_sandbox_creates_and_cleans_up():
    with worktree_sandbox(REPO_ROOT) as sandbox:
        assert sandbox.exists()
        assert (sandbox / ".git").exists()
        assert str(sandbox).startswith("/tmp/rsi_sandbox_") or "rsi_sandbox_" in str(sandbox)
    # after exit, gone
    assert not sandbox.exists()


@pytest.mark.skipif(not _has_git(), reason="git not available")
def test_sandbox_isolated_from_main_tree(tmp_path):
    with worktree_sandbox(REPO_ROOT) as sandbox:
        marker = sandbox / "SANDBOX_ONLY.txt"
        marker.write_text("scratch")
        assert marker.exists()
        # Not present in main repo
        assert not (REPO_ROOT / "SANDBOX_ONLY.txt").exists()
    assert not (REPO_ROOT / "SANDBOX_ONLY.txt").exists()


def test_sandbox_rejects_non_git_dir(tmp_path):
    with pytest.raises(RuntimeError):
        with worktree_sandbox(tmp_path):
            pass
