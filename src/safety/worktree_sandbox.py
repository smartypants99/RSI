"""Git worktree sandbox for testing patches without touching the live tree."""
from __future__ import annotations

import contextlib
import shutil
import subprocess  # noqa: S404 — sandbox needs it; this file is hand-written, not generated
import tempfile
import uuid
from pathlib import Path
from typing import Iterator


def _run(cmd: list[str], cwd: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


@contextlib.contextmanager
def worktree_sandbox(repo_root: str | Path, branch: str | None = None) -> Iterator[Path]:
    """Create a throwaway git worktree rooted at /tmp/rsi_sandbox_<id>.

    Yields the Path to the sandbox. On exit, force-removes the worktree and
    deletes the temp directory. Caller should run their callable inside the
    yielded directory.
    """
    repo_root = Path(repo_root).resolve()
    if not (repo_root / ".git").exists():
        raise RuntimeError(f"not a git repo: {repo_root}")

    sandbox_id = uuid.uuid4().hex[:8]
    sandbox = Path(tempfile.gettempdir()) / f"rsi_sandbox_{sandbox_id}"
    if sandbox.exists():
        shutil.rmtree(sandbox, ignore_errors=True)

    ref = branch or "HEAD"
    # Detached worktree — no branch creation, cheap.
    _run(["git", "worktree", "add", "--detach", str(sandbox), ref], cwd=str(repo_root))
    try:
        yield sandbox
    finally:
        # Best-effort cleanup; never raise from __exit__.
        try:
            _run(["git", "worktree", "remove", "--force", str(sandbox)], cwd=str(repo_root))
        except Exception:
            pass
        if sandbox.exists():
            shutil.rmtree(sandbox, ignore_errors=True)
        try:
            _run(["git", "worktree", "prune"], cwd=str(repo_root))
        except Exception:
            pass
