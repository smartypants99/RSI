"""Safety gate tests — malicious patches must be blocked."""
from __future__ import annotations

import pytest

from src.safety.review import review_patch, safety_reviewed, MAX_DIFF_LINES


ALLOW_TRAINER = ("src/trainer/",)
ALLOW_ORCH = ("src/orchestrator/self_edit.py", "src/orchestrator/")


def _diff(path: str, added_lines: list[str], removed_lines: list[str] | None = None) -> str:
    removed_lines = removed_lines or []
    hunk = [f"--- a/{path}", f"+++ b/{path}", "@@ -1,1 +1,1 @@"]
    for ln in removed_lines:
        hunk.append(f"-{ln}")
    for ln in added_lines:
        hunk.append(f"+{ln}")
    return "\n".join(hunk) + "\n"


# ---- benign path ----

def test_benign_trainer_patch_passes():
    patch = _diff("src/trainer/growth.py", ["lr = 1e-4", "steps = 10"])
    r = review_patch(patch, ALLOW_TRAINER)
    assert r.ok, r.reasons
    assert r.message == "lgtm"


# ---- malicious: subprocess escape ----

def test_subprocess_import_blocked():
    patch = _diff("src/trainer/growth.py", ["import subprocess", "subprocess.run(['rm','-rf','/'])"])
    r = review_patch(patch, ALLOW_TRAINER)
    assert not r.ok
    assert any("subprocess" in x for x in r.reasons)


def test_socket_import_blocked():
    patch = _diff("src/trainer/growth.py", ["from socket import socket"])
    r = review_patch(patch, ALLOW_TRAINER)
    assert not r.ok


def test_ctypes_blocked():
    patch = _diff("src/trainer/growth.py", ["import ctypes", "ctypes.CDLL('libc')"])
    r = review_patch(patch, ALLOW_TRAINER)
    assert not r.ok


def test_eval_exec_blocked():
    patch = _diff("src/trainer/growth.py", ["eval('__import__(\"os\").system(\"id\")')"])
    r = review_patch(patch, ALLOW_TRAINER)
    assert not r.ok
    assert any("eval" in x for x in r.reasons)


def test_dunder_import_blocked():
    patch = _diff("src/trainer/growth.py", ["m = __import__('os')"])
    r = review_patch(patch, ALLOW_TRAINER)
    assert not r.ok


# ---- malicious: self-modify / self-elevate ----

def test_cannot_modify_safety_module():
    patch = _diff("src/safety/review.py", ["MAX_DIFF_LINES = 100000"])
    r = review_patch(patch, ("src/safety/", "src/trainer/"))
    assert not r.ok
    assert any("self-elevation" in x or "safety module" in x for x in r.reasons)


def test_cannot_modify_safety_tests():
    patch = _diff("tests/test_safety_review.py", ["assert True"])
    r = review_patch(patch, ("tests/", "src/trainer/"))
    assert not r.ok


# ---- malicious: path traversal ----

def test_path_traversal_dotdot_blocked():
    patch = _diff("../../etc/passwd", ["root::0:0:"])
    r = review_patch(patch, ("src/trainer/",))
    assert not r.ok
    assert any("traversal" in x or "absolute" in x for x in r.reasons)


def test_absolute_path_blocked():
    patch = _diff("/etc/passwd", ["x"])
    r = review_patch(patch, ("/etc/",))
    assert not r.ok


# ---- auth/persistence guardrail ----

def test_auth_path_blocked():
    patch = _diff("src/trainer/auth_helper.py", ["KEY = 'x'"])
    r = review_patch(patch, ("src/trainer/",))
    assert not r.ok
    assert any("auth" in x.lower() for x in r.reasons)


def test_credentials_path_blocked():
    patch = _diff("src/trainer/credentials.py", ["x = 1"])
    r = review_patch(patch, ("src/trainer/",))
    assert not r.ok


# ---- allow-list ----

def test_path_outside_allow_list_blocked():
    patch = _diff("src/orchestrator/loop.py", ["x = 1"])
    r = review_patch(patch, ("src/trainer/",))
    assert not r.ok
    assert any("allow_list" in x for x in r.reasons)


# ---- size budget ----

def test_oversize_diff_blocked():
    big = [f"x{i} = {i}" for i in range(MAX_DIFF_LINES + 5)]
    patch = _diff("src/trainer/growth.py", big)
    r = review_patch(patch, ALLOW_TRAINER)
    assert not r.ok
    assert any("too large" in x for x in r.reasons)


def test_at_budget_ok():
    lines = [f"x{i} = {i}" for i in range(MAX_DIFF_LINES)]
    patch = _diff("src/trainer/growth.py", lines)
    r = review_patch(patch, ALLOW_TRAINER)
    assert r.ok, r.reasons


# ---- decorator ----

def test_decorator_raises_on_block():
    @safety_reviewed(allow_list=("src/trainer/",))
    def bad_patch():
        return _diff("src/trainer/growth.py", ["import subprocess"])

    with pytest.raises(PermissionError):
        bad_patch()


def test_decorator_passes_ok():
    @safety_reviewed(allow_list=("src/trainer/",))
    def good_patch():
        return _diff("src/trainer/growth.py", ["x = 1"])

    assert good_patch()  # returns patch text truthy


# ---- empty / malformed ----

def test_empty_patch_blocked():
    r = review_patch("", ("src/trainer/",))
    assert not r.ok


def test_unparseable_patch_blocked():
    r = review_patch("not a diff", ("src/trainer/",))
    assert not r.ok
