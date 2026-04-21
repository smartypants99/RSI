"""Safety review gate for weight-growth and self-edit patches.

Enforces static policy on unified diffs before they reach the repo.
Default stance: if we cannot prove a patch is safe, reject it.
"""
from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence


MAX_DIFF_LINES = 40

FORBIDDEN_IMPORTS = (
    "subprocess",
    "socket",
    "ctypes",
    "pty",
    "os.system",
    "pickle",  # deserialization RCE vector
    "marshal",
)

# Paths the safety module owns — patches must never touch these (no self-elevation).
# Second layer of defense beyond self_edit.HARD_DENY_LIST: even if the allow-list
# is somehow widened, safety.review still refuses these paths. Keep in sync with
# self_edit.HARD_DENY_LIST for the measurement-infrastructure entries.
SELF_PROTECTED = (
    "src/safety/",
    "tests/test_safety",
    # Measurement infrastructure (task #4). A self-edit that rewrites any of
    # these could corrupt the empirical record while reporting green numbers.
    "src/diagnostics/eval_partition.py",
    "src/diagnostics/solution_diversity.py",
    "src/utils/external_benchmarks.py",
)

# Names of config fields that form the measurement contract. Added lines in a
# patch that redefine any of these (e.g. `regression_revert_threshold = 99.0`)
# are rejected. Substring match on the whole added-lines blob — if the patch
# targets only allow-listed files, it has no business redefining these.
MEASUREMENT_CONTRACT_TOKENS = (
    "regression_revert_threshold",
    "anchor_eval_enabled",
    "anchor_eval_benchmarks",
    "verifier_capture_alarm_threshold",
    "heldout_questions_per_domain",
    "heldout_repetitions",
    "PARTITION_SEED",
    "HELD_OUT_ONLY",
    "SMOKE_EVAL",
)

# Code areas off-limits to auto-generated patches.
AUTH_PERSIST_PATTERNS = (
    re.compile(r"(?i)(^|/|_)auth(_|\.|entication|orize|/|$)"),
    re.compile(r"(?i)\bcredential"),
    re.compile(r"(?i)\bsecret"),
    re.compile(r"(?i)\btoken\b"),
    re.compile(r"(?i)\bpassword"),
    re.compile(r"(?i)/persistence/"),
    re.compile(r"(?i)/db/"),
    re.compile(r"(?i)migrations?/"),
)

DIFF_FILE_RE = re.compile(r"^\+\+\+ b/(.+?)\s*$", re.MULTILINE)
OLD_FILE_RE = re.compile(r"^--- a/(.+?)\s*$", re.MULTILINE)


@dataclass
class ReviewResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok

    @property
    def message(self) -> str:
        if self.ok:
            return "lgtm"
        return "blocked: " + "; ".join(self.reasons)


def _touched_paths(patch_text: str) -> list[str]:
    paths = set(DIFF_FILE_RE.findall(patch_text))
    paths.update(OLD_FILE_RE.findall(patch_text))
    paths.discard("/dev/null")
    return sorted(paths)


def _added_lines(patch_text: str) -> list[str]:
    out = []
    for line in patch_text.splitlines():
        if line.startswith("+++"):
            continue
        if line.startswith("+"):
            out.append(line[1:])
    return out


def _is_path_traversal(p: str) -> bool:
    if ".." in Path(p).parts:
        return True
    if p.startswith("/"):
        return True
    return False


def _matches_allow_list(path: str, allow_list: Sequence[str]) -> bool:
    for allowed in allow_list:
        if allowed.endswith("/"):
            if path.startswith(allowed):
                return True
        elif path == allowed:
            return True
    return False


def review_patch(patch_text: str, allow_list: Sequence[str]) -> ReviewResult:
    """Static review of a unified diff. Returns ReviewResult."""
    reasons: list[str] = []

    if not patch_text or not patch_text.strip():
        return ReviewResult(False, ["empty patch"])

    # Line budget — count only diff body (+/-), not context/header.
    body_lines = [
        ln for ln in patch_text.splitlines()
        if (ln.startswith("+") or ln.startswith("-"))
        and not ln.startswith(("+++", "---"))
    ]
    if len(body_lines) > MAX_DIFF_LINES:
        reasons.append(f"diff too large: {len(body_lines)} > {MAX_DIFF_LINES} lines")

    paths = _touched_paths(patch_text)
    if not paths:
        reasons.append("no files touched (unparseable diff)")

    for p in paths:
        if _is_path_traversal(p):
            reasons.append(f"path traversal / absolute path: {p}")
            continue
        for protected in SELF_PROTECTED:
            if p.startswith(protected):
                reasons.append(f"patch touches safety module (self-elevation): {p}")
                break
        for pat in AUTH_PERSIST_PATTERNS:
            if pat.search(p):
                reasons.append(f"patch touches auth/persistence path: {p}")
                break
        if not _matches_allow_list(p, allow_list):
            reasons.append(f"path not in allow_list: {p}")

    # Forbidden imports / calls in added lines.
    added = "\n".join(_added_lines(patch_text))
    for needle in FORBIDDEN_IMPORTS:
        # catch `import X`, `from X`, and attribute access `X.`
        pat = re.compile(
            rf"(?:^|\s)(?:import\s+{re.escape(needle)}\b"
            rf"|from\s+{re.escape(needle)}\b"
            rf"|\b{re.escape(needle)}\.)",
            re.MULTILINE,
        )
        if pat.search(added):
            reasons.append(f"forbidden import/usage: {needle}")

    # Also catch eval/exec/__import__ as common escape hatches.
    for escape in ("eval(", "exec(", "__import__(", "compile("):
        if escape in added:
            reasons.append(f"forbidden dynamic-exec primitive: {escape.rstrip('(')}")

    # Measurement-contract tokens: a patch targeting only allow-listed
    # (generator/verifier) code has no reason to mention these. If an added
    # line does, treat it as an attempt to re-define the measurement surface
    # from outside its owning modules.
    for tok in MEASUREMENT_CONTRACT_TOKENS:
        if tok in added:
            reasons.append(f"patch redefines measurement-contract token: {tok}")

    return ReviewResult(ok=len(reasons) == 0, reasons=reasons)


def safety_reviewed(allow_list: Iterable[str]) -> Callable:
    """Decorator: wrap a function that returns (patch_text, ...).

    The decorator calls review_patch and raises PermissionError on block.
    """
    allow = tuple(allow_list)

    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            patch = result[0] if isinstance(result, tuple) else result
            verdict = review_patch(patch, allow)
            if not verdict.ok:
                raise PermissionError(verdict.message)
            return result
        return wrapper
    return deco
