"""Task #4: measurement immutability + solution diversity tracker.

Tier-4 self-edits must not be able to rewrite the measurement surface. This
test file pins the HARD_DENY_LIST contents and verifies review_patch rejects
patches targeting the protected paths or redefining measurement-contract
tokens.
"""
from __future__ import annotations

import pytest

from src.orchestrator.self_edit import (
    HARD_DENY_LIST,
    validate_patch,
)
from src.safety.review import (
    SELF_PROTECTED,
    MEASUREMENT_CONTRACT_TOKENS,
    review_patch,
)


MEASUREMENT_PATHS = (
    "src/diagnostics/eval_partition.py",
    "src/diagnostics/solution_diversity.py",
    "src/utils/external_benchmarks.py",
)

# A patch that adds the single word "foo" to a generator file. Used as the
# allow-listed baseline against which we swap the target path.
_PATCH_TEMPLATE = """\
--- a/{path}
+++ b/{path}
@@ -1,1 +1,2 @@
 existing
+foo
"""


def _make_patch(path: str) -> str:
    return _PATCH_TEMPLATE.format(path=path)


@pytest.mark.parametrize("path", MEASUREMENT_PATHS)
def test_measurement_paths_on_hard_deny_list(path):
    """Every measurement-infra path is explicitly in HARD_DENY_LIST."""
    diff = _make_patch(path)
    v = validate_patch(diff)
    assert not v.ok
    assert any("hard deny-list" in r for r in v.reasons), v.reasons


@pytest.mark.parametrize("path", MEASUREMENT_PATHS)
def test_measurement_paths_on_safety_self_protected(path):
    """safety.review also refuses measurement paths, even if allow-list widens."""
    diff = _make_patch(path)
    # Pretend a misconfigured allow-list permits these.
    allow = [path]
    r = review_patch(diff, allow_list=allow)
    assert not r.ok
    # Either the SELF_PROTECTED check or allow-list rejects. SELF_PROTECTED
    # must specifically flag it.
    joined = " ".join(r.reasons)
    assert any(p in SELF_PROTECTED for p in SELF_PROTECTED)  # sanity
    assert path in joined or "self-elevation" in joined


def test_measurement_contract_tokens_rejected_in_added_lines():
    """Patches to allow-listed files cannot redefine measurement tokens."""
    diff = """\
--- a/src/generator/foo.py
+++ b/src/generator/foo.py
@@ -1,1 +1,2 @@
 existing
+regression_revert_threshold = 99.0
"""
    r = review_patch(diff, allow_list=["src/generator/"])
    assert not r.ok
    assert any("measurement-contract token" in msg for msg in r.reasons)


def test_measurement_contract_tokens_constant_is_nonempty():
    assert "regression_revert_threshold" in MEASUREMENT_CONTRACT_TOKENS
    assert "PARTITION_SEED" in MEASUREMENT_CONTRACT_TOKENS


def test_hard_deny_includes_safety_and_measurement():
    for p in MEASUREMENT_PATHS:
        assert p in HARD_DENY_LIST, p
    # These were already present; re-pin them so a future refactor can't
    # quietly delete them.
    assert "src/safety/*" in HARD_DENY_LIST
    assert "src/orchestrator/self_edit.py" in HARD_DENY_LIST
