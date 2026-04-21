"""Verifier adequacy — curated-bundle scoring for proposed property_fns.

Complements `verifier_of_verifiers.py` (which synthesizes corruptions on the
fly from the current reference solution) with a FIXED curated bundle per
domain. The bundle is frozen so we can measure drift: if a property used to
score well and now scores below threshold on the same bundle, the model has
produced new tricky-similar-to-correct solutions that the property no
longer catches — prune it.

Per-task bundle triple:
  (correct, wrong_similar, wrong_obvious)

A property_fn is a callable (solution, ctx) -> bool or (bool, str).
Adequate iff:
  TPR (flags wrong_* as FAIL) ≥ TPR_THRESHOLD     (default 0.60)
  TNR (accepts correct as PASS) ≥ TNR_THRESHOLD   (default 0.70)

The curated bundles live under `adequacy_fixtures/<domain>.json`. Each file
is a list of triples — we ship 20 per domain. Fixtures are authoritative
ground truth for this suite; adding entries hardens the gate.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).parent / "adequacy_fixtures"

TPR_THRESHOLD = 0.60
TNR_THRESHOLD = 0.70

PropertyFn = Callable[..., Any]


@dataclass(frozen=True)
class Triple:
    """One fixture triple: a correct solution plus two wrong variants."""
    correct: Any
    wrong_similar: Any   # plausible failure mode close to correct
    wrong_obvious: Any   # blatant bug the weakest property should still catch
    ctx: Any = None


@dataclass
class AdequacyScore:
    """TPR/TNR report for a property_fn against a curated bundle."""
    property_name: str
    domain: str
    n_triples: int
    correct_accepted: int       # property said PASS on correct
    wrong_similar_caught: int   # property said FAIL on wrong_similar
    wrong_obvious_caught: int   # property said FAIL on wrong_obvious
    tpr: float = 0.0            # (similar_caught + obvious_caught) / (2·n)
    tnr: float = 0.0            # correct_accepted / n
    crashed_on_correct: int = 0
    adequate: bool = False
    reason: str = ""


def _run_once(fn: PropertyFn, solution: Any, ctx: Any) -> tuple[bool, bool]:
    """Invoke a property_fn tolerantly.

    Supports (sol, ctx) or (sol,) signatures and (bool, msg) or bool returns.
    Returns (verdict, crashed).
    """
    try:
        try:
            result = fn(solution, ctx)
        except TypeError as e:
            msg = str(e)
            if "positional argument" in msg or "takes" in msg or "argument" in msg:
                result = fn(solution)
            else:
                raise
    except Exception:
        return (False, True)
    if isinstance(result, tuple) and len(result) == 2:
        return (bool(result[0]), False)
    return (bool(result), False)


def score_verifier(
    property_fn: PropertyFn,
    test_bundle: Iterable[Triple],
    *,
    property_name: str = "<unnamed>",
    domain: str = "unknown",
    tpr_threshold: float = TPR_THRESHOLD,
    tnr_threshold: float = TNR_THRESHOLD,
) -> AdequacyScore:
    """Score a property_fn against a curated bundle of triples.

    TPR counts wrong_* variants flagged as FAIL (crash counts as caught —
    a crash on a wrong input is a valid reject). TNR counts correct
    solutions accepted as PASS (crash on a correct input does NOT count
    as accepted — a property that universally crashes must not earn
    credit).
    """
    triples = list(test_bundle)
    n = len(triples)
    if n == 0:
        return AdequacyScore(
            property_name=property_name, domain=domain, n_triples=0,
            correct_accepted=0, wrong_similar_caught=0, wrong_obvious_caught=0,
            reason="empty bundle",
        )

    correct_accepted = 0
    crashed_on_correct = 0
    similar_caught = 0
    obvious_caught = 0

    for t in triples:
        verdict_c, crashed_c = _run_once(property_fn, t.correct, t.ctx)
        if crashed_c:
            crashed_on_correct += 1
        elif verdict_c:
            correct_accepted += 1

        verdict_s, crashed_s = _run_once(property_fn, t.wrong_similar, t.ctx)
        if crashed_s or not verdict_s:
            similar_caught += 1

        verdict_o, crashed_o = _run_once(property_fn, t.wrong_obvious, t.ctx)
        if crashed_o or not verdict_o:
            obvious_caught += 1

    tpr = (similar_caught + obvious_caught) / (2 * n)
    tnr = correct_accepted / n

    adequate = tpr >= tpr_threshold and tnr >= tnr_threshold
    reasons = []
    if tpr < tpr_threshold:
        reasons.append(f"TPR {tpr:.2f} < {tpr_threshold}")
    if tnr < tnr_threshold:
        reasons.append(f"TNR {tnr:.2f} < {tnr_threshold}")
    reason = "adequate" if adequate else "; ".join(reasons)

    return AdequacyScore(
        property_name=property_name,
        domain=domain,
        n_triples=n,
        correct_accepted=correct_accepted,
        wrong_similar_caught=similar_caught,
        wrong_obvious_caught=obvious_caught,
        tpr=tpr,
        tnr=tnr,
        crashed_on_correct=crashed_on_correct,
        adequate=adequate,
        reason=reason,
    )


def load_fixture_bundle(domain: str, *, fixtures_dir: Path = FIXTURES_DIR) -> list[Triple]:
    """Load curated triples for `domain` from the JSON fixture file."""
    path = fixtures_dir / f"{domain}.json"
    if not path.exists():
        raise FileNotFoundError(f"no adequacy fixture for domain {domain!r} at {path}")
    data = json.loads(path.read_text())
    triples = []
    for row in data:
        triples.append(Triple(
            correct=row["correct"],
            wrong_similar=row["wrong_similar"],
            wrong_obvious=row["wrong_obvious"],
            ctx=row.get("ctx"),
        ))
    return triples


# ─── decay / prune ────────────────────────────────────────────────────────


@dataclass
class LibraryEntry:
    """A property living in the long-term library with its adequacy history."""
    property_name: str
    domain: str
    property_fn: PropertyFn
    score_history: list[AdequacyScore] = field(default_factory=list)
    cycles_since_rescore: int = 0


def rescore_and_prune(
    entries: list[LibraryEntry],
    *,
    tpr_threshold: float = TPR_THRESHOLD,
    tnr_threshold: float = TNR_THRESHOLD,
    fixtures_dir: Path = FIXTURES_DIR,
) -> tuple[list[LibraryEntry], list[tuple[LibraryEntry, AdequacyScore]]]:
    """Re-run adequacy on every entry; partition into (kept, pruned).

    Called from the pipeline every N cycles (spec: 10). A property whose
    latest score falls below threshold is pruned — even if it was admitted
    originally — because the model has learned to produce solutions it no
    longer catches.
    """
    kept: list[LibraryEntry] = []
    pruned: list[tuple[LibraryEntry, AdequacyScore]] = []
    for entry in entries:
        try:
            bundle = load_fixture_bundle(entry.domain, fixtures_dir=fixtures_dir)
        except FileNotFoundError:
            # No fixture for this domain — conservative: keep the entry,
            # but log. We don't prune on missing-fixture.
            logger.warning("no fixture for domain=%s, skipping rescore for %s",
                           entry.domain, entry.property_name)
            kept.append(entry)
            continue
        score = score_verifier(
            entry.property_fn, bundle,
            property_name=entry.property_name, domain=entry.domain,
            tpr_threshold=tpr_threshold, tnr_threshold=tnr_threshold,
        )
        entry.score_history.append(score)
        entry.cycles_since_rescore = 0
        if score.adequate:
            kept.append(entry)
        else:
            pruned.append((entry, score))
    return kept, pruned
