"""Cross-cycle library: property + proposer few-shot banks.

The RSI critique: a pipeline where only model weights change is not
self-improving — it's just fine-tuning. For the pipeline itself to improve,
the library of verification properties and the pool of "good" proposer
exemplars must accumulate across cycles and feed back into the next cycle's
prompts. This module provides both banks.

Property bank
-------------
Ranks admitted properties by:
    score = rejected_adversarial_count * confirmer_pass_rate

    - rejected_adversarial_count: how many VoV corruption records this
      property correctly FAILed. Direct evidence of discriminative power.
    - confirmer_pass_rate: fraction of primary (non-adversarial) appearances
      in which it PASSed. A property that flags every real candidate is
      toothless — this term downweights it.

The top-k is rendered as short few-shot blurbs (name + description +
independence_class + kill count) that the proposer prompt includes BEFORE
the canonical problem example. We deliberately render the property's
*semantic* description, not its source code — the source code is 4 kB and
would blow the context. The bank is ADVISORY: the model still authors
fresh properties per problem; the bank is just inspiration for shape.

Proposer exemplar bank
----------------------
Ranks ProblemRegistry records by their quorum-accept count — a problem
that produced multiple accepted candidates has good (problem, reference,
tests) self-consistency and is worth showing as a shape template. We show
prompt text + entry/reference/tests when available.

Persistence audit
-----------------
Both banks read from the session registries (PropertyRegistry,
VerificationLog, ProblemRegistry, TrainingPool) — they are append-only
JSONL on disk, so readers are naturally pure (no in-memory caches to sync).
The caller passes the RSIRegistries instance; cross-session reading is the
orchestrator's responsibility (via a stable sid — see OrchestratorConfig.run_id).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


# ─── property bank ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class PropertyExemplar:
    """One rendered property entry for a few-shot prompt block."""
    property_id: str
    name: str
    description: str
    independence_class: str
    kind: str
    rejected_adversarial_count: int
    confirmer_pass_count: int
    confirmer_total: int
    score: float

    @property
    def confirmer_pass_rate(self) -> float:
        if self.confirmer_total <= 0:
            return 0.0
        return self.confirmer_pass_count / self.confirmer_total


def _iter_verdicts(verification_log: Any) -> Iterable[tuple[bool, bool, dict]]:
    """Yield (adversarial, quorum_accepted, verdict_dict) for each per-property row."""
    if verification_log is None:
        return
    for rec in verification_log.iter_records():
        adv = bool(rec.get("adversarial", False))
        qa = bool(rec.get("quorum_accepted", False))
        for v in rec.get("per_property_verdicts", []) or []:
            if isinstance(v, dict):
                yield adv, qa, v


def _score_properties(verification_log: Any) -> dict[str, dict]:
    """Aggregate per-property stats from a VerificationLog.

    Returns {property_id: {"rej_adv": int, "conf_pass": int, "conf_total": int}}.
    """
    stats: dict[str, dict] = defaultdict(lambda: {"rej_adv": 0, "conf_pass": 0, "conf_total": 0})
    for adversarial, _qa, v in _iter_verdicts(verification_log):
        pid = v.get("property_id") or ""
        if not pid:
            continue
        # per_property_verdicts rows can carry either bool `passed` (registry
        # record shape) or verdict string "PASS"/"FAIL"/"ERROR" (PropertyVerdict
        # shape). Normalize both.
        passed = v.get("passed")
        if passed is None:
            verdict = str(v.get("verdict", "")).upper()
            passed = (verdict == "PASS")
        passed = bool(passed)
        if adversarial:
            # In the adversarial pass the property's job is to FAIL the
            # corrupted candidate. A FAIL here is a "rejected adversarial".
            # ERRORs don't count either way.
            verdict_str = str(v.get("verdict", "")).upper()
            is_error = verdict_str == "ERROR"
            if not is_error and not passed:
                stats[pid]["rej_adv"] += 1
        else:
            stats[pid]["conf_total"] += 1
            if passed:
                stats[pid]["conf_pass"] += 1
    return stats


def top_k_admitted_properties(
    property_registry: Any,
    verification_log: Any,
    *,
    k: int = 5,
    min_vov_score: float = 1.0,
) -> list[PropertyExemplar]:
    """Rank admitted properties by rejected_adversarial_count * confirmer_pass_rate.

    Args:
        property_registry: PropertyRegistry with iter_records().
        verification_log: VerificationLog with iter_records().
        k: max number of exemplars to return.
        min_vov_score: drop properties below this score (default 1.0 —
          requires at least one adversarial rejection AND a non-zero pass rate).

    Returns a list sorted by descending score. Ties broken by name for stability.
    """
    if property_registry is None:
        return []
    stats = _score_properties(verification_log) if verification_log is not None else {}
    out: list[PropertyExemplar] = []
    seen: set[str] = set()
    # Walk registry in reverse so last-write-wins for duplicated ids.
    for rec in reversed(list(property_registry.iter_records())):
        pid = rec.get("property_id") or ""
        if not pid or pid in seen:
            continue
        seen.add(pid)
        s = stats.get(pid, {"rej_adv": 0, "conf_pass": 0, "conf_total": 0})
        conf_rate = (s["conf_pass"] / s["conf_total"]) if s["conf_total"] > 0 else 0.0
        score = s["rej_adv"] * conf_rate
        if score < min_vov_score:
            continue
        out.append(PropertyExemplar(
            property_id=pid,
            name=str(rec.get("name", ""))[:120],
            description=str(rec.get("description", "") or rec.get("payload", {}).get("description", ""))[:240],
            independence_class=str(rec.get("independence_class", "")),
            kind=str(rec.get("kind", "")),
            rejected_adversarial_count=s["rej_adv"],
            confirmer_pass_count=s["conf_pass"],
            confirmer_total=s["conf_total"],
            score=score,
        ))
    out.sort(key=lambda e: (-e.score, e.name, e.property_id))
    return out[: max(0, k)]


def count_admitted_properties(property_registry: Any) -> int:
    """Distinct property_id count across the registry (last-write-wins)."""
    if property_registry is None:
        return 0
    ids: set[str] = set()
    for rec in property_registry.iter_records():
        pid = rec.get("property_id") or ""
        if pid:
            ids.add(pid)
    return len(ids)


def render_property_exemplars_block(exemplars: list[PropertyExemplar]) -> str:
    """Render the property bank as a short ASCII block for prompt injection.

    Keeps each entry to ~one line so the full block is bounded well under a
    context-window-sensitive budget. Clearly delimited per §3.1 prompt
    robustness note — DeepSeek-R1 breaks on format changes, so we wrap in
    explicit `### EXAMPLES FROM PRIOR CYCLES` and `### END EXAMPLES` markers.
    """
    if not exemplars:
        return ""
    lines = [
        "### EXAMPLES FROM PRIOR CYCLES — property ideas that earned trust",
        "# (pattern inspiration only — your task still authors fresh property descriptors"
        "  for your specific problem, but these discriminate real candidates well.)",
    ]
    for i, e in enumerate(exemplars, 1):
        lines.append(
            f"# {i}. {e.name!r} [{e.independence_class}, {e.kind}] "
            f"— {e.description.splitlines()[0][:140] if e.description else '(no description)'} "
            f"(rejected {e.rejected_adversarial_count} corruptions, "
            f"{e.confirmer_pass_count}/{e.confirmer_total} confirmer-pass)"
        )
    lines.append("### END EXAMPLES")
    lines.append("")
    return "\n".join(lines)


# ─── proposer exemplar bank ─────────────────────────────────────────────


@dataclass(frozen=True)
class ProposerExemplar:
    """One rendered proposer entry for a few-shot prompt block."""
    problem_id: str
    problem_text: str
    domain: str
    quorum_accept_count: int
    entry_point: str = ""
    reference: str = ""
    tests: tuple[str, ...] = ()


def _count_problem_accepts(training_pool: Any) -> dict[str, int]:
    """Count accepted training-pool records per problem_id."""
    counts: dict[str, int] = defaultdict(int)
    if training_pool is None:
        return counts
    for rec in training_pool.iter_records():
        pid = rec.get("problem_id") or ""
        if pid:
            counts[pid] += 1
    return counts


def top_k_proposer_exemplars(
    problem_registry: Any,
    training_pool: Any,
    *,
    k: int = 3,
    min_accept_count: int = 1,
) -> list[ProposerExemplar]:
    """Rank ProblemRegistry entries by quorum-accept count from TrainingPool.

    Problems that produced ≥min_accept_count accepted candidates get shown
    as shape templates — proof that the (problem, reference, tests) triple
    is self-consistent enough to yield passing samples.

    Returns up to k; sorted by descending accept-count, then recency.
    """
    if problem_registry is None:
        return []
    accepts = _count_problem_accepts(training_pool)
    # Fold by problem_id: last record wins for mutations (retired patches, etc.).
    folded: dict[str, dict] = {}
    for rec in problem_registry.iter_records():
        pid = rec.get("problem_id") or ""
        if not pid:
            continue
        if rec.get("_patch"):
            # Patches only update specific fields; merge onto prior.
            if pid in folded:
                folded[pid].update({k: v for k, v in rec.items() if v is not None})
            continue
        folded[pid] = dict(rec)

    out: list[ProposerExemplar] = []
    for pid, rec in folded.items():
        if rec.get("retired"):
            # Retired problems have already minted training samples — they
            # ARE good exemplars by definition. Include unless accept count is 0.
            pass
        cnt = accepts.get(pid, 0)
        if cnt < min_accept_count:
            continue
        ctx = rec.get("problem_ctx") or {}
        if not isinstance(ctx, dict):
            ctx = {}
        tests = tuple(str(t) for t in (ctx.get("tests") or [])[:5])
        out.append(ProposerExemplar(
            problem_id=pid,
            problem_text=str(rec.get("problem_text", ""))[:600],
            domain=str(rec.get("domain", "")),
            quorum_accept_count=cnt,
            entry_point=str(ctx.get("entry_point", "")),
            reference=str(ctx.get("reference", ""))[:800],
            tests=tests,
        ))
    out.sort(key=lambda e: (-e.quorum_accept_count, e.problem_id))
    return out[: max(0, k)]


def render_proposer_exemplars_block(exemplars: list[ProposerExemplar]) -> str:
    """Render proposer exemplars as a delimited block BEFORE the canonical
    PROBLEM: template example.

    Each entry mirrors the CODE_PROPOSAL_TEMPLATE fields so the model sees
    the exact shape it will be asked to produce. Truncated aggressively to
    keep the prompt bounded (5 tests × 800-char reference × k≤3 ~= 4 kB).
    """
    if not exemplars:
        return ""
    lines = [
        "### EXAMPLES FROM PRIOR CYCLES — problems that passed quorum",
        "# These real proposals earned accepted training samples. Use them as",
        "# SHAPE inspiration for your own DIFFERENT problem — do NOT copy.",
    ]
    for i, e in enumerate(exemplars, 1):
        lines.append(f"# --- exemplar {i} (accepted {e.quorum_accept_count}×) ---")
        lines.append(f"# PROBLEM: {e.problem_text}")
        if e.entry_point:
            lines.append(f"# ENTRY: {e.entry_point}")
        if e.reference:
            ref_lines = e.reference.splitlines()[:12]   # cap at 12 source lines
            lines.append("# REFERENCE:")
            for rl in ref_lines:
                lines.append(f"#   {rl}")
        if e.tests:
            lines.append("# TESTS:")
            for t in e.tests:
                lines.append(f"#   - {t}")
    lines.append("### END EXAMPLES")
    lines.append("")
    return "\n".join(lines)


# ─── combined prompt helper ─────────────────────────────────────────────


def build_library_prefix(
    *,
    property_registry: Any,
    verification_log: Any,
    problem_registry: Any,
    training_pool: Any,
    min_admitted_for_gate: int = 20,
    k_properties: int = 5,
    k_proposer: int = 3,
    min_vov_score: float = 1.0,
) -> str:
    """Build the combined few-shot prefix (properties + proposer exemplars).

    Returns "" when the library is too small to warrant injection (guards
    against cold-start noise). This is the single entry point the
    task_synthesizer call-site uses; all ranking/rendering is encapsulated.
    """
    n_admitted = count_admitted_properties(property_registry)
    if n_admitted < min_admitted_for_gate:
        return ""
    prop_ex = top_k_admitted_properties(
        property_registry, verification_log,
        k=k_properties, min_vov_score=min_vov_score,
    )
    prop_ex_block = render_property_exemplars_block(prop_ex)
    prop_proposer = top_k_proposer_exemplars(
        problem_registry, training_pool, k=k_proposer,
    )
    prop_proposer_block = render_proposer_exemplars_block(prop_proposer)
    parts = [b for b in (prop_ex_block, prop_proposer_block) if b]
    return "\n".join(parts)


__all__ = [
    "PropertyExemplar",
    "ProposerExemplar",
    "top_k_admitted_properties",
    "top_k_proposer_exemplars",
    "count_admitted_properties",
    "render_property_exemplars_block",
    "render_proposer_exemplars_block",
    "build_library_prefix",
]
