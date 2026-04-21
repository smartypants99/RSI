"""Meta-meta: per-component contribution tracking + graduated self-edit allow-list.

After each cycle, the orchestrator records which components were active and the
held-out delta that cycle. Over time we estimate each component's contribution
by comparing held-out delta on cycles where it was active versus cycles where
it was not (simple difference-of-means isolation). Those estimates feed two
things:

  * `cycle_metrics.meta_meta.component_contributions` — forensic logging.
  * `effective_allow_list(history)` — the self-edit allow-list is GRADUATED:
    additional path tiers unlock only after `self_edit` itself has demonstrated
    sustained positive contribution. If it regresses two tier-N cycles in a
    row, tier N is revoked and sits in a cooldown for 5 cycles.

This module is pure / side-effect-free except for `append_history` /
`load_history` which read+write a JSONL file. Keeping the math pure makes the
tier-promotion and revert logic easy to unit-test.

The hard deny-list in `src/orchestrator/self_edit.py` remains authoritative —
meta_meta only ever EXPANDS the allow-list within bounds it was told about.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Optional


# ---------------------------------------------------------------------------
# Tier configuration
# ---------------------------------------------------------------------------
#
# safety-gate contract (2026-04-21): `src/orchestrator/self_edit.HARD_DENY_LIST`
# is authoritative. Tier graduation can only EXPAND the allow-list and must
# never propose paths that are on HARD_DENY (apply-time deny wins anyway).
# Current HARD_DENY covers orchestrator/loop.py, utils/config.py, trainer/*,
# self_edit.py, safety/*, tests/*, run.sh, .github/*. Tiers below pick paths
# that sit OUTSIDE HARD_DENY. Expanding to trainer/loop.py requires a separate
# reviewed PR to narrow HARD_DENY — not an auto-graduation from this module.

# Tier 0: always allowed. Matches self_edit.DEFAULT_ALLOW_LIST.
TIER_0: tuple[str, ...] = ("src/generator/*.py", "src/verifier/*.py")
# Tier 2: diagnostics + deeper generator tree. Not on HARD_DENY.
TIER_2: tuple[str, ...] = ("src/diagnostics/*.py",)
# Tier 3: orchestrator leaf modules that are NOT loop.py / self_edit.py /
# meta_meta.py. Still on the bounded surface (not on HARD_DENY).
TIER_3: tuple[str, ...] = ("src/orchestrator/decision_log.py", "src/orchestrator/registries.py")
# Tier 4 is HUMAN-GATED: meta_meta may record that qualification was reached
# and emit a proposal artifact, but the runtime allow-list is NEVER expanded
# automatically. Expansion requires an external approval step.
TIER_4_PROPOSAL: tuple[str, ...] = ()  # intentionally empty at runtime
HUMAN_GATED_TIERS: frozenset[int] = frozenset({4})

# Promotion thresholds: (required_successful_self_edit_cycles, required_mean_delta).
PROMOTION_THRESHOLDS: dict[int, tuple[int, float]] = {
    2: (5, 0.002),
    3: (10, 0.002),
    4: (20, 0.002),
}

# Revert semantics (precise, per safety-gate):
# A tier-N self-edit cycle is "bad" iff components_active["self_edit"] AND
# self_edit_tier == N AND held_out_delta < NEGATIVE_DELTA_THRESHOLD.
# Two consecutive bad tier-N cycles (no intervening positive tier-N cycle)
# revoke tier N and start REVERT_COOLDOWN cycles of cooldown.
NEGATIVE_DELTA_THRESHOLD = 0.0  # strictly < 0 counts as bad
SUCCESSFUL_DELTA_THRESHOLD = 0.0  # strictly > 0 counts as a self_edit win
CONSECUTIVE_BAD_TO_REVERT = 2
REVERT_COOLDOWN = 5


def tier_requires_human_approval(tier: int) -> bool:
    """Tier 4 can be QUALIFIED by evidence but never auto-unlocked."""
    return tier in HUMAN_GATED_TIERS

COMPONENT_KEYS: tuple[str, ...] = (
    "fast_student",
    "ood",
    "curriculum_ratchet",
    "growth",
    "self_edit",
    "grpo",
)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class CycleRecord:
    cycle_id: int
    components_active: dict[str, bool]
    held_out_delta: float
    # Tier that was UNLOCKED during this cycle's self-edit attempt, if any.
    # Used to attribute tier-N patches correctly for the revert rule.
    self_edit_tier: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CycleRecord":
        return cls(
            cycle_id=int(d["cycle_id"]),
            components_active={k: bool(v) for k, v in (d.get("components_active") or {}).items()},
            held_out_delta=float(d["held_out_delta"]),
            self_edit_tier=d.get("self_edit_tier"),
        )


@dataclass
class TierState:
    unlocked: list[int] = field(default_factory=lambda: [0])
    # Per-tier cooldown: cycle_id at which the cooldown ends.
    cooldown_until: dict[int, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def append_history(path: Path, record: CycleRecord) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record.to_dict()) + "\n")


def load_history(path: Path) -> list[CycleRecord]:
    path = Path(path)
    if not path.exists():
        return []
    out: list[CycleRecord] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(CycleRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return out


# ---------------------------------------------------------------------------
# Ablation / isolation: per-component contribution
# ---------------------------------------------------------------------------


def component_contributions(
    history: Iterable[CycleRecord],
    components: Iterable[str] = COMPONENT_KEYS,
) -> dict[str, dict]:
    """Estimate each component's contribution by difference-of-means.

    For component X: contribution = mean(held_out_delta | X active)
                                  - mean(held_out_delta | X inactive).

    Returns {component: {"active_n", "inactive_n", "active_mean",
    "inactive_mean", "contribution"}}. When either arm is empty the
    contribution is None (not enough data to isolate).
    """
    hist = list(history)
    out: dict[str, dict] = {}
    for c in components:
        active = [h.held_out_delta for h in hist if h.components_active.get(c)]
        inactive = [h.held_out_delta for h in hist if not h.components_active.get(c)]
        active_mean = sum(active) / len(active) if active else None
        inactive_mean = sum(inactive) / len(inactive) if inactive else None
        contribution = (
            active_mean - inactive_mean
            if active_mean is not None and inactive_mean is not None
            else None
        )
        out[c] = {
            "active_n": len(active),
            "inactive_n": len(inactive),
            "active_mean": active_mean,
            "inactive_mean": inactive_mean,
            "contribution": contribution,
        }
    return out


# ---------------------------------------------------------------------------
# Tier graduation logic (pure)
# ---------------------------------------------------------------------------


def _self_edit_cycles(history: list[CycleRecord]) -> list[CycleRecord]:
    return [h for h in history if h.components_active.get("self_edit")]


def _is_successful_self_edit(h: CycleRecord) -> bool:
    return (
        bool(h.components_active.get("self_edit"))
        and h.held_out_delta > SUCCESSFUL_DELTA_THRESHOLD
    )


def _is_bad_tier_cycle(h: CycleRecord, tier: int) -> bool:
    return (
        bool(h.components_active.get("self_edit"))
        and h.self_edit_tier == tier
        and h.held_out_delta < NEGATIVE_DELTA_THRESHOLD
    )


def _successful_self_edit_count(history: list[CycleRecord]) -> tuple[int, float]:
    wins = [h.held_out_delta for h in history if _is_successful_self_edit(h)]
    if not wins:
        return 0, 0.0
    return len(wins), sum(wins) / len(wins)


def _tier_recent_bad_streak(history: list[CycleRecord], tier: int) -> int:
    """Consecutive trailing tier-N self_edit cycles that meet _is_bad_tier_cycle.
    A positive tier-N cycle breaks the streak; non-self_edit / other-tier cycles
    are skipped (they don't advance or reset)."""
    streak = 0
    for h in reversed(history):
        if not h.components_active.get("self_edit"):
            continue
        if h.self_edit_tier != tier:
            continue
        if _is_bad_tier_cycle(h, tier):
            streak += 1
        else:
            break
    return streak


def compute_tier_state(
    history: list[CycleRecord],
    current_cycle: int,
) -> TierState:
    """Given full history, decide which tiers should be unlocked right now.

    Deterministic: run left-to-right over the history, promoting / reverting as
    evidence accumulates. Any tier currently in a cooldown window stays locked
    regardless of qualification until the window closes.
    """
    state = TierState(unlocked=[0])
    hist: list[CycleRecord] = []

    for rec in history:
        hist.append(rec)
        cyc = rec.cycle_id
        # Revert check for each currently-unlocked tier (except 0).
        for tier in list(state.unlocked):
            if tier == 0:
                continue
            bad = _tier_recent_bad_streak(hist, tier)
            if bad >= CONSECUTIVE_BAD_TO_REVERT:
                state.unlocked.remove(tier)
                state.cooldown_until[tier] = cyc + REVERT_COOLDOWN

        # Promotion check.
        wins, mean_delta = _successful_self_edit_count(hist)
        for tier, (need_n, need_mean) in sorted(PROMOTION_THRESHOLDS.items()):
            if tier in state.unlocked:
                continue
            if cyc < state.cooldown_until.get(tier, -1):
                continue  # still cooling down
            # Require all lower tiers to be unlocked first — graduated.
            prereqs = [t for t in PROMOTION_THRESHOLDS if t < tier]
            if any(t not in state.unlocked for t in prereqs):
                continue
            if wins >= need_n and mean_delta >= need_mean:
                state.unlocked.append(tier)

    # Also apply cooldown check relative to *current_cycle* for tiers that
    # were revoked on the last recorded cycle — caller may query before a new
    # record lands.
    for tier, until in list(state.cooldown_until.items()):
        if current_cycle >= until and tier not in state.unlocked:
            # Cooldown elapsed but tier stays locked until new evidence
            # re-qualifies it; just clear the marker so promotion can retry.
            state.cooldown_until.pop(tier, None)

    state.unlocked.sort()
    return state


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_TIER_GLOBS: dict[int, tuple[str, ...]] = {
    0: TIER_0,
    2: TIER_2,
    3: TIER_3,
    4: TIER_4_PROPOSAL,  # empty at runtime — tier 4 is human-gated
}


def _hard_deny_from_self_edit() -> tuple[str, ...]:
    """Read the authoritative HARD_DENY_LIST from self_edit at call time.

    Keeping this dynamic means meta_meta stays in sync if self_edit tightens
    HARD_DENY. We never shadow or redefine it here.
    """
    try:
        from src.orchestrator.self_edit import HARD_DENY_LIST  # type: ignore
        return tuple(HARD_DENY_LIST)
    except Exception:
        return ()


def _glob_overlaps_deny(glob: str, deny: Iterable[str]) -> bool:
    """Return True only when a deny rule would swallow the ENTIRE tier glob.

    A deny rule covering one file *inside* a directory glob (e.g.
    `src/diagnostics/eval_partition.py` vs `src/diagnostics/*.py`) does not
    collide the whole glob — self_edit.validate_patch still rejects the one
    file per-patch. We only filter the tier glob when the deny rule is at
    least as broad (e.g. `src/diagnostics/*` would swallow `src/diagnostics/*.py`).
    """
    import fnmatch as _fn
    for d in deny:
        # Whole-directory deny (trailing /* or /*.ext etc) subsumes the tier glob.
        if ("*" in d or "?" in d) and _fn.fnmatch(glob, d):
            return True
        # Exact-path deny where the tier glob is also that exact file.
        if "*" not in d and "?" not in d and d == glob:
            return True
    return False


def effective_allow_list(
    history: Iterable[CycleRecord],
    current_cycle: int = 0,
) -> tuple[str, ...]:
    """Return the allow-list globs for self-edit given the history so far.

    Always includes tier 0. Higher tiers appear only after graduation. Tier 4
    is HUMAN-GATED and never auto-unlocked regardless of evidence. Any glob
    that overlaps the authoritative HARD_DENY_LIST is filtered out — meta_meta
    must not silently contradict safety.
    """
    hist = list(history)
    state = compute_tier_state(hist, current_cycle)
    deny = _hard_deny_from_self_edit()
    globs: list[str] = []
    for tier in state.unlocked:
        if tier_requires_human_approval(tier):
            continue  # human-gated — evidence exists, but runtime stays locked
        for g in _TIER_GLOBS.get(tier, ()):
            if g in globs:
                continue
            if _glob_overlaps_deny(g, deny):
                continue
            globs.append(g)
    return tuple(globs)


def qualified_tiers(
    history: Iterable[CycleRecord],
    current_cycle: int = 0,
) -> list[int]:
    """Tiers that evidence says SHOULD be unlocked, including human-gated ones.
    Use this to surface tier-4 proposals to human reviewers via update-log."""
    return compute_tier_state(list(history), current_cycle).unlocked


def append_audit_log(
    audit_path: Path,
    cycle_id: int,
    event: str,
    tier: int,
    detail: str = "",
) -> None:
    """Append a human-readable audit line. Intended for update-log.txt.

    Events: 'tier_unlocked', 'tier_reverted', 'tier_proposed_human_gate'.
    """
    audit_path = Path(audit_path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    line = f"[meta_meta] cycle={cycle_id} event={event} tier={tier}"
    if detail:
        line += f" detail={detail}"
    with audit_path.open("a") as f:
        f.write(line + "\n")


def current_self_edit_tier(
    history: Iterable[CycleRecord],
    current_cycle: int = 0,
) -> int:
    """The highest tier currently unlocked — the tier a new self-edit would use."""
    state = compute_tier_state(list(history), current_cycle)
    return max(state.unlocked) if state.unlocked else 0


# ---------------------------------------------------------------------------
# Wall-time observability (task #10)
# ---------------------------------------------------------------------------
#
# Per-phase wall-time samples are appended to a JSONL sidecar. A rolling
# 10-cycle window feeds `wall_time_trend` which returns a percent-change
# between the older half and the newer half of the window. The caller logs
# "cycle time trending down by X%/10 cycles" at the end of each 10-cycle
# window. Pure: no stdout from this module, only structured data.


@dataclass
class WallTimeRecord:
    cycle_id: int
    phase: str
    ms: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WallTimeRecord":
        return cls(
            cycle_id=int(d["cycle_id"]),
            phase=str(d["phase"]),
            ms=float(d["ms"]),
        )


def record_wall_time(
    path: Path,
    cycle: int,
    phase: str,
    ms: float,
) -> WallTimeRecord:
    """Append one phase wall-time sample to a JSONL history."""
    rec = WallTimeRecord(cycle_id=int(cycle), phase=str(phase), ms=float(ms))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(rec.to_dict()) + "\n")
    return rec


def load_wall_time(path: Path) -> list[WallTimeRecord]:
    path = Path(path)
    if not path.exists():
        return []
    out: list[WallTimeRecord] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(WallTimeRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return out


def wall_time_trend(
    records: Iterable[WallTimeRecord],
    window: int = 10,
) -> Optional[dict]:
    """Return trend over the last `window` cycles (aggregated across phases).

    Compares the mean total-cycle-ms of the first half of the window against
    the second half. Positive `pct_change_down` means cycle time trended DOWN
    (faster). Returns None if fewer than `window` distinct cycles are present.
    """
    per_cycle: dict[int, float] = {}
    for r in records:
        per_cycle[r.cycle_id] = per_cycle.get(r.cycle_id, 0.0) + r.ms
    if len(per_cycle) < window:
        return None
    cycles = sorted(per_cycle)[-window:]
    half = window // 2
    older = cycles[:half]
    newer = cycles[half:]
    mean_older = sum(per_cycle[c] for c in older) / max(len(older), 1)
    mean_newer = sum(per_cycle[c] for c in newer) / max(len(newer), 1)
    if mean_older <= 0:
        return None
    pct_change_down = (mean_older - mean_newer) / mean_older * 100.0
    return {
        "window": window,
        "cycles": cycles,
        "mean_ms_older": mean_older,
        "mean_ms_newer": mean_newer,
        "pct_change_down": pct_change_down,
    }


def record_cycle(
    history_path: Path,
    cycle_id: int,
    components_active: dict[str, bool],
    held_out_delta: float,
    self_edit_tier: Optional[int] = None,
) -> CycleRecord:
    """Append one cycle's meta-meta record to the JSONL history."""
    rec = CycleRecord(
        cycle_id=cycle_id,
        components_active={k: bool(components_active.get(k, False)) for k in COMPONENT_KEYS},
        held_out_delta=float(held_out_delta),
        self_edit_tier=self_edit_tier,
    )
    append_history(history_path, rec)
    return rec
