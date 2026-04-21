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

# Tier 0 is the baseline — always allowed, matches self_edit.DEFAULT_ALLOW_LIST.
TIER_0: tuple[str, ...] = ("src/generator/*.py", "src/verifier/*.py")
TIER_2: tuple[str, ...] = ("src/utils/*.py", "src/trainer/grpo.py")
TIER_3: tuple[str, ...] = ("src/trainer/custom_lora.py",)
TIER_4: tuple[str, ...] = ("src/orchestrator/loop.py",)

# Promotion thresholds: successful self_edit cycles required to unlock a tier,
# AND the minimum average held-out-delta contribution of self_edit across those
# successful cycles. "Successful" = self_edit was active AND held_out_delta > 0.
PROMOTION_THRESHOLDS: dict[int, tuple[int, float]] = {
    2: (5, 0.002),
    3: (10, 0.002),
    4: (20, 0.002),
}

# Revert: 2 consecutive tier-N patches with negative delta -> revoke tier N,
# cooldown REVERT_COOLDOWN cycles before it can be re-unlocked.
CONSECUTIVE_BAD_TO_REVERT = 2
REVERT_COOLDOWN = 5

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


def _successful_self_edit_count(history: list[CycleRecord]) -> tuple[int, float]:
    """Return (count, mean_delta) over cycles where self_edit was active AND
    the held-out delta was positive."""
    wins = [h.held_out_delta for h in _self_edit_cycles(history) if h.held_out_delta > 0]
    if not wins:
        return 0, 0.0
    return len(wins), sum(wins) / len(wins)


def _tier_recent_bad_streak(history: list[CycleRecord], tier: int) -> int:
    """Consecutive trailing tier-N self_edit cycles with negative delta."""
    streak = 0
    for h in reversed(history):
        if not h.components_active.get("self_edit"):
            continue
        if h.self_edit_tier != tier:
            break
        if h.held_out_delta < 0:
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
    4: TIER_4,
}


def effective_allow_list(
    history: Iterable[CycleRecord],
    current_cycle: int = 0,
) -> tuple[str, ...]:
    """Return the allow-list globs for self-edit given the history so far.

    Always includes tier 0. Higher tiers appear only after graduation.
    """
    hist = list(history)
    state = compute_tier_state(hist, current_cycle)
    globs: list[str] = []
    for tier in state.unlocked:
        for g in _TIER_GLOBS.get(tier, ()):
            if g not in globs:
                globs.append(g)
    return tuple(globs)


def current_self_edit_tier(
    history: Iterable[CycleRecord],
    current_cycle: int = 0,
) -> int:
    """The highest tier currently unlocked — the tier a new self-edit would use."""
    state = compute_tier_state(list(history), current_cycle)
    return max(state.unlocked) if state.unlocked else 0


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
