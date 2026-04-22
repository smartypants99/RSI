"""Curriculum escalation: DifficultyTracker.

Records per-cycle which held-out questions the model got right/wrong, which
proposals were accepted/rejected, and exposes a ``frontier()`` API that
identifies the easiest skill-pair (subdomain) where the model currently
fails. Also maintains a difficulty ratchet: each cycle where held-out
improves ≥ min_improvement the minimum self-reported proposal difficulty
floor is raised; on regression it lowers. State persists across restarts
via ``outputs/difficulty_state.json``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


_RATCHET_STEP = 0.05
_RATCHET_MAX = 0.9
_RATCHET_MIN = 0.0
_MIN_DELTA = 0.01  # min held-out improvement to raise the ratchet


@dataclass
class _SubdomainStats:
    attempts: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float:
        return (self.correct / self.attempts) if self.attempts else 0.0


class DifficultyTracker:
    """Tracks held-out outcomes, proposals, and a difficulty ratchet.

    Methods:
      * record_heldout(per_question) — per-question records from a held-out eval
      * record_proposals(accepted, rejected) — counts for the current cycle
      * update_ratchet(heldout_delta) — adjusts difficulty floor by ±0.05 on
        ±_MIN_DELTA held-out changes
      * frontier() — skill-pair string for the easiest zone the model fails in
      * save() / load() — JSON persistence
    """

    def __init__(self, state_path: Optional[Path] = None) -> None:
        self.state_path = Path(state_path) if state_path is not None else None
        # Subdomain tag -> aggregate stats across recorded cycles.
        self._subdomain_stats: dict[str, _SubdomainStats] = {}
        # Last-cycle snapshot of what was right/wrong (for "currently fails").
        self._last_cycle_wrong: set[str] = set()
        self._last_cycle_right: set[str] = set()
        self.proposals_accepted_total: int = 0
        self.proposals_rejected_total: int = 0
        self.last_accepted: int = 0
        self.last_rejected: int = 0
        # Difficulty ratchet state (floor on DIFFICULTY: field in proposals).
        self.difficulty_floor: float = _RATCHET_MIN
        self.ratchet_history: list[dict] = []  # {cycle,delta,floor_before,floor_after}
        self.cycles_recorded: int = 0

    # ---- recording -----------------------------------------------------

    def record_heldout(self, per_question: Iterable[dict]) -> None:
        """Record a pass of held-out per-question results.

        Each entry should have ``correct`` (bool) and ideally ``domain`` +
        ``subdomain`` fields (the diagnostics engine populates both).
        """
        wrong: set[str] = set()
        right: set[str] = set()
        for rec in per_question or ():
            domain = str(rec.get("domain", "") or "")
            subdomain = str(rec.get("subdomain", "") or "")
            key = f"{domain}/{subdomain}" if subdomain else (domain or "unknown")
            stats = self._subdomain_stats.setdefault(key, _SubdomainStats())
            stats.attempts += 1
            correct = bool(rec.get("correct", False))
            if correct:
                stats.correct += 1
                right.add(key)
            else:
                wrong.add(key)
        self._last_cycle_wrong = wrong
        self._last_cycle_right = right
        self.cycles_recorded += 1

    def record_proposals(self, accepted: int, rejected: int) -> None:
        self.last_accepted = int(max(0, accepted))
        self.last_rejected = int(max(0, rejected))
        self.proposals_accepted_total += self.last_accepted
        self.proposals_rejected_total += self.last_rejected

    # ---- ratchet -------------------------------------------------------

    def update_ratchet(self, heldout_delta: Optional[float], cycle: int = -1) -> float:
        """Adjust the difficulty floor based on held-out score delta.

        Raises the floor by +0.05 (cap 0.9) when heldout_delta >= +0.01.
        Lowers it by -0.05 (floor 0.0) when heldout_delta <= -0.01.
        Returns the new floor.
        """
        before = self.difficulty_floor
        if heldout_delta is None:
            return before
        delta = float(heldout_delta)
        if delta >= _MIN_DELTA:
            self.difficulty_floor = min(_RATCHET_MAX, before + _RATCHET_STEP)
        elif delta <= -_MIN_DELTA:
            self.difficulty_floor = max(_RATCHET_MIN, before - _RATCHET_STEP)
        if self.difficulty_floor != before:
            self.ratchet_history.append({
                "cycle": int(cycle),
                "heldout_delta": delta,
                "floor_before": before,
                "floor_after": self.difficulty_floor,
            })
        return self.difficulty_floor

    # ---- queries -------------------------------------------------------

    def frontier(self, domain: Optional[str] = None) -> str:
        """Return the skill-pair string for the 'easiest zone currently failing'.

        Easiest = highest historical accuracy on this skill, among subdomains
        that the model got wrong in its most recent held-out pass. If no
        last-cycle wrongs are recorded, falls back to the lowest-accuracy
        recorded subdomain (below 1.0). Returns "" if nothing is recorded yet.

        When ``domain`` is provided, only subdomains matching that prefix
        (``domain/...``) are considered. This prevents the cross-domain
        leakage seen in cycles 3–7 of the overnight run, where frontier()
        returned "math/percentage" and was spliced into propose_batch_code's
        CODE-ONLY proposal prompt — biasing the model's code proposals toward
        a math subdomain it could not satisfy, causing 5 consecutive cycles
        of frontier drift onto the wrong domain.
        """
        def _in_domain(key: str) -> bool:
            if not domain:
                return True
            prefix = domain.strip()
            if not prefix:
                return True
            return key.split("/", 1)[0] == prefix

        if self._last_cycle_wrong:
            candidates = [
                (k, self._subdomain_stats.get(k, _SubdomainStats()).accuracy)
                for k in self._last_cycle_wrong if _in_domain(k)
            ]
            if candidates:
                candidates.sort(key=lambda kv: (-kv[1], kv[0]))
                return candidates[0][0]
            # Fall through to aggregate-stats fallback if the last-cycle
            # wrongs had nothing in the requested domain.
        # Fallback: weakest aggregate subdomain (below perfect) in domain.
        ranked = sorted(
            (
                (k, v.accuracy)
                for k, v in self._subdomain_stats.items()
                if v.attempts > 0 and v.accuracy < 1.0 and _in_domain(k)
            ),
            key=lambda kv: (kv[1], kv[0]),
        )
        return ranked[0][0] if ranked else ""

    # ---- persistence --------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "subdomain_stats": {
                k: {"attempts": v.attempts, "correct": v.correct}
                for k, v in self._subdomain_stats.items()
            },
            "last_cycle_wrong": sorted(self._last_cycle_wrong),
            "last_cycle_right": sorted(self._last_cycle_right),
            "proposals_accepted_total": self.proposals_accepted_total,
            "proposals_rejected_total": self.proposals_rejected_total,
            "last_accepted": self.last_accepted,
            "last_rejected": self.last_rejected,
            "difficulty_floor": self.difficulty_floor,
            "ratchet_history": list(self.ratchet_history),
            "cycles_recorded": self.cycles_recorded,
        }

    @classmethod
    def from_dict(cls, data: dict, state_path: Optional[Path] = None) -> "DifficultyTracker":
        t = cls(state_path=state_path)
        for k, v in (data.get("subdomain_stats") or {}).items():
            t._subdomain_stats[k] = _SubdomainStats(
                attempts=int(v.get("attempts", 0)),
                correct=int(v.get("correct", 0)),
            )
        t._last_cycle_wrong = set(data.get("last_cycle_wrong") or [])
        t._last_cycle_right = set(data.get("last_cycle_right") or [])
        t.proposals_accepted_total = int(data.get("proposals_accepted_total", 0))
        t.proposals_rejected_total = int(data.get("proposals_rejected_total", 0))
        t.last_accepted = int(data.get("last_accepted", 0))
        t.last_rejected = int(data.get("last_rejected", 0))
        floor = float(data.get("difficulty_floor", 0.0))
        t.difficulty_floor = max(_RATCHET_MIN, min(_RATCHET_MAX, floor))
        t.ratchet_history = list(data.get("ratchet_history") or [])
        t.cycles_recorded = int(data.get("cycles_recorded", 0))
        return t

    def save(self, path: Optional[Path] = None) -> Path:
        p = Path(path) if path is not None else self.state_path
        if p is None:
            raise ValueError("no state_path set; pass path= to save()")
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        tmp.replace(p)
        return p

    @classmethod
    def load_or_new(cls, path: Optional[Path]) -> "DifficultyTracker":
        if path is None:
            return cls()
        p = Path(path)
        if not p.exists():
            return cls(state_path=p)
        try:
            with open(p) as f:
                data = json.load(f)
            return cls.from_dict(data, state_path=p)
        except Exception as exc:
            logger.warning("DifficultyTracker: failed to load %s (%s); starting fresh", p, exc)
            return cls(state_path=p)
