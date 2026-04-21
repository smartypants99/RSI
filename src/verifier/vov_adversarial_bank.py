"""Persistent adversarial bank for VoV §1.4.

Grows the verifier as the model improves: every time post-training eval
regresses, the candidates admitted in that cycle are appended here. Future
VoV audits run admitted properties against every bank entry; a property
that ACCEPTS a bank entry is rejected as toothless. Bad samples become
permanent adversarial tests for the verification layer.

File format: JSONL at ``outputs/vov_adversarial_bank.jsonl``. One entry per
line::

    {"bank_id": "...", "problem_id": "...", "candidate": "...",
     "domain": "code", "problem_ctx": {...}, "added_at": 1700000000.0,
     "last_triggered_at": 1700000000.0, "cycle": 42, "reason": "..."}

Capacity: 500 entries. Eviction: LRU by ``last_triggered_at`` (falling back
to ``added_at``) so entries that keep catching properties stay; stale ones
roll out.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

DEFAULT_BANK_PATH = Path("outputs/vov_adversarial_bank.jsonl")
MAX_BANK_ENTRIES = 500


@dataclass
class BankEntry:
    """One adversarial entry — a candidate that made it through §2.1 quorum
    but was implicated in a post-training regression."""
    bank_id: str
    problem_id: str
    candidate: Any
    domain: str
    problem_ctx: dict = field(default_factory=dict)
    added_at: float = 0.0
    last_triggered_at: float = 0.0
    cycle: int = -1
    reason: str = ""

    @classmethod
    def from_json(cls, obj: dict) -> "BankEntry":
        return cls(
            bank_id=str(obj.get("bank_id") or uuid.uuid4().hex[:16]),
            problem_id=str(obj.get("problem_id", "")),
            candidate=obj.get("candidate"),
            domain=str(obj.get("domain", "code")),
            problem_ctx=dict(obj.get("problem_ctx") or {}),
            added_at=float(obj.get("added_at", 0.0)),
            last_triggered_at=float(obj.get("last_triggered_at", 0.0)),
            cycle=int(obj.get("cycle", -1)),
            reason=str(obj.get("reason", "")),
        )

    def to_json(self) -> dict:
        return asdict(self)


class AdversarialBank:
    """Persistent, LRU-capped adversarial-candidate bank."""

    def __init__(self, path: Path | str = DEFAULT_BANK_PATH, *, max_entries: int = MAX_BANK_ENTRIES):
        self.path = Path(path)
        self.max_entries = int(max_entries)
        self._entries: list[BankEntry] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self._entries = []
            return
        loaded: list[BankEntry] = []
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        loaded.append(BankEntry.from_json(json.loads(line)))
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.debug("skipping malformed bank line: %s", e)
        except OSError as e:
            logger.warning("VoV bank load failed (%s); starting empty", e)
            loaded = []
        self._entries = loaded

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for e in self._entries:
                fh.write(json.dumps(e.to_json(), default=str) + "\n")
        os.replace(tmp, self.path)

    def __len__(self) -> int:
        return len(self._entries)

    def entries(self) -> list[BankEntry]:
        return list(self._entries)

    def append(
        self,
        *,
        problem_id: str,
        candidate: Any,
        domain: str,
        problem_ctx: Optional[dict] = None,
        cycle: int = -1,
        reason: str = "",
    ) -> BankEntry:
        """Add a regression-implicated candidate. Applies LRU eviction."""
        now = time.time()
        entry = BankEntry(
            bank_id=uuid.uuid4().hex[:16],
            problem_id=problem_id,
            candidate=candidate,
            domain=domain,
            problem_ctx=dict(problem_ctx or {}),
            added_at=now,
            last_triggered_at=now,
            cycle=cycle,
            reason=reason,
        )
        self._entries.append(entry)
        self._evict_if_needed()
        try:
            self._persist()
        except OSError as e:
            logger.warning("VoV bank persist failed (%s)", e)
        return entry

    def append_many(self, records: Iterable[dict]) -> int:
        """Bulk append. Returns count added."""
        n = 0
        for r in records:
            self.append(
                problem_id=r.get("problem_id", ""),
                candidate=r.get("candidate"),
                domain=r.get("domain", "code"),
                problem_ctx=r.get("problem_ctx"),
                cycle=int(r.get("cycle", -1)),
                reason=str(r.get("reason", "")),
            )
            n += 1
        return n

    def _evict_if_needed(self) -> None:
        """LRU eviction: keep newest-triggered `max_entries`."""
        if len(self._entries) <= self.max_entries:
            return
        self._entries.sort(
            key=lambda e: (e.last_triggered_at or e.added_at),
        )
        drop = len(self._entries) - self.max_entries
        self._entries = self._entries[drop:]

    def mark_triggered(self, bank_id: str) -> None:
        """Update last_triggered_at for an entry that just rejected a
        property (keeps useful entries from being LRU-evicted)."""
        for e in self._entries:
            if e.bank_id == bank_id:
                e.last_triggered_at = time.time()
                break
        try:
            self._persist()
        except OSError:
            pass

    def filter_for_domain(self, domain: str) -> list[BankEntry]:
        """Entries whose domain matches (or bank-entry domain is empty)."""
        return [e for e in self._entries if not e.domain or e.domain == domain]


_DEFAULT_BANK: Optional[AdversarialBank] = None


def get_default_bank() -> AdversarialBank:
    """Module-level singleton so VoV audits and the orchestrator share state."""
    global _DEFAULT_BANK
    if _DEFAULT_BANK is None:
        _DEFAULT_BANK = AdversarialBank(DEFAULT_BANK_PATH)
    return _DEFAULT_BANK


def set_default_bank(bank: Optional[AdversarialBank]) -> None:
    """Override (tests) or reset (pass None) the module-level bank."""
    global _DEFAULT_BANK
    _DEFAULT_BANK = bank


__all__ = [
    "AdversarialBank", "BankEntry",
    "DEFAULT_BANK_PATH", "MAX_BANK_ENTRIES",
    "get_default_bank", "set_default_bank",
]
