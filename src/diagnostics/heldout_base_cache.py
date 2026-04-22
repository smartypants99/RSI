"""Base-model held-out prediction cache (task #23 wedge 3).

On cycle 1 the orchestrator needs a "pre" reference for paired_delta
computation. Previously this was paid by running a full held-out eval
on the base model every cycle-1 — ~40 min of inference that produces
the same answers every run.

This module caches per-question predictions keyed by (prompt, expected)
to a single jsonl file. The cache is:

  * content-addressed by model id + question key — so swapping the base
    model invalidates entries for that model
  * deterministic given HELDOUT_CYCLE_SEED (which freezes the question
    set) — cache hits are safe across runs
  * additive: if a cache file exists for a given model id, load it and
    only generate entries for questions NOT already cached

The cache is an optimization, not a correctness dependency — the loop
can always fall back to a fresh eval if the cache is missing or stale.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

logger = logging.getLogger(__name__)


CACHE_VERSION = 1


def _question_key(prompt: str, expected: str | None) -> str:
    """Stable key matching paired_eval._per_question_correct_map."""
    exp = expected if expected is not None else ""
    payload = f"{prompt}|{exp}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _model_fingerprint(model_id: str) -> str:
    """16-hex model-id hash. Swapping the base model invalidates entries."""
    return hashlib.sha256(str(model_id).encode("utf-8")).hexdigest()[:16]


@dataclass
class BaseHeldoutCache:
    """Single-file jsonl cache of {question_key: {correct, prediction}}.

    The cache is scoped by model fingerprint; querying a different model
    against the same cache file returns misses rather than stale answers.
    """
    path: Path
    model_id: str
    entries: dict[str, dict] = field(default_factory=dict)
    _fingerprint: str = ""

    def __post_init__(self) -> None:
        self._fingerprint = _model_fingerprint(self.model_id)
        self.path = Path(self.path)

    # ---- persistence ------------------------------------------------------

    @classmethod
    def load_or_new(cls, path: Path | str, model_id: str) -> "BaseHeldoutCache":
        cache = cls(path=Path(path), model_id=model_id)
        if not cache.path.exists():
            return cache
        try:
            with cache.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    # Skip rows from a different model or cache version.
                    if row.get("version") != CACHE_VERSION:
                        continue
                    if row.get("model_fp") != cache._fingerprint:
                        continue
                    qkey = row.get("qkey")
                    if not qkey:
                        continue
                    cache.entries[qkey] = {
                        "correct": bool(row.get("correct", False)),
                        "prediction": str(row.get("prediction", "")),
                        "prompt": str(row.get("prompt", "")),
                        "expected": str(row.get("expected", "")),
                    }
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("heldout_base_cache load failed (%s): %s", type(exc).__name__, exc)
        return cache

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for qkey, ent in self.entries.items():
                fh.write(json.dumps({
                    "version": CACHE_VERSION,
                    "model_fp": self._fingerprint,
                    "qkey": qkey,
                    **ent,
                }) + "\n")
        tmp.replace(self.path)

    # ---- API --------------------------------------------------------------

    def has(self, prompt: str, expected: str | None) -> bool:
        return _question_key(prompt, expected) in self.entries

    def get(self, prompt: str, expected: str | None) -> Optional[dict]:
        return self.entries.get(_question_key(prompt, expected))

    def put(self, *, prompt: str, expected: str | None,
            correct: bool, prediction: str = "") -> None:
        self.entries[_question_key(prompt, expected)] = {
            "correct": bool(correct),
            "prediction": str(prediction),
            "prompt": str(prompt),
            "expected": str(expected or ""),
        }

    def missing(self, questions: Iterable[tuple[str, str | None]]) -> list[tuple[str, str | None]]:
        """Return the subset of (prompt, expected) tuples not in cache."""
        return [q for q in questions if not self.has(q[0], q[1])]

    # ---- per_question conversion -----------------------------------------

    def to_per_question_records(self) -> list[dict]:
        """Export as a list of per-question records in the shape
        paired_eval.paired_delta expects: {'prompt','expected','correct'}."""
        return [
            {
                "prompt": ent.get("prompt", ""),
                "expected": ent.get("expected", ""),
                "correct": bool(ent.get("correct", False)),
            }
            for ent in self.entries.values()
        ]


def populate_from_eval(
    cache: BaseHeldoutCache,
    per_question_records: list[dict],
) -> int:
    """Ingest the per_question list from a DiagnosticResult into the cache.

    Returns the number of new entries added (pre-existing entries are not
    overwritten — once the base model has answered a question, that
    answer is definitional for the cache lifetime).
    """
    added = 0
    for rec in per_question_records or []:
        prompt = str(rec.get("prompt", rec.get("question", "")))
        expected = rec.get("expected")
        if not prompt:
            continue
        if cache.has(prompt, expected):
            continue
        cache.put(
            prompt=prompt,
            expected=expected,
            correct=bool(rec.get("correct", False)),
            prediction=str(rec.get("response", "")),
        )
        added += 1
    return added


__all__ = [
    "BaseHeldoutCache",
    "populate_from_eval",
    "CACHE_VERSION",
]
