"""Append-only JSONL registries for the RSI property-verification pipeline.

Per spec §4.1, all intermediate artifacts are persisted as append-only JSONL
files scoped to a session id (sid). Each store exposes:
  - append(record)  — write one record, flush immediately
  - iter_records()  — iterate all records from disk (latest state wins for keyed stores)
  - path            — the absolute Path to the underlying file

Session files are created on first write; the outputs/ directories are created
on store instantiation. This matches the spec requirement: "append-only jsonl
at outputs/<kind>/<sid>.jsonl".

Registry classes (all inherit _AppendOnlyStore):
  PropertyRegistry     — admissible Property records
  ProblemRegistry      — ProposedProblem records
  VerificationLog      — VerificationRecord + AdversarialRecord (adversarial=True tag)
  CalibrationLedger    — per-class calibration snapshots (one row per tick per class)
  TrainingPool         — TrainingSample records cleared for training

Portability: pure stdlib — no numpy/torch dependencies so registries can be
imported in any process including the preflight check.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

# ───────────────────────────────────────── helpers ──────────────────────────


def _now() -> float:
    return time.time()


def _sid_default() -> str:
    return uuid.uuid4().hex[:12]


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclass / Path / non-serializable types for json.dumps."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    return obj


# ──────────────────────────────────── base store ────────────────────────────


class _AppendOnlyStore:
    """Thread-compatible append-only JSONL file with immediate flush.

    Not thread-safe for concurrent writers; each session should own exactly
    one instance. The file is opened fresh for each append so the process
    can be killed at any point without losing committed records.
    """

    kind: str = "records"  # subclasses override

    def __init__(self, output_dir: Path, sid: str) -> None:
        self._dir = output_dir / self.kind
        self._dir.mkdir(parents=True, exist_ok=True)
        self.path = self._dir / f"{sid}.jsonl"
        self._sid = sid

    def append(self, record: Any) -> None:
        """Serialize record and append to the session file."""
        try:
            line = json.dumps(_to_jsonable(record), ensure_ascii=False, default=str)
        except (TypeError, ValueError) as exc:
            logger.error("registry %s: failed to serialize record (%s) — skipping", self.kind, exc)
            return
        try:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
        except OSError as exc:
            logger.error("registry %s: write failed (%s)", self.kind, exc)

    def iter_records(self) -> Iterator[dict]:
        """Yield all records from the session file in append order."""
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "registry %s line %d: JSON parse error (%s) — skipping",
                        self.kind, lineno, exc,
                    )

    def count(self) -> int:
        """Return the number of valid records in the session file."""
        return sum(1 for _ in self.iter_records())

    def __repr__(self) -> str:
        return f"<{type(self).__name__} sid={self._sid} path={self.path}>"


# ──────────────────────────────────── record types ──────────────────────────
# These are lightweight wrappers — property_verifier and task_synthesizer own
# the canonical dataclasses. We accept Any here so the store works with both
# the full dataclasses (which may evolve) and plain dicts during early dev.


@dataclass
class PropertyRecord:
    """Minimal envelope written to PropertyRegistry.

    The full Property dataclass (18 fields, spec §1.1) is owned by
    property_verifier. We store the serialized dict; consumers deserialize
    into their own Property type.
    """
    property_id: str
    problem_id: str
    author: str
    independence_class: str
    kind: str
    name: str
    admitted_at: float = field(default_factory=_now)
    # Full spec §1.1 payload — stored verbatim for round-trip fidelity.
    payload: dict = field(default_factory=dict)


@dataclass
class ProblemRecord:
    """Envelope written to ProblemRegistry."""
    problem_id: str
    domain: str
    problem_text: str
    declared_difficulty: float      # [0,1]; model's self-estimate of fail probability
    nearest_neighbor_dist: float    # distance from training corpus (spec §3.3)
    parent_skills: list[str] = field(default_factory=list)
    proposed_at: float = field(default_factory=_now)
    session_id: str = ""
    retired: bool = False           # True after first training-pool acceptance (§7)


@dataclass
class VerificationRecord:
    """One candidate evaluation — primary or adversarial pass.

    Required fields mirror the integrator's quorum decision.  Optional fields
    (all v0.2.1) carry the numeric quorum components from property_verifier's
    P7 shape so §6 health metrics can be computed without re-running
    verification.  Old writers that omit them produce records with defaults;
    new writers should populate them from QuorumVerdict fields.
    """
    record_id: str
    problem_id: str
    candidate_id: str
    property_ids: list[str]
    per_property_verdicts: list[dict]   # [{property_id, passed, reason, class}, ...]
    quorum_accepted: bool
    quorum_reason: str
    adversarial: bool = False           # True for §3.2.3 adversarial pass
    created_at: float = field(default_factory=_now)
    session_id: str = ""
    # v0.2.1 — quorum audit trail (spec §2.1 / P7 QuorumVerdict shape).
    # Defaults keep old writers compatible; populate from QuorumVerdict when available.
    quorum_distinct_classes_required: int = 3   # threshold used at decision time
    quorum_n: int = 0                           # total properties evaluated
    pass_count: int = 0
    fail_count: int = 0
    error_count: int = 0
    distinct_classes: tuple = field(default_factory=tuple)  # PASS-class names (§2.4 health reporting)


@dataclass
class CalibrationEntry:
    """One per-class calibration snapshot written every RSI tick (spec §2.4)."""
    tick: int
    independence_class: str
    true_accept_rate: float     # on known-good answers from ground_truth.py
    true_reject_rate: float     # on known-bad answers
    error_rate: float           # fraction of exceptions
    suspended: bool             # True if class is currently excluded from quorum
    n_probes: int               # number of (good, bad) pairs tested this tick
    recorded_at: float = field(default_factory=_now)
    session_id: str = ""


@dataclass
class TrainingPoolRecord:
    """One accepted sample cleared for LoRA training."""
    pool_record_id: str
    problem_id: str
    candidate_id: str
    verification_record_id: str
    domain: str
    prompt: str
    response: str
    source: str = "rsi_property"    # per spec §5.2 TrainingSample.source tag
    admitted_at: float = field(default_factory=_now)
    session_id: str = ""
    # Optional: payload for round-trip into TrainingSample (trainer reads this)
    training_sample_payload: dict = field(default_factory=dict)


# ──────────────────────────────────── store classes ─────────────────────────


class PropertyRegistry(_AppendOnlyStore):
    """Admissible properties (post §1.3 + §1.4 gates). Producer: property_verifier."""
    kind = "properties"

    def append_property(self, prop: Any, *, bundle_passed_vov: bool) -> None:
        """Write prop to the registry, enforcing the bundle-gate invariant.

        Spec v0.2.1: a property that passed §1.3 (admit()) is still only a
        candidate.  It must NOT be persisted until its bundle passes §1.4
        (VoV quorum_verdict).  Callers must pass bundle_passed_vov=True;
        passing False raises ValueError so an accidental pre-VoV write is
        caught at call time, not silently corrupted into the registry.

        Accepts a PropertyRecord dataclass, a full Property dataclass, or a
        plain dict — all serialized identically.
        """
        if not bundle_passed_vov:
            raise ValueError(
                "PropertyRegistry.append_property called with bundle_passed_vov=False. "
                "Properties may only be persisted after the bundle passes §1.4 VoV "
                "(quorum_verdict). Run VoV first and pass bundle_passed_vov=True."
            )
        if isinstance(prop, dict):
            self.append(prop)
        elif hasattr(prop, "__dataclass_fields__"):
            self.append(prop)
        else:
            logger.warning("PropertyRegistry.append_property: unknown type %s", type(prop))

    def get_by_problem(self, problem_id: str) -> list[dict]:
        return [r for r in self.iter_records() if r.get("problem_id") == problem_id]

    def get_by_id(self, property_id: str) -> dict | None:
        # Last-write-wins: scan all records, keep the last match.
        result = None
        for r in self.iter_records():
            if r.get("property_id") == property_id:
                result = r
        return result


class ProblemRegistry(_AppendOnlyStore):
    """Proposed problems. Producer: task_synthesizer."""
    kind = "problems"

    def append_problem(self, problem: Any) -> None:
        if isinstance(problem, dict):
            self.append(problem)
        elif hasattr(problem, "__dataclass_fields__"):
            self.append(problem)
        else:
            logger.warning("ProblemRegistry.append_problem: unknown type %s", type(problem))

    def get_by_id(self, problem_id: str) -> dict | None:
        """Return the most-recent-patch-folded view of the problem record.

        Because the file is append-only, mutations (e.g. retirement via
        mark_retired) are stored as separate patch records tagged _patch=True.
        This method returns the *last* record with the matching problem_id,
        which is either the original record (if never patched) or the latest
        patch (last-write-wins).  The `retired` field therefore reflects the
        current known state; callers do not need to replay patch history.
        """
        result = None
        for r in self.iter_records():
            if r.get("problem_id") == problem_id:
                result = r
        return result

    def mark_retired(self, problem_id: str, session_id: str = "") -> None:
        """Append a retirement patch record (spec §7: retire on first pool acceptance)."""
        self.append({
            "problem_id": problem_id,
            "retired": True,
            "retired_at": _now(),
            "session_id": session_id,
            "_patch": True,
        })


class VerificationLog(_AppendOnlyStore):
    """Primary + adversarial VerificationRecords. Producer: property_verifier."""
    kind = "verifications"

    def append_verification(self, record: Any) -> None:
        if isinstance(record, dict):
            self.append(record)
        elif hasattr(record, "__dataclass_fields__"):
            self.append(record)
        else:
            logger.warning("VerificationLog.append_verification: unknown type %s", type(record))

    def get_for_candidate(self, candidate_id: str) -> list[dict]:
        return [r for r in self.iter_records() if r.get("candidate_id") == candidate_id]

    def adversarial_records(self) -> Iterator[dict]:
        for r in self.iter_records():
            if r.get("adversarial"):
                yield r


class CalibrationLedger(_AppendOnlyStore):
    """Per-class calibration snapshots, one row per tick per class. Producer: orchestrator."""
    kind = "calibration"

    # A single shared session-scoped file is fine; but spec §4.1 says
    # "outputs/calibration.jsonl" (no <sid>) — we honour that by using a
    # fixed filename regardless of sid.

    def __init__(self, output_dir: Path, sid: str) -> None:
        self._dir = output_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self.path = self._dir / "calibration.jsonl"
        self._sid = sid

    def append_calibration(self, entry: CalibrationEntry | dict) -> None:
        self.append(entry)

    def suspended_classes(self, tick: int | None = None) -> set[str]:
        """Return the set of class names currently suspended.

        If tick is given, only considers entries up to that tick.
        Last entry per class wins (mirrors mark_retired pattern).
        """
        latest: dict[str, bool] = {}
        for r in self.iter_records():
            if tick is not None and r.get("tick", 0) > tick:
                continue
            cls = r.get("independence_class", "")
            if cls:
                latest[cls] = bool(r.get("suspended", False))
        return {cls for cls, suspended in latest.items() if suspended}

    def class_stats(self, independence_class: str) -> list[dict]:
        return [r for r in self.iter_records()
                if r.get("independence_class") == independence_class]


class TrainingPool(_AppendOnlyStore):
    """Candidates accepted for LoRA training. Producer: integrator."""
    kind = "training_pool"

    def append_sample(self, record: Any) -> None:
        if isinstance(record, dict):
            self.append(record)
        elif hasattr(record, "__dataclass_fields__"):
            self.append(record)
        else:
            logger.warning("TrainingPool.append_sample: unknown type %s", type(record))

    def pending_samples(self, min_batch: int = 1) -> list[dict]:
        """Return all records; caller decides when pool is large enough to train."""
        return list(self.iter_records())


# ──────────────────────────────────── factory ───────────────────────────────


@dataclass
class RSIRegistries:
    """Container for all five session registries. Instantiate once per RSI session."""
    property_registry: PropertyRegistry
    problem_registry: ProblemRegistry
    verification_log: VerificationLog
    calibration_ledger: CalibrationLedger
    training_pool: TrainingPool

    @classmethod
    def open(cls, output_dir: Path | str, sid: str | None = None) -> "RSIRegistries":
        """Open (or create) all registries for a session.

        Args:
            output_dir: base outputs/ directory (e.g. Path("./outputs")).
            sid: session id; defaults to a random 12-char hex string.
        """
        output_dir = Path(output_dir)
        if sid is None:
            sid = _sid_default()
        return cls(
            property_registry=PropertyRegistry(output_dir, sid),
            problem_registry=ProblemRegistry(output_dir, sid),
            verification_log=VerificationLog(output_dir, sid),
            calibration_ledger=CalibrationLedger(output_dir, sid),
            training_pool=TrainingPool(output_dir, sid),
        )

    @property
    def sid(self) -> str:
        return self.property_registry._sid
