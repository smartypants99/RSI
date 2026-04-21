"""Hard 4-way partition of the diagnostic question universe.

Prior to this module, the "held-out" eval set and the proposer's curriculum
state could overlap: the ground-truth bank split off a holdout slice using a
domain-local RNG (see `build_ground_truth_bank`), but the open-ended
curriculum / OOD proposer / template generators could — and did — surface
the same questions that the held-out eval would later draw. That collapsed
the held-out signal into a self-validation closed loop whenever a proposed
problem's prompt matched a curated entry.

This module enforces a hard partition by hashing question identity into one
of four disjoint buckets. Any question — curated, programmatic, or
template-synthesized — belongs to exactly one bucket, deterministically:

    HELD_OUT_ONLY   — only the Phase-5b held-out eval may draw these.
    PROPOSER_ONLY   — only the curriculum / OOD proposer may seed from these.
    TRAIN           — eligible for the training pool.
    SMOKE_EVAL      — only self_edit.smoke_eval may draw these.

Partition is defined by a single deterministic function of `question_id`
(a stable SHA256 over prompt + canonical answer) and a fixed module-level
seed. Changing `PARTITION_SEED` reshuffles the partition; treat it as a
versioned constant.

Bucket weights are chosen so that, for typical bank sizes, the held-out
bucket lands around 1200 questions. With 168 curated items we rely on
programmatic generators to fill the rest — each programmatic draw is
bucketed by its final prompt hash, so drawing 1200 held-out items only
requires oversampling the generators ~1/HELD_OUT_WEIGHT times.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional


PARTITION_SEED = "rsi-v1-eval-partition-2026-04"


class Partition(str, Enum):
    HELD_OUT_ONLY = "held_out_only"
    PROPOSER_ONLY = "proposer_only"
    TRAIN = "train"
    SMOKE_EVAL = "smoke_eval"


# Partition weights MUST sum to 1.0. 0.37 × typical bank fill → ~1200 held-out
# questions at p=0.5 gives SE ≈ 0.014, enough to clear |delta| > 0.02 at
# high confidence (user "eval-rigor" concern).
_WEIGHTS: tuple[tuple[Partition, float], ...] = (
    (Partition.HELD_OUT_ONLY, 0.37),
    (Partition.PROPOSER_ONLY, 0.28),
    (Partition.TRAIN, 0.28),
    (Partition.SMOKE_EVAL, 0.07),
)

HELD_OUT_TARGET = 1200


def question_id(prompt: str, canonical_answer: str | None = None) -> str:
    """Stable 16-hex-char question identity over prompt + canonical answer.

    Two questions with the same prompt but different canonical answers map
    to different ids, so programmatic generators emitting the same wording
    with different answers don't collide.
    """
    payload = f"{prompt}|{canonical_answer if canonical_answer is not None else ''}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def partition_for(qid: str) -> Partition:
    """Deterministically map a question id to its partition bucket.

    Uses a 53-bit unit-interval draw seeded by PARTITION_SEED so the
    partition is stable across processes and versions that share the seed.
    """
    h = hashlib.sha256(f"{PARTITION_SEED}|{qid}".encode("utf-8")).digest()
    # 56-bit uniform draw. Using a power-of-two divisor avoids the modulo
    # bias that `int(h, 16) % 100` would introduce.
    x = int.from_bytes(h[:7], "big") / float(1 << 56)
    cum = 0.0
    for bucket, w in _WEIGHTS:
        cum += w
        if x < cum:
            return bucket
    return _WEIGHTS[-1][0]


def partition_for_question(prompt: str, canonical_answer: str | None = None) -> Partition:
    return partition_for(question_id(prompt, canonical_answer))


def is_held_out(prompt: str, canonical_answer: str | None = None) -> bool:
    return partition_for_question(prompt, canonical_answer) is Partition.HELD_OUT_ONLY


def is_smoke_eval(prompt: str, canonical_answer: str | None = None) -> bool:
    return partition_for_question(prompt, canonical_answer) is Partition.SMOKE_EVAL


def is_proposer_eligible(prompt: str, canonical_answer: str | None = None) -> bool:
    """Proposer may seed from PROPOSER_ONLY or TRAIN, never HELD_OUT or SMOKE_EVAL."""
    p = partition_for_question(prompt, canonical_answer)
    return p is Partition.PROPOSER_ONLY or p is Partition.TRAIN


def is_train_eligible(prompt: str, canonical_answer: str | None = None) -> bool:
    """Training pool may only ingest TRAIN-bucket items."""
    return partition_for_question(prompt, canonical_answer) is Partition.TRAIN


@dataclass(frozen=True)
class PartitionCounts:
    held_out_only: int
    proposer_only: int
    train: int
    smoke_eval: int

    @property
    def total(self) -> int:
        return self.held_out_only + self.proposer_only + self.train + self.smoke_eval


def count_partitions(items: Iterable[tuple[str, Optional[str]]]) -> PartitionCounts:
    """For testing/observability: count items per bucket."""
    buckets = {p: 0 for p in Partition}
    for prompt, ans in items:
        buckets[partition_for_question(prompt, ans)] += 1
    return PartitionCounts(
        held_out_only=buckets[Partition.HELD_OUT_ONLY],
        proposer_only=buckets[Partition.PROPOSER_ONLY],
        train=buckets[Partition.TRAIN],
        smoke_eval=buckets[Partition.SMOKE_EVAL],
    )


def filter_to(items: Iterable[dict], bucket: Partition,
              prompt_key: str = "prompt",
              answer_key: str = "expected") -> list[dict]:
    """Filter a list of question-dicts to only those in `bucket`."""
    out: list[dict] = []
    for q in items:
        p = q.get(prompt_key, "")
        a = q.get(answer_key, None)
        if partition_for_question(p, a) is bucket:
            out.append(q)
    return out
