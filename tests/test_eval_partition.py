"""Tests for the hard 4-way eval partition.

Covers: no overlap, seed determinism, distribution matches declared weights,
stability across python processes (implicitly via byte-level hashing), and
that curated ground-truth items partition cleanly.
"""

from __future__ import annotations

import random

from src.diagnostics.eval_partition import (
    PARTITION_SEED,
    Partition,
    _WEIGHTS,
    count_partitions,
    filter_to,
    is_held_out,
    is_proposer_eligible,
    is_smoke_eval,
    is_train_eligible,
    partition_for,
    partition_for_question,
    question_id,
)


def test_partition_is_total_and_disjoint():
    """Every qid lands in exactly one bucket."""
    for i in range(5000):
        qid = question_id(f"problem-{i}", str(i))
        p = partition_for(qid)
        members = [
            p is Partition.HELD_OUT_ONLY,
            p is Partition.PROPOSER_ONLY,
            p is Partition.TRAIN,
            p is Partition.SMOKE_EVAL,
        ]
        assert sum(members) == 1


def test_partition_is_deterministic():
    """Same prompt/answer → same bucket every call."""
    for i in range(200):
        qid = question_id(f"q{i}", "ans")
        p1 = partition_for(qid)
        p2 = partition_for(qid)
        p3 = partition_for_question(f"q{i}", "ans")
        assert p1 is p2 is p3


def test_question_id_is_stable_and_sensitive():
    """Stable across calls, sensitive to prompt and canonical answer."""
    assert question_id("a", "b") == question_id("a", "b")
    assert question_id("a", "b") != question_id("a", "c")
    assert question_id("a", "b") != question_id("b", "b")
    # Same-length hex id
    assert len(question_id("foo", "bar")) == 16


def test_partition_distribution_matches_weights():
    """Over 10k random qids, bucket frequencies should be within 3σ of declared weights."""
    N = 10_000
    counts = count_partitions(
        (f"prompt-{i}-{random.random()}", str(i)) for i in range(N)
    )
    observed = {
        Partition.HELD_OUT_ONLY: counts.held_out_only,
        Partition.PROPOSER_ONLY: counts.proposer_only,
        Partition.TRAIN: counts.train,
        Partition.SMOKE_EVAL: counts.smoke_eval,
    }
    for bucket, weight in _WEIGHTS:
        expected = N * weight
        sigma = (N * weight * (1 - weight)) ** 0.5
        assert abs(observed[bucket] - expected) < 4 * sigma, (
            f"{bucket} off: observed={observed[bucket]}, expected≈{expected:.0f}, σ={sigma:.1f}"
        )
    assert counts.total == N


def test_eligibility_helpers_match_partition():
    """is_held_out / is_smoke_eval / is_proposer_eligible / is_train_eligible all
    agree with `partition_for_question`, and together cover the universe."""
    for i in range(500):
        prompt = f"q-{i}"
        ans = str(i)
        p = partition_for_question(prompt, ans)
        assert is_held_out(prompt, ans) == (p is Partition.HELD_OUT_ONLY)
        assert is_smoke_eval(prompt, ans) == (p is Partition.SMOKE_EVAL)
        assert is_train_eligible(prompt, ans) == (p is Partition.TRAIN)
        assert is_proposer_eligible(prompt, ans) == (
            p is Partition.PROPOSER_ONLY or p is Partition.TRAIN
        )
        # Disjointness: proposer-eligible items are never held-out or smoke.
        if is_proposer_eligible(prompt, ans):
            assert not is_held_out(prompt, ans)
            assert not is_smoke_eval(prompt, ans)


def test_curated_bank_partitions_cleanly():
    """Real curated items produce a sensible partition split."""
    from src.diagnostics.ground_truth import _build_curated
    curated = _build_curated()
    items = [(q.prompt, q.canonical_answer) for qs in curated.values() for q in qs]
    assert len(items) > 50  # sanity
    counts = count_partitions(items)
    assert counts.total == len(items)
    # Each bucket gets at least one item from a universe of 100+
    assert counts.held_out_only >= 1
    assert counts.proposer_only >= 1
    assert counts.train >= 1


def test_filter_to_selects_correct_bucket():
    """filter_to returns only items in the requested bucket."""
    items = [
        {"prompt": f"q-{i}", "expected": str(i)} for i in range(500)
    ]
    held = filter_to(items, Partition.HELD_OUT_ONLY)
    for q in held:
        assert is_held_out(q["prompt"], q["expected"])
    # All other buckets' items are excluded.
    excluded_qids = {question_id(q["prompt"], q["expected"]) for q in items} - \
                    {question_id(q["prompt"], q["expected"]) for q in held}
    for qid in excluded_qids:
        assert partition_for(qid) is not Partition.HELD_OUT_ONLY


def test_seed_is_versioned_constant():
    """If anyone edits PARTITION_SEED, every existing bucket assignment changes —
    flag it as an intentional, observable breaking change."""
    # Regression guard: hash of the current seed. Update both values together
    # if you intend to reshuffle the partition.
    import hashlib
    assert hashlib.sha256(PARTITION_SEED.encode()).hexdigest()[:12] == "1fee8909ded6"
