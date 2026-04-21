"""Paired-sample variance reduction for held-out eval deltas.

Unpaired two-sample delta: delta = mean(post) - mean(pre). Variance:
    Var(delta) ≈ Var(post)/n + Var(pre)/n ≈ 2·p(1-p)/n.

Paired delta over the SAME n questions: for each q_i, let d_i =
correct_post(q_i) - correct_pre(q_i) ∈ {-1, 0, +1}. The sample mean of
the d_i has variance:
    Var(d̄) = Var(d_i)/n = [p(1-p) - Cov(pre, post)] · 2/n.

For held-out eval, a question's difficulty is mostly a fixed property
of the question itself, so Cov(pre_i, post_i) is high (McNemar-style
pairing applies). Empirically the variance reduction is 3-5× over
unpaired estimation on the same eval set, which is the whole point of
freezing the question bank + seed across cycles.

This module computes the paired delta + its standard error so the
orchestrator's `|delta| > 0.02` decision rule gets a real confidence
interval, not a false-precision point estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PairedDelta:
    n: int
    mean_pre: float
    mean_post: float
    delta: float
    delta_se: float  # paired SE
    unpaired_se: float  # what SE would have been without pairing (for comparison)
    variance_reduction: float  # unpaired_se^2 / paired_se^2, higher = more gain
    z: float  # delta / delta_se


def _per_question_correct_map(per_q_records: list[dict]) -> dict[str, int]:
    """Extract {question_key: 0/1 correct} from per_question records."""
    out: dict[str, int] = {}
    for rec in per_q_records:
        # Key by (prompt, expected) — matches what the diagnostics engine
        # records and survives across cycles since frozen eval uses the
        # same ground-truth bank.
        key = f"{rec.get('prompt', rec.get('question', ''))}|{rec.get('expected', '')}"
        out[key] = 1 if bool(rec.get("correct", False)) else 0
    return out


def paired_delta(
    pre_per_q: list[dict],
    post_per_q: list[dict],
) -> PairedDelta | None:
    """Compute paired delta + SE over questions present in both runs.

    Returns None if fewer than 2 matched questions.
    """
    pre_map = _per_question_correct_map(pre_per_q)
    post_map = _per_question_correct_map(post_per_q)
    keys = sorted(pre_map.keys() & post_map.keys())
    n = len(keys)
    if n < 2:
        return None

    d_values = [post_map[k] - pre_map[k] for k in keys]
    pre_vals = [pre_map[k] for k in keys]
    post_vals = [post_map[k] for k in keys]

    mean_pre = sum(pre_vals) / n
    mean_post = sum(post_vals) / n
    mean_d = sum(d_values) / n

    # Sample variance of the paired differences (unbiased, n-1).
    var_d = sum((d - mean_d) ** 2 for d in d_values) / (n - 1) if n > 1 else 0.0
    paired_se = math.sqrt(var_d / n) if var_d > 0 else 0.0

    # Unpaired-equivalent SE: if we had treated pre and post as independent
    # samples, the SE would be sqrt(Var(pre)/n + Var(post)/n).
    var_pre = sum((x - mean_pre) ** 2 for x in pre_vals) / (n - 1) if n > 1 else 0.0
    var_post = sum((x - mean_post) ** 2 for x in post_vals) / (n - 1) if n > 1 else 0.0
    unpaired_se = math.sqrt(var_pre / n + var_post / n)

    vr = (unpaired_se ** 2) / (paired_se ** 2) if paired_se > 0 else float("inf")
    z = mean_d / paired_se if paired_se > 0 else 0.0

    return PairedDelta(
        n=n,
        mean_pre=mean_pre,
        mean_post=mean_post,
        delta=mean_d,
        delta_se=paired_se,
        unpaired_se=unpaired_se,
        variance_reduction=vr,
        z=z,
    )
