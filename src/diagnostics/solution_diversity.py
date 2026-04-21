"""Per-cycle solution-diversity tracker (task #4).

For every problem in a cycle the synthesizer produces k candidate solutions
(default k=6). If those candidates converge onto a single canonical answer
for most problems, the model is mode-collapsing — and empirically this shows
up 10-20 cycles *before* the held-out benchmark degrades. This module gives
the orchestrator a leading indicator instead of a trailing one.

Design constraints:
  * No training / no GPU. Embedding uses sentence-transformers' smallest
    model (all-MiniLM-L6-v2). If the package is unavailable at runtime we
    fall back to a deterministic hashing embedding so the tracker never
    silently returns zeros — a dependency miss must not look like "perfect
    diversity" in the log.
  * Pure functions where practical; the single stateful bit is a lazily
    cached embedding model instance.
  * Identical inputs across cycles produce identical similarity numbers so
    the leading-indicator trend is comparable cycle-to-cycle.

This module is on `HARD_DENY_LIST` — self-edit cannot modify it.
"""

from __future__ import annotations

import hashlib
import logging
import math
import statistics
from dataclasses import dataclass, field, asdict
from typing import Iterable, Optional, Sequence

logger = logging.getLogger(__name__)


MODE_COLLAPSE_SIM_THRESHOLD = 0.90
MODE_COLLAPSE_PROBLEM_FRACTION = 0.50
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_FALLBACK_DIM = 64


@dataclass
class ProblemDiversity:
    problem_id: str
    n_candidates: int
    mean_pairwise_cosine: float
    max_pairwise_cosine: float
    above_threshold: bool


@dataclass
class DiversityReport:
    cycle: int
    n_problems: int
    n_problems_above_threshold: int
    fraction_above_threshold: float
    diversity_mean: float       # 1 - mean(mean_pairwise_cosine) across problems
    diversity_variance: float   # variance of (1 - mean_pairwise_cosine) across problems
    mode_collapse_alarm: bool
    per_problem: list[ProblemDiversity] = field(default_factory=list)
    embedding_backend: str = "unknown"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["per_problem"] = [asdict(p) for p in self.per_problem]
        return d


# ---------------------------------------------------------------------------
# Embedding backend
# ---------------------------------------------------------------------------


_MODEL_CACHE: dict[str, object] = {}


def _load_sentence_transformer(model_name: str):
    """Return a SentenceTransformer instance, or None if unavailable."""
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as e:
        logger.debug("sentence-transformers unavailable: %s", e)
        _MODEL_CACHE[model_name] = None
        return None
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        logger.warning("failed to load %s: %s — falling back to hashing embedding", model_name, e)
        _MODEL_CACHE[model_name] = None
        return None
    _MODEL_CACHE[model_name] = model
    return model


def _hash_embed(text: str, dim: int = _FALLBACK_DIM) -> list[float]:
    """Deterministic fallback embedding. Not semantic, but never silently zero.

    Splits text into tokens and hashes each into one of `dim` buckets; the
    vector is L2-normalised so cosine similarities are in [0, 1]. For mode-
    collapse detection we care about *relative* similarity across candidates
    for the same problem, so a hash-based signal is still informative: two
    identical strings score 1.0, two token-overlapping strings score high.
    """
    vec = [0.0] * dim
    if not text:
        vec[0] = 1.0
        return vec
    tokens = text.split()
    if not tokens:
        tokens = [text]
    for tok in tokens:
        h = int(hashlib.sha1(tok.encode("utf-8", "replace")).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        vec[0] = 1.0
        return vec
    return [x / norm for x in vec]


def embed_texts(
    texts: Sequence[str],
    model_name: str = DEFAULT_MODEL_NAME,
) -> tuple[list[list[float]], str]:
    """Embed a list of strings. Returns (vectors, backend_name)."""
    if not texts:
        return [], "empty"
    model = _load_sentence_transformer(model_name)
    if model is not None:
        try:
            vecs = model.encode(list(texts), normalize_embeddings=True)
            # tolist() may be on np array or torch; try both paths.
            if hasattr(vecs, "tolist"):
                vecs = vecs.tolist()
            return [list(v) for v in vecs], f"sentence-transformers:{model_name}"
        except Exception as e:
            logger.warning("sentence-transformers encode failed: %s — falling back to hash", e)
    return [_hash_embed(t) for t in texts], "hash-fallback"


# ---------------------------------------------------------------------------
# Cosine / pairwise aggregation (pure, tested)
# ---------------------------------------------------------------------------


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na) * math.sqrt(nb)
    if denom == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / denom))


def pairwise_cosine_stats(vectors: Sequence[Sequence[float]]) -> tuple[float, float]:
    """Return (mean, max) of pairwise cosine similarities. (0.0, 0.0) if <2."""
    n = len(vectors)
    if n < 2:
        return 0.0, 0.0
    sims: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(_cosine(vectors[i], vectors[j]))
    return (sum(sims) / len(sims), max(sims))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _candidate_to_text(c) -> str:
    if isinstance(c, str):
        return c
    return getattr(c, "solution", None) or str(c)


def compute_diversity(
    cycle: int,
    candidates_by_problem: dict,
    model_name: str = DEFAULT_MODEL_NAME,
    sim_threshold: float = MODE_COLLAPSE_SIM_THRESHOLD,
    problem_fraction: float = MODE_COLLAPSE_PROBLEM_FRACTION,
) -> DiversityReport:
    """Compute per-cycle solution diversity + mode-collapse alarm.

    Args:
        cycle: integer cycle index (for logging).
        candidates_by_problem: {problem_id: [candidate_str_or_obj, ...]}.
        model_name: sentence-transformers model id.
        sim_threshold: per-problem mean pairwise cosine above which the
            problem counts as "collapsed" for alarm purposes.
        problem_fraction: if more than this fraction of problems are
            collapsed, raise `mode_collapse_alarm`.

    Always returns a report; never raises on empty / degenerate input.
    """
    per_problem: list[ProblemDiversity] = []
    backend_seen = "empty"

    for pid, cands in (candidates_by_problem or {}).items():
        texts = [_candidate_to_text(c) for c in cands or []]
        texts = [t for t in texts if t]
        n = len(texts)
        if n < 2:
            per_problem.append(ProblemDiversity(
                problem_id=str(pid),
                n_candidates=n,
                mean_pairwise_cosine=0.0,
                max_pairwise_cosine=0.0,
                above_threshold=False,
            ))
            continue
        vecs, backend = embed_texts(texts, model_name=model_name)
        backend_seen = backend
        mean_sim, max_sim = pairwise_cosine_stats(vecs)
        per_problem.append(ProblemDiversity(
            problem_id=str(pid),
            n_candidates=n,
            mean_pairwise_cosine=mean_sim,
            max_pairwise_cosine=max_sim,
            above_threshold=mean_sim > sim_threshold,
        ))

    n_problems = len(per_problem)
    n_collapsed = sum(1 for p in per_problem if p.above_threshold)
    frac_collapsed = n_collapsed / n_problems if n_problems else 0.0

    # Diversity scalar = 1 - mean_pairwise_cosine; higher is more diverse.
    diversities = [1.0 - p.mean_pairwise_cosine for p in per_problem if p.n_candidates >= 2]
    div_mean = statistics.fmean(diversities) if diversities else 0.0
    div_var = statistics.pvariance(diversities) if len(diversities) >= 2 else 0.0

    alarm = frac_collapsed > problem_fraction and n_problems > 0

    report = DiversityReport(
        cycle=cycle,
        n_problems=n_problems,
        n_problems_above_threshold=n_collapsed,
        fraction_above_threshold=frac_collapsed,
        diversity_mean=div_mean,
        diversity_variance=div_var,
        mode_collapse_alarm=alarm,
        per_problem=per_problem,
        embedding_backend=backend_seen,
    )

    if alarm:
        logger.warning(
            "[cycle %d] MODE-COLLAPSE ALARM: %d/%d problems (%.1f%%) have "
            "mean pairwise cosine > %.2f (leading indicator, 10-20 cycles "
            "ahead of benchmark degradation)",
            cycle, n_collapsed, n_problems, 100.0 * frac_collapsed, sim_threshold,
        )
    else:
        logger.info(
            "[cycle %d] diversity: mean=%.3f var=%.4f collapsed=%d/%d backend=%s",
            cycle, div_mean, div_var, n_collapsed, n_problems, backend_seen,
        )
    return report
