"""Task #4: solution-diversity tracker (leading indicator for mode collapse)."""
from __future__ import annotations

from src.diagnostics.solution_diversity import (
    compute_diversity,
    embed_texts,
    pairwise_cosine_stats,
    _hash_embed,
    MODE_COLLAPSE_SIM_THRESHOLD,
)


def test_hash_embed_is_deterministic_and_normalized():
    v1 = _hash_embed("return a + b")
    v2 = _hash_embed("return a + b")
    assert v1 == v2
    norm_sq = sum(x * x for x in v1)
    assert abs(norm_sq - 1.0) < 1e-6


def test_pairwise_cosine_identical_is_one():
    v = _hash_embed("print(hello)")
    mean, mx = pairwise_cosine_stats([v, v, v])
    assert abs(mean - 1.0) < 1e-6
    assert abs(mx - 1.0) < 1e-6


def test_pairwise_cosine_handles_less_than_two():
    mean, mx = pairwise_cosine_stats([])
    assert (mean, mx) == (0.0, 0.0)
    mean, mx = pairwise_cosine_stats([[1.0, 0.0]])
    assert (mean, mx) == (0.0, 0.0)


def test_mode_collapse_alarm_fires_on_identical_candidates():
    # Every problem has 6 identical candidates → mean cosine = 1.0 > 0.90
    # for >50% (100%) of problems → alarm.
    cands = {f"p{i}": ["def solve(): return 42"] * 6 for i in range(5)}
    report = compute_diversity(cycle=7, candidates_by_problem=cands)
    assert report.mode_collapse_alarm is True
    assert report.fraction_above_threshold == 1.0
    assert report.n_problems == 5
    assert report.diversity_mean < 0.05  # near zero diversity


def test_mode_collapse_alarm_quiet_on_diverse_candidates():
    # Highly distinct candidates per problem.
    cands = {
        "p0": [
            "return a + b",
            "print(sorted(xs))",
            "for i in range(n): yield i*i",
            "math.sqrt(x) if x >= 0 else None",
            "''.join(reversed(s))",
            "{k:v for k,v in items}",
        ],
        "p1": [
            "lambda x: x**2",
            "functools.reduce(operator.mul, nums)",
            "requests.get(url).json()",
            "re.findall(r'\\d+', text)",
            "collections.Counter(words)",
            "json.dumps({'ok': True})",
        ],
    }
    report = compute_diversity(cycle=3, candidates_by_problem=cands)
    assert report.mode_collapse_alarm is False
    assert report.fraction_above_threshold < MODE_COLLAPSE_SIM_THRESHOLD


def test_compute_diversity_empty_input_no_alarm():
    report = compute_diversity(cycle=0, candidates_by_problem={})
    assert report.mode_collapse_alarm is False
    assert report.n_problems == 0
    assert report.diversity_mean == 0.0


def test_compute_diversity_single_candidate_does_not_crash():
    report = compute_diversity(cycle=1, candidates_by_problem={"p0": ["only one"]})
    assert report.n_problems == 1
    assert report.per_problem[0].mean_pairwise_cosine == 0.0
    assert report.mode_collapse_alarm is False


def test_report_to_dict_roundtrip():
    report = compute_diversity(
        cycle=2,
        candidates_by_problem={"p0": ["a", "b", "c"]},
    )
    d = report.to_dict()
    assert d["cycle"] == 2
    assert "per_problem" in d
    assert isinstance(d["per_problem"], list)


def test_embed_texts_fallback_backend_is_labelled():
    # Passing a bogus model name forces the sentence-transformers load to
    # fail and falls back to hashing. The backend label must reflect that
    # so operators can tell from the log which embedding path ran.
    vecs, backend = embed_texts(["a", "b"], model_name="does-not-exist/xyz-123")
    assert len(vecs) == 2
    assert "hash-fallback" in backend or backend.startswith("sentence-transformers")
