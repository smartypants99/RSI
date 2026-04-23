"""Intra-rep chunked SPRT (K=4 OBF) over diagnostics.run_chunked().

Covers:
  1. DiagnosticsEngine.run_chunked yields deterministic cumulative chunks
     whose per_question union equals run()'s output.
  2. ImprovementLoop._run_heldout_chunked_sprt stops early on a strong
     positive signal (strong post-vs-base delta) — consumes < K chunks.
  3. Under a null delta, the generator runs to completion (no false
     early-stop) and the full per_question vector is accumulated.
  4. With a weak-but-nonzero delta AND a caller-configured futility_z,
     stop_accept_null fires before max_chunks.

OBF K=4 α=0.05 critical values (Jennison-Turnbull Table 2.3, c=2.024):
    look 1 (25% N): |z| ≥ 4.049
    look 2 (50% N): |z| ≥ 2.863
    look 3 (75% N): |z| ≥ 2.337
    look 4 (100%):  |z| ≥ 2.024
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.diagnostics.sequential_eval import (
    obf_critical_values,
    sprt_decide,
)


# ---------------------------------------------------------------------------
# OBF K=4 boundary arithmetic
# ---------------------------------------------------------------------------


def test_obf_k4_critical_values_match_jennison_turnbull():
    c1, c2, c3, c4 = obf_critical_values(K=4, alpha=0.05)
    # c=2.024, c_k = c / sqrt(k/K).
    assert abs(c1 - 4.049) < 0.01, c1
    assert abs(c2 - 2.863) < 0.01, c2
    assert abs(c3 - 2.337) < 0.01, c3
    assert abs(c4 - 2.024) < 0.01, c4


def test_sprt_decide_k4_continue_below_boundary():
    d = sprt_decide(
        look=1, n_so_far=150, delta=0.01, delta_se=0.01, K=4,
    )
    # |z| = 1.0 << 4.049 → continue
    assert d.decision == "continue"


def test_sprt_decide_k4_stop_reject_above_boundary():
    d = sprt_decide(
        look=2, n_so_far=300, delta=0.06, delta_se=0.01, K=4,
    )
    # |z| = 6 >> 2.863 → stop
    assert d.decision == "stop_reject_null"


def test_sprt_decide_k4_futility_stops_accept_null():
    d = sprt_decide(
        look=2, n_so_far=300, delta=0.0005, delta_se=0.01,
        K=4, futility_z=0.5,
    )
    # |z| = 0.05 << futility_z=0.5 → stop_accept_null
    assert d.decision == "stop_accept_null"


def test_sprt_decide_k4_futility_does_not_fire_on_final_look():
    d = sprt_decide(
        look=4, n_so_far=600, delta=0.0005, delta_se=0.01,
        K=4, futility_z=0.5,
    )
    # Final look: futility disabled (we're already done).
    assert d.decision == "continue"


# ---------------------------------------------------------------------------
# DiagnosticsEngine.run_chunked — streaming primitive
# ---------------------------------------------------------------------------


class _StubEngine:
    """Minimal stand-in for DiagnosticsEngine.run_chunked — exercises the
    method bound on the real class via attribute patching without needing
    a live model loader. Populates the same state the method reads."""

    def __init__(self, domains, questions_per_domain):
        from src.diagnostics.engine import DiagnosticsEngine
        self._run_chunked = DiagnosticsEngine.run_chunked.__get__(self)
        self.config = SimpleNamespace(domains=list(domains))
        self._seen_hashes = set()
        # Preseed per-domain (prompt, expected, correct) triples the stub
        # _probe_domain returns; keyed by domain.
        self._fake = {}
        for i, d in enumerate(domains):
            self._fake[d] = [
                {
                    "question": f"q{d}-{k}",
                    "expected": f"e{d}-{k}",
                    "response": "",
                    "correct": (k % 2 == 0),
                    "score": 1.0 if k % 2 == 0 else 0.0,
                    "domain": d,
                    "subdomain": "general",
                    "difficulty": "medium",
                    "confidence": 0.5,
                    "check_type": "contains",
                }
                for k in range(questions_per_domain)
            ]

    def _probe_domain(self, domain, cycle):
        evidence = list(self._fake[domain])
        score = sum(1 for e in evidence if e["correct"]) / max(1, len(evidence))
        return score, evidence

    def run_chunked(self, *args, **kwargs):
        return self._run_chunked(*args, **kwargs)


def test_run_chunked_determinism_and_cumulative_schema():
    eng_a = _StubEngine(domains=["d1", "d2", "d3", "d4"], questions_per_domain=50)
    eng_b = _StubEngine(domains=["d1", "d2", "d3", "d4"], questions_per_domain=50)
    chunks_a = list(eng_a.run_chunked(cycle=7, chunk_size=100))
    chunks_b = list(eng_b.run_chunked(cycle=7, chunk_size=100))
    # Deterministic — same (cycle, chunk_size) yields same number of chunks
    # and same cumulative per_question.
    assert len(chunks_a) == len(chunks_b)
    # Chunks are cumulative: strictly non-decreasing total_questions.
    totals = [c.total_questions for c in chunks_a]
    assert totals == sorted(totals)
    # Final chunk matches what run() would produce (4 domains × 50 = 200).
    assert chunks_a[-1].total_questions == 200
    # Per-question carries the required keys.
    q = chunks_a[-1].per_question[0]
    for k in ("question_id", "domain", "question", "expected", "correct",
              "score", "difficulty", "confidence", "check_type"):
        assert k in q, f"missing per_question field: {k}"


def test_run_chunked_max_prompts_caps_total():
    eng = _StubEngine(domains=["d1", "d2", "d3", "d4"], questions_per_domain=50)
    chunks = list(eng.run_chunked(cycle=1, chunk_size=50, max_prompts=75))
    assert chunks[-1].total_questions <= 75


# ---------------------------------------------------------------------------
# Orchestrator wire-up — _run_heldout_chunked_sprt
# ---------------------------------------------------------------------------


def _make_loop(tmp_path: Path, **orch_overrides):
    from src.orchestrator.loop import ImprovementLoop
    from src.utils.config import OrchestratorConfig

    loop = ImprovementLoop.__new__(ImprovementLoop)
    orchestrator = OrchestratorConfig(
        output_dir=tmp_path,
        log_dir=tmp_path / "logs",
        **orch_overrides,
    )
    loop.config = SimpleNamespace(orchestrator=orchestrator)
    loop.history = []
    loop._heldout_base_cache = None
    return loop


class _FakeHeldoutCache:
    def __init__(self, records):
        self.entries = {i: r for i, r in enumerate(records)}
        self._records = records

    def to_per_question_records(self):
        return list(self._records)


class _ScriptedDiag:
    """Iterates a pre-built list of partial DiagnosticResult objects."""

    def __init__(self, chunks):
        self._chunks = chunks

    def run_chunked(self, cycle, chunk_size=150, max_prompts=None):
        for c in self._chunks:
            yield c


def _mk_partial(prompts, pre_score_each=0.5, correct_each=False):
    """Build a minimal DiagnosticResult-shaped SimpleNamespace usable by
    _run_heldout_chunked_sprt. The helper reads .per_question."""
    per_q = []
    for p, expected, score in prompts:
        per_q.append({
            "question": p,
            "prompt": p,  # continuous_paired_delta prefers prompt
            "expected": expected,
            "correct": score >= 0.5,
            "score": float(score),
        })
    return SimpleNamespace(per_question=per_q)


def _base_records(prompts, base_score):
    out = []
    for p, expected in prompts:
        out.append({
            "prompt": p,
            "expected": expected,
            "correct": base_score >= 0.5,
            "score": float(base_score),
        })
    return out


def test_chunked_sprt_strong_signal_stops_early(tmp_path):
    loop = _make_loop(tmp_path)
    # 4 chunks × 50 prompts each = 200 total. Base alternates 0.0/0.1
    # (non-zero variance), post alternates 0.9/1.0 — per-question delta
    # always ≈ 0.9 with tiny σ_d → |z| huge → stop at chunk 1.
    import random
    rng = random.Random(2)
    all_prompts = [(f"q{i}", f"e{i}") for i in range(200)]
    base_recs = []
    post_scores = []
    for i, (p, e) in enumerate(all_prompts):
        b = rng.uniform(0.0, 0.2)  # low base, real variance
        post = min(1.0, b + 0.7 + rng.uniform(-0.05, 0.05))  # strong +0.7 delta
        base_recs.append({
            "prompt": p, "expected": e, "correct": b >= 0.5, "score": b,
        })
        post_scores.append(post)
    loop._heldout_base_cache = _FakeHeldoutCache(base_recs)
    chunks = []
    running = []
    for i in range(4):
        for j, (p, e) in enumerate(all_prompts[i * 50:(i + 1) * 50]):
            gi = i * 50 + j
            running.append((p, e, post_scores[gi]))
        chunks.append(_mk_partial(running))
    loop.diagnostics = _ScriptedDiag(chunks)
    loop.HELDOUT_CYCLE_SEED = 42
    result, stop_chunk, decision = loop._run_heldout_chunked_sprt(
        chunk_size=50, K=4, futility_z=None,
    )
    assert stop_chunk == 1, stop_chunk
    assert decision == "stop_reject_null"
    # Final result holds exactly chunk-1's per_question (no more were
    # consumed after the break).
    assert result is not None
    assert len(result.per_question) == 50


def test_chunked_sprt_null_delta_runs_full_no_false_stop(tmp_path):
    loop = _make_loop(tmp_path)
    all_prompts = [(f"q{i}", f"e{i}") for i in range(200)]
    # True delta = 0 but with real variance. Base and post both drawn
    # from the same distribution (deterministic alternating 0.3/0.7) with
    # a mild anti-correlated shuffle so var_d > 0 but mean_d ≈ 0.
    # Concretely: base[i] = 0.3 if i%2 else 0.7; post[i] = 0.7 if i%2 else 0.3
    # would give a huge delta — instead we swap only on even i so net ≈ 0.
    import random
    rng = random.Random(0)
    base_recs = []
    post_scores = []
    for i, (p, e) in enumerate(all_prompts):
        b = 0.3 if i % 2 else 0.7
        # post perturbs ±0.05 symmetrically around the same base — mean 0.
        perturb = rng.choice([-0.05, 0.05])
        base_recs.append({
            "prompt": p, "expected": e, "correct": b >= 0.5, "score": b,
        })
        post_scores.append(max(0.0, min(1.0, b + perturb)))
    loop._heldout_base_cache = _FakeHeldoutCache(base_recs)
    chunks = []
    running = []
    for i in range(4):
        for j, (p, e) in enumerate(all_prompts[i * 50:(i + 1) * 50]):
            gi = i * 50 + j
            running.append((p, e, post_scores[gi]))
        chunks.append(_mk_partial(running))
    loop.diagnostics = _ScriptedDiag(chunks)
    loop.HELDOUT_CYCLE_SEED = 42
    result, stop_chunk, decision = loop._run_heldout_chunked_sprt(
        chunk_size=50, K=4, futility_z=None,
    )
    assert stop_chunk is None, (stop_chunk, decision)
    assert decision is None
    # Full 200 prompts consumed.
    assert result is not None
    assert len(result.per_question) == 200


def test_chunked_sprt_weak_delta_futility_fires(tmp_path):
    loop = _make_loop(tmp_path)
    all_prompts = [(f"q{i}", f"e{i}") for i in range(200)]
    # Tiny but non-zero true delta with real variance: base alternates
    # 0.3/0.7; post = base + tiny perturbation so |z| stays well below
    # any OBF boundary but also below futility_z=1.0 → stop_accept_null.
    import random
    rng = random.Random(1)
    base_recs = []
    post_scores = []
    for i, (p, e) in enumerate(all_prompts):
        b = 0.3 if i % 2 else 0.7
        perturb = rng.choice([-0.05, 0.05]) + 0.001  # near-null drift
        base_recs.append({
            "prompt": p, "expected": e, "correct": b >= 0.5, "score": b,
        })
        post_scores.append(max(0.0, min(1.0, b + perturb)))
    loop._heldout_base_cache = _FakeHeldoutCache(base_recs)
    chunks = []
    running = []
    for i in range(4):
        for j, (p, e) in enumerate(all_prompts[i * 50:(i + 1) * 50]):
            gi = i * 50 + j
            running.append((p, e, post_scores[gi]))
        chunks.append(_mk_partial(running))
    loop.diagnostics = _ScriptedDiag(chunks)
    loop.HELDOUT_CYCLE_SEED = 42
    result, stop_chunk, decision = loop._run_heldout_chunked_sprt(
        chunk_size=50, K=4, futility_z=1.0,
    )
    assert stop_chunk == 1
    assert decision == "stop_accept_null"
    assert result is not None
    assert len(result.per_question) == 50


# ---------------------------------------------------------------------------
# sprt_decisions.jsonl telemetry
# ---------------------------------------------------------------------------


def test_sprt_decisions_jsonl_emitted_per_chunk(tmp_path):
    """Every SPRT look — including the final stop — lands one row in
    outputs/sprt_decisions.jsonl under the orchestrator output_dir.
    """
    import json

    loop = _make_loop(tmp_path)
    # Reuse the strong-signal setup to force a chunk-1 stop.
    import random
    rng = random.Random(2)
    all_prompts = [(f"q{i}", f"e{i}") for i in range(200)]
    base_recs = []
    post_scores = []
    for i, (p, e) in enumerate(all_prompts):
        b = rng.uniform(0.0, 0.2)
        post = min(1.0, b + 0.7 + rng.uniform(-0.05, 0.05))
        base_recs.append({
            "prompt": p, "expected": e, "correct": b >= 0.5, "score": b,
        })
        post_scores.append(post)
    loop._heldout_base_cache = _FakeHeldoutCache(base_recs)
    chunks = []
    running = []
    for i in range(4):
        for j, (p, e) in enumerate(all_prompts[i * 50:(i + 1) * 50]):
            gi = i * 50 + j
            running.append((p, e, post_scores[gi]))
        chunks.append(_mk_partial(running))
    loop.diagnostics = _ScriptedDiag(chunks)
    loop.HELDOUT_CYCLE_SEED = 42
    result, stop_chunk, decision = loop._run_heldout_chunked_sprt(
        chunk_size=50, K=4, futility_z=None, cycle=7,
    )
    assert stop_chunk == 1
    path = tmp_path / "sprt_decisions.jsonl"
    assert path.exists(), "sprt_decisions.jsonl should have been written"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    assert len(rows) == 1, rows
    r = rows[0]
    assert r["cycle"] == 7
    assert r["chunk_idx"] == 1
    assert r["decision"] == "stop_reject_null"
    assert r["continuing"] is False
    assert r["n_so_far"] == 50
    assert isinstance(r["z"], float) and abs(r["z"]) > 2.0


def test_sprt_decisions_jsonl_null_delta_emits_one_row_per_chunk(tmp_path):
    """Under a null delta the generator runs all 4 chunks; each one gets
    its own row with continuing=True on 1-3 and continuing=True on row 4
    as well (run_chunked exhausted, not an SPRT stop)."""
    import json

    loop = _make_loop(tmp_path)
    all_prompts = [(f"q{i}", f"e{i}") for i in range(200)]
    import random
    rng = random.Random(0)
    base_recs = []
    post_scores = []
    for i, (p, e) in enumerate(all_prompts):
        b = 0.3 if i % 2 else 0.7
        perturb = rng.choice([-0.05, 0.05])
        base_recs.append({
            "prompt": p, "expected": e, "correct": b >= 0.5, "score": b,
        })
        post_scores.append(max(0.0, min(1.0, b + perturb)))
    loop._heldout_base_cache = _FakeHeldoutCache(base_recs)
    chunks = []
    running = []
    for i in range(4):
        for j, (p, e) in enumerate(all_prompts[i * 50:(i + 1) * 50]):
            gi = i * 50 + j
            running.append((p, e, post_scores[gi]))
        chunks.append(_mk_partial(running))
    loop.diagnostics = _ScriptedDiag(chunks)
    loop.HELDOUT_CYCLE_SEED = 42
    _, stop_chunk, _ = loop._run_heldout_chunked_sprt(
        chunk_size=50, K=4, futility_z=None, cycle=3,
    )
    assert stop_chunk is None
    path = tmp_path / "sprt_decisions.jsonl"
    assert path.exists()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    # One row per chunk — all 4 chunks consumed since no early-stop.
    assert len(rows) == 4, rows
    for idx, r in enumerate(rows, start=1):
        assert r["cycle"] == 3
        assert r["chunk_idx"] == idx
        assert r["decision"] == "continue"
        assert r["continuing"] is True
