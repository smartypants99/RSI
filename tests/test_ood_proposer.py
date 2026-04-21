"""Tests for src/generator/ood_proposer.py (Task #7, curriculum-ood)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.generator.ood_proposer import (
    MAINSTREAM_THRESHOLD,
    DomainRecord,
    OODDomainTracker,
    OODProposer,
    OODSeedBatch,
    _normalize_domain,
    parse_domain_list,
    parse_seed_problems,
)


def test_normalize_domain_folds_punctuation_and_case():
    assert _normalize_domain("Graph Theory") == "graph_theory"
    assert _normalize_domain("protocol-verification") == "protocol_verification"
    assert _normalize_domain("  CRYPTANALYSIS  ") == "cryptanalysis"
    assert _normalize_domain("") == ""


def test_parse_domain_list_strips_bullets_and_dedupes():
    text = """- Graph Theory
* graph theory
1. Cryptanalysis
   Protocol Verification
not a domain line: includes: colons
"""
    out = parse_domain_list(text, limit=5)
    # "graph theory" appears twice in different casings; after normalization
    # dedup kicks in so we only keep the first.
    keys = [_normalize_domain(d) for d in out]
    assert keys == ["graph_theory", "cryptanalysis", "protocol_verification"]


def test_parse_seed_problems_only_keeps_prefixed_lines():
    text = """PROBLEM: find shortest path in DAG of 7 nodes.
random commentary
PROBLEM: count primes below 100.
"""
    out = parse_seed_problems(text, limit=5)
    assert out == [
        "find shortest path in DAG of 7 nodes.",
        "count primes below 100.",
    ]


def test_tracker_registers_and_records_outcomes(tmp_path):
    t = OODDomainTracker(state_path=tmp_path / "ood.jsonl")
    rec = t.register("Graph Theory", cycle=12)
    assert rec.domain == "graph_theory"
    assert rec.first_seen_cycle == 12
    # Register again — same object, no duplicate.
    rec2 = t.register("graph_theory", cycle=99)
    assert rec2 is rec
    assert rec2.first_seen_cycle == 12  # unchanged

    # Record 10 proposals, 1 accept → 10% accept rate, below mainstream.
    for i in range(10):
        t.record_outcome("Graph Theory", accepted=(i == 0))
    assert rec.cumulative_proposals == 10
    assert rec.cumulative_accepts == 1
    assert rec.accept_rate == pytest.approx(0.1)
    assert rec.mainstream is False
    assert rec.domain_maturity == pytest.approx(0.1 / MAINSTREAM_THRESHOLD)


def test_tracker_mainstream_threshold(tmp_path):
    t = OODDomainTracker(state_path=tmp_path / "ood.jsonl")
    t.register("cryptanalysis", cycle=0)
    for i in range(10):
        t.record_outcome("cryptanalysis", accepted=(i < 3))
    rec = t.get("cryptanalysis")
    assert rec.accept_rate == pytest.approx(0.3)
    assert rec.mainstream is True
    assert rec.domain_maturity == 1.0  # clipped


def test_record_outcome_unknown_domain_raises(tmp_path):
    t = OODDomainTracker(state_path=tmp_path / "ood.jsonl")
    with pytest.raises(KeyError):
        t.record_outcome("nothing", accepted=True)


def test_is_novel_domain_rejects_known_and_registered(tmp_path):
    t = OODDomainTracker(state_path=tmp_path / "ood.jsonl")
    t.register("graph_theory", cycle=1)
    assert not t.is_novel_domain("Graph Theory", known_domains=[])
    # "cryptanalysis/rsa" normalizes to "cryptanalysis_rsa" — distinct from "cryptanalysis".
    assert t.is_novel_domain("cryptanalysis", known_domains=["cryptanalysis/rsa"])
    # Exact normalized collision — "crypt analysis" → "crypt_analysis".
    assert not t.is_novel_domain("crypt analysis", known_domains=["crypt/analysis"])
    assert t.is_novel_domain("protocol verification", known_domains=["algebra/linear"])
    assert not t.is_novel_domain("   ", known_domains=[])


def test_snapshot_and_load_roundtrip(tmp_path):
    p = tmp_path / "ood_domains.jsonl"
    t = OODDomainTracker(state_path=p)
    t.register("graph_theory", cycle=12)
    t.record_outcome("graph_theory", accepted=True)
    t.record_outcome("graph_theory", accepted=False)
    t.register("cryptanalysis", cycle=24)
    t.snapshot_jsonl()

    assert p.exists()
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    keys = {d["domain"] for d in parsed}
    assert keys == {"graph_theory", "cryptanalysis"}

    t2 = OODDomainTracker.load_or_new(p)
    assert "graph_theory" in t2
    gr = t2.get("graph_theory")
    assert gr.cumulative_proposals == 2
    assert gr.cumulative_accepts == 1


def test_load_or_new_missing_file(tmp_path):
    p = tmp_path / "does_not_exist.jsonl"
    t = OODDomainTracker.load_or_new(p)
    assert t.all_domains() == []
    assert t.state_path == p


class _StubModel:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self._responses:
            return ""
        return self._responses.pop(0)


def test_proposer_should_run_cadence():
    stub = _StubModel([])
    tracker = OODDomainTracker()
    p = OODProposer(model_call=stub, tracker=tracker, period=12)
    assert p.should_run(0) is False
    assert p.should_run(11) is False
    assert p.should_run(12) is True
    assert p.should_run(24) is True
    assert p.should_run(13) is False


def test_proposer_period_must_be_positive():
    tracker = OODDomainTracker()
    with pytest.raises(ValueError):
        OODProposer(model_call=lambda _p: "", tracker=tracker, period=0)


def test_proposer_propose_happy_path(tmp_path):
    domain_text = """- Graph Theory
- Cryptanalysis
- Protocol Verification
"""
    seeds_graph = """PROBLEM: find shortest path in K_5.
PROBLEM: count spanning trees of C_6.
"""
    seeds_crypto = "PROBLEM: break Caesar cipher with key 3.\n"
    seeds_protocol = "PROBLEM: prove mutex satisfies safety.\n"
    stub = _StubModel([domain_text, seeds_graph, seeds_crypto, seeds_protocol])
    tracker = OODDomainTracker(state_path=tmp_path / "ood.jsonl")
    proposer = OODProposer(
        model_call=stub,
        tracker=tracker,
        period=12,
        domains_per_cycle=3,
        seeds_per_domain=4,
    )
    batches = proposer.propose(cycle=12, known_domains=["algebra/linear"])
    assert len(batches) == 3
    domains = [b.domain for b in batches]
    # parse_domain_list preserves original casing; batch.domain is the raw string
    assert [_normalize_domain(d) for d in domains] == [
        "graph_theory",
        "cryptanalysis",
        "protocol_verification",
    ]
    assert batches[0].problems == [
        "find shortest path in K_5.",
        "count spanning trees of C_6.",
    ]
    # All three domains registered with first_seen_cycle=12.
    for d in ["graph_theory", "cryptanalysis", "protocol_verification"]:
        rec = tracker.get(d)
        assert rec is not None
        assert rec.first_seen_cycle == 12
    # Snapshot written.
    assert (tmp_path / "ood.jsonl").exists()


def test_proposer_filters_known_domains():
    domain_text = "- graph theory\n- cryptanalysis\n"
    stub = _StubModel([domain_text, "PROBLEM: do X.\n"])
    tracker = OODDomainTracker()
    proposer = OODProposer(
        model_call=stub,
        tracker=tracker,
        period=12,
        domains_per_cycle=5,
        seeds_per_domain=2,
    )
    batches = proposer.propose(cycle=12, known_domains=["graph/theory", "other/skill"])
    # "graph theory" normalizes to "graph_theory", collides with "graph/theory"
    # which also normalizes to "graph_theory" → filtered out. Only cryptanalysis remains.
    assert len(batches) == 1
    assert _normalize_domain(batches[0].domain) == "cryptanalysis"


def test_proposer_skips_domain_with_zero_parseable_seeds():
    stub = _StubModel(["- Graph Theory\n- Cryptanalysis\n", "no problems here", "PROBLEM: ok.\n"])
    tracker = OODDomainTracker()
    proposer = OODProposer(
        model_call=stub,
        tracker=tracker,
        period=12,
        domains_per_cycle=2,
        seeds_per_domain=2,
    )
    batches = proposer.propose(cycle=12, known_domains=[])
    # Graph Theory returned no parseable seeds → skipped; cryptanalysis returned one.
    assert len(batches) == 1
    assert _normalize_domain(batches[0].domain) == "cryptanalysis"
    # But BOTH domains got registered (register happens before seed parse).
    assert tracker.get("graph_theory") is not None
    assert tracker.get("cryptanalysis") is not None


def test_seed_batch_metadata_reflects_maturity():
    tracker = OODDomainTracker()
    tracker.register("graph_theory", cycle=12)
    # 2 accepts out of 10 proposals → accept_rate = 0.2 = MAINSTREAM_THRESHOLD → maturity 1.0
    for i in range(10):
        tracker.record_outcome("graph_theory", accepted=(i < 2))
    batch = OODSeedBatch(cycle=12, domain="Graph Theory", problems=["x"])
    meta = batch.metadata_for("x", tracker)
    assert meta["ood"] is True
    assert meta["ood_domain"] == "graph_theory"
    assert meta["domain_maturity"] == pytest.approx(1.0)
    assert meta["cycle_proposed"] == 12
