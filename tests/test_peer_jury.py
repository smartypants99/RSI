"""Tests for src/verifier/peer_jury.py (Task #1A)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.verifier import peer_jury as pj
from src.verifier.property_engine import INDEPENDENCE_CLASSES, PropertyKind


# ───────── registry additions ─────────

def test_peer_consensus_independence_class_registered():
    assert "peer.consensus" in INDEPENDENCE_CLASSES


def test_peer_review_property_kind_registered():
    assert PropertyKind.PEER_REVIEW.value == "PEER_REVIEW"
    assert PropertyKind("PEER_REVIEW") is PropertyKind.PEER_REVIEW


# ───────── parsing ─────────

@pytest.mark.parametrize("raw,vote", [
    ("VALID", "VALID"),
    ("  VALID\n", "VALID"),
    ("The answer is VALID because ...", "VALID"),
    ("INVALID: step 3 uses wrong formula.", "INVALID"),
    ("INVALID - circular reasoning in step 2", "INVALID"),
    ("", "INVALID"),
    ("idk lol", "INVALID"),
])
def test_parse_peer_response(raw, vote):
    v, _reason = pj.parse_peer_response(raw)
    assert v == vote


def test_parse_peer_response_reason_extracted():
    _v, reason = pj.parse_peer_response("INVALID: final answer is off by one")
    assert "off by one" in reason.lower()


def test_build_jury_prompt_contains_parts():
    p = pj.build_jury_prompt("What is 2+2?", "4", "step 1: add")
    assert "2+2" in p
    assert "VALID" in p and "INVALID" in p
    assert "step 1: add" in p


def test_candidate_hash_stable_and_distinguishing():
    h1 = pj.candidate_hash("prob-1", "ans A", "reasoning")
    h2 = pj.candidate_hash("prob-1", "ans A", "reasoning")
    h3 = pj.candidate_hash("prob-1", "ans B", "reasoning")
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 64


# ───────── jury verdict (mocked subprocess) ─────────

def _make_fake_invoker(responses: dict[str, tuple[bool, str]]):
    """Build a callable matching _invoke_peer's signature from a static map."""
    def _invoke(peer: str, prompt: str, timeout_s: int):
        return responses.get(peer, (False, "no response configured"))
    return _invoke


def test_jury_verdict_unanimous_valid_passes(tmp_path: Path):
    cache = tmp_path / "cache.jsonl"
    fake = _make_fake_invoker({
        "codex": (True, "VALID"),
        "gemini": (True, "VALID"),
    })
    r = pj.jury_verdict(
        problem_id="p1", problem="2+2=?", candidate_response="4",
        ours_vote="VALID", peers=("codex", "gemini"),
        cache_path=cache, _subprocess_override=fake,
    )
    assert r.passed is True
    assert r.valid_votes == 3
    assert r.total_counted == 3


def test_jury_verdict_split_vote_passes_at_two(tmp_path: Path):
    fake = _make_fake_invoker({
        "codex": (True, "VALID"),
        "gemini": (True, "INVALID: wrong"),
    })
    r = pj.jury_verdict(
        problem_id="p2", problem="x", candidate_response="y",
        ours_vote="VALID", peers=("codex", "gemini"), min_agree=2,
        cache_path=tmp_path / "c.jsonl", _subprocess_override=fake,
    )
    assert r.passed is True
    assert r.valid_votes == 2


def test_jury_verdict_ours_invalid_cannot_resurrect(tmp_path: Path):
    fake = _make_fake_invoker({
        "codex": (True, "VALID"),
        "gemini": (True, "VALID"),
    })
    r = pj.jury_verdict(
        problem_id="p3", problem="x", candidate_response="y",
        ours_vote="INVALID", peers=("codex", "gemini"), min_agree=3,
        cache_path=tmp_path / "c.jsonl", _subprocess_override=fake,
    )
    assert r.passed is False  # only 2 valid votes, need 3


def test_jury_verdict_skip_when_peer_unavailable(tmp_path: Path):
    fake = _make_fake_invoker({
        "codex": (False, "codex not on PATH"),
        "gemini": (True, "VALID"),
    })
    r = pj.jury_verdict(
        problem_id="p4", problem="x", candidate_response="y",
        ours_vote="VALID", peers=("codex", "gemini"), min_agree=2,
        cache_path=tmp_path / "c.jsonl", _subprocess_override=fake,
    )
    # ours=VALID + gemini=VALID = 2 valid; codex=SKIP not counted
    assert r.passed is True
    assert any(v.vote == "SKIP" for v in r.peer_votes)


def test_jury_verdict_all_peers_skip_min_agree_two_fails(tmp_path: Path):
    fake = _make_fake_invoker({
        "codex": (False, "down"),
        "gemini": (False, "down"),
    })
    r = pj.jury_verdict(
        problem_id="p5", problem="x", candidate_response="y",
        ours_vote="VALID", peers=("codex", "gemini"), min_agree=2,
        cache_path=tmp_path / "c.jsonl", _subprocess_override=fake,
    )
    assert r.passed is False
    assert r.valid_votes == 1


def test_jury_cache_roundtrip(tmp_path: Path):
    cache_file = tmp_path / "jury.jsonl"
    calls = {"n": 0}

    def counting(peer, prompt, timeout_s):
        calls["n"] += 1
        return True, "VALID"

    kwargs = dict(
        problem_id="p-cache", problem="x", candidate_response="y",
        ours_vote="VALID", peers=("codex",), min_agree=1,
        cache_path=cache_file, _subprocess_override=counting,
    )
    r1 = pj.jury_verdict(**kwargs)
    r2 = pj.jury_verdict(**kwargs)
    assert r1.passed and r2.passed
    assert calls["n"] == 1   # second call served from cache
    assert r2.peer_votes[0].cached is True
    # On-disk format is jsonl
    rows = [json.loads(l) for l in cache_file.read_text().splitlines() if l.strip()]
    assert rows and rows[0]["peer"] == "codex"


def test_jury_rejects_bad_ours_vote(tmp_path: Path):
    with pytest.raises(ValueError):
        pj.jury_verdict(
            problem_id="p", problem="x", candidate_response="y",
            ours_vote="MAYBE", cache_path=tmp_path / "c.jsonl",
        )


def test_peer_available_returns_bool():
    # Smoke: doesn't raise, returns bool. Environment may or may not have
    # the CLIs, so we only check the type.
    assert isinstance(pj.peer_available("codex"), bool)
    assert pj.peer_available("definitely-not-a-real-cli-xyz") is False
