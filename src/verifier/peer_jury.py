"""Peer-LLM jury verification (Task #1A).

For open-ended or cross-domain candidates where formal verification cannot
fully reach, route the candidate through multiple independent peer LLMs via
CLI and require ≥2/3 consensus (ours + ≥1 external).

Independence class: ``peer.consensus`` (added to INDEPENDENCE_CLASSES).
PropertyKind: ``PEER_REVIEW`` (added to PropertyKind enum).

External peers are invoked via subprocess:
    - codex  :  `codex "<prompt>"`
    - gemini :  `gemini -p "<prompt>"`

Responses are cached at ``outputs/peer_jury_cache.jsonl`` keyed by
(problem_id, candidate_hash, peer) so repeat runs never re-charge. A peer
CLI that is not on PATH is treated as a graceful skip (its vote does not
count, and the remaining votes decide).

Design notes
------------
* This module is intentionally independent of the 18-field `Property`
  registry: it's called at the verifier boundary, not through admit().
  That keeps peer votes from entering quorum math for properties that
  already have formal backends.
* ``jury_verdict()`` returns a `JuryResult` dataclass — a PropertyVerdict-
  shaped object that can also be folded into a VerificationRecord if a
  caller wants jury consensus as a row.
* Parsing is strict-tolerant: the peer is asked to emit ``VALID`` or
  ``INVALID: <reason>`` (same protocol the internal model verifier uses,
  so responses don't drift between backends).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Peer CLI registry. Value is a callable that builds the argv list.
_PEER_COMMANDS: dict[str, callable] = {
    "codex": lambda prompt: ["codex", prompt],
    "gemini": lambda prompt: ["gemini", "-p", prompt],
}

PEER_VOTE_VALID = "VALID"
PEER_VOTE_INVALID = "INVALID"
PEER_VOTE_SKIP = "SKIP"   # CLI unavailable or errored; vote doesn't count


@dataclass
class PeerVote:
    peer: str
    vote: str                 # VALID | INVALID | SKIP
    reason: str = ""
    cached: bool = False
    raw: str = ""

    @property
    def counts(self) -> bool:
        return self.vote in (PEER_VOTE_VALID, PEER_VOTE_INVALID)


@dataclass
class JuryResult:
    """Outcome of a peer-LLM jury vote.

    ``passed`` is True iff ``valid_votes`` >= ``min_agree`` AND the ours
    vote is also VALID (ours is counted separately as the 1/3 vote).
    """
    problem_id: str
    candidate_hash: str
    ours_vote: str            # VALID | INVALID
    peer_votes: list[PeerVote] = field(default_factory=list)
    min_agree: int = 2
    passed: bool = False
    rationale: str = ""

    @property
    def valid_votes(self) -> int:
        n = 1 if self.ours_vote == PEER_VOTE_VALID else 0
        n += sum(1 for v in self.peer_votes if v.vote == PEER_VOTE_VALID)
        return n

    @property
    def total_counted(self) -> int:
        n = 1  # ours always counted
        n += sum(1 for v in self.peer_votes if v.counts)
        return n

    def summary(self) -> str:
        detail = ",".join(f"{v.peer}={v.vote}" for v in self.peer_votes)
        return (f"ours={self.ours_vote} {detail} "
                f"(valid={self.valid_votes}/{self.total_counted} need≥{self.min_agree})")


# ─────────────────────────── cache ───────────────────────────

class _JuryCache:
    """Append-only jsonl cache keyed by (problem_id, candidate_hash, peer)."""

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path)
        self._mem: dict[tuple[str, str, str], dict] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = (row.get("problem_id", ""), row.get("candidate_hash", ""), row.get("peer", ""))
                    if all(key):
                        self._mem[key] = row
        except OSError as e:  # pragma: no cover
            logger.warning("peer_jury cache load failed: %s", e)

    def get(self, problem_id: str, candidate_hash: str, peer: str) -> Optional[dict]:
        self._load()
        return self._mem.get((problem_id, candidate_hash, peer))

    def put(self, row: dict) -> None:
        self._load()
        key = (row.get("problem_id", ""), row.get("candidate_hash", ""), row.get("peer", ""))
        if not all(key):
            return
        self._mem[key] = row
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as e:  # pragma: no cover
            logger.warning("peer_jury cache write failed: %s", e)


# ─────────────────────────── parsing ───────────────────────────

_VERDICT_LINE = re.compile(r"\b(VALID|INVALID)\b", re.IGNORECASE)


def parse_peer_response(raw: str) -> tuple[str, str]:
    """Parse a peer-LLM response into (vote, reason).

    Accepts anywhere in the text: VALID / INVALID or INVALID: <reason>.
    Unrecognized responses fall through to INVALID with the first 200 chars
    as reason — a peer that cannot commit is treated as a negative vote.
    """
    txt = (raw or "").strip()
    if not txt:
        return PEER_VOTE_INVALID, "empty response"
    m = re.search(r"INVALID\s*[:\-–—]\s*(.{1,240})", txt, re.IGNORECASE | re.DOTALL)
    if m:
        return PEER_VOTE_INVALID, m.group(1).strip()[:200]
    m2 = _VERDICT_LINE.search(txt)
    if m2:
        v = m2.group(1).upper()
        return (PEER_VOTE_VALID, "") if v == "VALID" else (PEER_VOTE_INVALID, txt[:200])
    return PEER_VOTE_INVALID, txt[:200]


def build_jury_prompt(problem: str, candidate_response: str,
                      reasoning_chain: str = "") -> str:
    parts = [
        "You are an impartial reasoning-chain verifier on a peer jury. "
        "Another solver produced the CANDIDATE below for the PROBLEM. "
        "Decide whether the candidate's final answer AND reasoning are correct.",
        "",
        "PROBLEM:",
        problem.strip(),
        "",
    ]
    if reasoning_chain.strip():
        parts += ["REASONING CHAIN:", reasoning_chain.strip(), ""]
    parts += [
        "CANDIDATE FINAL ANSWER:",
        candidate_response.strip(),
        "",
        "Respond with EXACTLY one of:",
        "  VALID",
        "  INVALID: <one-sentence reason citing what's wrong>",
        "Do NOT add anything else. Output only VALID or INVALID: <reason>.",
    ]
    return "\n".join(parts)


def candidate_hash(problem_id: str, candidate_response: str,
                   reasoning_chain: str = "") -> str:
    h = hashlib.sha256()
    h.update(problem_id.encode("utf-8"))
    h.update(b"\x00")
    h.update(candidate_response.encode("utf-8"))
    h.update(b"\x00")
    h.update(reasoning_chain.encode("utf-8"))
    return h.hexdigest()


# ─────────────────────────── peer invocation ───────────────────────────

def peer_available(peer: str) -> bool:
    """True iff the peer CLI is on PATH."""
    return shutil.which(peer) is not None


def _invoke_peer(peer: str, prompt: str, timeout_s: int) -> tuple[bool, str]:
    """Run a peer CLI; return (ok, raw_output). Non-zero exit / timeout → (False, err)."""
    builder = _PEER_COMMANDS.get(peer)
    if builder is None:
        return False, f"unknown peer {peer}"
    if not peer_available(peer):
        return False, f"{peer} not on PATH"
    argv = builder(prompt)
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True,
            timeout=timeout_s, check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"{peer} timed out after {timeout_s}s"
    except (OSError, ValueError) as e:
        return False, f"{peer} failed: {type(e).__name__}: {e}"
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout or "")[:400]
    return True, proc.stdout or ""


# ─────────────────────────── public API ───────────────────────────

def jury_verdict(
    *,
    problem_id: str,
    problem: str,
    candidate_response: str,
    reasoning_chain: str = "",
    ours_vote: str = PEER_VOTE_VALID,
    peers: tuple[str, ...] = ("codex", "gemini"),
    min_agree: int = 2,
    cache_path: str | os.PathLike = "outputs/peer_jury_cache.jsonl",
    timeout_s: int = 30,
    _subprocess_override: Optional[callable] = None,  # test hook
) -> JuryResult:
    """Run a peer-LLM jury vote on one candidate. Returns a JuryResult.

    ``ours_vote`` is the decision the *in-process* verifier already reached
    (VALID or INVALID) — the jury augments this with independent peer
    opinions, not replaces it. If ours_vote is INVALID, passing still
    requires min_agree valid votes including ours (i.e. the jury cannot
    resurrect a rejection; it can only confirm or deny).

    Graceful degradation: peers that are unavailable / timeout / error
    return SKIP votes that don't count toward min_agree. If all peers skip
    and ours says VALID, the jury returns passed=True only when
    min_agree <= 1; otherwise passed=False with a clear rationale.
    """
    if ours_vote not in (PEER_VOTE_VALID, PEER_VOTE_INVALID):
        raise ValueError(f"ours_vote must be VALID or INVALID, got {ours_vote!r}")
    if not (1 <= min_agree <= 3):
        raise ValueError(f"min_agree must be in [1,3], got {min_agree}")

    cand_hash = candidate_hash(problem_id, candidate_response, reasoning_chain)
    cache = _JuryCache(cache_path)
    prompt = build_jury_prompt(problem, candidate_response, reasoning_chain)

    invoke = _subprocess_override or _invoke_peer

    peer_votes: list[PeerVote] = []
    for peer in peers:
        cached = cache.get(problem_id, cand_hash, peer)
        if cached is not None:
            peer_votes.append(PeerVote(
                peer=peer,
                vote=cached.get("vote", PEER_VOTE_SKIP),
                reason=cached.get("reason", ""),
                cached=True,
                raw=cached.get("raw", ""),
            ))
            continue
        ok, raw = invoke(peer, prompt, timeout_s)
        if not ok:
            vote_obj = PeerVote(peer=peer, vote=PEER_VOTE_SKIP,
                                reason=raw[:200], cached=False, raw=raw)
        else:
            v, reason = parse_peer_response(raw)
            vote_obj = PeerVote(peer=peer, vote=v, reason=reason,
                                cached=False, raw=raw[:2000])
        peer_votes.append(vote_obj)
        cache.put({
            "problem_id": problem_id,
            "candidate_hash": cand_hash,
            "peer": peer,
            "vote": vote_obj.vote,
            "reason": vote_obj.reason,
            "raw": vote_obj.raw,
        })

    result = JuryResult(
        problem_id=problem_id,
        candidate_hash=cand_hash,
        ours_vote=ours_vote,
        peer_votes=peer_votes,
        min_agree=min_agree,
    )
    result.passed = result.valid_votes >= min_agree
    result.rationale = result.summary()
    return result


__all__ = [
    "PeerVote",
    "JuryResult",
    "jury_verdict",
    "parse_peer_response",
    "build_jury_prompt",
    "candidate_hash",
    "peer_available",
    "PEER_VOTE_VALID",
    "PEER_VOTE_INVALID",
    "PEER_VOTE_SKIP",
]
