"""Out-of-distribution curriculum proposer.

Every O cycles (default 12) the orchestrator dedicates a propose-cycle to
problem CATEGORIES the model has never trained on. This partially breaks the
"model can only imagine problems within its own ceiling" constraint: instead
of sampling from the existing skill-pair frontier (handled by
DifficultyTracker.frontier), we ask the model — via a meta-prompt — to name
entirely new domains, then seed the next cycle's proposer with N problems
per new domain.

Responsibilities
----------------
* Meta-prompt the model for novel DOMAIN names (graph theory, cryptanalysis,
  protocol verification, …).
* Filter out domains already recorded in the in-dist bank (via a caller-
  provided `known_domains` callable — typically pulls subdomain keys from
  DifficultyTracker).
* Per newly-discovered domain, prompt for N seed problems and hand them back
  to the caller (who forwards them to TaskSynthesizer or DataGenerator).
* Track domain lifecycle in outputs/ood_domains.jsonl:
    {domain, first_seen_cycle, cumulative_accepts, cumulative_proposals,
     accept_rate, mainstream}
  A domain becomes "mainstream" (no longer OOD-tagged) when
  accept_rate >= MAINSTREAM_THRESHOLD (default 0.20).

Reward coordination (see rl-transition, Task #8)
------------------------------------------------
Emitted problems carry metadata {ood: True, ood_domain, domain_maturity}
where domain_maturity = min(1.0, accept_rate / MAINSTREAM_THRESHOLD).
GRPO uses OOD_bonus = beta * (1 - domain_maturity) * accepted.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

logger = logging.getLogger(__name__)


MAINSTREAM_THRESHOLD: float = 0.20
OOD_CYCLE_PERIOD_DEFAULT: int = 12
SEEDS_PER_DOMAIN_DEFAULT: int = 8
DOMAINS_PER_OOD_CYCLE_DEFAULT: int = 3


_DOMAIN_LINE = re.compile(r"^\s*(?:[-*\d+.)\s]*)\s*([A-Za-z][A-Za-z0-9 _/&-]{2,60})\s*$")


def _normalize_domain(raw: str) -> str:
    s = raw.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


@dataclass
class DomainRecord:
    domain: str
    first_seen_cycle: int
    cumulative_proposals: int = 0
    cumulative_accepts: int = 0

    @property
    def accept_rate(self) -> float:
        return (self.cumulative_accepts / self.cumulative_proposals) if self.cumulative_proposals else 0.0

    @property
    def mainstream(self) -> bool:
        return self.accept_rate >= MAINSTREAM_THRESHOLD and self.cumulative_proposals >= 5

    @property
    def domain_maturity(self) -> float:
        return min(1.0, self.accept_rate / MAINSTREAM_THRESHOLD) if MAINSTREAM_THRESHOLD > 0 else 1.0

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "first_seen_cycle": self.first_seen_cycle,
            "cumulative_proposals": self.cumulative_proposals,
            "cumulative_accepts": self.cumulative_accepts,
            "accept_rate": round(self.accept_rate, 4),
            "mainstream": self.mainstream,
        }


class OODDomainTracker:
    """Tracks lifecycle of OOD-discovered domains.

    This is intentionally a sibling of DifficultyTracker (not a subclass):
    DifficultyTracker indexes by domain/subdomain skill-pairs from the
    diagnostic bank; OODDomainTracker indexes by free-form category names
    proposed by the model. They are consulted together: a proposed OOD
    name is only KEPT if DifficultyTracker has no subdomain key whose
    normalized form matches (see ``is_novel_domain``).
    """

    def __init__(self, state_path: Optional[Path] = None) -> None:
        self.state_path = Path(state_path) if state_path is not None else None
        self._domains: dict[str, DomainRecord] = {}

    def __contains__(self, domain: str) -> bool:
        return _normalize_domain(domain) in self._domains

    def get(self, domain: str) -> Optional[DomainRecord]:
        return self._domains.get(_normalize_domain(domain))

    def all_domains(self) -> list[DomainRecord]:
        return list(self._domains.values())

    def register(self, domain: str, cycle: int) -> DomainRecord:
        key = _normalize_domain(domain)
        if not key:
            raise ValueError(f"empty domain name: {domain!r}")
        rec = self._domains.get(key)
        if rec is None:
            rec = DomainRecord(domain=key, first_seen_cycle=int(cycle))
            self._domains[key] = rec
        return rec

    def record_outcome(self, domain: str, accepted: bool, proposals: int = 1) -> DomainRecord:
        key = _normalize_domain(domain)
        rec = self._domains.get(key)
        if rec is None:
            raise KeyError(f"unknown domain {domain!r}; call register() first")
        rec.cumulative_proposals += int(max(0, proposals))
        if accepted:
            rec.cumulative_accepts += 1
        return rec

    def is_novel_domain(self, domain: str, known_domains: Iterable[str]) -> bool:
        key = _normalize_domain(domain)
        if not key:
            return False
        if key in self._domains:
            return False
        for k in known_domains:
            if _normalize_domain(k) == key:
                return False
        return True

    def snapshot_jsonl(self, path: Optional[Path] = None) -> Path:
        p = Path(path) if path is not None else self.state_path
        if p is None:
            raise ValueError("no state_path set; pass path= to snapshot_jsonl()")
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "w") as f:
            for rec in self._domains.values():
                f.write(json.dumps(rec.to_dict()) + "\n")
        tmp.replace(p)
        return p

    @classmethod
    def load_or_new(cls, path: Optional[Path]) -> "OODDomainTracker":
        t = cls(state_path=Path(path) if path else None)
        if path is None:
            return t
        p = Path(path)
        if not p.exists():
            return t
        try:
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    rec = DomainRecord(
                        domain=str(data["domain"]),
                        first_seen_cycle=int(data.get("first_seen_cycle", 0)),
                        cumulative_proposals=int(data.get("cumulative_proposals", 0)),
                        cumulative_accepts=int(data.get("cumulative_accepts", 0)),
                    )
                    t._domains[rec.domain] = rec
        except Exception as exc:
            logger.warning("OODDomainTracker: failed to load %s (%s); starting fresh", p, exc)
        return t


_META_PROMPT = """You are helping expand a reasoning model's training curriculum.

List {n} NEW problem-solving DOMAINS that a strong reasoning model should know
but which are NOT in this list of already-covered areas:

ALREADY COVERED:
{covered}

Respond with one domain per line, 2-5 words each. No numbering, no commentary.
Prefer domains that are rigorously verifiable (formal, symbolic, or
computational) over open-ended humanities.
"""

_SEED_PROMPT = """Propose {n} self-contained problems in the domain "{domain}".

Each problem must:
  - be solvable without external resources,
  - have a unique, machine-checkable answer (integer, string, list, or
    short proof),
  - state the answer format explicitly.

Respond as one problem per line, prefixed "PROBLEM: ".
"""


def parse_domain_list(text: str, limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for line in (text or "").splitlines():
        m = _DOMAIN_LINE.match(line)
        if not m:
            continue
        cand = m.group(1).strip()
        key = _normalize_domain(cand)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(cand)
        if len(out) >= limit:
            break
    return out


def parse_seed_problems(text: str, limit: int) -> list[str]:
    out: list[str] = []
    for line in (text or "").splitlines():
        s = line.strip()
        if s.upper().startswith("PROBLEM:"):
            body = s.split(":", 1)[1].strip()
            if body:
                out.append(body)
                if len(out) >= limit:
                    break
    return out


@dataclass
class OODSeedBatch:
    cycle: int
    domain: str
    problems: list[str] = field(default_factory=list)

    def metadata_for(self, problem: str, tracker: OODDomainTracker) -> dict:
        rec = tracker.get(self.domain)
        maturity = rec.domain_maturity if rec else 0.0
        return {
            "ood": True,
            "ood_domain": _normalize_domain(self.domain),
            "domain_maturity": round(maturity, 4),
            "cycle_proposed": self.cycle,
        }


ModelCall = Callable[[str], str]


class OODProposer:
    """Drives OOD-cycle domain discovery and seed-problem proposal.

    Usage:
        proposer = OODProposer(model_call=llm_fn, tracker=tracker)
        if proposer.should_run(cycle):
            batches = proposer.propose(cycle, known_domains=diff_tracker_keys())
            for batch in batches:
                for prob in batch.problems:
                    meta = batch.metadata_for(prob, tracker)
                    ...
    """

    def __init__(
        self,
        model_call: ModelCall,
        tracker: OODDomainTracker,
        *,
        period: int = OOD_CYCLE_PERIOD_DEFAULT,
        domains_per_cycle: int = DOMAINS_PER_OOD_CYCLE_DEFAULT,
        seeds_per_domain: int = SEEDS_PER_DOMAIN_DEFAULT,
    ) -> None:
        if period <= 0:
            raise ValueError("period must be positive")
        self._model_call = model_call
        self.tracker = tracker
        self.period = int(period)
        self.domains_per_cycle = int(domains_per_cycle)
        self.seeds_per_domain = int(seeds_per_domain)

    def should_run(self, cycle: int) -> bool:
        # Cycle 0 is a normal cycle; first OOD-cycle fires at `period`.
        return cycle > 0 and cycle % self.period == 0

    def propose(self, cycle: int, known_domains: Iterable[str]) -> list[OODSeedBatch]:
        known = list(known_domains)
        covered_str = ", ".join(sorted({_normalize_domain(k) for k in known if k})) or "(none)"
        meta_text = self._model_call(
            _META_PROMPT.format(n=self.domains_per_cycle * 2, covered=covered_str)
        )
        raw_domains = parse_domain_list(meta_text, limit=self.domains_per_cycle * 3)
        novel = [d for d in raw_domains if self.tracker.is_novel_domain(d, known)][: self.domains_per_cycle]
        batches: list[OODSeedBatch] = []
        for dom in novel:
            self.tracker.register(dom, cycle)
            seed_text = self._model_call(
                _SEED_PROMPT.format(n=self.seeds_per_domain, domain=dom)
            )
            problems = parse_seed_problems(seed_text, limit=self.seeds_per_domain)
            if not problems:
                logger.info("OOD: domain %r produced 0 parseable seeds, skipping batch", dom)
                continue
            batches.append(OODSeedBatch(cycle=cycle, domain=dom, problems=problems))
        if self.tracker.state_path is not None:
            try:
                self.tracker.snapshot_jsonl()
            except Exception as exc:
                logger.warning("OOD: snapshot_jsonl failed (%s)", exc)
        return batches
