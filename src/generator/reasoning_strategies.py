"""Reasoning-strategy library (Task #1B).

The model explicitly generates reasoning *templates* — system-prompt prefixes
like "decompose→subgoals→solve", "contradiction-seek then synthesize",
"answer first, justify backwards". These are treated as a first-class
object that can be (a) A/B-tested on a held-out problem slice,
(b) persisted across cycles, and (c) injected as few-shot prefixes for
future propose/solve calls.

Core ideas
----------
* A ``ReasoningStrategy`` is (name, template, provenance, stats).
* ``StrategyLibrary`` persists strategies as JSONL. It supports ``add``,
  ``record_result``, ``top_k``, and ``ab_holdout``.
* A strategy wins a slot when its Wilson-lower-bound acceptance rate on
  the held-out slice beats the library's current best by a margin. The
  Wilson bound avoids the small-n problem where one lucky win would lock
  a bad strategy into the library forever.
* The model can propose *genuinely new* meta-reasoning patterns: the
  library doesn't enumerate a fixed palette — anything the proposer
  emits in the expected template shape gets added, subject to the A/B
  gate before it's re-used.

This module is deliberately free of any RL / GRPO wiring — it exposes a
clean selection + update API that the synthesizer / trainer can plug
into without taking a dependency on the library's persistence format.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────── canonical seed strategies ───────────────────────────
#
# These are deliberately tiny so they work as few-shot *prefixes* without
# eating the context window. Each is domain-agnostic; domain-specific
# strategies can be authored by the model and added dynamically.

SEED_STRATEGIES: tuple[dict[str, str], ...] = (
    {
        "name": "decompose-subgoals",
        "template": (
            "Decompose the problem into 2-4 subgoals. For each subgoal, state "
            "what would have to be true for it to be solved. Solve each in order; "
            "if a subgoal is stuck, reformulate it before retrying."
        ),
        "origin": "seed",
    },
    {
        "name": "contradiction-seek",
        "template": (
            "Try to prove the negation of the target claim. Follow the "
            "derivation until it hits a contradiction or succeeds. If it "
            "contradicts, the original claim holds; if it succeeds, the "
            "original is wrong and you must revise it."
        ),
        "origin": "seed",
    },
    {
        "name": "worked-example-transfer",
        "template": (
            "Find an analogous, simpler instance of the problem whose solution "
            "you are confident in. State the structural map from that instance "
            "to this one, then transfer the steps, patching mismatches as they arise."
        ),
        "origin": "seed",
    },
    {
        "name": "invariant-first",
        "template": (
            "Identify a quantity or structure that must be preserved by any "
            "valid derivation (conservation, type, dimensional, parity). "
            "Solve by insisting every step preserves that invariant; reject any "
            "step that violates it."
        ),
        "origin": "seed",
    },
)


# ─────────────────────────── dataclass ───────────────────────────

_NAME_RE = re.compile(r"^[a-z][a-z0-9\-]{1,63}$")
_MAX_TEMPLATE_BYTES = 4096


def _now() -> float:
    return time.time()


def _hash_template(template: str) -> str:
    return hashlib.sha256(template.encode("utf-8")).hexdigest()[:16]


def wilson_lower_bound(successes: int, trials: int, z: float = 1.96) -> float:
    """Wilson lower bound of the success rate at a ~95% CI (z=1.96).

    Used to rank strategies so a small number of lucky wins doesn't lock
    a bad strategy in. ``trials == 0`` → 0.0 (no evidence).
    """
    if trials <= 0:
        return 0.0
    p = successes / trials
    denom = 1.0 + (z * z) / trials
    centre = p + (z * z) / (2 * trials)
    margin = z * math.sqrt((p * (1 - p) + (z * z) / (4 * trials)) / trials)
    return max(0.0, (centre - margin) / denom)


@dataclass
class ReasoningStrategy:
    name: str
    template: str
    origin: str = "model"            # "seed" | "model" | "hybrid"
    created_at: float = field(default_factory=_now)
    holdout_trials: int = 0
    holdout_successes: int = 0
    prod_trials: int = 0             # usage count when NOT on the A/B slice
    prod_successes: int = 0
    parent_strategy: Optional[str] = None
    template_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _NAME_RE.match(self.name):
            raise ValueError(
                f"ReasoningStrategy.name must match {_NAME_RE.pattern!r}, got {self.name!r}"
            )
        if not isinstance(self.template, str) or not self.template.strip():
            raise ValueError("ReasoningStrategy.template must be a non-empty string")
        if len(self.template.encode("utf-8")) > _MAX_TEMPLATE_BYTES:
            raise ValueError(
                f"ReasoningStrategy.template must be ≤ {_MAX_TEMPLATE_BYTES} bytes"
            )
        if self.origin not in ("seed", "model", "hybrid"):
            raise ValueError(f"origin must be seed|model|hybrid, got {self.origin!r}")
        for f in ("holdout_trials", "holdout_successes", "prod_trials", "prod_successes"):
            v = getattr(self, f)
            if not isinstance(v, int) or v < 0:
                raise ValueError(f"{f} must be non-negative int, got {v!r}")
        if self.holdout_successes > self.holdout_trials:
            raise ValueError("holdout_successes > holdout_trials")
        if self.prod_successes > self.prod_trials:
            raise ValueError("prod_successes > prod_trials")
        if not self.template_hash:
            # mutate via __dict__ to bypass frozen if we ever freeze later
            object.__setattr__(self, "template_hash", _hash_template(self.template))

    @property
    def holdout_score(self) -> float:
        return wilson_lower_bound(self.holdout_successes, self.holdout_trials)

    @property
    def prod_score(self) -> float:
        return wilson_lower_bound(self.prod_successes, self.prod_trials)

    @property
    def blended_score(self) -> float:
        """Weighted blend: holdout carries more signal per-trial than prod."""
        h = self.holdout_score * min(1.0, self.holdout_trials / 4.0)
        p = self.prod_score * min(1.0, self.prod_trials / 16.0)
        return max(h, p * 0.8)


# ─────────────────────────── library ───────────────────────────

class StrategyLibrary:
    """Persistent reasoning-strategy store with A/B admission.

    File format: JSONL with one strategy record per line. On ``save()`` we
    rewrite the entire file (simpler than append-with-tombstones; the
    library never grows large enough for that to matter).
    """

    def __init__(self, path: str | os.PathLike,
                 ab_holdout_size: int = 4,
                 k_few_shot: int = 2,
                 seed_if_empty: bool = True) -> None:
        self.path = Path(path)
        self.ab_holdout_size = ab_holdout_size
        self.k_few_shot = k_few_shot
        self._by_name: dict[str, ReasoningStrategy] = {}
        self._loaded = False
        self._seed_if_empty = seed_if_empty

    # ── lifecycle ──

    def load(self) -> "StrategyLibrary":
        self._loaded = True
        self._by_name.clear()
        if self.path.exists():
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
                        try:
                            s = ReasoningStrategy(**row)
                        except (TypeError, ValueError) as e:
                            logger.warning("skip malformed strategy row: %s", e)
                            continue
                        self._by_name[s.name] = s
            except OSError as e:  # pragma: no cover
                logger.warning("StrategyLibrary load failed: %s", e)
        if not self._by_name and self._seed_if_empty:
            for s in SEED_STRATEGIES:
                self._by_name[s["name"]] = ReasoningStrategy(**s)
        return self

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def save(self) -> None:
        self._ensure_loaded()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for s in self._by_name.values():
                    f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
            tmp.replace(self.path)
        except OSError as e:  # pragma: no cover
            logger.warning("StrategyLibrary save failed: %s", e)

    # ── read ──

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._by_name)

    def __iter__(self) -> Iterator[ReasoningStrategy]:
        self._ensure_loaded()
        return iter(self._by_name.values())

    def get(self, name: str) -> Optional[ReasoningStrategy]:
        self._ensure_loaded()
        return self._by_name.get(name)

    def all(self) -> list[ReasoningStrategy]:
        self._ensure_loaded()
        return list(self._by_name.values())

    def top_k(self, k: Optional[int] = None) -> list[ReasoningStrategy]:
        """Rank by blended_score desc; return top k (defaults to k_few_shot)."""
        self._ensure_loaded()
        k = self.k_few_shot if k is None else k
        ranked = sorted(
            self._by_name.values(),
            key=lambda s: (s.blended_score, s.holdout_trials + s.prod_trials),
            reverse=True,
        )
        return ranked[:max(0, k)]

    def few_shot_prefix(self, k: Optional[int] = None) -> str:
        """Format top-k strategies as a system-prompt prefix block."""
        chosen = self.top_k(k)
        if not chosen:
            return ""
        lines = ["# PROVEN REASONING STRATEGIES (use when applicable):"]
        for s in chosen:
            lines.append(f"## {s.name}")
            lines.append(s.template.strip())
        return "\n".join(lines) + "\n"

    # ── write ──

    def add(self, strategy: ReasoningStrategy, allow_overwrite: bool = False) -> bool:
        """Add a strategy. Returns True if added/updated, False on duplicate template."""
        self._ensure_loaded()
        # De-dupe on template hash
        for existing in self._by_name.values():
            if existing.template_hash == strategy.template_hash and existing.name != strategy.name:
                return False
        if strategy.name in self._by_name and not allow_overwrite:
            return False
        self._by_name[strategy.name] = strategy
        return True

    def propose_from_model(self, name: str, template: str,
                           parent: Optional[str] = None) -> Optional[ReasoningStrategy]:
        """Register a model-authored strategy with 0 trials. Returns the strategy
        or None if rejected (malformed / duplicate)."""
        self._ensure_loaded()
        try:
            s = ReasoningStrategy(
                name=name,
                template=template,
                origin="model",
                parent_strategy=parent,
            )
        except ValueError as e:
            logger.info("reject proposed strategy %r: %s", name, e)
            return None
        if not self.add(s):
            return None
        return s

    def record_result(self, name: str, success: bool, holdout: bool) -> None:
        """Record a single A/B or production trial for ``name``."""
        self._ensure_loaded()
        s = self._by_name.get(name)
        if s is None:
            return
        if holdout:
            s.holdout_trials += 1
            if success:
                s.holdout_successes += 1
        else:
            s.prod_trials += 1
            if success:
                s.prod_successes += 1

    # ── A/B gate ──

    def ab_admit(self, name: str, min_trials: Optional[int] = None,
                 margin: float = 0.05) -> bool:
        """A/B admission gate for a candidate strategy.

        A strategy is admitted to production (i.e. eligible for `top_k`
        injection) when:
            holdout_trials >= min_trials
            AND holdout wilson-lb score >= best_seed_prod_wilson_lb + margin
               OR library is still only seeds (no prior model-authored admit).

        Returns True iff admitted (and flips ``origin`` to "hybrid"). The
        function never demotes; losing A/B trials just keeps origin=model.
        """
        self._ensure_loaded()
        s = self._by_name.get(name)
        if s is None:
            return False
        n_min = self.ab_holdout_size if min_trials is None else min_trials
        if s.holdout_trials < n_min:
            return False
        seed_best = 0.0
        for other in self._by_name.values():
            if other.name == s.name:
                continue
            if other.origin in ("seed", "hybrid"):
                seed_best = max(seed_best, other.blended_score)
        if s.holdout_score < seed_best + margin and seed_best > 0.0:
            return False
        s.origin = "hybrid"
        return True

    def holdout_slice(self, problems: Iterable, rng=None) -> list:
        """Pick ab_holdout_size problems from ``problems`` for A/B testing.

        Deterministic subset when rng is None (first k items); randomized
        when rng is supplied (e.g. a seeded random.Random).
        """
        items = list(problems)
        k = min(self.ab_holdout_size, len(items))
        if k <= 0:
            return []
        if rng is None:
            return items[:k]
        return rng.sample(items, k)


# ─────────────────────────── parsing helpers ───────────────────────────

_STRATEGY_BLOCK = re.compile(
    r"STRATEGY\s*:\s*(?P<name>[a-z][a-z0-9\-]{1,63})\s*\n"
    r"TEMPLATE\s*:\s*(?P<tmpl>.+?)(?:\n\s*\n|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def parse_strategies_from_model_output(raw: str) -> list[tuple[str, str]]:
    """Extract (name, template) pairs from a free-form model emission.

    Accepts blocks of the form:
        STRATEGY: my-approach
        TEMPLATE: do X then Y.
    (Blank line separated.) Unparseable sections are skipped, not raised.
    """
    out: list[tuple[str, str]] = []
    for m in _STRATEGY_BLOCK.finditer(raw or ""):
        name = m.group("name").strip().lower()
        tmpl = m.group("tmpl").strip()
        if name and tmpl:
            out.append((name, tmpl))
    return out


__all__ = [
    "ReasoningStrategy",
    "StrategyLibrary",
    "SEED_STRATEGIES",
    "parse_strategies_from_model_output",
    "wilson_lower_bound",
]
