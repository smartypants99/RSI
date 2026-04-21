"""Self-compute-allocation: learned bandit over allocation strategies.

The orchestrator has knobs that trade compute for quality:

  * `k_candidates`:   candidates per problem
  * `token_budget`:   max tokens per candidate
  * `branch_vs_confirm`: 'branch' (explore wider) vs 'confirm' (re-sample a
                     promising candidate)
  * `fast_student`:   whether to activate the fast-student path this cycle
  * `train_mode`:     'grpo' vs 'sft' for the training step

An `AllocationStrategy` is one concrete setting of those knobs. The
`ComputeAllocator` is a contextual-free UCB1 bandit over a caller-supplied
set of strategies. Reward = improvement_per_compute — i.e.
(held_out_delta) / (wall_time_seconds or tokens_spent), clipped to a finite
range so a single outlier cannot dominate the posterior.

Budget enforcement is a hard cap, not a soft signal: a strategy whose
expected cost exceeds the remaining cycle budget is filtered out of the
selection pool before UCB1 runs. If every strategy exceeds budget, we
fall back to the cheapest one so the cycle still makes progress.

Pure / deterministic given its history JSONL. No imports from torch / CUDA.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence


ALLOWED_TRAIN_MODES: frozenset[str] = frozenset({"grpo", "sft"})
ALLOWED_BRANCH_MODES: frozenset[str] = frozenset({"branch", "confirm"})


# Clip reward per cycle so one crazy outlier (e.g. near-zero compute used)
# cannot blow up the bandit's mean.
REWARD_CLIP = (-1.0, 1.0)
# Prior pseudo-count: without this a strategy with one great cycle looks
# infallible. One pseudo-observation at reward=0 shrinks means toward 0.
PRIOR_PSEUDO_COUNT = 1.0
PRIOR_PSEUDO_REWARD = 0.0
# UCB1 exploration constant.
UCB_C = 1.4


@dataclass(frozen=True)
class AllocationStrategy:
    name: str
    k_candidates: int
    token_budget: int
    branch_vs_confirm: str  # 'branch' | 'confirm'
    fast_student: bool
    train_mode: str  # 'grpo' | 'sft'

    def __post_init__(self) -> None:
        if self.k_candidates <= 0:
            raise ValueError("k_candidates must be positive")
        if self.token_budget <= 0:
            raise ValueError("token_budget must be positive")
        if self.branch_vs_confirm not in ALLOWED_BRANCH_MODES:
            raise ValueError(f"branch_vs_confirm: {self.branch_vs_confirm}")
        if self.train_mode not in ALLOWED_TRAIN_MODES:
            raise ValueError(f"train_mode: {self.train_mode}")

    def expected_cost(self) -> float:
        """Rough cost proxy in tokens. Cheap, monotonic in the knobs we care
        about. Callers may pass a real cost via AllocationOutcome."""
        base = float(self.k_candidates) * float(self.token_budget)
        if self.fast_student:
            base *= 0.5  # fast path halves cost (rough)
        if self.train_mode == "grpo":
            base *= 1.2  # GRPO is heavier than SFT at equal samples
        return base

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AllocationOutcome:
    cycle_id: int
    strategy_name: str
    held_out_delta: float
    compute_used: float  # tokens or seconds — must be consistent per run
    wall_time_s: Optional[float] = None

    def reward(self) -> float:
        if self.compute_used <= 0:
            return 0.0
        r = self.held_out_delta / self.compute_used * 1e6  # scale for readability
        lo, hi = REWARD_CLIP
        return max(lo, min(hi, r))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AllocationOutcome":
        return cls(
            cycle_id=int(d["cycle_id"]),
            strategy_name=str(d["strategy_name"]),
            held_out_delta=float(d["held_out_delta"]),
            compute_used=float(d["compute_used"]),
            wall_time_s=(float(d["wall_time_s"]) if d.get("wall_time_s") is not None else None),
        )


@dataclass
class StrategyStats:
    n: int = 0
    reward_sum: float = 0.0

    @property
    def mean(self) -> float:
        eff_n = self.n + PRIOR_PSEUDO_COUNT
        return (self.reward_sum + PRIOR_PSEUDO_REWARD) / eff_n


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def append_outcome(path: Path, outcome: AllocationOutcome) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(outcome.to_dict()) + "\n")


def load_outcomes(path: Path) -> list[AllocationOutcome]:
    path = Path(path)
    if not path.exists():
        return []
    out: list[AllocationOutcome] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(AllocationOutcome.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return out


# --------------------------------------------------------------------------
# Bandit
# --------------------------------------------------------------------------


@dataclass
class ComputeAllocator:
    strategies: tuple[AllocationStrategy, ...]
    history: list[AllocationOutcome] = field(default_factory=list)

    def __post_init__(self) -> None:
        names = [s.name for s in self.strategies]
        if len(set(names)) != len(names):
            raise ValueError("strategy names must be unique")
        if not self.strategies:
            raise ValueError("at least one strategy required")

    def record(self, outcome: AllocationOutcome) -> None:
        self.history.append(outcome)

    def stats(self) -> dict[str, StrategyStats]:
        out: dict[str, StrategyStats] = {s.name: StrategyStats() for s in self.strategies}
        for o in self.history:
            if o.strategy_name not in out:
                continue
            s = out[o.strategy_name]
            s.n += 1
            s.reward_sum += o.reward()
        return out

    def _affordable(self, remaining_budget: float) -> list[AllocationStrategy]:
        return [s for s in self.strategies if s.expected_cost() <= remaining_budget]

    def select(self, remaining_budget: float) -> AllocationStrategy:
        """UCB1 over strategies that fit the remaining budget.

        If none fit, return the single cheapest so the cycle still runs — we
        prefer SOME progress over a stalled cycle, and the cost cap is a
        ranking tool, not a hard abort.
        """
        if remaining_budget <= 0:
            return min(self.strategies, key=lambda s: s.expected_cost())
        pool = self._affordable(remaining_budget) or [
            min(self.strategies, key=lambda s: s.expected_cost())
        ]
        stats = self.stats()
        total_n = sum(stats[s.name].n for s in pool)
        # If any strategy in the pool is untried, try it first (standard UCB1).
        untried = [s for s in pool if stats[s.name].n == 0]
        if untried:
            return untried[0]
        log_total = math.log(max(total_n, 1))
        def ucb(s: AllocationStrategy) -> float:
            st = stats[s.name]
            exploration = UCB_C * math.sqrt(log_total / max(st.n, 1))
            return st.mean + exploration
        return max(pool, key=ucb)

    def ranking(self) -> list[tuple[str, float, int]]:
        stats = self.stats()
        rows = [(s.name, stats[s.name].mean, stats[s.name].n) for s in self.strategies]
        rows.sort(key=lambda r: r[1], reverse=True)
        return rows


def default_strategies() -> tuple[AllocationStrategy, ...]:
    """A small, hand-tuned starter menu covering the main axes. Callers can
    replace or extend; this just ensures the allocator has SOMETHING to pick
    between on cycle 1."""
    return (
        AllocationStrategy("baseline_sft", k_candidates=4, token_budget=1024,
                           branch_vs_confirm="branch", fast_student=False, train_mode="sft"),
        AllocationStrategy("wide_explore_grpo", k_candidates=8, token_budget=1024,
                           branch_vs_confirm="branch", fast_student=False, train_mode="grpo"),
        AllocationStrategy("deep_confirm_sft", k_candidates=2, token_budget=2048,
                           branch_vs_confirm="confirm", fast_student=False, train_mode="sft"),
        AllocationStrategy("fast_student_sft", k_candidates=4, token_budget=1024,
                           branch_vs_confirm="branch", fast_student=True, train_mode="sft"),
        AllocationStrategy("cheap_confirm_sft", k_candidates=2, token_budget=512,
                           branch_vs_confirm="confirm", fast_student=True, train_mode="sft"),
    )


def allocator_from_history(
    history_path: Path,
    strategies: Optional[Sequence[AllocationStrategy]] = None,
) -> ComputeAllocator:
    strategies = tuple(strategies or default_strategies())
    alloc = ComputeAllocator(strategies=strategies)
    for o in load_outcomes(history_path):
        if o.strategy_name in {s.name for s in strategies}:
            alloc.record(o)
    return alloc


__all__ = [
    "AllocationStrategy",
    "AllocationOutcome",
    "ComputeAllocator",
    "StrategyStats",
    "default_strategies",
    "allocator_from_history",
    "append_outcome",
    "load_outcomes",
    "REWARD_CLIP",
]
