import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CycleMetric:
    cycle: int
    score: int
    previous_score: int
    directive: str
    directive_source: str  # "builtin" or "generated"
    branch_count: int
    best_variant_index: int  # -1 if no improvement
    elapsed_seconds: float


@dataclass
class RunMetrics:
    prompt: str = ""
    task_type: str = ""
    dilation_factor: int = 0
    branch_factor: int = 0
    start_time: float = 0.0
    cycles: list[CycleMetric] = field(default_factory=list)

    def record_cycle(self, **kwargs) -> None:
        self.cycles.append(CycleMetric(**kwargs))

    @property
    def score_history(self) -> list[int]:
        return [c.score for c in self.cycles]

    @property
    def improvement_rate(self) -> float:
        """Fraction of cycles that improved the score."""
        if not self.cycles:
            return 0.0
        improved = sum(1 for c in self.cycles if c.score > c.previous_score)
        return improved / len(self.cycles)

    @property
    def total_improvement(self) -> int:
        if not self.cycles:
            return 0
        return self.cycles[-1].score - self.cycles[0].previous_score

    @property
    def stagnant_streak(self) -> int:
        """Current consecutive cycles without improvement."""
        streak = 0
        for c in reversed(self.cycles):
            if c.score <= c.previous_score:
                streak += 1
            else:
                break
        return streak

    @property
    def score_inflation_rate(self) -> float:
        """Detect suspiciously consistent score increases.
        Returns fraction of cycles with improvement — values >0.8 suggest inflation."""
        if len(self.cycles) < 3:
            return 0.0
        improving = sum(1 for c in self.cycles if c.score > c.previous_score)
        return improving / len(self.cycles)

    @property
    def max_score_jump(self) -> int:
        """Largest single-cycle score increase."""
        if not self.cycles:
            return 0
        return max((c.score - c.previous_score) for c in self.cycles)

    @property
    def avg_cycle_time(self) -> float:
        """Average wall-clock seconds per cycle."""
        if not self.cycles:
            return 0.0
        return sum(c.elapsed_seconds for c in self.cycles) / len(self.cycles)

    @property
    def effective_dilation(self) -> float:
        """Actual dilation achieved: total_cycles / elapsed_wall_clock.
        Higher means more cycles packed per second."""
        if not self.cycles:
            return 0.0
        total_time = sum(c.elapsed_seconds for c in self.cycles)
        if total_time <= 0:
            return 0.0
        return len(self.cycles) / total_time

    @property
    def directive_effectiveness(self) -> dict[str, float]:
        """Track which directive sources lead to improvements.
        Returns {source: improvement_rate}."""
        by_source: dict[str, list[bool]] = {}
        for c in self.cycles:
            source = c.directive_source
            improved = c.score > c.previous_score
            by_source.setdefault(source, []).append(improved)
        return {
            source: sum(results) / len(results)
            for source, results in by_source.items()
            if results
        }

    @property
    def best_directive(self) -> str | None:
        """Return the directive text that produced the largest score jump."""
        if not self.cycles:
            return None
        best = max(self.cycles, key=lambda c: c.score - c.previous_score)
        if best.score <= best.previous_score:
            return None
        return best.directive

    @property
    def points_per_cycle(self) -> list[float]:
        """Score improvement per cycle — shows diminishing returns."""
        return [max(0, c.score - c.previous_score) for c in self.cycles]

    @property
    def diminishing_returns(self) -> bool:
        """True if last 3+ cycles averaged < 1 point improvement."""
        if len(self.cycles) < 3:
            return False
        recent = self.points_per_cycle[-3:]
        return sum(recent) / len(recent) < 1.0

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "task_type": self.task_type,
            "dilation_factor": self.dilation_factor,
            "branch_factor": self.branch_factor,
            "total_cycles": len(self.cycles),
            "improvement_rate": self.improvement_rate,
            "total_improvement": self.total_improvement,
            "avg_cycle_time": self.avg_cycle_time,
            "effective_dilation": self.effective_dilation,
            "score_inflation_rate": self.score_inflation_rate,
            "max_score_jump": self.max_score_jump,
            "directive_effectiveness": self.directive_effectiveness,
            "best_directive": self.best_directive,
            "score_history": self.score_history,
            "elapsed_seconds": time.time() - self.start_time if self.start_time else 0,
        }

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
