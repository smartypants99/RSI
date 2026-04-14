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
    best_variant_index: int  # -1 if no improvement, -2 if crossover
    elapsed_seconds: float
    output_delta: float = 0.0  # fraction of output that changed (0.0-1.0)
    output_length: int = 0
    comparative_overruled: bool = False  # True if comparative check rejected winner
    crossover_used: bool = False  # True if crossover produced the winning variant
    inference_calls: int = 0  # Number of LLM calls in this cycle


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
    def peak_score(self) -> int:
        """Highest score achieved at any point during the run."""
        if not self.cycles:
            return 0
        return max(c.score for c in self.cycles)

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
    def directive_effectiveness_by_score_range(self) -> dict[str, dict[str, float]]:
        """Track which directive sources work at different score levels.
        Returns {score_range: {source: improvement_rate}}."""
        ranges: dict[str, dict[str, list[bool]]] = {
            "low": {},    # 0-49
            "mid": {},    # 50-74
            "high": {},   # 75+
        }
        for c in self.cycles:
            if c.previous_score < 50:
                bucket = "low"
            elif c.previous_score < 75:
                bucket = "mid"
            else:
                bucket = "high"
            improved = c.score > c.previous_score
            ranges[bucket].setdefault(c.directive_source, []).append(improved)
        return {
            bucket: {
                source: sum(results) / len(results)
                for source, results in sources.items()
                if results
            }
            for bucket, sources in ranges.items()
            if sources
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
    def projected_final_score(self) -> int | None:
        """Simple linear projection of final score based on recent trend.
        Returns None if insufficient data."""
        if len(self.cycles) < 3:
            return None
        recent = self.cycles[-3:]
        gains = [c.score - c.previous_score for c in recent]
        avg_gain = sum(gains) / len(gains)
        remaining = self.dilation_factor - 1 - len(self.cycles)
        if remaining <= 0:
            return self.cycles[-1].score
        projected = self.cycles[-1].score + avg_gain * remaining
        return max(0, min(100, int(projected)))

    @property
    def should_early_terminate(self) -> bool:
        """Recommend early termination if projected gains are negligible."""
        proj = self.projected_final_score
        if proj is None:
            return False
        current = self.cycles[-1].score if self.cycles else 0
        remaining = self.dilation_factor - 1 - len(self.cycles)
        # Stop if projected gain is < 2 points over remaining cycles
        return remaining > 2 and (proj - current) < 2

    @property
    def wasted_cycles(self) -> int:
        """Count of cycles where no improvement was made."""
        return sum(1 for c in self.cycles if c.score <= c.previous_score)

    @property
    def efficiency(self) -> float:
        """Ratio of improving cycles to total cycles. Higher is better."""
        if not self.cycles:
            return 0.0
        return 1.0 - (self.wasted_cycles / len(self.cycles))

    @property
    def avg_output_delta(self) -> float:
        """Average output change magnitude across cycles."""
        if not self.cycles:
            return 0.0
        return sum(c.output_delta for c in self.cycles) / len(self.cycles)

    @property
    def score_variance(self) -> float:
        """Variance of score deltas across recent cycles. High variance
        suggests unreliable scoring — triggers ensemble scoring."""
        if len(self.cycles) < 3:
            return 0.0
        recent = self.cycles[-5:]
        deltas = [c.score - c.previous_score for c in recent]
        mean = sum(deltas) / len(deltas)
        return sum((d - mean) ** 2 for d in deltas) / len(deltas)

    @property
    def score_oscillating(self) -> bool:
        """True if scores are bouncing up and down — sign of unreliable scoring."""
        if len(self.cycles) < 4:
            return False
        recent = self.cycles[-4:]
        deltas = [c.score - c.previous_score for c in recent]
        sign_changes = sum(
            1 for i in range(1, len(deltas))
            if (deltas[i] > 0) != (deltas[i - 1] > 0) and deltas[i] != 0 and deltas[i - 1] != 0
        )
        return sign_changes >= 2

    @property
    def superficial_change_rate(self) -> float:
        """Fraction of improving cycles with tiny output changes (<5%).
        High values suggest the model is gaming scores with trivial edits."""
        improving = [c for c in self.cycles if c.score > c.previous_score]
        if not improving:
            return 0.0
        superficial = sum(1 for c in improving if c.output_delta < 0.05)
        return superficial / len(improving)

    @property
    def scoring_bias(self) -> str:
        """Detect systematic scoring bias. Returns 'high', 'low', or 'normal'.
        'high' = all scores >80 from the start (likely overrating).
        'low' = all scores <30 after 3+ cycles (likely underrating)."""
        if len(self.cycles) < 3:
            return "normal"
        scores = [c.score for c in self.cycles]
        if all(s > 80 for s in scores):
            return "high"
        if all(s < 30 for s in scores):
            return "low"
        return "normal"

    @property
    def output_bloat_ratio(self) -> float:
        """Ratio of final output length to initial. Values >3.0 suggest bloat."""
        lengths = [c.output_length for c in self.cycles if c.output_length > 0]
        if len(lengths) < 2:
            return 1.0
        return lengths[-1] / lengths[0] if lengths[0] > 0 else 1.0

    @property
    def comparative_overrule_rate(self) -> float:
        """Fraction of cycles where comparative validation rejected the winner.
        High rates suggest scoring is unreliable."""
        if not self.cycles:
            return 0.0
        overruled = sum(1 for c in self.cycles if c.comparative_overruled)
        return overruled / len(self.cycles)

    @property
    def crossover_win_rate(self) -> float:
        """Fraction of improving cycles where crossover produced the winner."""
        improving = [c for c in self.cycles if c.score > c.previous_score]
        if not improving:
            return 0.0
        crossover_wins = sum(1 for c in improving if c.crossover_used)
        return crossover_wins / len(improving)

    @property
    def total_inference_calls(self) -> int:
        """Total LLM inference calls across all cycles."""
        return sum(c.inference_calls for c in self.cycles)

    @property
    def score_ceiling(self) -> int | None:
        """Detect a scoring ceiling — if last 3+ cycles all hit the same max
        but never exceed it, return that ceiling value. None if no ceiling."""
        if len(self.cycles) < 3:
            return None
        recent_scores = [c.score for c in self.cycles[-3:]]
        peak = max(recent_scores)
        # All recent cycles hit the same peak (within 2 points)
        if all(abs(s - peak) <= 2 for s in recent_scores):
            return peak
        return None

    @property
    def points_per_inference(self) -> float:
        """Score improvement per inference call — measures cost-effectiveness.
        Lower values suggest diminishing returns on compute spend."""
        total_calls = self.total_inference_calls
        if total_calls == 0:
            return 0.0
        return max(0, self.total_improvement) / total_calls

    @property
    def diminishing_returns(self) -> bool:
        """True if last 3+ cycles averaged < 1 point improvement."""
        if len(self.cycles) < 3:
            return False
        recent = self.points_per_cycle[-3:]
        return sum(recent) / len(recent) < 1.0

    @property
    def recommendations(self) -> list[str]:
        """Suggest config changes for next run based on observed metrics."""
        recs = []
        if len(self.cycles) < 3:
            return recs
        if self.efficiency < 0.3:
            recs.append("Low efficiency — try increasing branch_factor for more candidates per cycle")
        if self.output_bloat_ratio > 3.0:
            recs.append("Output bloat detected — consider adding max_output_length constraint")
        if self.comparative_overrule_rate > 0.3:
            recs.append("Scoring unreliable — try ensemble scoring (force_ensemble=True)")
        if self.score_ceiling is not None and self.total_improvement < 10:
            recs.append("Hit ceiling early — try a higher dilation_factor or different task_type classification")
        if self.crossover_win_rate > 0.4:
            recs.append("Crossover is very effective — consider increasing branch_factor to feed it more candidates")
        if self.diminishing_returns and len(self.cycles) > 3:
            recs.append("Diminishing returns — use a lower dilation_factor to save compute")
        if self.superficial_change_rate > 0.5:
            recs.append("Too many superficial changes — scoring may need recalibration")
        eff = self.directive_effectiveness
        if eff.get("generated", 0) > eff.get("builtin", 0) + 0.2:
            recs.append("Generated directives outperform builtins — they'll be preferred automatically")
        if self.avg_cycle_time > 30:
            recs.append("Slow cycles — consider reducing branch_factor or using a faster model")
        return recs

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "task_type": self.task_type,
            "dilation_factor": self.dilation_factor,
            "branch_factor": self.branch_factor,
            "total_cycles": len(self.cycles),
            "peak_score": self.peak_score,
            "improvement_rate": self.improvement_rate,
            "total_improvement": self.total_improvement,
            "avg_cycle_time": self.avg_cycle_time,
            "effective_dilation": self.effective_dilation,
            "score_inflation_rate": self.score_inflation_rate,
            "max_score_jump": self.max_score_jump,
            "directive_effectiveness": self.directive_effectiveness,
            "best_directive": self.best_directive,
            "wasted_cycles": self.wasted_cycles,
            "efficiency": self.efficiency,
            "avg_output_delta": self.avg_output_delta,
            "output_bloat_ratio": round(self.output_bloat_ratio, 2),
            "superficial_change_rate": self.superficial_change_rate,
            "comparative_overrule_rate": self.comparative_overrule_rate,
            "crossover_win_rate": self.crossover_win_rate,
            "total_inference_calls": self.total_inference_calls,
            "points_per_inference": round(self.points_per_inference, 3),
            "score_ceiling": self.score_ceiling,
            "score_history": self.score_history,
            "elapsed_seconds": time.time() - self.start_time if self.start_time else 0,
            "recommendations": self.recommendations,
        }

    def summary(self) -> str:
        """Human-readable summary of the run."""
        d = self.to_dict()
        lines = [
            f"Task type: {d['task_type']}",
            f"Dilation: {d['dilation_factor']}x ({d['total_cycles']} cycles)",
            f"Improvement: +{d['total_improvement']} pts ({d['improvement_rate']:.0%} of cycles improved)",
            f"Score history: {' -> '.join(str(s) for s in d['score_history'])}",
            f"Avg cycle: {d['avg_cycle_time']:.2f}s",
            f"Efficiency: {d['efficiency']:.0%} ({d['wasted_cycles']} wasted cycles)",
            f"Inference calls: {d['total_inference_calls']}",
        ]
        if self.best_directive:
            lines.append(f"Best directive: {self.best_directive}")
        if self.score_ceiling is not None:
            lines.append(f"Warning: score ceiling detected at {self.score_ceiling}")
        if self.diminishing_returns:
            lines.append("Warning: diminishing returns detected")
        if self.superficial_change_rate > 0.5 and len(self.cycles) >= 3:
            lines.append(f"Warning: superficial changes ({self.superficial_change_rate:.0%} of improvements are tiny edits)")
        if self.output_bloat_ratio > 3.0:
            lines.append(f"Warning: output bloat ({self.output_bloat_ratio:.1f}x initial length)")
        if self.comparative_overrule_rate > 0.3 and len(self.cycles) >= 3:
            lines.append(f"Warning: high comparative overrule rate ({self.comparative_overrule_rate:.0%} — scoring may be unreliable)")
        if self.crossover_win_rate > 0.0:
            lines.append(f"Crossover win rate: {self.crossover_win_rate:.0%}")
        if self.score_inflation_rate > 0.8 and len(self.cycles) >= 3:
            lines.append(f"Warning: score inflation ({self.score_inflation_rate:.0%})")
        eff = self.directive_effectiveness
        if eff:
            eff_str = ", ".join(f"{k}={v:.0%}" for k, v in eff.items())
            lines.append(f"Directive effectiveness: {eff_str}")
        recs = self.recommendations
        if recs:
            lines.append("Recommendations:")
            for r in recs:
                lines.append(f"  - {r}")
        return "\n".join(lines)

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
