"""Meta-learning: track directive effectiveness across runs."""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MetaLearner:
    """Persists directive effectiveness data across runs."""

    def __init__(self, path: str = ".timedilate_meta.json"):
        self.path = Path(path)
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except (json.JSONDecodeError, OSError):
                logger.warning("Failed to load meta data, starting fresh")
        return {"directive_stats": {}}

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2))

    def record_directive(self, task_type: str, directive: str, improved: bool) -> None:
        """Record whether a directive led to improvement."""
        stats = self.data["directive_stats"]
        key = f"{task_type}:{directive[:50]}"
        if key not in stats:
            stats[key] = {"attempts": 0, "successes": 0}
        stats[key]["attempts"] += 1
        if improved:
            stats[key]["successes"] += 1

    def best_directives(self, task_type: str, top_n: int = 3) -> list[str]:
        """Return the top N directives by success rate for a task type."""
        stats = self.data["directive_stats"]
        candidates = []
        for key, val in stats.items():
            if key.startswith(f"{task_type}:") and val["attempts"] >= 2:
                rate = val["successes"] / val["attempts"]
                directive = key[len(task_type) + 1:]
                candidates.append((rate, directive))
        candidates.sort(reverse=True)
        return [d for _, d in candidates[:top_n]]

    def effectiveness(self, task_type: str) -> dict[str, float]:
        """Return {directive_prefix: success_rate} for a task type."""
        stats = self.data["directive_stats"]
        result = {}
        for key, val in stats.items():
            if key.startswith(f"{task_type}:") and val["attempts"] > 0:
                directive = key[len(task_type) + 1:]
                result[directive] = val["successes"] / val["attempts"]
        return result
