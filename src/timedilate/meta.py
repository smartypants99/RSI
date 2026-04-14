"""Meta-learning: track directive effectiveness across runs."""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MetaLearner:
    """Persists directive effectiveness data across runs."""

    def __init__(self, path: str = ".timedilate_meta.json"):
        self.path = Path(path)
        self._loaded_from_disk = self.path.exists()
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except (json.JSONDecodeError, OSError):
                logger.warning("Failed to load meta data, starting fresh")
        return {"directive_stats": {}}

    def save(self) -> None:
        """Atomic save: write to temp file then rename to avoid corruption."""
        self._prune_stale()
        tmp = self.path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(self.data, indent=2))
            tmp.replace(self.path)
        except OSError as e:
            logger.warning("Failed to save meta data: %s", e)
            tmp.unlink(missing_ok=True)

    def _prune_stale(self, max_entries: int = 200) -> None:
        """Remove low-value entries to keep the file bounded."""
        stats = self.data.get("directive_stats", {})
        if len(stats) <= max_entries:
            return
        # Sort by attempts (least used first) and remove the excess
        sorted_keys = sorted(stats, key=lambda k: stats[k]["attempts"])
        to_remove = len(stats) - max_entries
        for key in sorted_keys[:to_remove]:
            del stats[key]
        logger.info("Pruned %d stale meta-learning entries", to_remove)

    def record_directive(self, task_type: str, directive: str, improved: bool,
                         score_delta: int = 0) -> None:
        """Record whether a directive led to improvement and by how much."""
        stats = self.data["directive_stats"]
        key = f"{task_type}:{directive[:50]}"
        if key not in stats:
            stats[key] = {"attempts": 0, "successes": 0, "total_delta": 0}
        stats[key]["attempts"] += 1
        if improved:
            stats[key]["successes"] += 1
        stats[key]["total_delta"] = stats[key].get("total_delta", 0) + score_delta

    def best_directives(self, task_type: str, top_n: int = 3) -> list[str]:
        """Return the top N directives ranked by avg score delta (not just win rate).
        This captures magnitude — a directive that gains +8 avg is better than one
        that gains +1 avg even if both have 100% success rate.
        Only returns results if data was loaded from a previous run on disk."""
        if not self._loaded_from_disk:
            return []
        stats = self.data["directive_stats"]
        candidates = []
        for key, val in stats.items():
            if key.startswith(f"{task_type}:") and val["attempts"] >= 2:
                avg_delta = val.get("total_delta", 0) / val["attempts"]
                directive = key[len(task_type) + 1:]
                candidates.append((avg_delta, directive))
        candidates.sort(reverse=True)
        # Only return directives with positive avg delta
        return [d for delta, d in candidates[:top_n] if delta > 0]

    def worst_directives(self, task_type: str, top_n: int = 3) -> list[str]:
        """Return directives that consistently fail (0% success with 3+ attempts)."""
        stats = self.data["directive_stats"]
        candidates = []
        for key, val in stats.items():
            if key.startswith(f"{task_type}:") and val["attempts"] >= 3:
                rate = val["successes"] / val["attempts"]
                if rate < 0.1:  # less than 10% success
                    directive = key[len(task_type) + 1:]
                    candidates.append((rate, directive))
        candidates.sort()
        return [d for _, d in candidates[:top_n]]

    def avg_delta(self, task_type: str) -> dict[str, float]:
        """Return {directive: avg_score_delta} for a task type."""
        stats = self.data["directive_stats"]
        result = {}
        for key, val in stats.items():
            if key.startswith(f"{task_type}:") and val["attempts"] > 0:
                directive = key[len(task_type) + 1:]
                result[directive] = val.get("total_delta", 0) / val["attempts"]
        return result

    def effectiveness(self, task_type: str) -> dict[str, float]:
        """Return {directive_prefix: success_rate} for a task type."""
        stats = self.data["directive_stats"]
        result = {}
        for key, val in stats.items():
            if key.startswith(f"{task_type}:") and val["attempts"] > 0:
                directive = key[len(task_type) + 1:]
                result[directive] = val["successes"] / val["attempts"]
        return result
