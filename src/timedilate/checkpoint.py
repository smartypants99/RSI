import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class CheckpointManager:
    def __init__(self, checkpoint_dir: str):
        self.dir = Path(checkpoint_dir)
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning("Could not create checkpoint dir %s: %s", checkpoint_dir, e)

    def save(self, cycle: int, output: str, score: int,
             prompt: str = "", task_type: str = "", no_improvement_count: int = 0) -> None:
        path = self.dir / f"cycle_{cycle:06d}.json"
        tmp_path = path.with_suffix(".json.tmp")
        try:
            data = json.dumps({
                "cycle": cycle,
                "output": output,
                "score": score,
                "prompt": prompt,
                "task_type": task_type,
                "no_improvement_count": no_improvement_count,
                "timestamp": time.time(),
            })
            # Atomic write: write to tmp then rename to avoid corrupt checkpoints
            tmp_path.write_text(data)
            tmp_path.replace(path)
        except OSError as e:
            logger.warning("Failed to save checkpoint at cycle %d: %s", cycle, e)
            # Clean up tmp file if it exists
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def load_latest(self) -> dict | None:
        files = sorted(self.dir.glob("cycle_*.json"))
        if not files:
            return None
        # Try files in reverse order — if latest is corrupt, fall back to previous
        for f in reversed(files):
            try:
                data = json.loads(f.read_text())
                if "cycle" in data and "output" in data and "score" in data:
                    return data
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Corrupt checkpoint %s: %s, trying previous", f.name, e)
        return None

    def list_checkpoints(self) -> list[dict]:
        """List all checkpoints with metadata."""
        result = []
        for f in sorted(self.dir.glob("cycle_*.json")):
            try:
                result.append(json.loads(f.read_text()))
            except (json.JSONDecodeError, OSError):
                continue
        return result

    def cleanup(self) -> None:
        for f in self.dir.glob("cycle_*.json"):
            try:
                f.unlink()
            except OSError as e:
                logger.warning("Failed to remove checkpoint %s: %s", f.name, e)
