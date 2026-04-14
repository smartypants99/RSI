import json
import time
from pathlib import Path


class CheckpointManager:
    def __init__(self, checkpoint_dir: str):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, cycle: int, output: str, score: int,
             prompt: str = "", task_type: str = "", no_improvement_count: int = 0) -> None:
        path = self.dir / f"cycle_{cycle:06d}.json"
        path.write_text(json.dumps({
            "cycle": cycle,
            "output": output,
            "score": score,
            "prompt": prompt,
            "task_type": task_type,
            "no_improvement_count": no_improvement_count,
            "timestamp": time.time(),
        }))

    def load_latest(self) -> dict | None:
        files = sorted(self.dir.glob("cycle_*.json"))
        if not files:
            return None
        return json.loads(files[-1].read_text())

    def list_checkpoints(self) -> list[dict]:
        """List all checkpoints with metadata."""
        result = []
        for f in sorted(self.dir.glob("cycle_*.json")):
            try:
                result.append(json.loads(f.read_text()))
            except json.JSONDecodeError:
                continue
        return result

    def cleanup(self) -> None:
        for f in self.dir.glob("cycle_*.json"):
            f.unlink()
