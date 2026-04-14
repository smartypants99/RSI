import tempfile
from timedilate.checkpoint import CheckpointManager


def test_save_and_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)
        mgr.save(cycle=5, output="hello world", score=85)
        result = mgr.load_latest()
        assert result["cycle"] == 5
        assert result["output"] == "hello world"
        assert result["score"] == 85


def test_load_latest_picks_highest_cycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)
        mgr.save(cycle=5, output="v1", score=70)
        mgr.save(cycle=10, output="v2", score=85)
        result = mgr.load_latest()
        assert result["cycle"] == 10
        assert result["output"] == "v2"


def test_load_empty_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)
        assert mgr.load_latest() is None


def test_cleanup():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)
        mgr.save(cycle=5, output="v1", score=70)
        mgr.cleanup()
        assert mgr.load_latest() is None


def test_save_with_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)
        mgr.save(cycle=3, output="code", score=80, prompt="write sort",
                 task_type="code", no_improvement_count=2)
        result = mgr.load_latest()
        assert result["prompt"] == "write sort"
        assert result["task_type"] == "code"
        assert result["no_improvement_count"] == 2
        assert "timestamp" in result


def test_list_checkpoints():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)
        mgr.save(cycle=1, output="v1", score=60)
        mgr.save(cycle=5, output="v2", score=80)
        cps = mgr.list_checkpoints()
        assert len(cps) == 2
        assert cps[0]["cycle"] == 1
        assert cps[1]["cycle"] == 5


def test_load_latest_skips_corrupt():
    """If the latest checkpoint is corrupt, falls back to previous."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)
        mgr.save(cycle=5, output="good", score=80)
        mgr.save(cycle=10, output="also good", score=90)
        # Corrupt the latest checkpoint
        from pathlib import Path
        corrupt = Path(tmpdir) / "cycle_000010.json"
        corrupt.write_text("{invalid json")
        result = mgr.load_latest()
        assert result is not None
        assert result["cycle"] == 5
        assert result["score"] == 80


def test_list_checkpoints_skips_corrupt():
    """Corrupt checkpoints are silently skipped in listing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)
        mgr.save(cycle=1, output="v1", score=60)
        from pathlib import Path
        (Path(tmpdir) / "cycle_000002.json").write_text("not json")
        mgr.save(cycle=3, output="v3", score=80)
        cps = mgr.list_checkpoints()
        assert len(cps) == 2
        assert cps[0]["cycle"] == 1
        assert cps[1]["cycle"] == 3


def test_prune_keeps_recent():
    """Prune removes old checkpoints, keeping most recent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)
        for i in range(1, 8):
            mgr.save(cycle=i, output=f"v{i}", score=50 + i)
        removed = mgr.prune(keep=3)
        assert removed == 4
        remaining = mgr.list_checkpoints()
        assert len(remaining) == 3
        assert remaining[0]["cycle"] == 5
        assert remaining[-1]["cycle"] == 7


def test_save_with_score_history():
    """Checkpoint saves score_history for trajectory analysis on resume."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)
        mgr.save(cycle=5, output="code", score=80,
                 score_history=[50, 60, 70, 75, 80])
        result = mgr.load_latest()
        assert result["score_history"] == [50, 60, 70, 75, 80]


def test_save_without_score_history_defaults_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)
        mgr.save(cycle=1, output="x", score=50)
        result = mgr.load_latest()
        assert result["score_history"] == []


def test_prune_noop_when_few():
    """Prune does nothing when fewer checkpoints than keep limit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)
        mgr.save(cycle=1, output="v1", score=60)
        mgr.save(cycle=2, output="v2", score=70)
        removed = mgr.prune(keep=5)
        assert removed == 0
        assert len(mgr.list_checkpoints()) == 2
