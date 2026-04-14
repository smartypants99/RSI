import tempfile
from pathlib import Path
from timedilate.meta import MetaLearner


def test_record_and_retrieve():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "meta.json")
        ml = MetaLearner(path)
        ml.record_directive("code", "Fix bugs", True, score_delta=10)
        ml.record_directive("code", "Fix bugs", True, score_delta=8)
        ml.record_directive("code", "Optimize", False, score_delta=-2)
        ml.record_directive("code", "Optimize", True, score_delta=3)
        ml.save()

        # Reload from disk
        ml2 = MetaLearner(path)
        best = ml2.best_directives("code")
        assert len(best) == 2
        assert best[0] == "Fix bugs"  # avg delta 9 > 0.5


def test_best_directives_minimum_attempts():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "meta.json")
        ml = MetaLearner(path)
        ml.record_directive("code", "One shot", True, score_delta=5)  # only 1 attempt
        ml.record_directive("code", "Reliable", True, score_delta=5)
        ml.record_directive("code", "Reliable", True, score_delta=3)
        ml.save()
        ml2 = MetaLearner(path)
        best = ml2.best_directives("code")
        assert len(best) == 1  # "One shot" excluded (< 2 attempts)
        assert best[0] == "Reliable"


def test_effectiveness():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "meta.json")
        ml = MetaLearner(path)
        ml.record_directive("prose", "Improve clarity", True)
        ml.record_directive("prose", "Improve clarity", False)
        eff = ml.effectiveness("prose")
        assert eff["Improve clarity"] == 0.5


def test_empty_meta():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "meta.json")
        ml = MetaLearner(path)
        assert ml.best_directives("code") == []
        assert ml.effectiveness("code") == {}


def test_task_type_isolation():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "meta.json")
        ml = MetaLearner(path)
        ml.record_directive("code", "Fix bugs", True, score_delta=5)
        ml.record_directive("code", "Fix bugs", True, score_delta=5)
        ml.record_directive("prose", "Improve flow", True, score_delta=5)
        ml.record_directive("prose", "Improve flow", True, score_delta=5)
        ml.save()
        ml2 = MetaLearner(path)
        assert ml2.best_directives("code")[0] == "Fix bugs"
        assert ml2.best_directives("prose")[0] == "Improve flow"
        assert "Improve flow" not in ml2.best_directives("code")


def test_score_delta_tracking():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "meta.json")
        ml = MetaLearner(path)
        ml.record_directive("code", "Fix bugs", True, score_delta=10)
        ml.record_directive("code", "Fix bugs", True, score_delta=5)
        ml.record_directive("code", "Optimize", False, score_delta=-2)
        deltas = ml.avg_delta("code")
        assert abs(deltas["Fix bugs"] - 7.5) < 0.01
        assert abs(deltas["Optimize"] - (-2.0)) < 0.01


def test_worst_directives():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "meta.json")
        ml = MetaLearner(path)
        for _ in range(4):
            ml.record_directive("code", "Bad approach", False, score_delta=-3)
        for _ in range(3):
            ml.record_directive("code", "Good approach", True, score_delta=5)
        worst = ml.worst_directives("code")
        assert "Bad approach" in worst
        assert "Good approach" not in worst


def test_atomic_save_survives_reload():
    """Atomic save should produce a valid file that can be reloaded."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "meta.json")
        ml = MetaLearner(path)
        ml.record_directive("code", "Test", True, score_delta=5)
        ml.save()
        # No .tmp file should remain
        assert not Path(path).with_suffix(".tmp").exists()
        ml2 = MetaLearner(path)
        assert ml2.effectiveness("code")["Test"] == 1.0


def test_prune_stale_entries():
    """Pruning should remove lowest-attempt entries when over max."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "meta.json")
        ml = MetaLearner(path)
        # Add 5 entries with varying attempts
        for i in range(5):
            for _ in range(i + 1):
                ml.record_directive("code", f"directive_{i}", True)
        # Prune to keep only 3
        ml._prune_stale(max_entries=3)
        stats = ml.data["directive_stats"]
        assert len(stats) == 3
        # Should have kept the ones with most attempts (directive_2, 3, 4)
        keys = list(stats.keys())
        assert all(f"directive_{i}" in k for k, i in zip(sorted(keys), [2, 3, 4]))


def test_worst_directives_needs_min_attempts():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "meta.json")
        ml = MetaLearner(path)
        ml.record_directive("code", "Too few", False)
        ml.record_directive("code", "Too few", False)
        # Only 2 attempts, threshold is 3
        assert ml.worst_directives("code") == []
