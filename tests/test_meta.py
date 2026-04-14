import tempfile
from pathlib import Path
from timedilate.meta import MetaLearner


def test_record_and_retrieve():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "meta.json")
        ml = MetaLearner(path)
        ml.record_directive("code", "Fix bugs", True)
        ml.record_directive("code", "Fix bugs", True)
        ml.record_directive("code", "Optimize", False)
        ml.record_directive("code", "Optimize", True)
        ml.save()

        # Reload from disk
        ml2 = MetaLearner(path)
        best = ml2.best_directives("code")
        assert len(best) == 2
        assert best[0] == "Fix bugs"  # 100% success rate


def test_best_directives_minimum_attempts():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "meta.json")
        ml = MetaLearner(path)
        ml.record_directive("code", "One shot", True)  # only 1 attempt
        ml.record_directive("code", "Reliable", True)
        ml.record_directive("code", "Reliable", True)
        best = ml.best_directives("code")
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
        ml.record_directive("code", "Fix bugs", True)
        ml.record_directive("code", "Fix bugs", True)
        ml.record_directive("prose", "Improve flow", True)
        ml.record_directive("prose", "Improve flow", True)
        assert ml.best_directives("code")[0] == "Fix bugs"
        assert ml.best_directives("prose")[0] == "Improve flow"
        assert "Improve flow" not in ml.best_directives("code")
