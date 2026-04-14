import json
import tempfile
import time
from timedilate.metrics import RunMetrics, CycleMetric


def test_record_cycle():
    m = RunMetrics(prompt="test", task_type="code", dilation_factor=5, branch_factor=3, start_time=time.time())
    m.record_cycle(cycle=1, score=80, previous_score=70, directive="fix bugs",
                   directive_source="builtin", branch_count=3, best_variant_index=1, elapsed_seconds=0.5)
    assert len(m.cycles) == 1
    assert m.cycles[0].score == 80


def test_score_history():
    m = RunMetrics(start_time=time.time())
    m.record_cycle(cycle=1, score=70, previous_score=50, directive="d", directive_source="builtin",
                   branch_count=1, best_variant_index=0, elapsed_seconds=0.1)
    m.record_cycle(cycle=2, score=85, previous_score=70, directive="d", directive_source="builtin",
                   branch_count=1, best_variant_index=0, elapsed_seconds=0.1)
    assert m.score_history == [70, 85]


def test_improvement_rate():
    m = RunMetrics(start_time=time.time())
    m.record_cycle(cycle=1, score=80, previous_score=70, directive="d", directive_source="builtin",
                   branch_count=1, best_variant_index=0, elapsed_seconds=0.1)
    m.record_cycle(cycle=2, score=80, previous_score=80, directive="d", directive_source="builtin",
                   branch_count=1, best_variant_index=-1, elapsed_seconds=0.1)
    assert m.improvement_rate == 0.5


def test_total_improvement():
    m = RunMetrics(start_time=time.time())
    m.record_cycle(cycle=1, score=70, previous_score=50, directive="d", directive_source="builtin",
                   branch_count=1, best_variant_index=0, elapsed_seconds=0.1)
    m.record_cycle(cycle=2, score=90, previous_score=70, directive="d", directive_source="builtin",
                   branch_count=1, best_variant_index=0, elapsed_seconds=0.1)
    assert m.total_improvement == 40


def test_stagnant_streak():
    m = RunMetrics(start_time=time.time())
    m.record_cycle(cycle=1, score=80, previous_score=70, directive="d", directive_source="builtin",
                   branch_count=1, best_variant_index=0, elapsed_seconds=0.1)
    m.record_cycle(cycle=2, score=75, previous_score=80, directive="d", directive_source="builtin",
                   branch_count=1, best_variant_index=-1, elapsed_seconds=0.1)
    m.record_cycle(cycle=3, score=75, previous_score=75, directive="d", directive_source="builtin",
                   branch_count=1, best_variant_index=-1, elapsed_seconds=0.1)
    assert m.stagnant_streak == 2


def test_save_and_load():
    m = RunMetrics(prompt="test", task_type="code", dilation_factor=2, branch_factor=1, start_time=time.time())
    m.record_cycle(cycle=1, score=80, previous_score=70, directive="d", directive_source="builtin",
                   branch_count=1, best_variant_index=0, elapsed_seconds=0.1)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        m.save(f.name)
        data = json.loads(open(f.name).read())
    assert data["total_cycles"] == 1
    assert data["score_history"] == [80]
    assert data["improvement_rate"] == 1.0


def test_avg_cycle_time():
    m = RunMetrics(start_time=time.time())
    m.record_cycle(cycle=1, score=80, previous_score=70, directive="d", directive_source="builtin",
                   branch_count=1, best_variant_index=0, elapsed_seconds=1.0)
    m.record_cycle(cycle=2, score=90, previous_score=80, directive="d", directive_source="builtin",
                   branch_count=1, best_variant_index=0, elapsed_seconds=3.0)
    assert m.avg_cycle_time == 2.0


def test_effective_dilation():
    m = RunMetrics(start_time=time.time())
    m.record_cycle(cycle=1, score=80, previous_score=70, directive="d", directive_source="builtin",
                   branch_count=1, best_variant_index=0, elapsed_seconds=0.5)
    m.record_cycle(cycle=2, score=90, previous_score=80, directive="d", directive_source="builtin",
                   branch_count=1, best_variant_index=0, elapsed_seconds=0.5)
    assert m.effective_dilation == 2.0  # 2 cycles / 1 second


def test_empty_metrics():
    m = RunMetrics()
    assert m.improvement_rate == 0.0
    assert m.total_improvement == 0
    assert m.stagnant_streak == 0
    assert m.score_history == []
    assert m.avg_cycle_time == 0.0
    assert m.effective_dilation == 0.0
