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


def test_score_inflation_rate():
    m = RunMetrics(start_time=time.time())
    # All 4 cycles improve — suspicious
    for i in range(4):
        m.record_cycle(cycle=i+1, score=60+i*10, previous_score=50+i*10, directive="d",
                       directive_source="builtin", branch_count=1, best_variant_index=0, elapsed_seconds=0.1)
    assert m.score_inflation_rate == 1.0  # 100% improvement rate


def test_max_score_jump():
    m = RunMetrics(start_time=time.time())
    m.record_cycle(cycle=1, score=70, previous_score=50, directive="d", directive_source="builtin",
                   branch_count=1, best_variant_index=0, elapsed_seconds=0.1)
    m.record_cycle(cycle=2, score=75, previous_score=70, directive="d", directive_source="builtin",
                   branch_count=1, best_variant_index=0, elapsed_seconds=0.1)
    assert m.max_score_jump == 20


def test_directive_effectiveness():
    m = RunMetrics(start_time=time.time())
    m.record_cycle(cycle=1, score=80, previous_score=70, directive="fix bugs", directive_source="builtin",
                   branch_count=1, best_variant_index=0, elapsed_seconds=0.1)
    m.record_cycle(cycle=2, score=80, previous_score=80, directive="optimize", directive_source="builtin",
                   branch_count=1, best_variant_index=-1, elapsed_seconds=0.1)
    m.record_cycle(cycle=3, score=90, previous_score=80, directive="custom idea", directive_source="generated",
                   branch_count=1, best_variant_index=0, elapsed_seconds=0.1)
    eff = m.directive_effectiveness
    assert eff["builtin"] == 0.5  # 1 of 2 improved
    assert eff["generated"] == 1.0  # 1 of 1 improved


def test_best_directive():
    m = RunMetrics(start_time=time.time())
    m.record_cycle(cycle=1, score=75, previous_score=70, directive="small fix", directive_source="builtin",
                   branch_count=1, best_variant_index=0, elapsed_seconds=0.1)
    m.record_cycle(cycle=2, score=90, previous_score=75, directive="big refactor", directive_source="generated",
                   branch_count=1, best_variant_index=0, elapsed_seconds=0.1)
    assert m.best_directive == "big refactor"


def test_best_directive_no_improvement():
    m = RunMetrics(start_time=time.time())
    m.record_cycle(cycle=1, score=70, previous_score=70, directive="noop", directive_source="builtin",
                   branch_count=1, best_variant_index=-1, elapsed_seconds=0.1)
    assert m.best_directive is None


def test_points_per_cycle():
    m = RunMetrics(start_time=time.time())
    m.record_cycle(cycle=1, score=80, previous_score=70, directive="d", directive_source="builtin",
                   branch_count=1, best_variant_index=0, elapsed_seconds=0.1)
    m.record_cycle(cycle=2, score=82, previous_score=80, directive="d", directive_source="builtin",
                   branch_count=1, best_variant_index=0, elapsed_seconds=0.1)
    assert m.points_per_cycle == [10, 2]


def test_diminishing_returns_detected():
    m = RunMetrics(start_time=time.time())
    for i in range(4):
        m.record_cycle(cycle=i+1, score=70, previous_score=70, directive="d", directive_source="builtin",
                       branch_count=1, best_variant_index=-1, elapsed_seconds=0.1)
    assert m.diminishing_returns is True


def test_diminishing_returns_not_detected():
    m = RunMetrics(start_time=time.time())
    for i in range(3):
        m.record_cycle(cycle=i+1, score=70+i*5, previous_score=65+i*5, directive="d", directive_source="builtin",
                       branch_count=1, best_variant_index=0, elapsed_seconds=0.1)
    assert m.diminishing_returns is False


def test_summary():
    m = RunMetrics(prompt="write sort", task_type="code", dilation_factor=5, branch_factor=1, start_time=time.time())
    m.record_cycle(cycle=1, score=70, previous_score=50, directive="fix bugs", directive_source="builtin",
                   branch_count=1, best_variant_index=0, elapsed_seconds=0.5)
    m.record_cycle(cycle=2, score=85, previous_score=70, directive="optimize", directive_source="builtin",
                   branch_count=1, best_variant_index=0, elapsed_seconds=0.3)
    summary = m.summary()
    assert "code" in summary
    assert "+35" in summary
    assert "70 -> 85" in summary


def test_empty_metrics():
    m = RunMetrics()
    assert m.improvement_rate == 0.0
    assert m.total_improvement == 0
    assert m.stagnant_streak == 0
    assert m.score_history == []
    assert m.avg_cycle_time == 0.0
    assert m.effective_dilation == 0.0
    assert m.score_inflation_rate == 0.0
    assert m.max_score_jump == 0
