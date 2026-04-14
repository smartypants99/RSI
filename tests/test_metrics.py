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


def test_projected_final_score():
    m = RunMetrics(dilation_factor=10, start_time=time.time())
    for i in range(4):
        m.record_cycle(cycle=i+1, score=60+i*5, previous_score=55+i*5, directive="d",
                       directive_source="builtin", branch_count=1, best_variant_index=0, elapsed_seconds=0.1)
    proj = m.projected_final_score
    assert proj is not None
    assert proj > 75  # trending up by 5/cycle with 5 remaining


def test_should_early_terminate_stagnant():
    m = RunMetrics(dilation_factor=20, start_time=time.time())
    for i in range(5):
        m.record_cycle(cycle=i+1, score=80, previous_score=80, directive="d",
                       directive_source="builtin", branch_count=1, best_variant_index=-1, elapsed_seconds=0.1)
    assert m.should_early_terminate is True  # no gains projected


def test_should_not_early_terminate_improving():
    m = RunMetrics(dilation_factor=10, start_time=time.time())
    for i in range(4):
        m.record_cycle(cycle=i+1, score=60+i*5, previous_score=55+i*5, directive="d",
                       directive_source="builtin", branch_count=1, best_variant_index=0, elapsed_seconds=0.1)
    assert m.should_early_terminate is False  # still improving


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


def test_avg_output_delta():
    m = RunMetrics(start_time=time.time())
    m.record_cycle(cycle=1, score=80, previous_score=70, directive="d", directive_source="builtin",
                   branch_count=1, best_variant_index=0, elapsed_seconds=0.1, output_delta=0.3)
    m.record_cycle(cycle=2, score=85, previous_score=80, directive="d", directive_source="builtin",
                   branch_count=1, best_variant_index=0, elapsed_seconds=0.1, output_delta=0.1)
    assert m.avg_output_delta == 0.2


def test_superficial_change_rate():
    m = RunMetrics(start_time=time.time())
    # Two improving cycles: one with big delta, one with tiny delta
    m.record_cycle(cycle=1, score=80, previous_score=70, directive="d", directive_source="builtin",
                   branch_count=1, best_variant_index=0, elapsed_seconds=0.1, output_delta=0.3)
    m.record_cycle(cycle=2, score=85, previous_score=80, directive="d", directive_source="builtin",
                   branch_count=1, best_variant_index=0, elapsed_seconds=0.1, output_delta=0.02)
    assert m.superficial_change_rate == 0.5  # 1 of 2 improving cycles is superficial


def test_superficial_change_rate_ignores_non_improving():
    m = RunMetrics(start_time=time.time())
    # Non-improving cycle with tiny delta should not count
    m.record_cycle(cycle=1, score=70, previous_score=70, directive="d", directive_source="builtin",
                   branch_count=1, best_variant_index=-1, elapsed_seconds=0.1, output_delta=0.01)
    m.record_cycle(cycle=2, score=80, previous_score=70, directive="d", directive_source="builtin",
                   branch_count=1, best_variant_index=0, elapsed_seconds=0.1, output_delta=0.4)
    assert m.superficial_change_rate == 0.0  # only 1 improving cycle, it has big delta


def test_score_variance():
    m = RunMetrics(start_time=time.time())
    # Consistent deltas: variance should be low
    for i in range(4):
        m.record_cycle(cycle=i+1, score=60+i*5, previous_score=55+i*5, directive="d",
                       directive_source="builtin", branch_count=1, best_variant_index=0, elapsed_seconds=0.1)
    assert m.score_variance < 1.0  # very consistent +5 per cycle


def test_score_oscillating():
    m = RunMetrics(start_time=time.time())
    # Oscillating: up, down, up, down
    scores = [(70, 60), (65, 70), (75, 65), (70, 75)]
    for i, (score, prev) in enumerate(scores):
        m.record_cycle(cycle=i+1, score=score, previous_score=prev, directive="d",
                       directive_source="builtin", branch_count=1, best_variant_index=0, elapsed_seconds=0.1)
    assert m.score_oscillating is True


def test_not_oscillating_when_improving():
    m = RunMetrics(start_time=time.time())
    for i in range(4):
        m.record_cycle(cycle=i+1, score=60+i*5, previous_score=55+i*5, directive="d",
                       directive_source="builtin", branch_count=1, best_variant_index=0, elapsed_seconds=0.1)
    assert m.score_oscillating is False


def test_directive_effectiveness_by_score_range():
    m = RunMetrics(start_time=time.time())
    # Low score range: builtin works
    m.record_cycle(cycle=1, score=60, previous_score=40, directive="fix", directive_source="builtin",
                   branch_count=1, best_variant_index=0, elapsed_seconds=0.1)
    # Mid score range: generated works
    m.record_cycle(cycle=2, score=75, previous_score=60, directive="custom", directive_source="generated",
                   branch_count=1, best_variant_index=0, elapsed_seconds=0.1)
    # High score range: builtin fails
    m.record_cycle(cycle=3, score=75, previous_score=75, directive="polish", directive_source="builtin",
                   branch_count=1, best_variant_index=-1, elapsed_seconds=0.1)
    eff = m.directive_effectiveness_by_score_range
    assert "low" in eff
    assert eff["low"]["builtin"] == 1.0
    assert "mid" in eff
    assert eff["mid"]["generated"] == 1.0
    assert "high" in eff
    assert eff["high"]["builtin"] == 0.0


def test_wasted_cycles():
    m = RunMetrics(start_time=time.time())
    m.record_cycle(cycle=1, score=80, previous_score=70, directive="d", directive_source="builtin",
                   branch_count=1, best_variant_index=0, elapsed_seconds=0.1)
    m.record_cycle(cycle=2, score=75, previous_score=80, directive="d", directive_source="builtin",
                   branch_count=1, best_variant_index=-1, elapsed_seconds=0.1)
    m.record_cycle(cycle=3, score=75, previous_score=75, directive="d", directive_source="builtin",
                   branch_count=1, best_variant_index=-1, elapsed_seconds=0.1)
    assert m.wasted_cycles == 2
    assert abs(m.efficiency - 1.0 / 3.0) < 0.01


def test_scoring_bias_high():
    m = RunMetrics(start_time=time.time())
    for i in range(4):
        m.record_cycle(cycle=i+1, score=85+i, previous_score=84+i, directive="d",
                       directive_source="builtin", branch_count=1, best_variant_index=0, elapsed_seconds=0.1)
    assert m.scoring_bias == "high"


def test_scoring_bias_low():
    m = RunMetrics(start_time=time.time())
    for i in range(4):
        m.record_cycle(cycle=i+1, score=20, previous_score=20, directive="d",
                       directive_source="builtin", branch_count=1, best_variant_index=-1, elapsed_seconds=0.1)
    assert m.scoring_bias == "low"


def test_scoring_bias_normal():
    m = RunMetrics(start_time=time.time())
    m.record_cycle(cycle=1, score=50, previous_score=30, directive="d", directive_source="builtin",
                   branch_count=1, best_variant_index=0, elapsed_seconds=0.1)
    m.record_cycle(cycle=2, score=70, previous_score=50, directive="d", directive_source="builtin",
                   branch_count=1, best_variant_index=0, elapsed_seconds=0.1)
    m.record_cycle(cycle=3, score=75, previous_score=70, directive="d", directive_source="builtin",
                   branch_count=1, best_variant_index=0, elapsed_seconds=0.1)
    assert m.scoring_bias == "normal"


def test_output_bloat_ratio():
    m = RunMetrics(start_time=time.time())
    m.record_cycle(cycle=1, score=60, previous_score=50, directive="d", directive_source="builtin",
                   branch_count=1, best_variant_index=0, elapsed_seconds=0.1, output_length=100)
    m.record_cycle(cycle=2, score=70, previous_score=60, directive="d", directive_source="builtin",
                   branch_count=1, best_variant_index=0, elapsed_seconds=0.1, output_length=400)
    assert m.output_bloat_ratio == 4.0


def test_output_bloat_ratio_no_bloat():
    m = RunMetrics(start_time=time.time())
    m.record_cycle(cycle=1, score=60, previous_score=50, directive="d", directive_source="builtin",
                   branch_count=1, best_variant_index=0, elapsed_seconds=0.1, output_length=100)
    m.record_cycle(cycle=2, score=70, previous_score=60, directive="d", directive_source="builtin",
                   branch_count=1, best_variant_index=0, elapsed_seconds=0.1, output_length=110)
    assert m.output_bloat_ratio == 1.1


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
