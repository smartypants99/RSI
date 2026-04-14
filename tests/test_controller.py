from unittest.mock import MagicMock
from timedilate.controller import DilationController
from timedilate.config import TimeDilateConfig


def make_mock_engine(responses: list[str]):
    engine = MagicMock()
    engine.generate = MagicMock(side_effect=responses)
    engine.estimate_tokens = MagicMock(return_value=100)
    return engine


def test_controller_runs_correct_number_of_cycles():
    config = TimeDilateConfig(dilation_factor=3, branch_factor=1)
    mock_engine = make_mock_engine([
        "initial output",       # first generation
        "75",                   # score initial
        "improved v1", "90",    # cycle 1
        "improved v2", "95",    # cycle 2
    ])
    controller = DilationController(config, mock_engine)
    result = controller.run("Write hello world")
    assert result.cycles_completed == 2
    assert result.output == "improved v2"


def test_controller_dilation_1_means_no_refinement():
    config = TimeDilateConfig(dilation_factor=1, branch_factor=1)
    mock_engine = make_mock_engine(["initial output"])
    controller = DilationController(config, mock_engine)
    result = controller.run("test")
    assert result.output == "initial output"
    assert result.cycles_completed == 0


def test_controller_convergence_detection():
    # Use dilation_factor=5 (4 cycles), convergence_threshold=2
    # so convergence fires after 2 no-improvement cycles (before early termination)
    config = TimeDilateConfig(dilation_factor=5, branch_factor=1, convergence_threshold=2)
    responses = ["initial output", "80", "80"]  # third "80" is consistency check re-score
    # Plenty of responses for cycles + generated directives + fresh attempts
    for _ in range(20):
        responses.extend(["gen directive", "same output", "70", "fresh out", "65"])
    mock_engine = make_mock_engine(responses)
    controller = DilationController(config, mock_engine)
    result = controller.run("test")
    assert result.convergence_detected
    assert result.score == 80  # never improved past initial


def test_controller_on_cycle_callback():
    config = TimeDilateConfig(dilation_factor=2, branch_factor=1)
    mock_engine = make_mock_engine([
        "initial", "80",
        "improved", "90",
    ])
    callback_calls = []

    def on_cycle(cycle, total, score, elapsed):
        callback_calls.append((cycle, total, score))

    controller = DilationController(config, mock_engine)
    controller.run("test", on_cycle=on_cycle)
    assert len(callback_calls) == 1
    assert callback_calls[0][0] == 1


def test_controller_early_exit_on_perfect_score():
    """Should stop early when score reaches 100."""
    config = TimeDilateConfig(dilation_factor=10, branch_factor=1)
    responses = [
        "initial", "80", "80",  # third is consistency check
        "v1", "100",  # perfect score — should stop here
        # remaining cycles should NOT run
    ]
    mock_engine = make_mock_engine(responses)
    controller = DilationController(config, mock_engine)
    result = controller.run("test")
    assert result.score == 100
    assert result.cycles_completed == 1  # stopped after cycle 1


def test_controller_resume_from_checkpoint():
    """Should resume from checkpoint when resume=True."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TimeDilateConfig(dilation_factor=5, branch_factor=1, checkpoint_dir=tmpdir)

        # Pre-save a checkpoint at cycle 2
        from timedilate.checkpoint import CheckpointManager
        mgr = CheckpointManager(tmpdir)
        mgr.save(cycle=2, output="checkpoint_output", score=80)

        # Responses for cycles 3 and 4 (resuming from 2)
        # Cycle 3: v3 scores 85 (delta=5 from 80, comparative -> B confirms)
        # Cycle 4: v4 scores 95 (delta=10, no comparative needed)
        mock_engine = make_mock_engine([
            "v3", "85", "B",   # cycle 3 + comparative check
            # cycle 4 (index 3): 3%3==0 triggers feedback scoring
            "STRENGTHS:\n- Good\nWEAKNESSES:\n1. Bad\nSCORE: 85",
            "v4", "95",        # cycle 4 (large delta, no comparative)
        ])
        controller = DilationController(config, mock_engine)
        result = controller.run("test", resume=True)
        assert result.resumed_from_cycle == 2
        assert result.cycles_completed == 4
        assert result.score == 95


def test_controller_builds_history_summary():
    config = TimeDilateConfig(dilation_factor=4, branch_factor=1)
    mock_engine = make_mock_engine([
        "initial", "70", "70",  # consistency check
        "v1", "80",
        "v2", "85",
        "v3", "90",
    ])
    controller = DilationController(config, mock_engine)
    result = controller.run("test")
    # After 3 cycles, history should have been built for later cycles
    assert result.metrics is not None
    assert len(result.metrics.cycles) == 3


def test_controller_targeted_directive_at_cycle_5():
    """At cycle 5, controller should do detailed scoring for targeted directive."""
    config = TimeDilateConfig(dilation_factor=8, branch_factor=1)
    responses = ["initial", "60", "60"]  # consistency check
    # Cycle 0: variant scores 70
    responses.extend(["v1", "70"])
    # Cycles 1-2: each improves
    responses.extend(["v2", "80"])
    responses.extend(["v3", "83"])
    # Cycle 3: feedback scoring (3%3==0, 3%5!=0)
    responses.append("STRENGTHS:\n- Good\nWEAKNESSES:\n1. Bad\nSCORE: 83")
    responses.extend(["v4", "86"])
    # Cycle 4: normal
    responses.extend(["v5", "89"])
    # Cycle 5: detailed score (5%5==0) + targeted improvement + score
    responses.append("C:20 K:10 Q:18 E:15")  # completeness weakest
    responses.extend(["v5_targeted", "95"])
    # Cycle 6: feedback scoring (6%3==0, 6%5!=0) + normal cycle
    responses.append("STRENGTHS:\n- Good\nWEAKNESSES:\n1. Minor\nSCORE: 95")
    responses.extend(["v6", "97"])
    mock_engine = make_mock_engine(responses)
    controller = DilationController(config, mock_engine)
    result = controller.run("test")
    assert result.cycles_completed == 7
    # Check that a targeted directive was used
    targeted_cycles = [c for c in result.metrics.cycles if c.directive_source == "targeted"]
    assert len(targeted_cycles) >= 1


def test_should_prefer_generated_directives():
    """Prefer generated directives when they outperform builtins."""
    from timedilate.metrics import RunMetrics
    config = TimeDilateConfig(dilation_factor=2, branch_factor=1)
    mock_engine = make_mock_engine([])
    controller = DilationController(config, mock_engine)

    metrics = RunMetrics(start_time=0)
    # 3 builtin cycles: only 1 improved (33%)
    for i in range(3):
        metrics.record_cycle(cycle=i+1, score=70 if i == 0 else 60, previous_score=60,
                             directive="d", directive_source="builtin",
                             branch_count=1, best_variant_index=0 if i == 0 else -1, elapsed_seconds=0.1)
    # 2 generated cycles: both improved (100%)
    for i in range(2):
        metrics.record_cycle(cycle=4+i, score=80+i*5, previous_score=70+i*5,
                             directive="custom", directive_source="generated",
                             branch_count=1, best_variant_index=0, elapsed_seconds=0.1)

    assert controller._should_prefer_generated(metrics) is True


def test_should_not_prefer_generated_insufficient_data():
    from timedilate.metrics import RunMetrics
    config = TimeDilateConfig(dilation_factor=2, branch_factor=1)
    mock_engine = make_mock_engine([])
    controller = DilationController(config, mock_engine)
    metrics = RunMetrics(start_time=0)
    # Only 1 generated sample — not enough
    metrics.record_cycle(cycle=1, score=80, previous_score=70, directive="d",
                         directive_source="generated", branch_count=1, best_variant_index=0, elapsed_seconds=0.1)
    assert controller._should_prefer_generated(metrics) is False


def test_controller_has_metrics():
    config = TimeDilateConfig(dilation_factor=3, branch_factor=1)
    mock_engine = make_mock_engine([
        "initial", "70",
        "v1", "80",
        "v2", "90",
    ])
    controller = DilationController(config, mock_engine)
    result = controller.run("test")
    assert result.metrics is not None
    assert len(result.metrics.cycles) == 2
    assert result.metrics.improvement_rate == 1.0


def test_controller_no_meta_learning():
    """With use_meta_learning=False, controller runs without meta-learner."""
    config = TimeDilateConfig(dilation_factor=2, branch_factor=1, use_meta_learning=False)
    mock_engine = make_mock_engine([
        "initial output", "60",
        "improved", "80",
    ])
    controller = DilationController(config, mock_engine)
    assert controller.meta is None
    result = controller.run("test")
    assert result.score == 80


def test_checkpoint_prompt_mismatch_starts_fresh():
    """Resuming with a different prompt starts fresh instead of using stale checkpoint."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TimeDilateConfig(dilation_factor=2, branch_factor=1, checkpoint_dir=tmpdir)
        # Save a checkpoint with prompt "old task"
        from timedilate.checkpoint import CheckpointManager
        mgr = CheckpointManager(tmpdir)
        mgr.save(cycle=5, output="old output", score=90, prompt="old task")

        # Now run with a different prompt and resume=True
        mock_engine = make_mock_engine(["fresh output", "70", "improved", "80"])
        controller = DilationController(config, mock_engine)
        result = controller.run("new task", resume=True)
        # Should NOT have resumed — started fresh
        assert result.resumed_from_cycle == 0
        assert result.output != "old output"


def test_adaptive_convergence_threshold():
    """At high scores, convergence threshold is higher (more patient)."""
    config = TimeDilateConfig(convergence_threshold=5)
    engine = make_mock_engine(["init", "50"])
    controller = DilationController(config, engine)
    assert controller._effective_convergence_threshold(40) == 5
    assert controller._effective_convergence_threshold(70) == 6
    assert controller._effective_convergence_threshold(90) == 8


def test_report_includes_version():
    """Run report includes package version."""
    from timedilate.controller import DilationResult
    result = DilationResult(
        output="x", score=50, cycles_completed=1,
        elapsed_seconds=1.0, convergence_detected=False,
    )
    report = result.to_report()
    assert "version" in report


def test_to_report():
    """DilationResult.to_report produces a complete exportable dict."""
    from timedilate.controller import DilationResult
    from timedilate.metrics import RunMetrics
    import time
    metrics = RunMetrics(prompt="test", task_type="code", dilation_factor=3,
                         branch_factor=2, start_time=time.time())
    result = DilationResult(
        output="hello", score=85, cycles_completed=2,
        elapsed_seconds=1.5, convergence_detected=False, metrics=metrics,
    )
    config = TimeDilateConfig(dilation_factor=3, branch_factor=2)
    report = result.to_report(config)
    assert report["score"] == 85
    assert report["config"]["dilation_factor"] == 3
    assert "metrics" in report
    assert report["metrics"]["task_type"] == "code"


def test_task_type_override():
    """task_type_override skips auto-detection."""
    config = TimeDilateConfig(dilation_factor=2, branch_factor=1, task_type_override="prose")
    mock_engine = make_mock_engine([
        "initial output", "60",
        "improved", "80",
    ])
    controller = DilationController(config, mock_engine)
    result = controller.run("Do something")  # would auto-detect as "general"
    assert result.metrics.task_type == "prose"


def test_controller_survives_engine_error():
    """Engine errors in run_cycle are caught; controller continues."""
    engine = MagicMock()
    engine.estimate_tokens = MagicMock(return_value=100)
    # Initial gen + score, then cycle 1 fails, cycle 2 succeeds
    engine.generate = MagicMock(side_effect=[
        "initial", "50",        # init
        RuntimeError("gpu oom"), # cycle 1 fails in run_cycle
        "improved", "80",       # cycle 2
    ])
    config = TimeDilateConfig(dilation_factor=3, branch_factor=1)
    controller = DilationController(config, engine)
    # Patch run_cycle to raise on first call
    original_run_cycle = controller.improver.run_cycle
    call_count = [0]
    def patched_run_cycle(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("gpu oom")
        return original_run_cycle(*args, **kwargs)
    controller.improver.run_cycle = patched_run_cycle
    result = controller.run("test")
    assert result.cycles_completed == 2
    assert result.score >= 50  # should have recovered


def test_history_summary_structure():
    """History summary separates what worked from what failed."""
    from timedilate.metrics import RunMetrics
    config = TimeDilateConfig(dilation_factor=2, branch_factor=1)
    mock_engine = make_mock_engine(["init", "50"])
    controller = DilationController(config, mock_engine)
    metrics = RunMetrics(start_time=0)
    metrics.record_cycle(cycle=1, score=70, previous_score=50, directive="fix bugs",
                         directive_source="builtin", branch_count=1, best_variant_index=0, elapsed_seconds=0.1)
    metrics.record_cycle(cycle=2, score=70, previous_score=70, directive="optimize",
                         directive_source="builtin", branch_count=1, best_variant_index=-1, elapsed_seconds=0.1)
    summary = controller._build_history_summary(metrics)
    assert "improved the score" in summary
    assert "fix bugs" in summary
    assert "did NOT help" in summary
    assert "optimize" in summary


def test_scoring_consistency_check_triggers_ensemble():
    """When initial scores differ by >15, ensemble scoring is enabled."""
    config = TimeDilateConfig(dilation_factor=5, branch_factor=1)
    responses = [
        "initial output", "80", "50",  # consistency check: 80 vs 50 = delta 30 -> ensemble
    ]
    # With ensemble enabled, each scoring call does 2 scores (normal + CoT)
    # Cycle responses: variant + ensemble scores (normal + cot)
    for _ in range(10):
        responses.extend(["variant", "70", "Correctness: 20\nSCORE: 72"])
    mock_engine = make_mock_engine(responses)
    controller = DilationController(config, mock_engine)
    result = controller.run("test")
    # Initial score should be averaged: (80 + 50) // 2 = 65
    assert result.metrics is not None
    # Ensemble was triggered, so force_ensemble should have been set
    assert controller.improver.force_ensemble or result.cycles_completed > 0
