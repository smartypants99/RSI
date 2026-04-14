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
    config = TimeDilateConfig(dilation_factor=6, branch_factor=1, convergence_threshold=3)
    responses = ["initial output", "80"]
    # 5 cycles where nothing improves
    for _ in range(5):
        responses.extend(["same output", "70"])
    mock_engine = make_mock_engine(responses)
    controller = DilationController(config, mock_engine)
    result = controller.run("test")
    assert result.cycles_completed == 5
    assert result.convergence_detected


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
        "initial", "80",
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

        # Only need responses for cycles 3 and 4 (resuming from 2)
        mock_engine = make_mock_engine([
            "v3", "85",
            "v4", "90",
        ])
        controller = DilationController(config, mock_engine)
        result = controller.run("test", resume=True)
        assert result.resumed_from_cycle == 2
        assert result.cycles_completed == 4
        assert result.score == 90


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
