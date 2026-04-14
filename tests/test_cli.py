import tempfile
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
from timedilate.cli import main, parse_budget


def test_parse_budget_seconds():
    assert parse_budget("5s") == 5.0
    assert parse_budget("30s") == 30.0


def test_parse_budget_minutes():
    assert parse_budget("5m") == 300.0


def test_parse_budget_hours():
    assert parse_budget("2h") == 7200.0


def test_parse_budget_bare_number():
    assert parse_budget("10") == 10.0


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output
    assert "benchmark" in result.output


def test_run_subcommand_help():
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--help"])
    assert result.exit_code == 0
    assert "budget" in result.output
    assert "factor" in result.output


def test_run_subcommand_parses_args():
    runner = CliRunner()
    with patch("timedilate.cli._run_dilation") as mock_run:
        mock_metrics = MagicMock()
        mock_metrics.improvement_rate = 0.8
        mock_metrics.total_improvement = 35
        mock_metrics.avg_cycle_time = 0.5
        mock_metrics.score_inflation_rate = 0.5
        mock_metrics.cycles = [MagicMock(), MagicMock(), MagicMock()]
        mock_run.return_value = MagicMock(
            output="result",
            score=85,
            cycles_completed=5,
            elapsed_seconds=2.5,
            convergence_detected=False,
            interrupted=False,
            resumed_from_cycle=0,
            metrics=mock_metrics,
        )
        result = runner.invoke(main, ["run", "Write hello", "--factor", "5", "--budget", "10s"])
        assert result.exit_code == 0
        mock_run.assert_called_once()


def test_status_no_checkpoints():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = runner.invoke(main, ["status", "--checkpoint-dir", tmpdir])
        assert result.exit_code == 0
        assert "No checkpoints" in result.output


def test_status_with_checkpoints():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        from timedilate.checkpoint import CheckpointManager
        mgr = CheckpointManager(tmpdir)
        mgr.save(cycle=5, output="test", score=80, prompt="write sort", task_type="code")
        result = runner.invoke(main, ["status", "--checkpoint-dir", tmpdir])
        assert result.exit_code == 0
        assert "Cycle   5" in result.output
        assert "80" in result.output


def test_dry_run():
    runner = CliRunner()
    result = runner.invoke(main, ["run", "Write hello", "--factor", "3", "--dry-run"])
    assert result.exit_code == 0
    assert "Dry run" in result.output
    assert "Factor: 3x" in result.output


def test_history_no_reports():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = runner.invoke(main, ["history", tmpdir])
        assert result.exit_code == 0
        assert "No run reports" in result.output


def test_history_with_reports():
    import json
    from pathlib import Path
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        report = {"score": 85, "elapsed_seconds": 2.5, "cycles_completed": 5, "task_type": "code", "timestamp": 1000}
        Path(tmpdir, "run1.json").write_text(json.dumps(report))
        result = runner.invoke(main, ["history", tmpdir])
        assert result.exit_code == 0
        assert "run1.json" in result.output
        assert "85" in result.output


def test_benchmark_subcommand_help():
    runner = CliRunner()
    result = runner.invoke(main, ["benchmark", "--help"])
    assert result.exit_code == 0
    assert "factors" in result.output
    assert "output-dir" in result.output
