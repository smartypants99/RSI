"""Tests for the CLI."""
import sys
from unittest.mock import MagicMock

# Mock vllm — reuse if already mocked by test_engine
if "vllm" not in sys.modules or not isinstance(sys.modules["vllm"], MagicMock):
    sys.modules["vllm"] = MagicMock()
mock_vllm = sys.modules["vllm"]

from click.testing import CliRunner
from timedilate.cli import main


def _setup_mock():
    mock_vllm.reset_mock()
    mock_output = MagicMock()
    mock_output.outputs = [MagicMock(text="generated result")]
    mock_vllm.LLM.return_value.generate.return_value = [mock_output]
    mock_vllm.SamplingParams.return_value = MagicMock()


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output
    assert "benchmark" in result.output
    assert "explain" in result.output


def test_run_help():
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--help"])
    assert result.exit_code == 0
    assert "--factor" in result.output
    assert "--model" in result.output


def test_explain_command():
    runner = CliRunner()
    result = runner.invoke(main, ["explain", "--factor", "1000"])
    assert result.exit_code == 0
    assert "1000" in result.output


def test_dry_run():
    runner = CliRunner()
    result = runner.invoke(main, ["run", "Write hello", "--factor", "100", "--dry-run"])
    assert result.exit_code == 0
    assert "Dry run" in result.output


def test_run_quiet():
    _setup_mock()
    runner = CliRunner()
    result = runner.invoke(main, ["run", "Say hi", "--factor", "1", "--quiet"])
    assert result.exit_code == 0
    assert "generated result" in result.output


def test_run_with_output_file(tmp_path):
    _setup_mock()
    out = tmp_path / "out.txt"
    runner = CliRunner()
    result = runner.invoke(main, ["run", "test", "--factor", "1", "--quiet", "--output", str(out)])
    assert result.exit_code == 0
    assert out.read_text() == "generated result"


def test_benchmark_help():
    runner = CliRunner()
    result = runner.invoke(main, ["benchmark", "--help"])
    assert result.exit_code == 0
    assert "--factors" in result.output
