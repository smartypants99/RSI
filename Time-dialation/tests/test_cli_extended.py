"""Extended CLI tests covering flags and error paths."""
import sys
import json
from unittest.mock import MagicMock
from pathlib import Path

if "vllm" not in sys.modules or not isinstance(sys.modules["vllm"], MagicMock):
    sys.modules["vllm"] = MagicMock()
mock_vllm = sys.modules["vllm"]

from click.testing import CliRunner
from timedilate.cli import main


def _setup_mock():
    mock_vllm.reset_mock()
    out = MagicMock()
    out.outputs = [MagicMock(text="generated result")]
    mock_vllm.LLM.return_value.generate.return_value = [out]
    mock_vllm.SamplingParams.return_value = MagicMock()


def test_version_flag():
    """--version only works when package is installed; skip if not."""
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    # Accept either a clean exit or a package-not-installed RuntimeError
    if result.exit_code != 0:
        assert "not installed" in str(result.exception)
    else:
        assert result.exit_code == 0


def test_explain_includes_cycles_text():
    runner = CliRunner()
    result = runner.invoke(main, ["explain", "--factor", "100"])
    assert result.exit_code == 0
    assert "100" in result.output
    assert "cycles" in result.output.lower()


def test_explain_default_factor():
    runner = CliRunner()
    result = runner.invoke(main, ["explain"])
    assert result.exit_code == 0
    assert "1000" in result.output


def test_run_no_prompt_fails():
    runner = CliRunner()
    result = runner.invoke(main, ["run"])
    assert result.exit_code != 0


def test_run_dry_run_shows_config_without_invoking_engine():
    _setup_mock()
    runner = CliRunner()
    result = runner.invoke(main, ["run", "hello", "--factor", "100", "--dry-run"])
    assert result.exit_code == 0
    assert "Dry run" in result.output
    # Engine not invoked in dry run
    mock_vllm.LLM.return_value.generate.assert_not_called()


def test_run_model_flag_propagates():
    _setup_mock()
    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "hello", "--factor", "1", "--model", "foo/bar", "--quiet"]
    )
    assert result.exit_code == 0
    kwargs = mock_vllm.LLM.call_args.kwargs
    assert kwargs["model"] == "foo/bar"


def test_run_time_budget_flag_accepted():
    _setup_mock()
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["run", "hello", "--factor", "1", "--time-budget", "0.01", "--quiet"],
    )
    assert result.exit_code == 0


def test_run_max_tokens_flag():
    _setup_mock()
    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "hi", "--factor", "1", "--max-tokens", "77", "--quiet"]
    )
    assert result.exit_code == 0
    kwargs = mock_vllm.SamplingParams.call_args.kwargs
    assert kwargs["max_tokens"] == 77


def test_run_output_file_writes_result(tmp_path):
    _setup_mock()
    out_path = tmp_path / "result.txt"
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["run", "hi", "--factor", "1", "--output", str(out_path), "--quiet"],
    )
    assert result.exit_code == 0
    assert out_path.read_text() == "generated result"


def test_run_report_flag_writes_json(tmp_path):
    _setup_mock()
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(
            main, ["run", "hi", "--factor", "1", "--report", "--quiet"]
        )
        assert result.exit_code == 0
        reports = list(Path(".").glob("timedilate_report_*.json"))
        assert len(reports) == 1
        data = json.loads(reports[0].read_text())
        assert "dilation_factor" in data
        assert "version" in data


def test_run_verbose_flag_accepted():
    _setup_mock()
    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "hi", "--factor", "1", "--verbose", "--quiet"]
    )
    assert result.exit_code == 0


def test_run_nonquiet_shows_config_description():
    _setup_mock()
    runner = CliRunner()
    result = runner.invoke(main, ["run", "hi", "--factor", "1"])
    assert result.exit_code == 0
    assert "Time Dilation" in result.output


def test_run_invalid_factor_negative_rejected():
    """Factor < 1.0 triggers ConfigError from validate() in controller init."""
    _setup_mock()
    runner = CliRunner()
    result = runner.invoke(main, ["run", "hi", "--factor", "0.5", "--quiet"])
    # Non-zero exit expected due to ConfigError
    assert result.exit_code != 0


def test_main_no_subcommand_shows_help():
    runner = CliRunner()
    result = runner.invoke(main, [])
    assert result.exit_code == 0
    assert "run" in result.output
    assert "explain" in result.output
