import tempfile
from unittest.mock import MagicMock
from pathlib import Path
from timedilate.benchmark import run_benchmark, format_results, BENCHMARK_PROMPTS


def make_mock_engine(default_score="75"):
    """Mock engine that returns predictable outputs."""
    engine = MagicMock()
    call_count = [0]

    def mock_generate(prompt, **kwargs):
        call_count[0] += 1
        if "score" in prompt.lower() or "rate" in prompt.lower():
            return default_score
        if "compare" in prompt.lower() or "which is better" in prompt.lower():
            return "B"
        return f"mock output #{call_count[0]}"

    engine.generate = MagicMock(side_effect=mock_generate)
    engine.estimate_tokens = MagicMock(return_value=100)
    return engine


def test_benchmark_runs():
    engine = make_mock_engine()
    with tempfile.TemporaryDirectory() as tmpdir:
        results = run_benchmark(
            engine,
            factors=[1, 2],
            prompts=BENCHMARK_PROMPTS[:1],
            output_dir=tmpdir,
        )
        assert len(results) == 2  # 1 prompt x 2 factors
        assert results[0].dilation_factor == 1
        assert results[1].dilation_factor == 2
        assert (Path(tmpdir) / "results.json").exists()


def test_benchmark_higher_factor_completes_more_cycles():
    engine = make_mock_engine()
    with tempfile.TemporaryDirectory() as tmpdir:
        results = run_benchmark(
            engine,
            factors=[1, 5],
            prompts=BENCHMARK_PROMPTS[:1],
            output_dir=tmpdir,
        )
        assert results[0].cycles_completed == 0
        assert results[1].cycles_completed > 0


def test_format_results():
    engine = make_mock_engine()
    with tempfile.TemporaryDirectory() as tmpdir:
        results = run_benchmark(
            engine,
            factors=[1, 2],
            prompts=BENCHMARK_PROMPTS[:1],
            output_dir=tmpdir,
        )
        table = format_results(results)
        assert "sort_function" in table
        assert "1x" in table
        assert "2x" in table
