"""Benchmark harness for measuring time dilation effectiveness.

Run with a real model to measure quality improvement across dilation factors.
"""
import json
import time
from dataclasses import dataclass
from pathlib import Path

from timedilate.config import TimeDilateConfig
from timedilate.controller import DilationController


BENCHMARK_PROMPTS = [
    {
        "name": "sort_function",
        "prompt": "Write a Python function that sorts a list of integers using merge sort.",
        "type": "code",
    },
    {
        "name": "fibonacci",
        "prompt": "Write a Python function that returns the nth Fibonacci number efficiently.",
        "type": "code",
    },
    {
        "name": "web_scraper",
        "prompt": "Write a Python script that scrapes a webpage and extracts all links.",
        "type": "code",
    },
    {
        "name": "climate_essay",
        "prompt": "Write a short essay explaining the greenhouse effect to a high school student.",
        "type": "prose",
    },
    {
        "name": "api_design",
        "prompt": "Design a REST API for a todo list application. Include endpoints, methods, and example request/response bodies.",
        "type": "general",
    },
]


@dataclass
class BenchmarkResult:
    prompt_name: str
    dilation_factor: int
    score: int
    cycles_completed: int
    elapsed_seconds: float
    improvement_rate: float
    output_length: int


def run_benchmark(
    engine,
    factors: list[int] | None = None,
    prompts: list[dict] | None = None,
    output_dir: str = "benchmark_results",
) -> list[BenchmarkResult]:
    """Run benchmark across multiple dilation factors and prompts."""
    if factors is None:
        factors = [1, 2, 5, 10]
    if prompts is None:
        prompts = BENCHMARK_PROMPTS

    results = []
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for prompt_info in prompts:
        for factor in factors:
            config = TimeDilateConfig(
                dilation_factor=factor,
                branch_factor=min(3, factor),
            )
            controller = DilationController(config, engine)

            start = time.time()
            result = controller.run(prompt_info["prompt"])
            elapsed = time.time() - start

            bench_result = BenchmarkResult(
                prompt_name=prompt_info["name"],
                dilation_factor=factor,
                score=result.score,
                cycles_completed=result.cycles_completed,
                elapsed_seconds=elapsed,
                improvement_rate=result.metrics.improvement_rate if result.metrics else 0,
                output_length=len(result.output),
            )
            results.append(bench_result)

    # Save results
    results_data = [
        {
            "prompt": r.prompt_name,
            "factor": r.dilation_factor,
            "score": r.score,
            "cycles": r.cycles_completed,
            "elapsed_s": round(r.elapsed_seconds, 2),
            "improvement_rate": round(r.improvement_rate, 3),
            "output_chars": r.output_length,
        }
        for r in results
    ]
    (output_path / "results.json").write_text(json.dumps(results_data, indent=2))

    return results


def format_results(results: list[BenchmarkResult]) -> str:
    """Format benchmark results as a readable table."""
    lines = [
        f"{'Prompt':<20} {'Factor':>6} {'Score':>5} {'Cycles':>6} {'Time':>8} {'Imp%':>5}",
        "-" * 60,
    ]
    for r in results:
        lines.append(
            f"{r.prompt_name:<20} {r.dilation_factor:>5}x {r.score:>5} {r.cycles_completed:>6} "
            f"{r.elapsed_seconds:>7.1f}s {r.improvement_rate:>4.0%}"
        )
    return "\n".join(lines)
