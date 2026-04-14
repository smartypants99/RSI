import re
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from timedilate.config import TimeDilateConfig
from timedilate.engine import InferenceEngine
from timedilate.controller import DilationController, DilationResult
from timedilate.logging_config import setup_logging

console = Console()


def parse_budget(value: str) -> float:
    """Parse budget strings like '5s', '30s', '5m', '2h' or bare numbers."""
    match = re.match(r"^(\d+(?:\.\d+)?)\s*(s|m|h)?$", str(value).strip())
    if not match:
        raise click.BadParameter(f"Invalid budget format: {value}")
    num = float(match.group(1))
    unit = match.group(2) or "s"
    multipliers = {"s": 1, "m": 60, "h": 3600}
    return num * multipliers[unit]


def _run_dilation(prompt: str, config: TimeDilateConfig, resume: bool = False) -> DilationResult:
    engine = InferenceEngine(config)
    controller = DilationController(config, engine)
    total_cycles = config.dilation_factor - 1

    if total_cycles <= 0:
        result = controller.run(prompt)
        console.print(result.output)
        return result

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[bold]{task.completed}/{task.total} cycles"),
        TextColumn("score: {task.fields[score]}"),
        TextColumn("[dim]{task.fields[eta]}[/dim]"),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Dilating time...", total=total_cycles, score=0, eta=""
        )

        def on_cycle(cycle, total, score, elapsed):
            if cycle > 0:
                per_cycle = elapsed / cycle
                remaining = (total - cycle) * per_cycle
                eta = f"~{remaining:.0f}s left"
            else:
                eta = ""
            budget_pct = (elapsed / config.budget_seconds * 100) if config.budget_seconds > 0 else 0
            eta_str = f"{eta} | {elapsed:.1f}s/{config.budget_seconds:.0f}s ({budget_pct:.0f}%)"
            progress.update(task, completed=cycle, score=score, eta=eta_str)

        result = controller.run(prompt, on_cycle=on_cycle, resume=resume)

    return result


# Keep run_dilation as public alias for backward compat with tests
run_dilation = _run_dilation


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    """AI Time Dilation Runtime -- make AI think longer in less time."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command()
@click.argument("prompt")
@click.option("--factor", default=2, help="Dilation factor (e.g., 2, 100, 1000000)")
@click.option("--budget", default="30s", help="Advisory time budget (e.g., 5s, 30s, 5m)")
@click.option("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model name or path")
@click.option("--draft-model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Draft model for speculative decoding")
@click.option("--branches", default=3, help="Branch factor per cycle")
@click.option("--output", "output_file", default=None, help="Save output to file")
@click.option("--metrics", "metrics_file", default=None, help="Save run metrics to JSON file")
@click.option("--resume", is_flag=True, help="Resume from last checkpoint")
@click.option("--verbose", is_flag=True, help="Show detailed progress")
def run(prompt, factor, budget, model, draft_model, branches, output_file, metrics_file, resume, verbose):
    """Run time dilation on a prompt."""
    budget_seconds = parse_budget(budget)
    setup_logging(verbose=verbose)

    config = TimeDilateConfig(
        model=model,
        draft_model=draft_model,
        dilation_factor=factor,
        budget_seconds=budget_seconds,
        branch_factor=branches,
    )

    console.print("[bold green]Time Dilation Runtime[/]")
    console.print(f"  Model: {config.model}")
    console.print(f"  Factor: {factor}x ({factor - 1} refinement cycles)")
    console.print(f"  Branches: {branches} per cycle")
    console.print(f"  Budget: {budget} (advisory)")
    console.print()

    result = _run_dilation(prompt, config, resume=resume)

    if resume and hasattr(result, 'resumed_from_cycle') and result.resumed_from_cycle > 0:
        console.print(f"[dim]Resumed from cycle {result.resumed_from_cycle}[/]")

    console.print()
    console.print(
        f"[bold green]Complete![/] {result.cycles_completed} cycles in "
        f"{result.elapsed_seconds:.1f}s | Score: {result.score}/100"
    )
    if result.interrupted:
        console.print("[yellow]Interrupted -- returning best result so far[/]")
    if result.convergence_detected:
        console.print("[yellow]Note: convergence detected -- output may have plateaued[/]")
    console.print()
    console.print(result.output)

    if result.metrics:
        m = result.metrics
        console.print(f"  Improvement rate: {m.improvement_rate:.0%}")
        console.print(f"  Total improvement: +{m.total_improvement} points")

    if output_file:
        with open(output_file, "w") as f:
            f.write(result.output)
        console.print(f"\n[dim]Saved to {output_file}[/]")

    if metrics_file and result.metrics:
        result.metrics.save(metrics_file)
        console.print(f"[dim]Metrics saved to {metrics_file}[/]")


@main.command()
@click.option("--factors", default="1,2,5,10", help="Comma-separated dilation factors to test")
@click.option("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model name or path")
@click.option("--draft-model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Draft model for speculative decoding")
@click.option("--output-dir", default="benchmark_results", help="Directory for results")
@click.option("--verbose", is_flag=True, help="Show detailed progress")
def benchmark(factors, model, draft_model, output_dir, verbose):
    """Run benchmark across multiple dilation factors."""
    from timedilate.benchmark import run_benchmark, format_results

    setup_logging(verbose=verbose)
    factor_list = [int(f.strip()) for f in factors.split(",")]

    console.print("[bold green]Time Dilation Benchmark[/]")
    console.print(f"  Model: {model}")
    console.print(f"  Factors: {factor_list}")
    console.print()

    engine = InferenceEngine(TimeDilateConfig(model=model, draft_model=draft_model))
    results = run_benchmark(engine, factors=factor_list, output_dir=output_dir)

    console.print(format_results(results))
    console.print(f"\n[dim]Results saved to {output_dir}/results.json[/]")


if __name__ == "__main__":
    main()
