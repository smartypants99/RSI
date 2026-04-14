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
@click.version_option(package_name="timedilate")
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
@click.option("--reflection", is_flag=True, help="Enable reflect-then-act for high-score refinement")
@click.option("--structured-logs", is_flag=True, help="Emit JSON-structured logs")
@click.option("--score-weights", default=None, help="Custom score weights as JSON, e.g. '{\"correctness\":60,\"completeness\":20,\"quality\":10,\"elegance\":10}'")
@click.option("--task-type", type=click.Choice(["code", "prose", "general"]), default=None, help="Override task type auto-detection")
@click.option("--meta-learning/--no-meta-learning", default=True, help="Enable/disable cross-run meta-learning")
@click.option("--target-score", default=0, type=int, help="Stop early when this score is reached (0 = disabled)")
@click.option("--max-output-tokens", default=0, type=int, help="Cap output length in tokens (0 = no cap)")
@click.option("--quiet", is_flag=True, help="Only output the final result (for piping)")
@click.option("--verbose", is_flag=True, help="Show detailed progress")
@click.option("--report", is_flag=True, help="Auto-generate a timestamped JSON report")
@click.option("--dry-run", is_flag=True, help="Show configuration without running")
def run(prompt, factor, budget, model, draft_model, branches, output_file, metrics_file, resume, reflection, structured_logs, score_weights, task_type, meta_learning, target_score, max_output_tokens, quiet, verbose, report, dry_run):
    """Run time dilation on a prompt."""
    import json as _json
    budget_seconds = parse_budget(budget)
    setup_logging(verbose=verbose, structured=structured_logs)

    parsed_weights = None
    if score_weights:
        try:
            parsed_weights = _json.loads(score_weights)
        except _json.JSONDecodeError:
            raise click.BadParameter(f"Invalid JSON for --score-weights: {score_weights}")

    config = TimeDilateConfig(
        model=model,
        draft_model=draft_model,
        dilation_factor=factor,
        budget_seconds=budget_seconds,
        branch_factor=branches,
        use_reflection=reflection,
        use_meta_learning=meta_learning,
        task_type_override=task_type,
        score_weights=parsed_weights,
        target_score=target_score,
        max_output_tokens=max_output_tokens,
    )

    if not quiet:
        console.print("[bold green]Time Dilation Runtime[/]")
        console.print(f"  Model: {config.model}")
        console.print(f"  Factor: {factor}x ({factor - 1} refinement cycles)")
        console.print(f"  Branches: {branches} per cycle")
        console.print(f"  Budget: {budget} (advisory)")
        if reflection:
            console.print("  Reflection: enabled")
        if task_type:
            console.print(f"  Task type: {task_type}")
        if score_weights:
            console.print(f"  Score weights: {parsed_weights}")
        if target_score > 0:
            console.print(f"  Target score: {target_score}")
        console.print()

    if dry_run:
        console.print("[dim]Dry run — no inference will be performed.[/]")
        return

    result = _run_dilation(prompt, config, resume=resume)

    if quiet:
        click.echo(result.output)
    else:
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
            console.print(f"\n[dim]{result.metrics.summary()}[/]")

    if output_file:
        with open(output_file, "w") as f:
            f.write(result.output)
        console.print(f"\n[dim]Saved to {output_file}[/]")

    # Auto-report: generate timestamped filename
    if report and not metrics_file:
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = f"timedilate_report_{ts}.json"

    if metrics_file:
        import json as _json2
        report_data = result.to_report(config)
        from pathlib import Path
        Path(metrics_file).write_text(_json2.dumps(report_data, indent=2))
        console.print(f"[dim]Run report saved to {metrics_file}[/]")


@main.command()
@click.option("--checkpoint-dir", default=".timedilate_checkpoints", help="Checkpoint directory")
def status(checkpoint_dir):
    """Show status of checkpoints and last run."""
    from timedilate.checkpoint import CheckpointManager
    mgr = CheckpointManager(checkpoint_dir)
    checkpoints = mgr.list_checkpoints()
    if not checkpoints:
        console.print("[dim]No checkpoints found.[/]")
        return
    console.print(f"[bold]Found {len(checkpoints)} checkpoint(s)[/]")
    for cp in checkpoints:
        ts = cp.get("timestamp", 0)
        import datetime
        dt = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "unknown"
        console.print(
            f"  Cycle {cp['cycle']:3d} | Score: {cp['score']:3d} | "
            f"Task: {cp.get('task_type', '?'):8s} | {dt}"
        )
    latest = checkpoints[-1]
    console.print(f"\n[bold]Latest:[/] cycle {latest['cycle']}, score {latest['score']}")
    if latest.get("prompt"):
        console.print(f"  Prompt: {latest['prompt'][:80]}...")


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


@main.command()
@click.argument("report_a", type=click.Path(exists=True))
@click.argument("report_b", type=click.Path(exists=True))
def compare(report_a, report_b):
    """Compare two run reports side by side."""
    import json as _json
    a = _json.loads(open(report_a).read())
    b = _json.loads(open(report_b).read())

    console.print(f"[bold]Comparing runs[/]")
    console.print(f"  A: {report_a}")
    console.print(f"  B: {report_b}")
    console.print()

    fields = [
        ("Score", "score", ""),
        ("Cycles", "cycles_completed", ""),
        ("Time", "elapsed_seconds", "s"),
        ("Convergence", "convergence_detected", ""),
    ]
    console.print(f"{'Metric':<20} {'A':>12} {'B':>12} {'Delta':>12}")
    console.print("-" * 58)
    for label, key, suffix in fields:
        va = a.get(key, "?")
        vb = b.get(key, "?")
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            delta = vb - va
            sign = "+" if delta > 0 else ""
            console.print(f"{label:<20} {va:>11}{suffix} {vb:>11}{suffix} {sign}{delta:>10}{suffix}")
        else:
            console.print(f"{label:<20} {str(va):>12} {str(vb):>12}")

    # Compare metrics if present
    for report_label, report in [("A", a), ("B", b)]:
        m = report.get("metrics", {})
        if m:
            console.print(f"\n[bold]{report_label} metrics:[/]")
            for key in ["improvement_rate", "efficiency", "peak_score", "avg_cycle_time",
                         "output_bloat_ratio", "score_inflation_rate"]:
                val = m.get(key)
                if val is not None:
                    if isinstance(val, float):
                        console.print(f"  {key}: {val:.3f}")
                    else:
                        console.print(f"  {key}: {val}")


@main.command()
@click.argument("report_dir", default=".", type=click.Path(exists=True))
@click.option("--limit", default=10, help="Max reports to show")
def history(report_dir, limit):
    """List past run reports from a directory."""
    import json as _json
    from pathlib import Path

    reports = []
    for f in Path(report_dir).glob("*.json"):
        try:
            data = _json.loads(f.read_text())
            if "score" in data and "elapsed_seconds" in data:
                data["_file"] = f.name
                reports.append(data)
        except (ValueError, KeyError):
            continue

    if not reports:
        console.print("[dim]No run reports found.[/]")
        return

    reports.sort(key=lambda r: r.get("timestamp", 0), reverse=True)
    reports = reports[:limit]

    console.print(f"[bold]Run History ({len(reports)} reports)[/]")
    console.print(f"{'File':<30} {'Score':>6} {'Cycles':>7} {'Time':>8} {'Task':>8}")
    console.print("-" * 65)
    for r in reports:
        console.print(
            f"{r['_file']:<30} {r.get('score', '?'):>6} "
            f"{r.get('cycles_completed', '?'):>7} "
            f"{r.get('elapsed_seconds', 0):>7.1f}s "
            f"{r.get('task_type', '?'):>8}"
        )


if __name__ == "__main__":
    main()
