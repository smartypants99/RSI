"""CLI for the Time Dilation Runtime."""
import click
from rich.console import Console
from timedilate.config import TimeDilateConfig
from timedilate.engine import DilationEngine
from timedilate.controller import DilationController
from timedilate.logging_config import setup_logging

console = Console()


@click.group(invoke_without_command=True)
@click.version_option(package_name="timedilate")
@click.pass_context
def main(ctx):
    """AI Time Dilation Runtime — make AI inference faster."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command()
@click.argument("prompt")
@click.option("--factor", default=1.0, type=float, help="Dilation factor (e.g., 10, 1000, 1000000)")
@click.option("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Base model")
@click.option("--draft-model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Draft model for speculative decoding")
@click.option("--max-tokens", default=4096, type=int, help="Max output tokens")
@click.option("--output", "output_file", default=None, help="Save output to file")
@click.option("--report", is_flag=True, help="Save JSON report")
@click.option("--quiet", is_flag=True, help="Only output the result")
@click.option("--verbose", is_flag=True, help="Detailed logging")
@click.option("--dry-run", is_flag=True, help="Show acceleration config without running")
def run(prompt, factor, model, draft_model, max_tokens, output_file, report, quiet, verbose, dry_run):
    """Run time-dilated inference on a prompt.

    Examples:
        timedilate run "Write a sort function" --factor 100
        timedilate run "Explain quantum physics" --factor 1000000
    """
    setup_logging(verbose=verbose)

    config = TimeDilateConfig(
        model=model,
        draft_model=draft_model,
        dilation_factor=factor,
        max_tokens=max_tokens,
    )

    # Auto-configure acceleration
    config = config.auto_configure()

    if not quiet:
        console.print("[bold green]Time Dilation Runtime[/]")
        console.print(f"  Target speedup: {factor}x")
        console.print(f"  Base model: {model}")
        console.print()
        console.print(config.describe_acceleration())
        console.print()

    if dry_run:
        console.print("[dim]Dry run — showing configuration only.[/]")
        return

    controller = DilationController(config)
    result = controller.run(prompt)

    if quiet:
        click.echo(result.output)
    else:
        console.print(result.output)
        console.print()
        console.print(
            f"[bold green]Done![/] {result.actual_latency:.3f}s "
            f"(estimated {result.base_latency_estimate:.1f}s without dilation) "
            f"= [bold]{result.achieved_speedup:.1f}x[/] speedup"
        )
        if result.achieved_speedup < factor * 0.5:
            console.print(
                f"[yellow]Note: achieved {result.achieved_speedup:.1f}x "
                f"vs target {factor}x — hardware-limited[/]"
            )

    if output_file:
        with open(output_file, "w") as f:
            f.write(result.output)
        if not quiet:
            console.print(f"[dim]Saved to {output_file}[/]")

    if report:
        import json
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"timedilate_report_{ts}.json"
        report_data = result.to_report(config)
        from pathlib import Path
        Path(report_file).write_text(json.dumps(report_data, indent=2))
        if not quiet:
            console.print(f"[dim]Report saved to {report_file}[/]")


@main.command()
@click.argument("prompt")
@click.option("--factors", default="1,10,100,1000", help="Comma-separated dilation factors")
@click.option("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Base model")
@click.option("--verbose", is_flag=True, help="Detailed logging")
def benchmark(prompt, factors, model, verbose):
    """Benchmark a prompt across multiple dilation factors."""
    setup_logging(verbose=verbose)
    factor_list = [float(f.strip()) for f in factors.split(",")]

    console.print("[bold green]Time Dilation Benchmark[/]")
    console.print(f"  Prompt: {prompt[:80]}...")
    console.print(f"  Factors: {factor_list}")
    console.print()

    config = TimeDilateConfig(model=model)
    controller = DilationController(config)
    results = controller.benchmark(prompt, factor_list)

    console.print(f"{'Factor':>10} {'Latency':>10} {'Speedup':>10} {'Model':>30}")
    console.print("-" * 65)
    for r in results:
        console.print(
            f"{r.dilation_factor:>9}x {r.actual_latency:>9.3f}s "
            f"{r.achieved_speedup:>9.1f}x {r.model_used:>30}"
        )


@main.command()
@click.option("--factor", default=1000.0, type=float, help="Dilation factor to inspect")
def explain(factor):
    """Explain what acceleration techniques would be used for a given factor."""
    config = TimeDilateConfig(dilation_factor=factor).auto_configure()
    console.print(f"[bold]Acceleration plan for {factor}x dilation:[/]")
    console.print()
    console.print(config.describe_acceleration())


if __name__ == "__main__":
    main()
