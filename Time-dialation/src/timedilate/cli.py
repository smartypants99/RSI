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
    """AI Time Dilation Runtime — give AI more thinking time."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command()
@click.argument("prompt")
@click.option("--factor", default=1.0, type=float, help="Dilation factor (e.g., 10, 1000, 1000000)")
@click.option("--model", default="Qwen/Qwen3-8B", help="Model to use")
@click.option("--time-budget", default=None, type=float, help="Wall-clock seconds the AI gets (e.g., 5 = 5 real seconds)")
@click.option("--max-tokens", default=4096, type=int, help="Max output tokens")
@click.option("--output", "output_file", default=None, help="Save output to file")
@click.option("--report", is_flag=True, help="Save JSON report")
@click.option("--quiet", is_flag=True, help="Only output the result")
@click.option("--verbose", is_flag=True, help="Detailed logging")
@click.option("--dry-run", is_flag=True, help="Show config without running")
def run(prompt, factor, model, time_budget, max_tokens, output_file, report, quiet, verbose, dry_run):
    """Run time-dilated inference on a prompt.

    The dilation factor controls how much subjective thinking time the AI gets.
    Combined with --time-budget, it gives the AI factor * budget seconds of thinking.

    Examples:
        timedilate run "Write a sort function" --factor 1000
        timedilate run "Solve this" --factor 1000000 --time-budget 5
    """
    setup_logging(verbose=verbose)

    config = TimeDilateConfig(
        model=model,
        dilation_factor=factor,
        max_tokens=max_tokens,
        time_budget_seconds=time_budget,
    )

    if not quiet:
        console.print("[bold green]Time Dilation Runtime[/]")
        console.print(config.describe())
        console.print()

    if dry_run:
        console.print("[dim]Dry run — showing configuration only.[/]")
        return

    def on_cycle(cycle, total, score, elapsed):
        if not quiet:
            console.print(f"  [dim]Cycle {cycle}/{total} — score: {score} — {elapsed:.1f}s[/]")

    controller = DilationController(config)
    result = controller.run(prompt, on_cycle=on_cycle)

    if quiet:
        click.echo(result.output)
    else:
        console.print()
        console.print(result.output)
        console.print()
        console.print(
            f"[bold green]Done![/] {result.cycles_completed} cycles in {result.elapsed_seconds:.1f}s "
            f"— final score: [bold]{result.score}[/]/100"
        )
        if result.convergence_resets > 0:
            console.print(f"[dim]Fresh approach attempts: {result.convergence_resets}[/]")

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
@click.option("--factor", default=1000.0, type=float, help="Dilation factor to inspect")
def explain(factor):
    """Explain what happens at a given dilation factor."""
    config = TimeDilateConfig(dilation_factor=factor)
    console.print(f"[bold]Time dilation at {factor}x:[/]")
    console.print()
    console.print(config.describe())
    console.print()
    console.print(f"The AI will get [bold]{config.num_cycles}[/] reasoning cycles.")
    console.print("Each cycle: score -> critique -> refine -> repeat.")
    console.print("No quality loss. No shortcuts. Just more thinking.")


if __name__ == "__main__":
    main()
