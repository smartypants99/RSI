"""CLI for the Time Dilation Runtime."""
import click
from rich.console import Console
from timedilate.config import TimeDilateConfig, ConfigError
from timedilate.controller import DilationController
from timedilate.logging_config import setup_logging

console = Console()

SPARK_CHARS = "▁▂▃▄▅▆▇█"


def sparkline(values: list[int], lo: int = 0, hi: int = 100) -> str:
    if not values:
        return ""
    span = max(1, hi - lo)
    out = []
    for v in values:
        idx = int((max(lo, min(hi, v)) - lo) / span * (len(SPARK_CHARS) - 1))
        out.append(SPARK_CHARS[idx])
    return "".join(out)


def format_subjective_time(seconds: float) -> str:
    """Break subjective time into days/weeks/months/years for human readability."""
    minute, hour, day = 60, 3600, 86400
    week, month, year = 7 * day, 30 * day, 365 * day
    if seconds < minute:
        return f"{seconds:.1f} seconds"
    parts = [
        (year, "year"),
        (month, "month"),
        (week, "week"),
        (day, "day"),
        (hour, "hour"),
        (minute, "minute"),
    ]
    for unit_s, name in parts:
        if seconds >= unit_s:
            val = seconds / unit_s
            return f"{val:,.2f} {name}{'s' if val != 1 else ''}"
    return f"{seconds:.1f} seconds"


@click.group(invoke_without_command=True)
@click.version_option(package_name="timedilate")
@click.pass_context
def main(ctx):
    """AI Time Dilation Runtime — give AI more subjective thinking time.

    \b
    Examples:
      timedilate run "Write a quicksort" --factor 100
      timedilate run "Solve this" --factor 1000000 --time-budget 5
      timedilate explain --factor 1000000 --time-budget 5
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command()
@click.argument("prompt")
# Core
@click.option("--factor", default=1.0, type=float,
              help="Dilation factor — subjective time multiplier (e.g., 10, 1000, 1000000).")
@click.option("--model", default="Qwen/Qwen3-8B", help="Model name or path.")
@click.option("--time-budget", default=None, type=float,
              help="Wall-clock seconds the AI may spend (e.g., 5). With --factor, gives subjective time.")
@click.option("--max-tokens", default=4096, type=int, help="Max output tokens per generation.")
# Hardware / vLLM
@click.option("--gpu-mem-util", "gpu_mem_util", default=None, type=float,
              help="Fraction of GPU memory vLLM may reserve (0.1-0.99). "
                   "Engine auto-retries init at 0.60 -> 0.45 -> 0.30 on OOM.")
@click.option("--max-model-len", default=None, type=int,
              help="Cap model context length (reduces KV cache memory).")
@click.option("--dtype", default=None,
              type=click.Choice(["auto", "float16", "bfloat16", "float32", "half", "bf16"],
                                case_sensitive=False),
              help="Model weights dtype.")
@click.option("--enforce-eager", is_flag=True, default=False,
              help="Skip CUDA graph capture (saves VRAM; slower).")
@click.option("--swap-space", "swap_space", default=None, type=float,
              help="CPU swap space for KV cache overflow (GiB).")
# Sampling
@click.option("--temperature", default=None, type=float, help="Sampling temperature (0.0-2.0).")
@click.option("--seed", default=None, type=int, help="RNG seed for deterministic sampling.")
# Dilation strategy
@click.option("--branch-factor", default=None, type=int,
              help="Parallel candidate branches per cycle (>=1).")
@click.option("--patience", "patience", default=None, type=int,
              help="Cycles without improvement before trying a fresh approach.")
@click.option("--early-stop", "early_stop", default=None, type=int,
              help="Stop once score >= this (0-100).")
@click.option("--no-critique", is_flag=True, default=False,
              help="Disable self-critique step each cycle.")
@click.option("--no-cot", is_flag=True, default=False,
              help="Disable chain-of-thought reasoning in critique.")
# Output
@click.option("--output", "output_file", default=None, help="Save final output to file.")
@click.option("--report", is_flag=True, help="Save JSON report alongside run.")
@click.option("--stream-progress", is_flag=True, default=False,
              help="Show a live per-cycle score sparkline.")
@click.option("--quiet", is_flag=True, help="Print only the final result.")
@click.option("--verbose", is_flag=True, help="Detailed logging.")
@click.option("--dry-run", is_flag=True, help="Show resolved config and exit.")
def run(prompt, factor, model, time_budget, max_tokens,
        gpu_mem_util, max_model_len, dtype, enforce_eager, swap_space,
        temperature, seed,
        branch_factor, patience, early_stop, no_critique, no_cot,
        output_file, report, stream_progress, quiet, verbose, dry_run):
    """Run time-dilated inference on a prompt.

    \b
    Examples:
      timedilate run "Write a sort function" --factor 1000
      timedilate run "Solve this" --factor 1000000 --time-budget 5
      timedilate run "Prove X" --factor 500 --branch-factor 4 --patience 10
      timedilate run "Essay" --factor 200 --no-cot --temperature 0.3 --seed 42
    """
    setup_logging(verbose=verbose)

    kwargs = dict(
        model=model,
        dilation_factor=factor,
        max_tokens=max_tokens,
        time_budget_seconds=time_budget,
    )
    if gpu_mem_util is not None:
        kwargs["gpu_memory_utilization"] = gpu_mem_util
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len
    if dtype is not None:
        kwargs["dtype"] = dtype
    if enforce_eager:
        kwargs["enforce_eager"] = True
    if swap_space is not None:
        kwargs["swap_space_gb"] = swap_space
    if temperature is not None:
        kwargs["temperature"] = temperature
    if seed is not None:
        kwargs["seed"] = seed
    if branch_factor is not None:
        kwargs["branch_factor"] = branch_factor
    if patience is not None:
        kwargs["convergence_patience"] = patience
    if early_stop is not None:
        kwargs["early_stop_score"] = early_stop
    if no_critique:
        kwargs["use_self_critique"] = False
    if no_cot:
        kwargs["use_chain_of_thought"] = False

    config = TimeDilateConfig(**kwargs)
    try:
        config.validate()
    except ConfigError as e:
        raise click.BadParameter(str(e))

    if not quiet:
        console.print("[bold green]Time Dilation Runtime[/]")
        console.print(config.describe())
        console.print()

    if dry_run:
        console.print("[dim]Dry run — configuration only.[/]")
        return

    score_trajectory: list[int] = []

    def on_cycle(cycle, total, score, elapsed):
        if quiet:
            return
        score_trajectory.append(int(score))
        progress = f"{cycle}/{total}" if total else f"{cycle} [{elapsed:.1f}s budget]"
        if stream_progress:
            spark = sparkline(score_trajectory)
            console.print(
                f"  [dim]cycle {progress}[/] "
                f"[bold]{score:3d}[/]/100  {spark}  [dim]{elapsed:.1f}s[/]"
            )
        else:
            console.print(f"  [dim]Cycle {progress} — score: {score} — {elapsed:.1f}s[/]")

    controller = DilationController(config)
    result = controller.run(prompt, on_cycle=on_cycle)
    cache_hits = getattr(controller, "_score_cache_hits", 0)

    if quiet:
        click.echo(result.output)
    else:
        console.print()
        console.print(result.output)
        console.print()
        scores = [c.score_after for c in result.cycle_history if c.score_after is not None]
        spark = sparkline(scores) if scores else ""
        console.print("[bold green]Run complete[/]")
        console.print(
            f"  cycles:           {result.cycles_completed}"
            + (f"/{result.total_cycles}" if result.total_cycles else "")
        )
        console.print(f"  elapsed:          {result.elapsed_seconds:.1f}s")
        console.print(f"  avg cycle:        {result.avg_cycle_seconds:.2f}s")
        console.print(
            f"  score:            [bold]{result.initial_score}[/] -> "
            f"[bold]{result.score}[/]  (gain {result.score_gain:+d})"
        )
        console.print(f"  improvement rate: {result.improvement_rate * 100:.0f}%")
        console.print(f"  convergence resets: {result.convergence_resets}")
        console.print(f"  score cache hits: {cache_hits}")
        if spark:
            console.print(f"  trajectory:       {spark}")

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
        report_data = result.to_report(config, score_cache_hits=cache_hits)
        from pathlib import Path
        Path(report_file).write_text(json.dumps(report_data, indent=2))
        if not quiet:
            console.print(f"[dim]Report saved to {report_file}[/]")


@main.command()
@click.option("--factor", default=1000.0, type=float, help="Dilation factor to inspect.")
@click.option("--time-budget", default=None, type=float,
              help="Wall-clock seconds; with --factor shows full subjective time breakdown.")
def explain(factor, time_budget):
    """Explain what happens at a given dilation factor.

    \b
    Examples:
      timedilate explain --factor 1000
      timedilate explain --factor 1000000 --time-budget 5
    """
    config = TimeDilateConfig(dilation_factor=factor, time_budget_seconds=time_budget)
    console.print(f"[bold]Time dilation at {factor:,}x[/]")
    console.print()
    console.print(config.describe())
    console.print()
    if time_budget is not None:
        sub = config.subjective_time or 0.0
        console.print(f"[bold]Subjective time breakdown[/] ({time_budget}s wall-clock x {factor:,}x):")
        console.print(f"  = {sub:,.0f} seconds")
        console.print(f"  ≈ {sub / 86400:,.2f} days")
        console.print(f"  ≈ {sub / (7 * 86400):,.2f} weeks")
        console.print(f"  ≈ {sub / (30 * 86400):,.2f} months")
        console.print(f"  ≈ {sub / (365 * 86400):,.4f} years")
        console.print(f"  human-readable: {format_subjective_time(sub)}")
    else:
        console.print(f"The AI will run up to [bold]{config.num_cycles:,}[/] reasoning cycles.")
    console.print()
    console.print("Each cycle: score -> critique -> refine (N branches) -> keep best.")
    console.print("On plateau: try a fresh approach. No quality loss — just more thinking.")


if __name__ == "__main__":
    main()
