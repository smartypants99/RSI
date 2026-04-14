import json
import logging
import sys


class StructuredFormatter(logging.Formatter):
    """JSON-structured log formatter for machine-readable output."""

    def format(self, record):
        log_entry = {
            "ts": self.formatTime(record, "%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if hasattr(record, "cycle_data"):
            log_entry["cycle"] = record.cycle_data
        return json.dumps(log_entry)


def setup_logging(verbose: bool = False, structured: bool = False) -> None:
    """Configure logging for the timedilate package."""
    level = logging.DEBUG if verbose else logging.WARNING
    handler = logging.StreamHandler(sys.stderr)
    if structured:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
    root = logging.getLogger("timedilate")
    root.setLevel(level)
    # Avoid duplicate handlers on repeated calls
    root.handlers.clear()
    root.addHandler(handler)
    root.propagate = False


def log_cycle_summary(logger: logging.Logger, cycle: int, score: int, previous_score: int,
                      directive_source: str, elapsed: float, branch_count: int,
                      output_delta: float = 0.0, peak_score: int = 0,
                      improvement_rate: float = 0.0,
                      budget_used_pct: float = 0.0) -> None:
    """Emit a structured cycle summary log with rich per-cycle data."""
    delta = score - previous_score
    data = {
        "cycle": cycle,
        "score": score,
        "delta": delta,
        "source": directive_source,
        "elapsed_s": round(elapsed, 3),
        "branches": branch_count,
        "output_delta": round(output_delta, 3),
        "peak_score": peak_score,
        "cumulative_improvement_rate": round(improvement_rate, 3),
        "budget_used_pct": round(budget_used_pct, 1),
    }
    budget_str = f" budget={budget_used_pct:.0f}%" if budget_used_pct > 0 else ""
    msg = (
        f"Cycle {cycle}: {score} ({delta:+d}) [{directive_source}] "
        f"{elapsed:.2f}s | change={output_delta:.0%} peak={peak_score}{budget_str}"
    )
    record = logger.makeRecord(
        logger.name, logging.INFO, "", 0, msg, (), None
    )
    record.cycle_data = data
    logger.handle(record)
