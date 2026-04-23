"""Structured observability logs — append-only JSONL, zero-cost on happy path.

Five sinks land under ``<output_dir>/``:

  - training_steps.jsonl     — one row per optimizer step (trainer)
  - heldout_per_prompt.jsonl — one row per held-out prompt eval (loop / diag)
  - verify_decisions.jsonl   — one row per verified candidate (verifier)
  - propose_attempts.jsonl   — one row per proposal attempt (generator)
  - cycle_summary.jsonl      — one denormalized row per cycle (orchestrator)

Design contract:
  * `emit(sink_name, record, cfg)` is the only public entry point.
  * Any exception is swallowed and logged at DEBUG — a logging bug MUST NOT
    crash training. Instrumentation is best-effort telemetry.
  * Flag gating: `OrchestratorConfig.structured_observability_enabled` plus
    a per-sink sub-flag. Missing flags default to the master flag's value.
  * Atomic append: `open(path, "a")` + json.dumps + flush. Each line is
    self-contained; partial files are still readable.

Expected sub-flag names on OrchestratorConfig (all default True when the
master flag is True, False when the master flag is False):
    structured_log_training_steps
    structured_log_heldout_per_prompt
    structured_log_verify_decisions
    structured_log_propose_attempts
    structured_log_cycle_summary
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

SINK_FILENAMES: dict[str, str] = {
    "training_steps": "training_steps.jsonl",
    "heldout_per_prompt": "heldout_per_prompt.jsonl",
    "verify_decisions": "verify_decisions.jsonl",
    "propose_attempts": "propose_attempts.jsonl",
    "cycle_summary": "cycle_summary.jsonl",
}

_SUBFLAG_PREFIX = "structured_log_"


def _is_enabled(cfg: Any, sink_name: str) -> bool:
    """Return True iff master flag AND sink sub-flag are both on.

    Resolution order:
      1. If cfg is None → disabled.
      2. Master flag `structured_observability_enabled` (default False when missing).
      3. Sub-flag `structured_log_<sink>` (default True when missing — i.e. sinks
         are opt-out, not opt-in, once the master flag is on).
    """
    if cfg is None:
        return False
    master = bool(getattr(cfg, "structured_observability_enabled", False))
    if not master:
        return False
    sub_attr = f"{_SUBFLAG_PREFIX}{sink_name}"
    sub = getattr(cfg, sub_attr, True)
    return bool(sub)


def _resolve_output_dir(cfg: Any, override: Optional[Path]) -> Optional[Path]:
    if override is not None:
        return Path(override)
    out = getattr(cfg, "output_dir", None)
    if out is None:
        return None
    return Path(out)


def emit(
    sink_name: str,
    record: Mapping[str, Any],
    cfg: Any,
    *,
    output_dir: Optional[Path] = None,
) -> None:
    """Append one JSON record to the named sink.

    Never raises. Logs at DEBUG on any failure. When the flag is off the
    function returns immediately with zero cost (one getattr + bool).

    Parameters
    ----------
    sink_name : must be a key in SINK_FILENAMES; unknown sinks are dropped
                with a debug log.
    record    : any JSON-serializable mapping. Non-serializable values are
                coerced via ``default=str``.
    cfg       : an object (typically OrchestratorConfig) that carries the
                gating flags and ``output_dir``. May be None.
    output_dir: optional override for tests / call sites without cfg access.
    """
    try:
        if sink_name not in SINK_FILENAMES:
            logger.debug("structured_logs: unknown sink %r", sink_name)
            return
        if not _is_enabled(cfg, sink_name):
            return
        out_dir = _resolve_output_dir(cfg, output_dir)
        if out_dir is None:
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / SINK_FILENAMES[sink_name]
        line = json.dumps(record, default=str, ensure_ascii=False)
        # Atomic append. On POSIX, O_APPEND guarantees the whole write lands
        # at EOF atomically for payloads under PIPE_BUF, which our JSON lines
        # virtually always are (< 4KB). For larger payloads this is still
        # safe for single-writer workloads — each emit call is one write().
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
    except Exception as e:  # pragma: no cover — defensive
        logger.debug(
            "structured_logs: emit(%s) failed (%s): %s",
            sink_name, type(e).__name__, e,
        )


def is_enabled(cfg: Any, sink_name: str) -> bool:
    """Public wrapper around the gate — used by call sites that want to
    skip expensive metric computation when the sink is off."""
    return _is_enabled(cfg, sink_name)
