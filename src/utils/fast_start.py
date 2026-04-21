"""Fast-start helpers (Task #11).

Three cycle-1-only levers to cut time-to-first-trained-held-out:

1. prestash_prior_training_samples — seed the rolling RSI pool from prior-run
   training_pool JSONL files so cycle 1 can actually train instead of waiting
   3-4 cycles for the pool to accumulate.

2. default_weakness_diag — uniform DiagnosticResult used on cycle 1 when
   OrchestratorConfig.skip_first_diagnostics is True. Saves the ~6 min cold
   diagnostic probe; real diagnostics runs cycle 2+.

3. bootstrap_tasks_per_cycle — return the first-cycle propose budget
   (SynthesisConfig.synthesis_tasks_per_cycle_bootstrap) for cycle==1, else
   the steady-state `tasks_per_cycle`.

None of these affect cycle 2+ behavior. Quality is preserved because the
quorum thresholds, regression_revert_threshold, and skip_if_initial_loss
guards in the normal path still catch bad training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


def _iter_training_pool_files(output_dir: Path) -> Iterable[Path]:
    """Yield every training_pool/*.jsonl under output_dir AND sibling
    outputs_run_*/training_pool/*.jsonl directories."""
    out = Path(output_dir)
    parent = out.parent
    candidate_roots: list[Path] = []
    pool_dir = out / "training_pool"
    if pool_dir.is_dir():
        candidate_roots.append(pool_dir)
    if parent.exists():
        for sib in parent.iterdir():
            if not sib.is_dir():
                continue
            # Match outputs_run_*, outputs*, outputs_*
            name = sib.name
            if sib == out:
                continue
            if not (name == "outputs" or name.startswith("outputs_") or name.startswith("outputs-")):
                continue
            tp = sib / "training_pool"
            if tp.is_dir():
                candidate_roots.append(tp)
    for root in candidate_roots:
        for p in sorted(root.glob("*.jsonl")):
            yield p


def _load_pool_records(
    path: Path, current_sid: str
) -> list[dict]:
    """Read a single pool JSONL, skipping records whose session_id matches
    current_sid (so we never double-train on samples from a prior incarnation
    of THIS session)."""
    out: list[dict] = []
    try:
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                sid = rec.get("session_id", "")
                if sid and current_sid and sid == current_sid:
                    continue
                prompt = rec.get("prompt") or ""
                response = rec.get("response") or ""
                if not prompt or not response:
                    continue
                out.append(rec)
    except OSError as exc:
        logger.debug("fast_start: cannot read %s (%s)", path, exc)
    return out


def prestash_prior_training_samples(
    output_dir: Path | str,
    current_sid: str,
    max_samples: int = 30,
) -> list[Any]:
    """Build a list of TrainingSample objects from prior training_pool JSONL.

    Returns up to `max_samples` TrainingSample instances suitable for
    extending `_rsi_pending_pool`. Returns [] on any error — fast-start is
    best-effort, never fatal.

    Records are keyed by (prompt, response) to de-dupe across files;
    first occurrence wins. The normal 0.03 regression_revert + pre-loss
    probe in loop.py protect against a bad stash poisoning training.
    """
    if max_samples <= 0:
        return []
    try:
        from ..generator.data_generator import TrainingSample
    except Exception as exc:  # pragma: no cover — import only fails if repo broken
        logger.warning("fast_start: cannot import TrainingSample (%s)", exc)
        return []

    seen: set[tuple[str, str]] = set()
    samples: list[Any] = []
    for path in _iter_training_pool_files(Path(output_dir)):
        if len(samples) >= max_samples:
            break
        for rec in _load_pool_records(path, current_sid):
            if len(samples) >= max_samples:
                break
            key = (rec.get("prompt", ""), rec.get("response", ""))
            if key in seen:
                continue
            seen.add(key)
            try:
                ts = TrainingSample(
                    prompt=rec.get("prompt", ""),
                    response=rec.get("response", ""),
                    domain=rec.get("domain", "unknown"),
                    verified=True,
                    source=rec.get("source", "rsi_property_prestash"),
                )
            except TypeError:
                continue
            samples.append(ts)
    logger.info(
        "fast_start: pre-stashed %d prior-run training samples from %s (cap=%d, excluding sid=%s)",
        len(samples), output_dir, max_samples, current_sid,
    )
    return samples


def default_weakness_diag(
    domains: list[str],
    cycle: int = 1,
) -> Any:
    """Return a DiagnosticResult with uniform mastery across `domains`.

    Used on cycle 1 when `OrchestratorConfig.skip_first_diagnostics` is True,
    so we can skip the ~6 min cold probe. Each domain gets score 0.5
    (neutral — not asserting weakness or mastery) and one sentinel
    WeaknessReport per domain so the synthesizer has SOMETHING to propose
    against. Subdomain scores are intentionally empty; the synthesizer
    falls back to generic prompts, which is the correct behavior when we
    have no probe data yet.
    """
    from ..diagnostics.engine import DiagnosticResult, WeaknessReport
    import time as _time

    weaknesses = [
        WeaknessReport(
            domain=d,
            subdomain="bootstrap",
            severity=0.5,
            description=f"bootstrap weakness for {d} — real diagnostics deferred to cycle 2",
            n_questions=0,
            n_failures=0,
            calibrated_confidence=0.5,
        )
        for d in domains
    ]
    return DiagnosticResult(
        cycle=cycle,
        timestamp=_time.time(),
        weaknesses=weaknesses,
        domain_scores={d: 0.5 for d in domains},
        subdomain_scores={},
        domain_question_counts={d: 0 for d in domains},
        layer_health={},
        total_questions=0,
        total_correct=0,
        per_question=[],
    )


def bootstrap_tasks_per_cycle(
    synthesis_cfg: Any,
    cycle: int,
) -> int:
    """Return the propose budget for this cycle.

    Cycle 1: synthesis_tasks_per_cycle_bootstrap (smaller → faster first
    propose; lower but acceptable accept count because the quorum gate
    still runs).
    Cycle ≥ 2: tasks_per_cycle (steady-state budget).
    """
    if cycle <= 1:
        bs = getattr(synthesis_cfg, "synthesis_tasks_per_cycle_bootstrap", None)
        if bs is not None and int(bs) > 0:
            return int(bs)
    return int(getattr(synthesis_cfg, "tasks_per_cycle", 20))
