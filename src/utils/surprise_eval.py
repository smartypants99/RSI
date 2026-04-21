"""Human-surprise eval bank — anti-contamination anchor source.

HumanEval/MBPP/GSM8K/MATH are all pre-2023; DeepSeek-R1 and most current
base checkpoints have almost certainly trained on them. Scores there
reflect memorization as much as capability, so they are a weak anchor
for detecting verifier capture or real capability drift.

This module supplies two fresh anchor sources:

1. ``codex``-generated novel code problems with reference solutions,
   cached on disk, regenerated on a weekly cadence. These are held-out
   by construction (they did not exist at R1's training cutoff).

2. Third-party public eval sets that R1 cannot have seen:
   - ARC-AGI-2 (via HF ``datasets`` if available) — visual reasoning
     on synthetic grids; almost no overlap with code pretraining.
   - LiveCodeBench with a post-cutoff date filter — contest problems
     released after the model's cutoff.

All three flow into the existing anchor_eval hub via the same
``BenchmarkItem`` shape that ``external_benchmarks.py`` exposes, so the
orchestrator's capture-canary logic picks them up without changes.

Graceful degradation: every loader falls back to a tiny bundled
fixture when the network / ``datasets`` / ``codex`` CLI is unavailable.
Tests stay deterministic and CI-green; prod runs use the live sources.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from src.utils.external_benchmarks import BenchmarkItem, _stable_hash

logger = logging.getLogger(__name__)


SURPRISE_CACHE_DIR = Path("outputs/surprise_eval")
SURPRISE_REFRESH_INTERVAL = timedelta(days=7)
DEFAULT_MODEL_CUTOFF_ISO = "2024-07-01"  # DeepSeek-R1 approx cutoff
SURPRISE_BENCHMARK_NAMES = ("codex_surprise", "arc_agi", "livecodebench_postcut")


# ---------------------------------------------------------------------------
# (1) codex-generated novel code problems
# ---------------------------------------------------------------------------

_CODEX_PROMPT = (
    "Generate {n} novel, self-contained Python coding problems that are "
    "unlikely to appear verbatim in any pretraining corpus. For each, "
    "emit a JSON object on its own line with keys: "
    '"item_id" (short slug), "prompt" (problem statement + function signature), '
    '"answer" (reference Python solution as a complete function), '
    '"tests" (array of 2-4 assert-style test strings). '
    "Output ONLY the JSONL rows, no prose, no code fences."
)


@dataclass
class SurpriseCacheMeta:
    generated_at: float
    n: int
    source: str  # "codex" | "offline"

    def is_stale(self, now: Optional[float] = None) -> bool:
        now = now if now is not None else time.time()
        age = now - float(self.generated_at)
        return age > SURPRISE_REFRESH_INTERVAL.total_seconds()


def _surprise_cache_files(cache_dir: Path) -> tuple[Path, Path]:
    return cache_dir / "codex_surprise.jsonl", cache_dir / "codex_surprise.meta.json"


def _parse_codex_jsonl(text: str) -> list[dict]:
    rows: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and "prompt" in row and "answer" in row:
            rows.append(row)
    return rows


def _run_codex(n: int, timeout_s: int) -> Optional[list[dict]]:
    """Invoke the codex CLI. Returns parsed rows, or None on failure."""
    if shutil.which("codex") is None:
        logger.debug("codex CLI not on PATH; skipping live generation")
        return None
    prompt = _CODEX_PROMPT.format(n=n)
    try:
        proc = subprocess.run(
            ["codex", "exec", "--skip-git-repo-check", prompt],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning("codex invocation failed (%s); using offline fallback", e)
        return None
    if proc.returncode != 0:
        logger.warning("codex returned %d: %s", proc.returncode, proc.stderr[:200])
        return None
    rows = _parse_codex_jsonl(proc.stdout)
    if len(rows) < max(1, n // 4):
        logger.warning("codex produced only %d rows (wanted %d); rejecting batch", len(rows), n)
        return None
    return rows


def _offline_surprise_fixture(n: int) -> list[dict]:
    """Deterministic, small fixture so tests and offline runs work."""
    fixture: list[dict] = []
    for i in range(n):
        fixture.append({
            "item_id": f"surprise_offline/{i}",
            "prompt": (
                f"def rotate_list_{i}(xs, k):\n"
                f"    '''Return xs rotated left by k positions (k may exceed len).'''\n"
            ),
            "answer": (
                f"def rotate_list_{i}(xs, k):\n"
                f"    if not xs:\n        return xs\n"
                f"    k = k % len(xs)\n"
                f"    return xs[k:] + xs[:k]\n"
            ),
            "tests": [
                f"assert rotate_list_{i}([1,2,3,4], 1) == [2,3,4,1]",
                f"assert rotate_list_{i}([], 3) == []",
            ],
        })
    return fixture


def _write_surprise(cache_dir: Path, rows: list[dict], source: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_path, meta_path = _surprise_cache_files(cache_dir)
    tmp = data_path.with_suffix(data_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    tmp.replace(data_path)
    meta_path.write_text(json.dumps({
        "generated_at": time.time(),
        "n": len(rows),
        "source": source,
    }))


def _read_surprise(cache_dir: Path) -> tuple[list[dict], Optional[SurpriseCacheMeta]]:
    data_path, meta_path = _surprise_cache_files(cache_dir)
    if not data_path.exists():
        return [], None
    rows: list[dict] = []
    with data_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    meta: Optional[SurpriseCacheMeta] = None
    if meta_path.exists():
        try:
            m = json.loads(meta_path.read_text())
            meta = SurpriseCacheMeta(
                generated_at=float(m.get("generated_at", 0.0)),
                n=int(m.get("n", len(rows))),
                source=str(m.get("source", "offline")),
            )
        except (OSError, ValueError, json.JSONDecodeError):
            meta = None
    return rows, meta


def load_codex_surprise(
    *,
    n: int = 20,
    cache_dir: Path | str = SURPRISE_CACHE_DIR,
    force_refresh: bool = False,
    force_offline: bool = False,
    codex_timeout_s: int = 120,
) -> list[BenchmarkItem]:
    """Return cached codex-generated surprise problems, refreshing weekly.

    - Reads cache if present and fresh (< 7 days old).
    - Calls ``codex exec`` to regenerate when stale / missing / forced.
    - Falls back to a deterministic offline fixture on any failure.
    """
    cache_dir = Path(cache_dir)
    rows, meta = _read_surprise(cache_dir)
    stale = meta is None or meta.is_stale()
    if rows and not (stale or force_refresh):
        return [_row_to_item(r, "codex_surprise") for r in rows]

    live_rows: Optional[list[dict]] = None
    if not force_offline:
        live_rows = _run_codex(n, codex_timeout_s)

    if live_rows:
        _write_surprise(cache_dir, live_rows, source="codex")
        return [_row_to_item(r, "codex_surprise") for r in live_rows]

    if rows:
        # Live call failed but we have an older cache — use it rather than
        # regressing to fixture. Log so the drift is visible.
        logger.info("codex refresh failed; serving stale cache (age ok for anchor)")
        return [_row_to_item(r, "codex_surprise") for r in rows]

    offline = _offline_surprise_fixture(n)
    _write_surprise(cache_dir, offline, source="offline")
    return [_row_to_item(r, "codex_surprise") for r in offline]


def _row_to_item(row: dict, bench: str) -> BenchmarkItem:
    return BenchmarkItem(
        benchmark=bench,
        item_id=str(row.get("item_id") or row.get("id") or _stable_hash(row.get("prompt", ""))),
        prompt=str(row.get("prompt", "")),
        answer=str(row.get("answer", "")),
        domain=str(row.get("domain", "code")),
        meta={"tests": row.get("tests", []), **{k: v for k, v in row.items()
              if k not in ("item_id", "prompt", "answer", "domain", "tests")}},
    )


# ---------------------------------------------------------------------------
# (2) ARC-AGI loader
# ---------------------------------------------------------------------------

_ARC_OFFLINE: list[dict] = [
    {
        "item_id": f"arc_offline/{i}",
        "prompt": (
            "Given training pairs of input/output grids, infer the rule and "
            f"predict the output. Train: [[{i},0],[0,{i}]] -> [[0,{i}],[{i},0]]. "
            f"Test input: [[{i+1},0],[0,{i+1}]]"
        ),
        "answer": f"[[0,{i+1}],[{i+1},0]]",
    }
    for i in range(8)
]


def load_arc_agi(
    *,
    cache_dir: Path | str = SURPRISE_CACHE_DIR,
    force_offline: bool = False,
    limit: int = 100,
) -> list[BenchmarkItem]:
    """ARC-AGI loader via HF datasets; offline fixture when unavailable.

    Tries a small set of known HF dataset names for ARC-AGI variants; first
    one that loads wins. All failures fall through to the offline fixture.
    """
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / "arc_agi.jsonl"
    if cache_path.exists():
        try:
            return _load_items_cache(cache_path, "arc_agi")
        except (OSError, json.JSONDecodeError):
            pass

    items: list[BenchmarkItem] = []
    if not force_offline:
        items = _try_load_arc_from_hf(limit)

    if not items:
        items = [
            BenchmarkItem(
                benchmark="arc_agi",
                item_id=r["item_id"],
                prompt=r["prompt"],
                answer=r["answer"],
                domain="reasoning",
            )
            for r in _ARC_OFFLINE
        ]

    _save_items_cache(cache_path, items)
    return items


def _try_load_arc_from_hf(limit: int) -> list[BenchmarkItem]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        return []
    candidates = [
        ("dataneel/arc-agi", None, "train"),
        ("nkasmanoff/arc-agi", None, "train"),
    ]
    for name, config, split in candidates:
        try:
            ds = load_dataset(name, config, split=split) if config else load_dataset(name, split=split)
        except Exception as e:
            logger.debug("ARC-AGI load attempt %s failed: %s", name, e)
            continue
        out: list[BenchmarkItem] = []
        for i, r in enumerate(ds):
            if i >= limit:
                break
            prompt = json.dumps({k: r.get(k) for k in ("train", "test") if k in r}) or str(r.get("input", ""))
            answer = str(r.get("output") or r.get("answer") or r.get("test_output") or "")
            if not prompt or not answer:
                continue
            out.append(BenchmarkItem(
                benchmark="arc_agi",
                item_id=str(r.get("task_id", f"arc/{i}")),
                prompt=prompt,
                answer=answer,
                domain="reasoning",
                meta={"source_dataset": name},
            ))
        if out:
            return out
    return []


# ---------------------------------------------------------------------------
# (3) LiveCodeBench post-cutoff loader
# ---------------------------------------------------------------------------

_LCB_OFFLINE: list[dict] = [
    {
        "item_id": f"lcb_offline/{i}",
        "prompt": (
            f"Given a list of integers, return the {i+2}-th smallest distinct value. "
            "If fewer than that many distinct values exist, return -1. "
            f"def kth_distinct_{i}(xs): ..."
        ),
        "answer": (
            f"def kth_distinct_{i}(xs):\n"
            f"    uniq = sorted(set(xs))\n"
            f"    return uniq[{i+1}] if len(uniq) > {i+1} else -1\n"
        ),
        "release_date": "2025-03-01",
    }
    for i in range(8)
]


def load_livecodebench_postcutoff(
    *,
    cutoff_iso: str = DEFAULT_MODEL_CUTOFF_ISO,
    cache_dir: Path | str = SURPRISE_CACHE_DIR,
    force_offline: bool = False,
    limit: int = 100,
) -> list[BenchmarkItem]:
    """LiveCodeBench problems released strictly after ``cutoff_iso``.

    The cutoff filter is the whole point: only problems the base model
    couldn't have trained on are valid anchors. Offline fixture dates
    are all 2025-xx so they pass any realistic cutoff.
    """
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"lcb_post_{cutoff_iso}.jsonl"
    if cache_path.exists():
        try:
            return _load_items_cache(cache_path, "livecodebench_postcut")
        except (OSError, json.JSONDecodeError):
            pass

    items: list[BenchmarkItem] = []
    if not force_offline:
        items = _try_load_lcb_from_hf(cutoff_iso, limit)

    if not items:
        items = [
            BenchmarkItem(
                benchmark="livecodebench_postcut",
                item_id=r["item_id"],
                prompt=r["prompt"],
                answer=r["answer"],
                domain="code",
                meta={"release_date": r["release_date"]},
            )
            for r in _LCB_OFFLINE
            if _after_cutoff(r["release_date"], cutoff_iso)
        ]

    _save_items_cache(cache_path, items)
    return items


def _after_cutoff(release_iso: str, cutoff_iso: str) -> bool:
    try:
        rel = datetime.fromisoformat(release_iso.replace("Z", "+00:00"))
        cut = datetime.fromisoformat(cutoff_iso.replace("Z", "+00:00"))
        if rel.tzinfo is None:
            rel = rel.replace(tzinfo=timezone.utc)
        if cut.tzinfo is None:
            cut = cut.replace(tzinfo=timezone.utc)
        return rel > cut
    except ValueError:
        return False


def _try_load_lcb_from_hf(cutoff_iso: str, limit: int) -> list[BenchmarkItem]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        return []
    candidates = [
        ("livecodebench/code_generation_lite", "release_v5"),
        ("livecodebench/code_generation", None),
    ]
    for name, config in candidates:
        try:
            ds = load_dataset(name, config, split="test") if config else load_dataset(name, split="test")
        except Exception as e:
            logger.debug("LCB load attempt %s failed: %s", name, e)
            continue
        out: list[BenchmarkItem] = []
        for i, r in enumerate(ds):
            if len(out) >= limit:
                break
            rel = str(r.get("contest_date") or r.get("release_date") or r.get("date") or "")
            if rel and not _after_cutoff(rel, cutoff_iso):
                continue
            prompt = str(r.get("question_content") or r.get("prompt") or r.get("problem", ""))
            answer = str(r.get("canonical_solution") or r.get("solution") or r.get("answer", ""))
            if not prompt:
                continue
            out.append(BenchmarkItem(
                benchmark="livecodebench_postcut",
                item_id=str(r.get("question_id") or f"lcb/{i}"),
                prompt=prompt,
                answer=answer,
                domain="code",
                meta={"release_date": rel, "source_dataset": name},
            ))
        if out:
            return out
    return []


# ---------------------------------------------------------------------------
# Cache helpers shared across the three loaders.
# ---------------------------------------------------------------------------

def _save_items_cache(path: Path, items: list[BenchmarkItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it.to_json()) + "\n")
    tmp.replace(path)


def _load_items_cache(path: Path, bench: str) -> list[BenchmarkItem]:
    out: list[BenchmarkItem] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(BenchmarkItem.from_json(json.loads(line)))
    # Tolerate legacy caches: coerce benchmark if drifted.
    for it in out:
        if it.benchmark != bench:
            it.benchmark = bench
    return out


# ---------------------------------------------------------------------------
# Unified anchor-hub integration.
# ---------------------------------------------------------------------------

def load_surprise_anchor_items(
    *,
    codex_n: int = 20,
    cache_dir: Path | str = SURPRISE_CACHE_DIR,
    cutoff_iso: str = DEFAULT_MODEL_CUTOFF_ISO,
    force_offline: bool = False,
) -> dict[str, list[BenchmarkItem]]:
    """Return all three surprise-anchor sources keyed by benchmark name.

    Shape matches ``items_by_benchmark`` expected by
    ``external_benchmarks.sample_anchor_set`` / ``run_anchor_eval``, so the
    orchestrator wires it in with one line::

        items = {**load_external(...), **load_surprise_anchor_items(...)}
    """
    return {
        "codex_surprise": load_codex_surprise(
            n=codex_n, cache_dir=cache_dir, force_offline=force_offline),
        "arc_agi": load_arc_agi(cache_dir=cache_dir, force_offline=force_offline),
        "livecodebench_postcut": load_livecodebench_postcutoff(
            cutoff_iso=cutoff_iso, cache_dir=cache_dir, force_offline=force_offline),
    }


__all__ = [
    "SURPRISE_BENCHMARK_NAMES",
    "SURPRISE_REFRESH_INTERVAL",
    "DEFAULT_MODEL_CUTOFF_ISO",
    "SurpriseCacheMeta",
    "load_codex_surprise",
    "load_arc_agi",
    "load_livecodebench_postcutoff",
    "load_surprise_anchor_items",
]
