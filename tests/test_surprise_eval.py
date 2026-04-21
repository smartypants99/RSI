"""Tests for the human-surprise eval bank + ARC-AGI + LiveCodeBench loaders."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from src.utils.external_benchmarks import BenchmarkItem, sample_anchor_set
from src.utils.surprise_eval import (
    DEFAULT_MODEL_CUTOFF_ISO,
    SURPRISE_REFRESH_INTERVAL,
    SurpriseCacheMeta,
    _after_cutoff,
    load_arc_agi,
    load_codex_surprise,
    load_livecodebench_postcutoff,
    load_surprise_anchor_items,
)


def test_codex_surprise_offline_fallback(tmp_path):
    items = load_codex_surprise(n=5, cache_dir=tmp_path, force_offline=True)
    assert len(items) == 5
    assert all(isinstance(it, BenchmarkItem) for it in items)
    assert all(it.benchmark == "codex_surprise" for it in items)
    assert all(it.prompt and it.answer for it in items)

    # Second call uses the cache.
    data_path = tmp_path / "codex_surprise.jsonl"
    assert data_path.exists()
    items2 = load_codex_surprise(n=5, cache_dir=tmp_path, force_offline=True)
    assert [i.item_id for i in items] == [i.item_id for i in items2]


def test_codex_surprise_refresh_when_stale(tmp_path):
    load_codex_surprise(n=4, cache_dir=tmp_path, force_offline=True)
    meta_path = tmp_path / "codex_surprise.meta.json"
    meta = json.loads(meta_path.read_text())
    # Simulate a stale cache.
    meta["generated_at"] = time.time() - SURPRISE_REFRESH_INTERVAL.total_seconds() - 60
    meta_path.write_text(json.dumps(meta))

    # Stale + force_offline: prefer serving the existing cache over regenerating
    # a duplicate offline fixture. Items still load cleanly.
    items = load_codex_surprise(n=4, cache_dir=tmp_path, force_offline=True)
    assert len(items) == 4


def test_surprise_cache_meta_staleness():
    fresh = SurpriseCacheMeta(generated_at=time.time(), n=10, source="codex")
    stale = SurpriseCacheMeta(
        generated_at=time.time() - SURPRISE_REFRESH_INTERVAL.total_seconds() - 1,
        n=10, source="codex",
    )
    assert fresh.is_stale() is False
    assert stale.is_stale() is True


def test_arc_agi_offline_fallback(tmp_path):
    items = load_arc_agi(cache_dir=tmp_path, force_offline=True)
    assert len(items) > 0
    assert all(it.benchmark == "arc_agi" for it in items)
    assert all(it.domain == "reasoning" for it in items)


def test_livecodebench_respects_cutoff(tmp_path):
    # Cutoff in the past → all 2025 offline items pass.
    items = load_livecodebench_postcutoff(
        cutoff_iso="2024-01-01", cache_dir=tmp_path, force_offline=True)
    assert len(items) > 0
    # Cutoff in the future → everything filtered out.
    out_dir = tmp_path / "future"
    items_future = load_livecodebench_postcutoff(
        cutoff_iso="2099-01-01", cache_dir=out_dir, force_offline=True)
    assert len(items_future) == 0


def test_after_cutoff_parsing():
    assert _after_cutoff("2025-03-01", "2024-07-01") is True
    assert _after_cutoff("2023-01-01", "2024-07-01") is False
    assert _after_cutoff("not-a-date", "2024-07-01") is False


def test_load_surprise_anchor_items_shape(tmp_path):
    items_by_bench = load_surprise_anchor_items(
        codex_n=3, cache_dir=tmp_path, force_offline=True)
    assert set(items_by_bench.keys()) == {"codex_surprise", "arc_agi", "livecodebench_postcut"}
    for bench, items in items_by_bench.items():
        assert all(isinstance(it, BenchmarkItem) for it in items)
        assert all(it.benchmark == bench for it in items)


def test_surprise_integrates_with_anchor_sampler(tmp_path):
    """The surprise loaders must feed external_benchmarks.sample_anchor_set."""
    items = load_surprise_anchor_items(
        codex_n=4, cache_dir=tmp_path, force_offline=True)
    sample = sample_anchor_set(items, per_benchmark=2, seed=42)
    # One sample per surprise-source × 2 = 6.
    by_bench = {}
    for it in sample:
        by_bench.setdefault(it.benchmark, []).append(it)
    assert set(by_bench.keys()) == {"codex_surprise", "arc_agi", "livecodebench_postcut"}
    for bench in by_bench:
        assert len(by_bench[bench]) == 2
