"""Tests for src/utils/fast_student.py — unit-level only, no GPU/vLLM."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.fast_student import (
    DistillPair,
    FastStudentConfig,
    FastStudentManager,
    StudentCheckpoint,
)


def _fake_distill(pairs, out_dir: Path, cfg) -> StudentCheckpoint:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fake_weights").write_text("ok")
    return StudentCheckpoint(
        cycle=-1,
        path=out_dir,
        model_name=cfg.model_name,
        num_pairs=len(pairs),
        distill_seconds=0.0,
    )


def _fake_loader(ckpt: StudentCheckpoint):
    def _gen(prompts):
        return [f"STUDENT[{ckpt.num_pairs}]:{p[:8]}" for p in prompts]

    return _gen


def _make_manager(tmp_path: Path, **overrides) -> FastStudentManager:
    cfg = FastStudentConfig(
        enabled=True,
        redistill_every=overrides.pop("redistill_every", 2),
        min_pairs_for_distill=overrides.pop("min_pairs_for_distill", 3),
        max_pairs_for_distill=overrides.pop("max_pairs_for_distill", 10),
        checkpoint_root=tmp_path / "fs",
        **overrides,
    )
    return FastStudentManager(cfg, distill_fn=_fake_distill, student_loader=_fake_loader)


def test_disabled_manager_is_noop(tmp_path: Path):
    mgr = FastStudentManager(
        FastStudentConfig(enabled=False, checkpoint_root=tmp_path / "fs"),
        distill_fn=_fake_distill,
        student_loader=_fake_loader,
    )
    mgr.record_teacher_generation(["p1"], ["c1"], cycle=0)
    mgr.on_trained_cycle(0)
    assert mgr.buffer_size() == 0
    assert mgr.generate_fn_for_cycle(0) is None


def test_config_rejects_bad_values(tmp_path: Path):
    with pytest.raises(ValueError):
        FastStudentConfig(redistill_every=0, checkpoint_root=tmp_path)
    with pytest.raises(ValueError):
        FastStudentConfig(
            min_pairs_for_distill=10, max_pairs_for_distill=5, checkpoint_root=tmp_path
        )
    with pytest.raises(ValueError):
        FastStudentConfig(student_gpu_memory_utilization=0.0, checkpoint_root=tmp_path)


def test_record_skips_empty_completions(tmp_path: Path):
    mgr = _make_manager(tmp_path)
    mgr.record_teacher_generation(
        ["p1", "p2", "p3"], ["c1", "", "   "], cycle=0
    )
    assert mgr.buffer_size() == 1


def test_buffer_is_fifo_capped(tmp_path: Path):
    mgr = _make_manager(tmp_path, max_pairs_for_distill=3, min_pairs_for_distill=1)
    for i in range(5):
        mgr.record_teacher_generation([f"p{i}"], [f"c{i}"], cycle=0)
    assert mgr.buffer_size() == 3
    # Oldest dropped first — should have p2, p3, p4.
    assert [p.prompt for p in mgr._pairs] == ["p2", "p3", "p4"]


def test_distill_fires_after_threshold_cycles_with_enough_pairs(tmp_path: Path):
    mgr = _make_manager(tmp_path, redistill_every=2, min_pairs_for_distill=2)
    mgr.record_teacher_generation(["p1", "p2", "p3"], ["c1", "c2", "c3"], cycle=0)
    mgr.on_trained_cycle(0)  # count=1, no fire
    assert mgr.current_checkpoint() is None
    mgr.on_trained_cycle(1)  # count=2, fires
    ckpt = mgr.current_checkpoint()
    assert ckpt is not None
    assert ckpt.num_pairs == 3
    assert (ckpt.path / "metadata.json").exists()
    assert ckpt.distill_seconds >= 0.0


def test_distill_skipped_when_not_enough_pairs(tmp_path: Path):
    mgr = _make_manager(tmp_path, redistill_every=1, min_pairs_for_distill=5)
    mgr.record_teacher_generation(["p1"], ["c1"], cycle=0)
    mgr.on_trained_cycle(0)
    assert mgr.current_checkpoint() is None


def test_generate_fn_routes_through_student_after_distill(tmp_path: Path):
    mgr = _make_manager(tmp_path, redistill_every=1, min_pairs_for_distill=1)
    mgr.record_teacher_generation(["prompt_abc"], ["comp"], cycle=0)
    assert mgr.generate_fn_for_cycle(0) is None  # not distilled yet
    mgr.on_trained_cycle(0)
    fn = mgr.generate_fn_for_cycle(1)
    assert fn is not None
    outs = fn(["prompt_xyz_longer"])
    assert outs == ["STUDENT[1]:prompt_x"]


def test_status_shape(tmp_path: Path):
    mgr = _make_manager(tmp_path)
    s = mgr.status()
    assert s["enabled"] is True
    assert s["buffer_size"] == 0
    assert s["current_checkpoint"] is None


def test_distill_stamps_real_cycle_on_checkpoint(tmp_path: Path):
    mgr = _make_manager(tmp_path, redistill_every=1, min_pairs_for_distill=1)
    mgr.record_teacher_generation(["p1"], ["c1"], cycle=0)
    mgr.on_trained_cycle(42)
    ckpt = mgr.current_checkpoint()
    assert ckpt is not None
    assert ckpt.cycle == 42
    meta = (ckpt.path / "metadata.json").read_text()
    assert '"cycle": 42' in meta


def test_distill_counter_resets_after_fire(tmp_path: Path):
    mgr = _make_manager(tmp_path, redistill_every=2, min_pairs_for_distill=1)
    mgr.record_teacher_generation(["p1", "p2"], ["c1", "c2"], cycle=0)
    mgr.on_trained_cycle(0)
    mgr.on_trained_cycle(1)  # fires
    first_ckpt = mgr.current_checkpoint()
    assert first_ckpt is not None
    mgr.record_teacher_generation(["p3"], ["c3"], cycle=2)
    mgr.on_trained_cycle(2)  # counter back to 1, no fire
    assert mgr.current_checkpoint() is first_ckpt
    mgr.on_trained_cycle(3)  # counter=2, fires again
    second = mgr.current_checkpoint()
    assert second is not None and second is not first_ckpt
