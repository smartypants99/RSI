"""Integration tests for foom-component wire-up in ImprovementLoop.

Covers the contract between src/orchestrator/loop.py and the foom modules:

  1. meta_meta.record_cycle writes exactly the components_active keys the
     loop constructs — fast_student is no longer hardcoded False.
  2. FastStudentManager, as wired, actually mutates state when the loop
     drives it (record_teacher_generation → buffer grows; on_trained_cycle
     at redistill_every → distill fires, producing a checkpoint).
  3. The loop's synthesized components_active dict produces a valid JSONL
     row loadable by meta_meta.load_history.

These tests do NOT instantiate the full ImprovementLoop (that requires a
model loader and diagnostics engine). They drive the exact code paths the
loop drives, using the same signatures, so regressions in wiring surface
here instead of at live-GPU time.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.orchestrator import meta_meta as mm
from src.utils.fast_student import (
    DistillPair,
    FastStudentConfig,
    FastStudentManager,
    StudentCheckpoint,
)


# --- helpers -----------------------------------------------------------------


def _fake_distill_fn(pairs, out_dir: Path, cfg: FastStudentConfig) -> StudentCheckpoint:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "weights.bin").write_bytes(b"fake")
    return StudentCheckpoint(
        cycle=-1,
        path=out_dir,
        model_name=cfg.model_name,
        num_pairs=len(pairs),
        distill_seconds=0.0,
    )


def _fake_loader(ckpt: StudentCheckpoint):
    def _gen(prompts):
        return [f"student_out:{p}" for p in prompts]
    return _gen


def _loop_components_active(
    *,
    fast_student_enabled: bool,
    ood_enabled: bool = False,
    curriculum: bool = False,
    grow_every: int = 0,
    self_edit_every: int = 0,
    grpo: bool = False,
) -> dict[str, bool]:
    """Mirror the loop.py _record_meta_meta block's components_active."""
    return {
        "fast_student": fast_student_enabled,
        "ood": bool(ood_enabled),
        "curriculum_ratchet": bool(curriculum),
        "growth": int(grow_every) > 0,
        "self_edit": int(self_edit_every) > 0,
        "grpo": bool(grpo),
    }


# --- tests -------------------------------------------------------------------


def test_meta_meta_records_fast_student_true_when_manager_enabled(tmp_path: Path):
    """One "cycle" → one row in outputs/meta_meta_history.jsonl with
    fast_student=True when the manager is enabled, matching the loop's
    post-#3 wire-up (no longer hardcoded False)."""
    history_path = tmp_path / "meta_meta_history.jsonl"
    comps = _loop_components_active(fast_student_enabled=True, grow_every=15)

    mm.record_cycle(
        history_path,
        cycle_id=1,
        components_active=comps,
        held_out_delta=0.012,
        self_edit_tier=None,
        gradient_health=None,
    )

    assert history_path.exists()
    rows = history_path.read_text().strip().splitlines()
    assert len(rows) == 1
    rec = json.loads(rows[0])
    assert rec["cycle_id"] == 1
    assert rec["components_active"]["fast_student"] is True
    assert rec["components_active"]["growth"] is True
    assert rec["components_active"]["self_edit"] is False

    # Round-trip via the public loader.
    loaded = mm.load_history(history_path)
    assert len(loaded) == 1
    assert loaded[0].components_active["fast_student"] is True


def test_meta_meta_records_fast_student_false_when_disabled(tmp_path: Path):
    history_path = tmp_path / "meta_meta_history.jsonl"
    comps = _loop_components_active(fast_student_enabled=False)
    mm.record_cycle(
        history_path,
        cycle_id=3,
        components_active=comps,
        held_out_delta=0.0,
    )
    loaded = mm.load_history(history_path)
    assert len(loaded) == 1
    assert loaded[0].components_active["fast_student"] is False


def test_fast_student_wire_up_mutates_state_on_cycle(tmp_path: Path):
    """Drive the exact FastStudentManager calls loop.py makes and confirm
    state actually changes: buffer grows on harvest, distill fires on
    on_trained_cycle at the redistill boundary, checkpoint lands on disk."""
    cfg = FastStudentConfig(
        enabled=True,
        redistill_every=1,       # fire on the first trained cycle
        min_pairs_for_distill=2,
        checkpoint_root=tmp_path / "fast_student",
    )
    mgr = FastStudentManager(cfg, distill_fn=_fake_distill_fn, student_loader=_fake_loader)

    # Simulate the loop's post-solve harvest: (prompt, completion) pairs.
    prompts = [
        "Write a Python function to solve this problem.\n\nPROBLEM: add two numbers",
        "Write a Python function to solve this problem.\n\nPROBLEM: reverse string",
    ]
    completions = [
        "def solve(a, b):\n    return a + b\n",
        "def solve(s):\n    return s[::-1]\n",
    ]
    assert mgr.buffer_size() == 0
    mgr.record_teacher_generation(prompts, completions, cycle=1)
    assert mgr.buffer_size() == 2, "buffer must grow on harvest"

    # Simulate the loop's post-training hook. With redistill_every=1 this
    # fires the fake distill and produces a checkpoint on disk.
    assert mgr.current_checkpoint() is None
    mgr.on_trained_cycle(1)
    ckpt = mgr.current_checkpoint()
    assert ckpt is not None, "on_trained_cycle at boundary must fire distill"
    assert ckpt.cycle == 1, "checkpoint must be stamped with the trained cycle id"
    assert ckpt.num_pairs == 2
    assert (tmp_path / "fast_student" / "ckpt_1" / "weights.bin").exists()
    assert (tmp_path / "fast_student" / "ckpt_1" / "metadata.json").exists()

    # With a live checkpoint, generate_fn_for_cycle returns a callable — the
    # integration point that would route future propose/solve through the
    # student when the loop wires inference.
    gen = mgr.generate_fn_for_cycle(2)
    assert gen is not None
    out = gen(["hello"])
    assert out == ["student_out:hello"]


def test_fast_student_disabled_noops_harvest(tmp_path: Path):
    cfg = FastStudentConfig(enabled=False, checkpoint_root=tmp_path / "fs")
    mgr = FastStudentManager(cfg, distill_fn=_fake_distill_fn, student_loader=_fake_loader)
    mgr.record_teacher_generation(["p"], ["c"], cycle=1)
    assert mgr.buffer_size() == 0, "disabled manager must not buffer"
    mgr.on_trained_cycle(1)
    assert mgr.current_checkpoint() is None


def test_default_distill_fn_runs_end_to_end_on_cpu_fixture(tmp_path: Path, monkeypatch):
    """Exercise src.utils.fast_student._default_distill_fn on a CPU-sized
    fixture by injecting fake transformers/peft into sys.modules. Catches
    the regression class "production distill path has never run" that an
    injected-distill_fn test can't cover (cross-review issue B).

    Fake transformers exposes AutoTokenizer.from_pretrained →
    callable-tokenizer and AutoModelForCausalLM.from_pretrained → a tiny
    nn.Module with a .loss-returning forward + save_pretrained +
    merge_and_unload. The real torch.AdamW step runs on CPU in <1s; the
    test asserts a checkpoint lands on disk.
    """
    import sys

    import torch
    import torch.nn as nn
    from src.utils.fast_student import (
        DistillPair,
        FastStudentConfig,
        _default_distill_fn,
    )

    transformers = pytest.importorskip("transformers")

    class _FakeTokenizer:
        pad_token_id = 0
        eos_token = "<eos>"
        pad_token = None

        def __call__(self, text, return_tensors=None, truncation=False, max_length=None):
            n = max(2, min(32, len(text) // 4))
            ids = torch.arange(n, dtype=torch.long).unsqueeze(0)
            return {"input_ids": ids}

        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "tokenizer.json").write_text("{}")

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name, trust_remote_code=False):
            return _FakeTokenizer()

    class _FakeCausalOut:
        def __init__(self, loss):
            self.loss = loss

    class _FakeCausalLM(nn.Module):
        def __init__(self, vocab=64, dim=8):
            super().__init__()
            self.emb = nn.Embedding(vocab, dim)
            self.head = nn.Linear(dim, vocab)
            self._vocab = vocab

        def forward(self, input_ids=None, labels=None, **_):
            h = self.emb(input_ids)
            logits = self.head(h)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self._vocab), labels.view(-1))
            return _FakeCausalOut(loss)

        def merge_and_unload(self):
            return self

        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "model.safetensors").write_bytes(b"fake")

    class _FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(name, torch_dtype=None, trust_remote_code=False):
            return _FakeCausalLM()

    # Patch the real transformers module's attributes so the
    # `from transformers import AutoModelForCausalLM, AutoTokenizer` line
    # inside _default_distill_fn picks up our fakes, but the rest of
    # transformers (tokenization_utils_tokenizers etc.) remains intact.
    monkeypatch.setattr(transformers, "AutoTokenizer", _FakeAutoTokenizer, raising=True)
    monkeypatch.setattr(transformers, "AutoModelForCausalLM", _FakeAutoModelForCausalLM, raising=True)

    # Force peft ImportError branch (full fine-tune path) — exercises the
    # documented fallback. Setting sys.modules['peft']=None makes
    # `from peft import ...` raise ImportError even if peft is installed.
    monkeypatch.setitem(sys.modules, "peft", None)

    # ---- run ---------------------------------------------------------------
    cfg = FastStudentConfig(
        enabled=True,
        redistill_every=1,
        min_pairs_for_distill=1,
        student_num_epochs=1,
        student_learning_rate=1e-3,
        checkpoint_root=tmp_path / "fs",
        model_name="fake-student",
    )
    pairs = [
        DistillPair(prompt="PROMPT: add", completion="def solve(a,b): return a+b", cycle=1),
    ]
    out_dir = tmp_path / "ckpt"
    ckpt = _default_distill_fn(pairs, out_dir, cfg)

    assert (out_dir / "model.safetensors").exists()
    assert (out_dir / "tokenizer.json").exists()
    assert ckpt.num_pairs == 1
    assert ckpt.model_name == "fake-student"
    assert ckpt.path == out_dir


def test_harvest_still_runs_when_distill_inline_disabled(tmp_path: Path):
    """Validates cross-review issue A fix: with distill-inline OFF, harvest
    still accumulates pairs but on_trained_cycle is never called by the loop,
    so the buffer grows across cycles without triggering a GPU-heavy distill.
    This mirrors the gated path in loop.py when
    orchestrator.fast_student_distill_inline=False.
    """
    cfg = FastStudentConfig(
        enabled=True,
        redistill_every=1,
        min_pairs_for_distill=1,
        checkpoint_root=tmp_path / "fs",
    )
    mgr = FastStudentManager(cfg, distill_fn=_fake_distill_fn, student_loader=_fake_loader)
    # Loop would harvest on every solve_batch but suppress on_trained_cycle.
    for c in range(1, 4):
        mgr.record_teacher_generation([f"p{c}"], [f"c{c}"], cycle=c)
    assert mgr.buffer_size() == 3
    assert mgr.current_checkpoint() is None, (
        "no distill should have fired when on_trained_cycle is suppressed"
    )


def test_loop_style_cycle_writes_history_then_reloads(tmp_path: Path):
    """Full round-trip: loop-style record_cycle for three synthetic cycles,
    including one with fast_student=True, then re-read via load_history.
    Exercises the exact payload shape loop.py produces."""
    history_path = tmp_path / "meta_meta_history.jsonl"

    mm.record_cycle(
        history_path,
        cycle_id=1,
        components_active=_loop_components_active(fast_student_enabled=False),
        held_out_delta=0.0,
    )
    mm.record_cycle(
        history_path,
        cycle_id=2,
        components_active=_loop_components_active(
            fast_student_enabled=True, ood_enabled=True, curriculum=True,
        ),
        held_out_delta=0.015,
    )
    mm.record_cycle(
        history_path,
        cycle_id=3,
        components_active=_loop_components_active(
            fast_student_enabled=True, grow_every=15, self_edit_every=8,
        ),
        held_out_delta=-0.002,
    )

    history = mm.load_history(history_path)
    assert [r.cycle_id for r in history] == [1, 2, 3]
    assert [r.components_active["fast_student"] for r in history] == [False, True, True]
    # component_contributions runs without error on the synthesized history
    # (proves the shape is valid for downstream meta_meta consumers).
    contribs = mm.component_contributions(history)
    assert "fast_student" in contribs
