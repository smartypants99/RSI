"""Task #10 speed-pass regression guards.

Each test pins a config knob or wire-up that was introduced/modified by the
speed-pass commit. If any of these fail, the cycle-time win is at risk of
silent regression.
"""
from __future__ import annotations

import pytest

from src.utils.config import (
    GeneratorConfig,
    OrchestratorConfig,
    SynthesisConfig,
    VLLMConfig,
)
from src.utils.fast_student import FastStudentConfig


# ─── Config defaults ─────────────────────────────────────────────────────────

def test_synthesis_defaults_speed_pass():
    cfg = SynthesisConfig()
    assert cfg.tasks_per_cycle == 12
    # Task #22 speed-round-2: 3 → 2 (reference + 1 fresh sample).
    assert cfg.candidates_per_problem == 2
    assert cfg.proposer_max_new_tokens == 600
    assert cfg.solver_max_new_tokens == 1200


def test_generator_defaults_speed_pass():
    cfg = GeneratorConfig()
    assert cfg.samples_per_weakness == 24


def test_orchestrator_quick_eval_defaults():
    cfg = OrchestratorConfig()
    # Task #22 speed-round-2: 128 → 96 (SE still matches revert threshold).
    assert cfg.heldout_quick_subsample_n == 96
    # Diff-analysis cycle 3 vs cycle 5: pinned to QUICK every cycle so
    # paired_delta is measuring the same population (same domain mix)
    # across cycles. See config comment on heldout_full_every for math.
    assert cfg.heldout_full_every == 9999


def test_orchestrator_quick_eval_validation():
    with pytest.raises(ValueError, match="heldout_quick_subsample_n"):
        OrchestratorConfig(heldout_quick_subsample_n=-1)
    with pytest.raises(ValueError, match="heldout_full_every"):
        OrchestratorConfig(heldout_full_every=0)


def test_orchestrator_quick_eval_disabled_zero():
    # 0 is a sentinel for "always full"; must not raise.
    cfg = OrchestratorConfig(heldout_quick_subsample_n=0)
    assert cfg.heldout_quick_subsample_n == 0


def test_vllm_max_num_seqs_default():
    # Task #18 step 1: 32 → 48 on the corrected Qwen2-32B GQA KV budget.
    # Pinned so any accidental revert (e.g., to vLLM default 256, which
    # overcommits KV on 48GB A6000) surfaces here.
    cfg = VLLMConfig()
    assert cfg.max_num_seqs == 48


def test_vllm_max_num_seqs_zero_is_vllm_default():
    cfg = VLLMConfig(max_num_seqs=0)
    assert cfg.max_num_seqs == 0


def test_vllm_max_num_seqs_negative_raises():
    with pytest.raises(ValueError, match="max_num_seqs"):
        VLLMConfig(max_num_seqs=-1)


def test_vllm_chunked_prefill_default_true():
    """Task #18 step 2: chunked prefill default ON for the 120-prompt
    solve batch. Regressing this loses ~2-3s per prefill wave."""
    cfg = VLLMConfig()
    assert cfg.enable_chunked_prefill is True


def test_vllm_log_throughput_stats_default_false():
    """Task #18 step 3: default OFF — log volume non-trivial on long runs.
    Flip on for a single diagnostic cycle to verify prefix-cache hit-rate."""
    cfg = VLLMConfig()
    assert cfg.log_throughput_stats is False


def test_vllm_log_throughput_stats_flips_disable_log_stats():
    """When log_throughput_stats=True, vLLM's disable_log_stats kwarg must
    land as False so the engine emits prompt_throughput + num_cached_tokens."""
    import sys, types
    fake_vllm = types.ModuleType("vllm")
    captured: dict = {}

    class _FakeLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def get_tokenizer(self):
            class _T:
                pad_token = "[PAD]"
                eos_token = "[EOS]"
                pad_token_id = 0
            return _T()

    fake_vllm.LLM = _FakeLLM
    fake_vllm.SamplingParams = object
    sys.modules["vllm"] = fake_vllm
    try:
        import src.utils.vllm_backend as vb
        vb._VLLM_AVAILABLE = True

        loader_off = vb.VLLMModelLoader(
            model_path="stub/base", log_throughput_stats=False,
        )
        loader_off._load_vllm()
        assert captured.get("disable_log_stats") is True

        captured.clear()
        loader_on = vb.VLLMModelLoader(
            model_path="stub/base", log_throughput_stats=True,
        )
        loader_on._load_vllm()
        assert captured.get("disable_log_stats") is False
    finally:
        sys.modules.pop("vllm", None)


def test_vllm_chunked_prefill_threaded_to_loader_kwargs():
    """The VLLMModelLoader stores the chunked-prefill flag and would pass
    it to vllm.LLM(). Stubs `vllm` import so no GPU is touched."""
    import sys, types
    fake_vllm = types.ModuleType("vllm")
    captured: dict = {}

    class _FakeLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def get_tokenizer(self):
            class _T:
                pad_token = "[PAD]"
                eos_token = "[EOS]"
                pad_token_id = 0
            return _T()

    fake_vllm.LLM = _FakeLLM
    fake_vllm.SamplingParams = object
    sys.modules["vllm"] = fake_vllm
    try:
        from src.utils.vllm_backend import VLLMModelLoader, _check_vllm  # noqa: F401
        # Force the module to re-check vLLM availability via the fake.
        import src.utils.vllm_backend as vb
        vb._VLLM_AVAILABLE = True

        loader = VLLMModelLoader(
            model_path="stub/base",
            enable_chunked_prefill=True,
        )
        # Bypass CUDA check: directly call _load_vllm.
        loader._load_vllm()
        assert captured.get("enable_chunked_prefill") is True

        captured.clear()
        loader2 = VLLMModelLoader(
            model_path="stub/base",
            enable_chunked_prefill=False,
        )
        loader2._load_vllm()
        # When disabled, the loader simply doesn't add the kwarg (vLLM's own
        # default applies). Assert the key is absent, not False.
        assert "enable_chunked_prefill" not in captured
    finally:
        sys.modules.pop("vllm", None)


def test_fast_student_redistill_every_speed_pass():
    cfg = FastStudentConfig()
    assert cfg.redistill_every == 2


# ─── Wire-up: proposer/solver caps propagate to generate_batch ───────────────

class _CapturingLoader:
    """Records the max_new_tokens each generate_batch call received."""
    def __init__(self):
        self.calls: list[int] = []
        self.config = type("M", (), {"max_seq_length": 4096})()

    def generate_batch(self, prompts, *, max_new_tokens, temperature, top_p, stop=None):
        self.calls.append(int(max_new_tokens))
        return ["" for _ in prompts]


def test_proposer_cap_propagates_to_generate_batch():
    from src.generator.task_synthesizer import TaskSynthesizer
    loader = _CapturingLoader()
    synth_cfg = SynthesisConfig(proposer_max_new_tokens=600)
    synth = TaskSynthesizer(synth_cfg, model_loader=loader)
    synth._generate_many(["p1", "p2"])  # proposer path
    assert loader.calls == [600]


def test_solver_cap_propagates_to_generate_batch():
    from src.generator.task_synthesizer import TaskSynthesizer
    loader = _CapturingLoader()
    synth_cfg = SynthesisConfig(solver_max_new_tokens=1200)
    synth = TaskSynthesizer(synth_cfg, model_loader=loader)
    synth._generate_many_code(["p1"])  # solver path via ```python fence
    assert loader.calls == [1200]


# ─── Wire-up: quick-eval cycle-schedule logic ────────────────────────────────

def test_quick_eval_cycle_1_is_full():
    # Cycle 1 always runs full so the base reference lands on a full draw.
    cfg = OrchestratorConfig(heldout_quick_subsample_n=128, heldout_full_every=5)
    cycle = 1
    is_full = (
        cycle == 1
        or cfg.heldout_full_every <= 1
        or (cycle % max(1, cfg.heldout_full_every) == 0)
    )
    assert is_full is True


def test_quick_eval_cycle_schedule_mid():
    cfg = OrchestratorConfig(heldout_quick_subsample_n=128, heldout_full_every=5)
    # Cycle 2, 3, 4 are quick; cycle 5, 10 are full; cycle 6..9 are quick.
    full = lambda c: (
        c == 1
        or cfg.heldout_full_every <= 1
        or (c % max(1, cfg.heldout_full_every) == 0)
    )
    assert full(2) is False
    assert full(4) is False
    assert full(5) is True
    assert full(6) is False
    assert full(10) is True


def test_quick_eval_disabled_always_full():
    cfg = OrchestratorConfig(heldout_full_every=1)  # every cycle full
    for c in range(1, 20):
        assert (
            c == 1
            or cfg.heldout_full_every <= 1
            or (c % max(1, cfg.heldout_full_every) == 0)
        ) is True
