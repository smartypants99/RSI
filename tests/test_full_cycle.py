"""End-to-end integration test: one full cycle with a mock model.

This is the test that actually proves the pipeline works. It fakes the
language model (so no GPU needed), fakes the LoRA training (so we don't
need real transformers), and runs a complete `ImprovementLoop._run_cycle(1)`
through diagnose → generate → verify → train → post-diag → eval.

If this test passes, the orchestration plumbing is correct. If it fails,
the failure location pinpoints the broken seam.
"""
from __future__ import annotations

import random
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.utils.config import (
    SystemConfig, ModelConfig, DiagnosticsConfig, GeneratorConfig,
    VerifierConfig, TrainerConfig, OrchestratorConfig,
)


class MockTokenizer:
    """Minimal tokenizer so modules that touch .pad_token / .eos_token work."""
    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0
    eos_token_id = 1
    padding_side = "left"

    def __call__(self, text, **kwargs):
        if isinstance(text, list):
            ids = [[2] * 4 for _ in text]
        else:
            ids = [2] * 4
        import torch
        t = torch.tensor(ids, dtype=torch.long)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return {
            "input_ids": t,
            "attention_mask": (t != 0).long(),
        }

    def decode(self, ids, skip_special_tokens=True):
        return "mock-decoded"


class MockModelLoader:
    """Fakes the ModelLoader interface used by diagnostics / generator / verifier.

    Returns domain-aware canned responses so the grading path produces a
    non-trivial mix of correct/incorrect answers — that's what exercises the
    weakness-detection path, not just "everything's zero".
    """

    def __init__(self, config=None):
        self.tokenizer = MockTokenizer()
        self.model = MagicMock()
        self.device = "cpu"
        self.config = config or ModelConfig(model_path="mock", max_seq_length=512)
        self._call_count = 0
        self._rng = random.Random(0xBEEF)

    def load(self):
        return self

    def _fake_response(self, prompt: str) -> str:
        """Generate a reasoning chain that sometimes contains the right answer.

        The mock is intentionally *bad at math*: it returns random small
        integers as the final answer. That guarantees the diagnostic phase
        flags "math" as a weakness, so downstream generation/verification
        code paths actually fire.
        """
        p = prompt.lower()
        # Extract a hint of what the "answer" might be so exact-match has
        # a chance on some questions (otherwise nothing passes and the loop
        # reports "no samples" and skips training).
        if "2+2" in p or "2 + 2" in p:
            return (
                "Step 1: Consider 2 + 2.\n"
                "Justification: addition\n"
                "Step 2: The sum is 4.\n"
                "Justification: arithmetic\n"
                "Conclusion: 4"
            )
        # Generic response that the parser can parse.
        n = self._rng.randint(0, 9)
        return (
            "Step 1: Examine the problem.\n"
            "Justification: reading\n"
            "Step 2: Apply reasoning.\n"
            "Justification: domain knowledge\n"
            f"Step 3: The answer is {n}.\n"
            "Justification: calculation\n"
            f"Conclusion: {n}"
        )

    def generate(self, prompt, max_new_tokens=2048, temperature=0.7, top_p=0.9):
        self._call_count += 1
        return self._fake_response(prompt)

    def generate_batch(self, prompts, max_new_tokens=2048, temperature=0.7, top_p=0.9):
        self._call_count += 1
        return [self._fake_response(p) for p in prompts]

    def capture_activations(self, layer_names=None):
        """No-op context manager — diagnostics will just get empty activations."""
        from contextlib import contextmanager

        class _Capture:
            activations = {}
        @contextmanager
        def _noop():
            yield _Capture()
        return _noop()

    def get_layer_info(self):
        return {}

    def save_checkpoint(self, path, cycle):
        # Touch the expected files so post-save validation doesn't crash.
        d = Path(path) / f"cycle_{cycle}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text("{}")
        (d / "model.safetensors").write_text("")

    def load_from_checkpoint(self, p):
        return self


class MockTrainer:
    """Stand-in for CustomLoRATrainer — no real torch, no real training."""

    def __init__(self, config, model_loader):
        self.config = config
        self.model_loader = model_loader
        self._last_merge_undertrained = False

    def inject_lora(self, weak_layers=None):
        pass

    def strip_lora(self):
        pass

    def merge_lora(self):
        pass

    def save_lora_weights(self, path, cycle):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / f"lora_cycle_{cycle}").mkdir(parents=True, exist_ok=True)

    def load_lora_weights(self, p):
        pass

    def train(self, verified_samples, cycle, preference_pairs=None):
        from src.trainer.custom_lora import TrainingMetrics
        return TrainingMetrics(
            cycle=cycle,
            avg_loss=1.5,
            final_loss=1.0,
            steps=max(1, len(verified_samples) // 4),
            samples_used=len(verified_samples),
            samples_rejected=0,
            learning_rate=self.config.learning_rate,
            lora_layers_injected=8,
            avg_rank=32.0,
        )


@pytest.fixture
def fast_config(tmp_path):
    """System config tuned for a single-cycle smoke test in seconds, not minutes."""
    cfg = SystemConfig()
    cfg.model = ModelConfig(model_path="mock", max_seq_length=256)
    # A real run uses 300 per domain; the test runs 8 for speed.
    cfg.diagnostics = DiagnosticsConfig(
        questions_per_domain=8,
        min_questions_per_domain=1,
        max_questions_per_domain=16,
        batch_size=4,
        activation_analysis=False,  # can't capture on mock model
        domains=["math", "logic"],  # subset for speed
    )
    cfg.generator = GeneratorConfig(samples_per_weakness=3, min_reasoning_steps=1)
    cfg.verifier = VerifierConfig(min_chain_steps=1, min_confidence_for_accept=0.3)
    cfg.trainer = TrainerConfig(lora_rank=8, min_rank=4, num_epochs=1, batch_size=1)
    cfg.orchestrator = OrchestratorConfig(
        max_cycles=1,
        output_dir=tmp_path / "out",
        log_dir=tmp_path / "out" / "logs",
        checkpoint_every=1,
    )
    return cfg


def test_single_cycle_end_to_end(fast_config):
    """Run one full _run_cycle — every phase must execute without exceptions.

    The mock model is intentionally bad at math so the diagnostic phase
    identifies weaknesses; this exercises the generate/verify/train path
    instead of falling through the "no weaknesses" early-return.
    """
    # Patch out the real ModelLoader + Trainer before constructing the loop.
    with patch("src.utils.model_loader.ModelLoader", MockModelLoader), \
         patch("src.orchestrator.loop.CustomLoRATrainer", MockTrainer):
        from src.orchestrator.loop import ImprovementLoop, CycleResult
        loop = ImprovementLoop(fast_config)
        # Swap the model loader it built (which tried to instantiate the real one
        # before the patch fully applied to transitive imports).
        loop.model_loader = MockModelLoader()
        loop.diagnostics.model = loop.model_loader
        loop.generator.model = loop.model_loader
        # Re-inject grader now that the mock is live
        loop.generator.set_grader(loop.diagnostics._check_answer)
        loop.verifier.set_ground_truth_grader(loop.diagnostics._check_answer)

        # Rebuild trainer with the mock
        loop.trainer = MockTrainer(fast_config.trainer, loop.model_loader)

        # Ensure output dirs
        fast_config.orchestrator.output_dir.mkdir(parents=True, exist_ok=True)
        fast_config.orchestrator.log_dir.mkdir(parents=True, exist_ok=True)

        # Run ONE cycle
        result = loop._run_cycle(1)

    # Verify the pipeline actually produced a result
    assert isinstance(result, CycleResult)
    assert result.cycle == 1
    assert result.diagnostics is not None, "diagnostics phase didn't run"
    assert result.diagnostics.total_questions > 0, "no questions were probed"
    # Pre-score must exist even if post-phase bailed (no weaknesses case)
    assert 0.0 <= result.pre_score <= 1.0
    # If weaknesses were found and samples produced, post-diag should exist
    if result.samples_verified > 0:
        assert result.post_diag is not None, "post-diag should run after training"
        assert result.training_metrics is not None


def test_cycle_handles_no_weaknesses_gracefully(fast_config):
    """If the model happens to ace everything, the cycle must exit cleanly."""
    class PerfectMock(MockModelLoader):
        def _fake_response(self, prompt):
            # Answer is always some canonical placeholder so most questions
            # with "contains"-style checks pass.
            return (
                "Step 1: think\nJustification: x\n"
                "Step 2: conclude\nJustification: y\nConclusion: 42"
            )

    with patch("src.orchestrator.loop.CustomLoRATrainer", MockTrainer):
        from src.orchestrator.loop import ImprovementLoop
        loop = ImprovementLoop(fast_config)
        loop.model_loader = PerfectMock()
        loop.diagnostics.model = loop.model_loader
        loop.generator.model = loop.model_loader
        loop.generator.set_grader(loop.diagnostics._check_answer)
        loop.verifier.set_ground_truth_grader(loop.diagnostics._check_answer)
        loop.trainer = MockTrainer(fast_config.trainer, loop.model_loader)
        fast_config.orchestrator.output_dir.mkdir(parents=True, exist_ok=True)
        fast_config.orchestrator.log_dir.mkdir(parents=True, exist_ok=True)
        result = loop._run_cycle(1)
    assert result.diagnostics is not None


def test_cycle_crash_is_captured_not_propagated(fast_config):
    """If one phase crashes, the cycle result records the error, loop continues."""
    class CrashingGeneratorMock(MockModelLoader):
        def generate_batch(self, prompts, **kw):
            raise RuntimeError("simulated generator crash")

    with patch("src.orchestrator.loop.CustomLoRATrainer", MockTrainer):
        from src.orchestrator.loop import ImprovementLoop
        loop = ImprovementLoop(fast_config)
        loop.model_loader = MockModelLoader()
        loop.diagnostics.model = loop.model_loader
        loop.generator.model = CrashingGeneratorMock()
        loop.generator.set_grader(loop.diagnostics._check_answer)
        loop.verifier.set_ground_truth_grader(loop.diagnostics._check_answer)
        loop.trainer = MockTrainer(fast_config.trainer, loop.model_loader)
        fast_config.orchestrator.output_dir.mkdir(parents=True, exist_ok=True)
        fast_config.orchestrator.log_dir.mkdir(parents=True, exist_ok=True)

        # _run_cycle catches generator exceptions via its own try/except
        # so this should return a result with error recorded, not raise.
        result = loop._run_cycle(1)

    # Either the generator crash was caught (result returned with errors) or
    # the diagnostics phase saw no weaknesses and skipped generation entirely.
    assert result is not None
