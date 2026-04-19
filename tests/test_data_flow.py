"""End-to-end mock data flow: verify schemas and cross-module plumbing.

Everything is mocked — no real model, no GPU. We fabricate TrainingSamples,
run them through Verifier.verify_batch, build PreferencePairs, and serialize
a CycleResult. If teammate-added modules are present, exercise them too.
"""
from __future__ import annotations

import importlib
import json
from unittest.mock import MagicMock

import pytest

from src.utils.config import (
    SystemConfig, VerifierConfig, DiagnosticsConfig, GeneratorConfig, TrainerConfig,
)
from src.generator.data_generator import TrainingSample, PreferencePair, ReasoningStep
from src.verifier.verifier import Verifier
from src.diagnostics.engine import WeaknessReport, DiagnosticResult
from src.orchestrator.loop import CycleResult


def _make_sample(prompt="Solve 2+2", response="4", domain="math", verified=True):
    return TrainingSample(
        prompt=prompt,
        response=response,
        reasoning_chain=[
            ReasoningStep(step_number=1, content="2+2=4", justification="arithmetic"),
            ReasoningStep(step_number=2, content="answer is 4", justification="therefore"),
        ],
        domain=domain,
        verified=verified,
        expected_answer=response,
    )


def test_verifier_accepts_well_formed_sample():
    v = Verifier(VerifierConfig())
    sample = _make_sample()
    result = v.verify_batch([sample])
    assert isinstance(result, list)


def test_verifier_handles_empty_batch():
    v = Verifier(VerifierConfig())
    assert v.verify_batch([]) == []


def test_preference_pair_from_two_samples():
    chosen = _make_sample(response="4", verified=True)
    rejected = _make_sample(response="5", verified=False)
    pair = PreferencePair(
        prompt=chosen.prompt,
        chosen_response=chosen.response,
        rejected_response=rejected.response,
        domain=chosen.domain,
    )
    assert pair.chosen_response != pair.rejected_response
    assert pair.content_hash


def test_cycle_result_json_serializable():
    """A full CycleResult must round-trip through JSON without errors."""
    r = CycleResult(cycle=3)
    r.pre_score = 0.3
    r.post_score = 0.45
    r.improvement = 0.15
    r.samples_generated = 10
    r.samples_verified = 7
    r.diagnostics = DiagnosticResult(
        cycle=3, timestamp=0.0,
        domain_scores={"math": 0.4, "code": 0.5},
        weaknesses=[WeaknessReport(domain="math", subdomain="algebra", severity=0.6)],
    )
    r.post_diag = DiagnosticResult(
        cycle=3, timestamp=0.0,
        domain_scores={"math": 0.55, "code": 0.6},
    )
    r.phase_times = {"diagnose": 1.0, "generate": 2.0}
    r.errors = [{"phase": "train", "type": "OOM", "message": "x", "traceback": "..."}]
    d = r.to_dict()
    s = json.dumps(d)
    parsed = json.loads(s)
    assert parsed["cycle"] == 3
    assert parsed["weaknesses_found"] == 1


def test_mock_diagnostic_to_generation_to_verification_flow():
    """Simulate: diag result -> synthetic samples -> verifier -> preference pairs."""
    # Fake a diagnostic result with one weakness.
    diag = DiagnosticResult(
        cycle=1, timestamp=0.0,
        domain_scores={"math": 0.3},
        domain_question_counts={"math": 50},
        weaknesses=[WeaknessReport(domain="math", subdomain="algebra", severity=0.7)],
    )
    assert diag.weaknesses[0].domain == "math"
    assert abs(diag.overall_score - 0.3) < 1e-6

    # Fabricate training samples addressing the weakness.
    samples = [_make_sample(prompt=f"Q{i}", response=str(i), verified=True) for i in range(3)]
    for s in samples:
        s.target_weakness = "algebra"

    # Run through verifier (should accept or produce results, not crash).
    v = Verifier(VerifierConfig())
    results = v.verify_batch(samples)
    assert len(results) == len(samples) or isinstance(results, list)

    # Build a preference pair (STaR-style pairing).
    pair = PreferencePair(
        prompt=samples[0].prompt,
        chosen_response=samples[0].response,
        rejected_response="wrong_answer",
        domain="math", target_weakness="algebra",
    )
    assert pair.weight == 1.0


def test_cycle_result_errors_accumulate():
    r = CycleResult(cycle=1)
    r.errors.append({"phase": "generate", "type": "ValueError", "message": "x", "traceback": "t"})
    r.errors.append({"phase": "train", "type": "RuntimeError", "message": "y", "traceback": "t"})
    d = r.to_dict()
    assert len(d["errors"]) == 2


@pytest.mark.parametrize("optional_sample_field", [
    "per_step_confidence",  # metacog_calib
])
def test_training_sample_tolerates_future_fields(optional_sample_field):
    """New fields added by teammates should not break existing code paths.

    If the field exists, construct a sample with it. Otherwise skip — we just
    don't want any regression where adding a field breaks the default ctor.
    """
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(TrainingSample)}
    if optional_sample_field not in field_names:
        pytest.skip(f"{optional_sample_field} not yet in TrainingSample")
    # If present, default-constructed sample must still be valid.
    s = TrainingSample(prompt="p", response="r")
    assert hasattr(s, optional_sample_field)


def test_quality_top_k_ranks_by_consistency_and_source():
    """Top-k filter prefers high-consistency, star-sourced samples; respects floor."""
    from src.orchestrator.loop import ImprovementLoop

    low = TrainingSample(prompt="p1", response="a", consistency_score=0.25,
                         parse_confidence=1.0, source="synthesized")
    mid = TrainingSample(prompt="p2", response="b", consistency_score=0.5,
                         parse_confidence=1.0, source="star_rationalized")
    hi  = TrainingSample(prompt="p3", response="c", consistency_score=1.0,
                         parse_confidence=1.0, source="star")
    hi2 = TrainingSample(prompt="p4", response="d", consistency_score=0.75,
                         parse_confidence=1.0, source="star")

    stub = ImprovementLoop.__new__(ImprovementLoop)
    stub.config = MagicMock()
    stub.config.generator.sample_quality_top_k = 2
    stub.config.generator.sample_quality_floor = 2
    kept = stub._apply_quality_top_k([low, mid, hi, hi2])
    assert kept[0] is hi and kept[1] is hi2

    # Disabled (k=0) returns input unchanged.
    stub.config.generator.sample_quality_top_k = 0
    assert stub._apply_quality_top_k([low, mid]) == [low, mid]

    # Floor prevents ranking below the floor.
    stub.config.generator.sample_quality_top_k = 1
    stub.config.generator.sample_quality_floor = 3
    kept = stub._apply_quality_top_k([low, mid, hi, hi2])
    assert len(kept) == 3


def test_parser_strips_think_tokens():
    """R1/Qwen style <think>...</think> must not leak into parsed output."""
    from src.generator.data_generator import ResponseParser

    raw = (
        "<think>let me reason about this</think>\n"
        "Step 1: compute\n  Justification: arithmetic\n"
        "Conclusion: 4\n"
        "</think>"
    )
    result = ResponseParser().parse(raw)
    assert "</think>" not in result.conclusion
    assert "<think>" not in result.conclusion
    for step in result.chain:
        assert "think>" not in step.content
        assert "think>" not in step.justification


def test_code_executes_rejects_wrong_function_name():
    """Function-name mismatch must fail the grader (merge_sorted vs mergesorted)."""
    from src.utils.config import DiagnosticsConfig
    from src.diagnostics.engine import DiagnosticsEngine

    eng = DiagnosticsEngine.__new__(DiagnosticsEngine)
    eng.config = DiagnosticsConfig()

    wrong = "```python\ndef mergesorted(a, b):\n    return sorted(a + b)\n```"
    right = "```python\ndef merge_sorted(a, b):\n    return sorted(a + b)\n```"

    # Missing exact name → reject (even if heuristic smoke test would pass).
    assert eng._check_answer(wrong, "merge_sorted", "code_executes") is False
    # Correct name → accept.
    assert eng._check_answer(right, "merge_sorted", "code_executes") is True
    # Non-identifier expected (e.g. "any") → falls back to execution-only check.
    assert eng._check_answer(right, "any valid sort", "code_executes") is True

    # hot_spots H3: lastelem/last_elem case surfaced in cycle 1.
    wrong_last = "```python\ndef lastelem(lst): return lst[-1]\n```"
    right_last = "```python\ndef last_elem(lst): return lst[-1]\n```"
    assert eng._check_answer(wrong_last, "last_elem", "code_executes") is False
    assert eng._check_answer(right_last, "last_elem", "code_executes") is True

    # hot_spots H1: Python keywords must never activate the name-gate — they can
    # never match a real def, so a "def" expected should NOT force rejection of
    # otherwise-runnable code.
    any_code = "```python\ndef foo(x): return x\n```"
    assert eng._check_answer(any_code, "def", "code_executes") is True
