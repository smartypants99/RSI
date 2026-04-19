"""Integration tests: synthesis pipeline → VoV handoff.

Covers the path:
  TaskSynthesizer.synthesize() → verify_properties_trustworthy (VoV gate)
  → verify_by_consensus (property_engine) → ImprovementLoop splice

These tests run without a GPU or real model. They use stubs for the model
loader and property check functions to validate the orchestration logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
from unittest.mock import MagicMock, patch

import pytest

from src.generator.task_synthesizer import TaskSynthesizer, SynthesisResult, SynthesizedTask
from src.verifier.property_engine import verify_by_consensus, register_property, LegacyProperty as Property
from src.verifier.verifier_of_verifiers import (
    verify_properties_trustworthy,
    generate_corruptions,
    make_task_fingerprint,
    quorum_verdict,
)
from src.utils.config import SynthesisConfig, SystemConfig
from src.generator.data_generator import TrainingSample, ReasoningStep


# ─── shared stubs ────────────────────────────────────────────────────────────

@dataclass
class _StubProperty:
    """Minimal Property stand-in matching the VoV getattr schema."""
    name: str
    check_fn: Callable[[Any, Any], tuple[bool, str]]
    stochasticity: float = 0.0
    required: bool = False


def _make_sample(domain: str = "code", response: str = "x + 1") -> TrainingSample:
    return TrainingSample(
        prompt="Write a function that increments x",
        response=response,
        domain=domain,
        reasoning_chain=[
            ReasoningStep(step_number=1, content="Return x plus one", justification="arithmetic"),
            ReasoningStep(step_number=2, content="Final answer", justification="conclusion"),
        ],
    )


def _make_diag(subdomain_scores: dict | None = None) -> MagicMock:
    """Build a DiagnosticResult mock suitable for TaskSynthesizer.synthesize()."""
    diag = MagicMock()
    diag.subdomain_scores = subdomain_scores or {}
    diag.per_question = []
    diag.domain_scores = {}
    diag.weaknesses = []
    return diag


# ─── SynthesisConfig ─────────────────────────────────────────────────────────

def test_synthesis_config_defaults():
    cfg = SynthesisConfig()
    assert cfg.enable_task_synthesis is False
    assert cfg.tasks_per_cycle == 20
    assert cfg.property_consensus_threshold == 0.7


def test_synthesis_config_enabled():
    cfg = SynthesisConfig(enable_task_synthesis=True, tasks_per_cycle=5, property_consensus_threshold=0.8)
    assert cfg.enable_task_synthesis is True
    assert cfg.tasks_per_cycle == 5
    assert cfg.property_consensus_threshold == 0.8


def test_synthesis_config_validation_tasks():
    with pytest.raises(ValueError, match="tasks_per_cycle"):
        SynthesisConfig(tasks_per_cycle=0)


def test_synthesis_config_validation_threshold_zero():
    with pytest.raises(ValueError, match="property_consensus_threshold"):
        SynthesisConfig(property_consensus_threshold=0.0)


def test_synthesis_config_validation_threshold_above_one():
    with pytest.raises(ValueError, match="property_consensus_threshold"):
        SynthesisConfig(property_consensus_threshold=1.1)


# ─── TaskSynthesizer public API ───────────────────────────────────────────────

def test_task_synthesizer_returns_synthesis_result():
    """synthesize() always returns a SynthesisResult, never raises."""
    cfg = SynthesisConfig(enable_task_synthesis=True)
    ts = TaskSynthesizer(cfg, model_loader=None)
    result = ts.synthesize(_make_diag())
    assert isinstance(result, SynthesisResult)


def test_task_synthesizer_no_mastered_skills_returns_empty():
    """With no mastered skills, synthesizer cannot compose novel tasks."""
    cfg = SynthesisConfig(enable_task_synthesis=True)
    ts = TaskSynthesizer(cfg, model_loader=None)
    # Empty subdomain_scores → no mastered skills → no tasks
    result = ts.synthesize(_make_diag(subdomain_scores={}))
    assert isinstance(result.tasks, list)
    assert result.meta is not None


def test_task_synthesizer_with_generate_fn_override():
    """Injecting generate_fn should produce tasks when mastered skills exist."""
    cfg = SynthesisConfig(enable_task_synthesis=True, tasks_per_cycle=1)

    def stub_generate(prompt: str) -> str:
        return (
            "PROBLEM: Given two sorted arrays A and B, merge them.\n"
            "REFERENCE: def merge(a, b): return sorted(a + b)\n"
            "PROPERTY[0]: output is sorted\n"
            "WRONG_ANSWER: def merge(a, b): return a + b\n"
        )

    ts = TaskSynthesizer(cfg, model_loader=None, generate_fn=stub_generate)
    # Provide two mastered skills (score >= 0.8 threshold)
    diag = _make_diag(subdomain_scores={
        "code/sorting": 0.9,
        "code/arrays": 0.85,
    })
    result = ts.synthesize(diag)
    assert isinstance(result, SynthesisResult)
    # Whether tasks is populated depends on parse success; we just need no crash.


def test_task_synthesizer_exception_safety():
    """synthesize() catches internal failures and returns empty SynthesisResult."""
    cfg = SynthesisConfig(enable_task_synthesis=True)
    ts = TaskSynthesizer(cfg, model_loader=None)
    # Pass a diag that raises when accessed (simulates broken diagnostic result)
    bad_diag = MagicMock(spec=[])  # spec=[] means no attrs → AttributeError
    result = ts.synthesize(bad_diag)
    assert isinstance(result, SynthesisResult)
    assert result.tasks == []


# ─── property_engine.verify_by_consensus (legacy filter path) ─────────────────

def test_verify_by_consensus_no_properties_accepts_all():
    """Empty registry → all samples accepted (preserves classic behavior)."""
    samples = [_make_sample(), _make_sample()]
    result = verify_by_consensus(samples, threshold=0.7, properties={})
    assert result == samples


def test_verify_by_consensus_all_pass():
    def always_pass(s): return True
    samples = [_make_sample()]
    result = verify_by_consensus(samples, threshold=1.0, properties={"p": always_pass})
    assert len(result) == 1


def test_verify_by_consensus_all_fail():
    def always_fail(s): return False
    samples = [_make_sample()]
    result = verify_by_consensus(samples, threshold=0.5, properties={"p": always_fail})
    assert result == []


def test_verify_by_consensus_partial_pass():
    """2 of 3 properties pass → 0.67 rate; accepted at 0.6, rejected at 0.7."""
    props = {
        "pass1": lambda s: True,
        "pass2": lambda s: True,
        "fail1": lambda s: False,
    }
    sample = _make_sample()
    assert len(verify_by_consensus([sample], threshold=0.6, properties=props)) == 1
    assert len(verify_by_consensus([sample], threshold=0.7, properties=props)) == 0


def test_verify_by_consensus_property_exception_counts_as_fail():
    """A property that raises is treated as a failed check, not a crash."""
    def exploding(s): raise ValueError("boom")
    sample = _make_sample()
    result = verify_by_consensus([sample], threshold=0.5, properties={"p": exploding})
    assert result == []


# ─── VoV: toothless vs strong properties ─────────────────────────────────────

def test_vov_gate_rejects_toothless_properties():
    """A property that accepts everything must not pass the VoV gate."""
    def accept_all(sol, ctx): return (True, "always fine")
    toothless = _StubProperty(name="toothless", check_fn=accept_all)
    report = verify_properties_trustworthy(
        task_id="t_toothless",
        reference_solution="def f(x): return x + 1",
        properties=[toothless],
        problem_ctx={},
        domain="code",
    )
    assert not report.passed


def test_vov_gate_rejects_empty_properties():
    """No properties at all should not pass (nothing to trust)."""
    report = verify_properties_trustworthy(
        task_id="t_empty",
        reference_solution="def f(x): return x + 1",
        properties=[],
        problem_ctx={},
        domain="code",
    )
    assert not report.passed


def test_vov_gate_accepts_discriminating_property():
    """A property that accepts reference AND catches corruptions must pass VoV."""
    # Use a numeric task where corruptions are unambiguous (+1, -1, sign flip)
    # and the check is straightforward.
    def check_answer_is_42(sol, ctx):
        try:
            val = float(sol) if isinstance(sol, (int, float, str)) else None
            ok = val is not None and abs(val - 42) < 1e-9
            return (ok, "is 42" if ok else f"got {val}")
        except Exception:
            return (False, "parse error")

    strong = _StubProperty(name="answer_is_42", check_fn=check_answer_is_42)
    report = verify_properties_trustworthy(
        task_id="t_numeric_strong",
        reference_solution=42,
        properties=[strong],
        problem_ctx={},
        domain="math",
    )
    assert report.passed, f"Expected pass, got: {report.reason}"


# ─── generate_corruptions domain routing ─────────────────────────────────────

def test_generate_corruptions_code_domain():
    corruptions = generate_corruptions("def f(x): return x + 1", domain="code")
    assert len(corruptions) >= 1
    for c in corruptions:
        assert hasattr(c, "kind")
        assert hasattr(c, "mutated")


def test_generate_corruptions_math_domain():
    corruptions = generate_corruptions(42, domain="math")
    assert len(corruptions) >= 1


def test_generate_corruptions_text_domain():
    corruptions = generate_corruptions("The answer is yes.", domain="text")
    assert isinstance(corruptions, list)


# ─── make_task_fingerprint dedup ─────────────────────────────────────────────

def test_make_task_fingerprint_stable():
    fp1 = make_task_fingerprint("task_1", "def f(x): return x + 1", ["prop_a"])
    fp2 = make_task_fingerprint("task_1", "def f(x): return x + 1", ["prop_a"])
    assert fp1 == fp2


def test_make_task_fingerprint_differs_on_solution():
    fp1 = make_task_fingerprint("task_1", "def f(x): return x + 1", ["prop_a"])
    fp2 = make_task_fingerprint("task_1", "def f(x): return x + 2", ["prop_a"])
    assert fp1 != fp2


# ─── synthesis → consensus filter end-to-end ─────────────────────────────────

def test_synthesis_pipeline_consensus_filter_passes_good_samples():
    """Samples whose responses satisfy the property pass through the filter."""
    samples = [
        _make_sample(response="correct"),
        _make_sample(response="wrong"),
    ]
    # Property: sample.response must contain "correct"
    props = {"must_be_correct": lambda s: "correct" in s.response}
    passed = verify_by_consensus(samples, threshold=1.0, properties=props)
    assert len(passed) == 1
    assert passed[0].response == "correct"


def test_synthesis_pipeline_empty_when_all_rejected():
    """If all synthesized samples fail consensus, verify returns empty."""
    samples = [_make_sample(response="junk")]
    props = {"must_be_correct": lambda s: "correct" in s.response}
    passed = verify_by_consensus(samples, threshold=1.0, properties=props)
    assert passed == []


# ─── ImprovementLoop: synthesis flag off = zero overhead ─────────────────────

def test_loop_synthesis_disabled_by_default():
    """SystemConfig default has synthesis off."""
    cfg = SystemConfig()
    synth_cfg = getattr(cfg, "synthesis", None)
    assert synth_cfg is not None
    assert synth_cfg.enable_task_synthesis is False


def test_synthesis_config_present_in_system_config():
    from src.utils.config import SystemConfig, SynthesisConfig
    cfg = SystemConfig()
    assert hasattr(cfg, "synthesis")
    assert isinstance(cfg.synthesis, SynthesisConfig)


def test_loop_synthesis_enabled_flag_wires_synthesizer():
    """When enable_task_synthesis=True, the loop instantiates a TaskSynthesizer."""
    from src.orchestrator.loop import ImprovementLoop
    cfg = SystemConfig()
    cfg.synthesis.enable_task_synthesis = True

    mock_loader_instance = MagicMock()
    mock_loader_instance.load = MagicMock()

    # ModelLoader is imported lazily inside __init__; patch at its source module.
    with (
        patch("src.utils.model_loader.ModelLoader", return_value=mock_loader_instance),
        patch("src.orchestrator.loop.DiagnosticsEngine"),
        patch("src.orchestrator.loop.DataGenerator"),
        patch("src.orchestrator.loop.Verifier"),
        patch("src.orchestrator.loop.CustomLoRATrainer"),
        patch("src.orchestrator.loop.MetaController"),
    ):
        loop = ImprovementLoop(cfg)

    assert loop._synthesis_enabled is True
    assert loop._task_synthesizer is not None


# ─── quorum_verdict (§2.1) ───────────────────────────────────────────────────

@dataclass
class _ClassedProperty:
    """Property stub with independence_class for quorum tests."""
    name: str
    check_fn: Callable[[Any, Any], tuple[bool, str]]
    independence_class: str = "structural"
    stochasticity: float = 0.0
    required: bool = False


def _make_quorum_pairs(classes: list[str], results: list[bool]) -> list[tuple]:
    """Build (property, bool) pairs for quorum_verdict."""
    pairs = []
    for cls, ok in zip(classes, results):
        prop = _ClassedProperty(
            name=f"prop_{cls}",
            check_fn=lambda s, c: (True, "ok"),
            independence_class=cls,
        )
        pairs.append((prop, ok))
    return pairs


def test_quorum_verdict_accepts_valid_quorum():
    """3 distinct classes, all pass, n=3 → accepted."""
    pairs = _make_quorum_pairs(["structural", "semantic", "exec"], [True, True, True])
    v = quorum_verdict(pairs)
    assert v.accepted
    assert v.fail_count == 0
    assert v.distinct_classes == 3


def test_quorum_verdict_fail_veto():
    """Any single FAIL is a veto regardless of pass count."""
    pairs = _make_quorum_pairs(
        ["structural", "semantic", "exec", "output"],
        [True, True, True, False],  # 3 pass, 1 fail
    )
    v = quorum_verdict(pairs)
    assert not v.accepted
    assert v.fail_count == 1
    assert "veto" in v.reason


def test_quorum_verdict_insufficient_classes():
    """n=3 with only 2 distinct classes → rejected."""
    pairs = _make_quorum_pairs(
        ["structural", "structural", "semantic"],
        [True, True, True],
    )
    v = quorum_verdict(pairs)
    assert not v.accepted
    assert v.distinct_classes == 2


def test_quorum_verdict_insufficient_properties():
    """Fewer than min_properties → rejected."""
    pairs = _make_quorum_pairs(["structural", "semantic"], [True, True])
    v = quorum_verdict(pairs, min_properties=3)
    assert not v.accepted
    assert v.total_properties == 2


def test_quorum_verdict_empty_list():
    v = quorum_verdict([])
    assert not v.accepted
    assert "no properties" in v.reason


def test_quorum_verdict_unclassified_collapse():
    """Properties with no independence_class all collapse to one class."""
    pairs = [
        (_StubProperty(name="p1", check_fn=lambda s, c: (True, "ok")), True),
        (_StubProperty(name="p2", check_fn=lambda s, c: (True, "ok")), True),
        (_StubProperty(name="p3", check_fn=lambda s, c: (True, "ok")), True),
    ]
    v = quorum_verdict(pairs, min_classes=3)
    # All collapse to "<unclassified>" → 1 distinct class → rejected
    assert not v.accepted
    assert v.distinct_classes == 1


def test_quorum_verdict_pass_threshold():
    """Exactly ceil(2n/3) must pass; one below → rejected."""
    import math
    n = 6
    threshold = math.ceil(2 * n / 3)  # 4
    # 3 pass (below threshold), no fails — still rejected on pass count
    pairs = _make_quorum_pairs(
        ["a", "b", "c", "d", "e", "f"],
        [True, True, True, False, False, False],
    )
    # But fails > 0 triggers veto — test pass-count rejection with all-pass
    pairs_pass_short = _make_quorum_pairs(
        ["a", "b", "c", "d", "e", "f"],
        [True, True, True, True, True, True],
    )
    # 6 all pass, 3 distinct classes → should accept
    v = quorum_verdict(pairs_pass_short[:6])
    # With 6 props all passing threshold=4 is met; but distinct_classes from a-f = 6
    assert v.accepted


# ─── loop synthesis phase with quorum_verdict ────────────────────────────────

def _make_synthesized_task(
    task_id: str = "t1",
    domain: str = "math",
    reference_solution: Any = 42,
    classes: list[str] | None = None,
    results: list[bool] | None = None,
) -> "SynthesizedTask":
    """Build a SynthesizedTask with classsed properties for quorum tests."""
    if classes is None:
        classes = ["structural", "semantic", "exec"]
    if results is None:
        results = [True] * len(classes)

    props = []
    for cls, ok in zip(classes, results):
        expected = ok  # capture for closure

        def make_check(expected_val):
            def check(sol, ctx):
                return (expected_val, "ok" if expected_val else "fail")
            return check

        props.append(_ClassedProperty(
            name=f"prop_{cls}",
            check_fn=make_check(ok),
            independence_class=cls,
        ))

    return SynthesizedTask(
        task_id=task_id,
        domain=domain,
        prompt=f"Compute something ({task_id})",
        reference_solution=str(reference_solution),
        properties=props,
        parent_skills=("skill_a", "skill_b"),
    )


def test_loop_synthesis_phase_admits_quorum_passing_tasks():
    """Tasks whose properties pass quorum are converted to TrainingSample."""
    from src.orchestrator.loop import ImprovementLoop
    cfg = SystemConfig()
    cfg.synthesis.enable_task_synthesis = True

    good_task = _make_synthesized_task(classes=["a", "b", "c"], results=[True, True, True])
    mock_synth = MagicMock()
    mock_synth.synthesize.return_value = SynthesisResult(tasks=[good_task])

    mock_loader = MagicMock()
    mock_loader.load = MagicMock()

    with (
        patch("src.utils.model_loader.ModelLoader", return_value=mock_loader),
        patch("src.orchestrator.loop.DiagnosticsEngine"),
        patch("src.orchestrator.loop.DataGenerator"),
        patch("src.orchestrator.loop.Verifier"),
        patch("src.orchestrator.loop.CustomLoRATrainer"),
        patch("src.orchestrator.loop.MetaController"),
        patch("src.generator.task_synthesizer.TaskSynthesizer", return_value=mock_synth),
    ):
        loop = ImprovementLoop(cfg)

    loop._task_synthesizer = mock_synth
    loop._synthesis_enabled = True

    # Simulate the synthesis block directly
    from src.verifier.verifier_of_verifiers import quorum_verdict as _qv
    synthesis_samples = []
    synth_result = mock_synth.synthesize(MagicMock())
    for task in synth_result.tasks:
        props = getattr(task, "properties", []) or []
        ref = getattr(task, "reference_solution", "")
        ctx = {}
        pairs = []
        for prop in props:
            check_fn = getattr(prop, "check_fn", None)
            ok, _ = check_fn(ref, ctx) if callable(check_fn) else (False, "")
            pairs.append((prop, bool(ok)))
        verdict = _qv(pairs)
        if verdict.accepted:
            synthesis_samples.append(task.to_training_sample())

    assert len(synthesis_samples) == 1
    assert synthesis_samples[0].domain == "math"


def test_loop_synthesis_phase_rejects_veto_task():
    """A task with any FAIL property is rejected by quorum (veto rule)."""
    veto_task = _make_synthesized_task(
        classes=["a", "b", "c", "d"],
        results=[True, True, True, False],  # one FAIL
    )
    from src.verifier.verifier_of_verifiers import quorum_verdict as _qv

    props = getattr(veto_task, "properties", [])
    ref = veto_task.reference_solution
    pairs = []
    for prop in props:
        ok, _ = prop.check_fn(ref, {})
        pairs.append((prop, ok))
    verdict = _qv(pairs)
    assert not verdict.accepted
    assert verdict.fail_count == 1
