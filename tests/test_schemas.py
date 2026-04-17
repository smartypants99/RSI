"""Schema serialization/deserialization tests for core dataclasses."""
from __future__ import annotations

from dataclasses import asdict

from src.generator.data_generator import TrainingSample, PreferencePair, ReasoningStep
from src.diagnostics.engine import WeaknessReport, DiagnosticResult


def test_training_sample_roundtrip_via_asdict():
    step = ReasoningStep(
        step_number=1, content="x=1", justification="given",
        assumptions=["x is integer"], step_id=1, depends_on=[], rule="premise",
    )
    s = TrainingSample(
        prompt="What is x?", response="x=1", reasoning_chain=[step],
        domain="math", target_weakness="algebra", verified=True,
        confidence=0.9, expected_answer="1", consistency_score=1.0,
    )
    d = asdict(s)
    assert d["prompt"] == "What is x?"
    assert d["reasoning_chain"][0]["content"] == "x=1"
    assert d["content_hash"]

    # Rebuild from the dict — filtering reasoning_chain separately.
    chain = [ReasoningStep(**rs) for rs in d.pop("reasoning_chain")]
    rebuilt = TrainingSample(**d, reasoning_chain=chain)
    assert rebuilt.prompt == s.prompt
    assert rebuilt.reasoning_chain[0].content == "x=1"


def test_training_sample_to_training_format():
    s = TrainingSample(prompt="p", response="r", domain="math")
    fmt = s.to_training_format()
    assert fmt["prompt"] == "p"
    assert "completion" in fmt
    assert fmt["metadata"]["domain"] == "math"


def test_preference_pair_roundtrip():
    p = PreferencePair(
        prompt="p", chosen_response="c", rejected_response="r",
        domain="math", weight=1.5,
    )
    d = asdict(p)
    rebuilt = PreferencePair(**d)
    assert rebuilt.prompt == p.prompt
    assert rebuilt.weight == 1.5
    assert rebuilt.content_hash == p.content_hash


def test_weakness_report_roundtrip():
    w = WeaknessReport(
        domain="math", subdomain="algebra", severity=0.7,
        evidence=[{"q": "1+1", "a": "3"}], weak_layers=["layer.0"],
        description="bad at algebra",
    )
    d = asdict(w)
    assert d["domain"] == "math"
    rebuilt = WeaknessReport(**d)
    assert rebuilt.severity == 0.7
    assert rebuilt.evidence[0]["q"] == "1+1"


def test_diagnostic_result_overall_score_weighted():
    d = DiagnosticResult(
        cycle=1, timestamp=0.0,
        domain_scores={"math": 0.5, "code": 1.0},
        domain_question_counts={"math": 100, "code": 100},
    )
    assert abs(d.overall_score - 0.75) < 1e-6

    d2 = DiagnosticResult(
        cycle=1, timestamp=0.0,
        domain_scores={"math": 0.5, "code": 1.0},
        domain_question_counts={"math": 100, "code": 300},
    )
    # weighted: (0.5*100 + 1.0*300) / 400 = 0.875
    assert abs(d2.overall_score - 0.875) < 1e-6


def test_cycle_result_to_dict():
    from src.orchestrator.loop import CycleResult
    r = CycleResult(cycle=5)
    r.pre_score = 0.4
    r.post_score = 0.5
    r.improvement = 0.1
    d = r.to_dict()
    assert d["cycle"] == 5
    assert d["pre_score"] == 0.4
    assert d["post_score"] == 0.5
    assert d["improvement"] == 0.1
    assert "training" in d
    assert "errors" in d
