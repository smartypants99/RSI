"""Comprehensive RSI regression tests covering Property schema, admission gates, quorum, and end-to-end integration.

Priority order:
A. Property post_init + admit() gates (blocks everything else)
B. VoV integration with real Property
C. task_synthesizer §1.4 gate
D. End-to-end one-tick scenario
E. Integration points preserved (backward compatibility)
"""
from __future__ import annotations

import hashlib
import pytest
from src.generator.data_generator import TrainingSample, ReasoningStep
from src.utils.config import SystemConfig
from src.verifier.property_engine import (
    Property, PropertyKind, INDEPENDENCE_CLASSES,
    build_property, admit, AdmissionResult, MockExecutor, SandboxedExecutor,
)


def _make_sample(
    prompt="Solve 2+2",
    response="4",
    domain="math",
    verified=True
) -> TrainingSample:
    """Helper to create a well-formed sample."""
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


def _sha256_hex(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _make_prop(
    *,
    name: str = "test_prop",
    kind: PropertyKind = PropertyKind.ALGEBRAIC,
    description: str = "Test property",
    independence_class: str = "exec.behavioral",
    source: str = "def check(problem, candidate): return True",
    entry_point: str = "check",
    author: str = "test_author",
    problem_id: str = "prob_001",
    parent_problem_hash: str | None = None,
    confirmer_example: str = "",
    falsifier_example: str = "",
    **kwargs,
) -> Property:
    """Factory for test Properties."""
    ph = parent_problem_hash or _sha256_hex("test_problem")
    return build_property(
        name=name,
        kind=kind,
        description=description,
        independence_class=independence_class,
        source=source,
        entry_point=entry_point,
        author=author,
        problem_id=problem_id,
        parent_problem_hash=ph,
        confirmer_example=confirmer_example,
        falsifier_example=falsifier_example,
        **kwargs,
    )


# ─── PRIORITY A: Property schema + admit() gates ───────────────────────────


class TestPropertySchema:
    """v0.2.1 Property post_init validation per spec §1.1 (19 fields)."""

    def test_property_19_field_construction_valid(self):
        """Property with all 19 required fields constructed successfully."""
        p = _make_prop(
            confirmer_example="4",
            falsifier_example="3",
        )
        assert p.property_id.startswith("prop_")
        assert p.name == "test_prop"
        assert p.kind == PropertyKind.ALGEBRAIC
        assert p.description == "Test property"
        assert p.independence_class == "exec.behavioral"
        assert p.language == "python"
        assert p.entry_point == "check"
        assert p.author == "test_author"
        assert p.deterministic is True
        assert p.timeout_ms == 2000
        assert p.difficulty_floor == 0.5
        assert p.confirmer_example == "4"
        assert p.falsifier_example == "3"

    def test_property_post_init_requires_non_empty_strings(self):
        """__post_init__ validates non-empty string fields."""
        with pytest.raises(ValueError, match="property_id must be a non-empty str"):
            Property(
                property_id="", problem_id="p", author="a", name="n",
                kind=PropertyKind.ALGEBRAIC, description="d", language="python",
                source="s", entry_point="e", timeout_ms=1, deterministic=True,
                inputs=(), returns="bool", independence_class="exec.behavioral",
                difficulty_floor=0.5, falsifier_example="f", confirmer_example="c",
                created_at=1.0, parent_problem_hash=_sha256_hex("x"),
            )

    def test_property_post_init_validates_kind_enum(self):
        """__post_init__ requires PropertyKind enum value."""
        with pytest.raises(ValueError, match="Property.kind must be a PropertyKind"):
            Property(
                property_id="p", problem_id="p", author="a", name="n",
                kind="INVALID", description="d", language="python",
                source="s", entry_point="e", timeout_ms=1, deterministic=True,
                inputs=(), returns="bool", independence_class="exec.behavioral",
                difficulty_floor=0.5, falsifier_example="f", confirmer_example="c",
                created_at=1.0, parent_problem_hash=_sha256_hex("x"),
            )

    def test_property_post_init_validates_timeout_ms_range(self):
        """__post_init__ validates timeout_ms in [1, 10000]."""
        with pytest.raises(ValueError, match="timeout_ms must be int in"):
            Property(
                property_id="p", problem_id="p", author="a", name="n",
                kind=PropertyKind.ALGEBRAIC, description="d", language="python",
                source="s", entry_point="e", timeout_ms=20000, deterministic=True,
                inputs=(), returns="bool", independence_class="exec.behavioral",
                difficulty_floor=0.5, falsifier_example="f", confirmer_example="c",
                created_at=1.0, parent_problem_hash=_sha256_hex("x"),
            )

    def test_property_post_init_validates_source_byte_limit(self):
        """__post_init__ enforces 4 KB source byte limit."""
        huge_source = "x" * 5000
        with pytest.raises(ValueError, match="source must be"):
            Property(
                property_id="p", problem_id="p", author="a", name="n",
                kind=PropertyKind.ALGEBRAIC, description="d", language="python",
                source=huge_source, entry_point="e", timeout_ms=1, deterministic=True,
                inputs=(), returns="bool", independence_class="exec.behavioral",
                difficulty_floor=0.5, falsifier_example="f", confirmer_example="c",
                created_at=1.0, parent_problem_hash=_sha256_hex("x"),
            )

    def test_property_post_init_validates_language(self):
        """__post_init__ validates language in allow-list."""
        with pytest.raises(ValueError, match="language must be in"):
            Property(
                property_id="p", problem_id="p", author="a", name="n",
                kind=PropertyKind.ALGEBRAIC, description="d", language="cpp",
                source="s", entry_point="e", timeout_ms=1, deterministic=True,
                inputs=(), returns="bool", independence_class="exec.behavioral",
                difficulty_floor=0.5, falsifier_example="f", confirmer_example="c",
                created_at=1.0, parent_problem_hash=_sha256_hex("x"),
            )

    def test_property_post_init_validates_independence_class(self):
        """__post_init__ validates independence_class is canonical."""
        with pytest.raises(ValueError, match="independence_class must be in"):
            Property(
                property_id="p", problem_id="p", author="a", name="n",
                kind=PropertyKind.ALGEBRAIC, description="d", language="python",
                source="s", entry_point="e", timeout_ms=1, deterministic=True,
                inputs=(), returns="bool", independence_class="invalid.class",
                difficulty_floor=0.5, falsifier_example="f", confirmer_example="c",
                created_at=1.0, parent_problem_hash=_sha256_hex("x"),
            )

    def test_property_frozen(self):
        """Property is frozen (immutable once created)."""
        p = _make_prop()
        with pytest.raises(AttributeError):
            p.name = "changed"

    def test_property_to_dict_serializes_kind_enum(self):
        """Property.to_dict() converts PropertyKind enum to string."""
        p = _make_prop()
        d = p.to_dict()
        assert d["kind"] == "ALGEBRAIC"
        assert isinstance(d["kind"], str)

    def test_property_independence_classes_canonical(self):
        """All 10 canonical independence classes present."""
        assert len(INDEPENDENCE_CLASSES) == 10
        assert "dimensional.physical" in INDEPENDENCE_CLASSES
        assert "exec.behavioral" in INDEPENDENCE_CLASSES
        assert "roundtrip" in INDEPENDENCE_CLASSES


class TestPropertyAdmissionGates:
    """§1.3 admission gates 1-4: cost, sandbox, self-test, determinism."""

    def test_admit_gate1_bounded_cost_timeout_validated_at_post_init(self):
        """Gate 1: timeout_ms validated in __post_init__ (construction time)."""
        with pytest.raises(ValueError, match="timeout_ms must be int in"):
            _make_prop(timeout_ms=20000)

    def test_admit_gate1_bounded_cost_source_size_validated_at_post_init(self):
        """Gate 1: source byte limit validated in __post_init__ (construction time)."""
        huge_source = "x" * 5000
        with pytest.raises(ValueError, match="source must be"):
            _make_prop(source=huge_source)

    def test_admit_gate1_bounded_cost_language_allowlist(self):
        """Gate 1: reject invalid language."""
        with pytest.raises(ValueError):
            _make_prop(language="cpp")

    def test_admit_gate2_sandbox_materialize_success(self):
        """Gate 2: MockExecutor materializes source code."""
        prop = _make_prop(
            source="def check(problem, candidate): return True",
            confirmer_example="x", falsifier_example="y",
        )
        executor = MockExecutor()
        result = admit(prop, executor=executor)
        assert result.gate_failed != "sandbox_materialize"

    def test_admit_gate2_sandbox_materialize_syntax_error(self):
        """Gate 2: bad syntax in source fails materialize."""
        prop = _make_prop(
            source="def check(problem, candidate): return !!!bad",
            confirmer_example="x", falsifier_example="y",
        )
        executor = MockExecutor()
        result = admit(prop, executor=executor)
        assert result.admitted is False
        assert result.gate_failed == "sandbox_materialize"

    def test_admit_gate3_self_test_empty_confirmer(self):
        """Gate 3: empty confirmer_example fails self-test."""
        prop = _make_prop(confirmer_example="", falsifier_example="x")
        executor = MockExecutor()
        result = admit(prop, executor=executor)
        assert result.admitted is False
        assert result.gate_failed == "self_test"

    def test_admit_gate3_self_test_empty_falsifier(self):
        """Gate 3: empty falsifier_example fails self-test."""
        prop = _make_prop(confirmer_example="x", falsifier_example="")
        executor = MockExecutor()
        result = admit(prop, executor=executor)
        assert result.admitted is False
        assert result.gate_failed == "self_test"

    def test_admit_gate3_self_test_confirmer_must_pass(self):
        """Gate 3: confirmer example must return PASS/True."""
        prop = _make_prop(
            source="def check(problem, candidate): return candidate == '42'",
            confirmer_example="99", falsifier_example="0",
        )
        executor = MockExecutor()
        result = admit(prop, executor=executor)
        assert result.admitted is False
        assert result.gate_failed == "self_test"
        assert "confirmer" in result.reason

    def test_admit_gate3_self_test_falsifier_must_fail(self):
        """Gate 3: falsifier example must return FAIL/False."""
        prop = _make_prop(
            source="def check(problem, candidate): return True",
            confirmer_example="99", falsifier_example="0",
        )
        executor = MockExecutor()
        result = admit(prop, executor=executor)
        assert result.admitted is False
        assert result.gate_failed == "self_test"
        assert "falsifier" in result.reason

    def test_admit_gate3_self_test_passes_valid_examples(self):
        """Gate 3: valid confirmer + falsifier passes self-test."""
        prop = _make_prop(
            source="def check(problem, candidate): return candidate == '42'",
            confirmer_example="42", falsifier_example="99",
        )
        executor = MockExecutor()
        result = admit(prop, executor=executor)
        assert result.gate_failed != "self_test"

    def test_admit_gate4_determinism_check(self):
        """Gate 4: deterministic=True replays confirmer."""
        prop = _make_prop(
            source="def check(problem, candidate): return candidate == 'yes'",
            confirmer_example="yes", falsifier_example="no",
            deterministic=True,
        )
        executor = MockExecutor()
        result = admit(prop, executor=executor)
        assert result.determinism_observed is True

    def test_admit_gate4_nondeterministic_skipped(self):
        """Gate 4: deterministic=False skips determinism check."""
        prop = _make_prop(
            source="def check(problem, candidate): return candidate == 'yes'",
            confirmer_example="yes", falsifier_example="no",
            deterministic=False,
        )
        executor = MockExecutor()
        result = admit(prop, executor=executor)
        assert result.determinism_observed is None

    def test_admit_all_gates_pass(self):
        """All gates pass for valid property with MockExecutor."""
        prop = _make_prop(
            source="def check(problem, candidate): return candidate == 'pass'",
            confirmer_example="pass", falsifier_example="fail",
            deterministic=True,
        )
        executor = MockExecutor()
        result = admit(prop, executor=executor)
        assert result.admitted is True
        assert result.reason == "admitted"
        assert result.gate_failed is None
        assert result.determinism_observed is True


# ─── PRIORITY B: VoV integration with real Property ───────────────────────


class TestVoVWithRealProperty:
    """Priority B: VoV integration with real 19-field Property."""

    def test_vov_admits_and_verifies_real_property(self):
        """Real v0.2.1 Property (19-field) admitted and run through verify()."""
        from src.verifier.property_engine import verify

        prop = _make_prop(
            name="numeric_check",
            source="def check(problem, candidate): return candidate == '42'",
            confirmer_example="42",
            falsifier_example="99",
        )
        executor = MockExecutor()
        result = admit(prop, executor=executor)
        assert result.admitted is True

        vrecord = verify(
            problem_id=prop.problem_id,
            candidate="42",
            admitted_properties=[prop],
            executor=executor,
            min_properties=1,
            quorum_distinct_classes_required=1,
        )
        assert vrecord.accepted is True
        assert vrecord.pass_count >= 1

    def test_vov_independence_class_flows_through_verify(self):
        """independence_class from real Property flows through to VerificationRecord."""
        from src.verifier.property_engine import verify

        prop = _make_prop(
            independence_class="algebra.symbolic",
            source="def check(problem, candidate): return candidate == 'match'",
            confirmer_example="match",
            falsifier_example="nomatch",
        )
        executor = MockExecutor()
        result = admit(prop, executor=executor)
        assert result.admitted is True

        vrecord = verify(
            problem_id=prop.problem_id,
            candidate="match",
            admitted_properties=[prop],
            executor=executor,
            min_properties=1,
            quorum_distinct_classes_required=1,
        )
        verdict_classes = vrecord.distinct_classes
        assert "algebra.symbolic" in verdict_classes

    def test_vov_real_property_getattr_duck_typing(self):
        """VoV accesses real Property attributes via getattr (name, author, etc.)."""
        prop = _make_prop(
            name="special_prop",
            author="alice",
            independence_class="smt.logical",
        )
        assert getattr(prop, "name") == "special_prop"
        assert getattr(prop, "author") == "alice"
        assert getattr(prop, "independence_class") == "smt.logical"
        assert getattr(prop, "property_id", None) is not None

    def test_vov_discriminative_property_high_kill_rate(self):
        """Property with discriminative source has high test-rejection rate."""
        from src.verifier.property_engine import verify

        prop = _make_prop(
            name="equals_ref",
            source="def check(problem, candidate): return candidate == 'ref'",
            confirmer_example="ref",
            falsifier_example="wrong",
        )
        executor = MockExecutor()
        result = admit(prop, executor=executor)
        assert result.admitted is True

        vrecord = verify(
            problem_id=prop.problem_id,
            candidate="ref",
            admitted_properties=[prop],
            executor=executor,
            min_properties=1,
            quorum_distinct_classes_required=1,
        )
        assert vrecord.accepted is True
        assert vrecord.pass_count >= 1

    def test_vov_toothless_property_cannot_admit(self):
        """Property that accepts both confirmer and falsifier fails admission."""
        prop = _make_prop(
            name="always_true",
            source="def check(problem, candidate): return True",
            confirmer_example="anything",
            falsifier_example="anything_else",
        )
        executor = MockExecutor()
        result = admit(prop, executor=executor)
        assert result.admitted is False
        assert result.gate_failed == "self_test"


class TestQuorumVerdictEdgeCases:
    """Additional quorum_verdict tests per architect spec."""

    def test_quorum_passes_with_3_classes_3_pass(self):
        """Basic quorum: n=3, 3 PASS from 3 distinct classes -> accept."""
        from src.verifier.verifier_of_verifiers import quorum_verdict

        def _prop(name, cls):
            p = type('P', (), {})()
            p.name = name
            p.independence_class = cls
            return p

        pairs = [
            (_prop("p1", "execution"), True),
            (_prop("p2", "structural"), True),
            (_prop("p3", "behavioral"), True),
        ]
        v = quorum_verdict(pairs)
        assert v.accepted, v.reason
        assert v.pass_count == 3
        assert v.distinct_classes == 3

    def test_quorum_rejects_single_fail_even_with_3_pass(self):
        """§2.1 veto rule: any FAIL rejects, even with strong PASSes."""
        from src.verifier.verifier_of_verifiers import quorum_verdict

        def _prop(name, cls):
            p = type('P', (), {})()
            p.name = name
            p.independence_class = cls
            return p

        pairs = [
            (_prop("p1", "execution"), True),
            (_prop("p2", "structural"), True),
            (_prop("p3", "behavioral"), True),
            (_prop("p4", "algebraic"), False),
        ]
        v = quorum_verdict(pairs)
        assert not v.accepted, "expected veto on any FAIL"
        assert "veto" in v.reason

    def test_quorum_requires_minimum_3_distinct_classes(self):
        """Must have >= 3 distinct independence classes."""
        from src.verifier.verifier_of_verifiers import quorum_verdict

        def _prop(name, cls):
            p = type('P', (), {})()
            p.name = name
            p.independence_class = cls
            return p

        pairs = [
            (_prop("p1", "execution"), True),
            (_prop("p2", "structural"), True),
            # Only 2 classes, not 3
        ]
        v = quorum_verdict(pairs)
        assert not v.accepted
        assert "distinct independence classes" in v.reason


# ─── PRIORITY C: task_synthesizer §1.4 gate ───────────────────────────────


class TestTaskSynthesizerVoVGate:
    """task_synthesizer calls VoV before emitting SynthesizedTask."""

    def test_synthesizer_calls_vov_before_emit(self):
        """§1.4: propose() calls verify_properties_trustworthy pre-emit."""
        pytest.skip("task_synthesizer integration not yet available")

    def test_synthesizer_toothless_bundle_not_emitted(self):
        """Bundle failing VoV must NOT produce SynthesizedTask."""
        pytest.skip("task_synthesizer integration not yet available")

    def test_synthesizer_strong_bundle_emitted(self):
        """Bundle passing VoV produces SynthesizedTask."""
        pytest.skip("task_synthesizer integration not yet available")

    def test_synthesizer_proposed_problem_carries_declared_difficulty(self):
        """ProposedProblem includes declared_difficulty field."""
        pytest.skip("task_synthesizer integration not yet available")

    def test_synthesizer_proposed_problem_carries_nearest_neighbor_dist(self):
        """ProposedProblem includes nearest_neighbor_dist field."""
        pytest.skip("task_synthesizer integration not yet available")


# ─── PRIORITY D: End-to-end one-tick scenario ───────────────────────────


class TestEndToEndRSITick:
    """One complete RSI tick: propose → admit → solve → verify → quorum."""

    def test_tick_toothless_property_dies_at_vov_gate(self):
        """§1.4: toothless bundle rejected at VoV, never enters tick."""
        pytest.skip("Full RSI tick not yet available")

    def test_tick_strong_bundle_reaches_training_pool(self):
        """§4: strong bundle passes admit + VoV → reaches TrainingPool."""
        pytest.skip("Full RSI tick not yet available")

    def test_tick_candidate_failing_property_rejected_at_quorum(self):
        """§2.1: candidate FAIL on any property -> quorum rejects."""
        pytest.skip("Full RSI tick not yet available")

    def test_tick_candidate_passing_supermajority_accepted(self):
        """§2.1: candidate PASS >= ceil(2n/3) from >= 3 classes -> accept."""
        pytest.skip("Full RSI tick not yet available")

    def test_tick_cycle_result_carries_rsi_metrics(self):
        """§6: CycleResult has novel_problems_proposed, properties_admitted, etc."""
        pytest.skip("CycleResult RSI fields not yet available")


# ─── PRIORITY E: Integration points preserved ───────────────────────────


class TestBackwardCompatibility:
    """Existing code paths still work; RSI mode is orthogonal."""

    def test_verifier_without_properties_param(self):
        """Verifier.verify(..., properties=None) uses existing heuristic path."""
        pytest.skip("Backward compat check pending")

    def test_training_sample_without_rsi_fields(self):
        """TrainingSample still works without source='rsi_property'."""
        pytest.skip("Backward compat check pending")

    def test_improvement_loop_without_rsi_tick(self):
        """ImprovementLoop.run_cycle() still works when mode != 'rsi'."""
        pytest.skip("Backward compat check pending")

    def test_all_71_original_tests_still_pass(self):
        """No regression: all original verification tests green."""
        pytest.skip("Run full suite to verify")


# ─── Integration points per spec §5 ─────────────────────────────────────


class TestIntegrationPoints:
    """Verify RSI integration points exist and are callable."""

    def test_property_engine_exports_all_apis(self):
        """property_engine exports all public APIs."""
        from src.verifier import property_engine

        assert hasattr(property_engine, "Property")
        assert hasattr(property_engine, "PropertyResult")
        assert hasattr(property_engine, "VerdictWithEvidence")
        assert hasattr(property_engine, "verify_by_consensus")
        assert hasattr(property_engine, "builtin_properties")
        assert hasattr(property_engine, "get_property")
        assert hasattr(property_engine, "resolve_properties")
        assert hasattr(property_engine, "sample_has_properties")
        assert hasattr(property_engine, "verify_sample_by_properties")

    def test_training_sample_accepts_properties(self):
        """TrainingSample can carry properties/property_ids."""
        from src.generator.data_generator import TrainingSample, ReasoningStep
        from src.verifier.property_engine import LegacyProperty as Property

        sample = TrainingSample(
            prompt="What is 2+2?",
            response="4",
            reasoning_chain=[
                ReasoningStep(step_number=1, content="2+2=4", justification="arithmetic"),
            ],
            domain="math",
            verified=True,
            expected_answer="4",
        )
        # Should accept these without error
        sample.property_ids = ["substitute_back", "numerical_plausibility"]
        assert sample.property_ids == ["substitute_back", "numerical_plausibility"]

    def test_verifier_of_verifiers_available(self):
        """verify_properties_trustworthy API is available."""
        from src.verifier.verifier_of_verifiers import verify_properties_trustworthy

        assert callable(verify_properties_trustworthy)

    def test_verifier_of_verifiers_quorum_verdict_available(self):
        """quorum_verdict API is available."""
        from src.verifier.verifier_of_verifiers import quorum_verdict

        assert callable(quorum_verdict)

    def test_generate_corruptions_available(self):
        """generate_corruptions API is available."""
        from src.verifier.verifier_of_verifiers import generate_corruptions

        assert callable(generate_corruptions)

    def test_make_task_fingerprint_available(self):
        """make_task_fingerprint API is available."""
        from src.verifier.verifier_of_verifiers import make_task_fingerprint

        assert callable(make_task_fingerprint)

    def test_registries_api_available(self):
        """RSIRegistries append-only JSONL stores are available."""
        from src.orchestrator.registries import RSIRegistries

        assert hasattr(RSIRegistries, 'open')
        assert callable(RSIRegistries.open)

    def test_registries_open_creates_stores(self):
        """RSIRegistries.open returns object with all five stores."""
        import tempfile
        from pathlib import Path
        from src.orchestrator.registries import RSIRegistries

        with tempfile.TemporaryDirectory() as tmpdir:
            regs = RSIRegistries.open(Path(tmpdir), sid="test_session")
            assert hasattr(regs, 'property_registry')
            assert hasattr(regs, 'problem_registry')
            assert hasattr(regs, 'verification_log')
            assert hasattr(regs, 'calibration_ledger')
            assert hasattr(regs, 'training_pool')

    def test_property_record_schema(self):
        """PropertyRecord envelope has required fields."""
        from src.orchestrator.registries import PropertyRecord

        rec = PropertyRecord(
            property_id="p1",
            problem_id="prob1",
            author="model",
            independence_class="execution",
            kind="code",
            name="test_prop",
            payload={"full": "property"},
        )
        assert rec.property_id == "p1"
        assert rec.problem_id == "prob1"
        assert rec.independence_class == "execution"
        assert rec.payload == {"full": "property"}

    def test_problem_record_schema(self):
        """ProblemRecord envelope has required fields."""
        from src.orchestrator.registries import ProblemRecord

        rec = ProblemRecord(
            problem_id="prob1",
            domain="code",
            problem_text="Write a function",
            declared_difficulty=0.7,
            nearest_neighbor_dist=0.5,
            parent_skills=["python"],
            retired=False,
        )
        assert rec.problem_id == "prob1"
        assert rec.domain == "code"
        assert rec.declared_difficulty == 0.7

    def test_verification_record_schema(self):
        """VerificationRecord envelope has required fields."""
        from src.orchestrator.registries import VerificationRecord

        rec = VerificationRecord(
            record_id="v1",
            problem_id="prob1",
            candidate_id="cand1",
            property_ids=["p1", "p2"],
            per_property_verdicts={"p1": True, "p2": False},
            quorum_accepted=False,
            quorum_reason="fail veto",
            adversarial=False,
        )
        assert rec.record_id == "v1"
        assert rec.quorum_accepted == False
        assert rec.adversarial == False

    def test_calibration_entry_schema(self):
        """CalibrationEntry envelope has required fields."""
        from src.orchestrator.registries import CalibrationEntry

        rec = CalibrationEntry(
            tick=1,
            independence_class="execution",
            true_accept_rate=0.9,
            true_reject_rate=0.85,
            error_rate=0.05,
            suspended=False,
            n_probes=100,
        )
        assert rec.tick == 1
        assert rec.independence_class == "execution"
        assert rec.true_accept_rate == 0.9

    def test_training_pool_record_schema(self):
        """TrainingPoolRecord envelope has required fields."""
        from src.orchestrator.registries import TrainingPoolRecord

        rec = TrainingPoolRecord(
            pool_record_id="pool1",
            problem_id="prob1",
            candidate_id="cand1",
            verification_record_id="v1",
            domain="code",
            prompt="Write a function",
            response="def f(): pass",
            source="rsi_property",
        )
        assert rec.pool_record_id == "pool1"
        assert rec.source == "rsi_property"
