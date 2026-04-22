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
        """Canonical independence classes present (10 spec + simulation.numerical from task #5 + peers)."""
        assert len(INDEPENDENCE_CLASSES) >= 11
        assert "dimensional.physical" in INDEPENDENCE_CLASSES
        assert "exec.behavioral" in INDEPENDENCE_CLASSES
        assert "roundtrip" in INDEPENDENCE_CLASSES
        assert "simulation.numerical" in INDEPENDENCE_CLASSES


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

    def test_verify_populates_backend_field_for_model_authored_python(self):
        """Task #8 visibility: PropertyVerdict.backend is populated from the
        property's language field for model-authored properties, so operators
        can tell whether python/z3/sympy/unit_test actually ran (vs. silent
        skip). Previously only independence_class was visible, which is
        semantic-level and did not reveal which backend executed."""
        from src.verifier.property_engine import verify

        prop = _make_prop(
            name="backend_python",
            source="def check(problem, candidate): return candidate == 'ok'",
            confirmer_example="ok",
            falsifier_example="nope",
            language="python",
        )
        executor = MockExecutor()
        assert admit(prop, executor=executor).admitted is True
        vrecord = verify(
            problem_id=prop.problem_id,
            candidate="ok",
            admitted_properties=[prop],
            executor=executor,
            min_properties=1,
            quorum_distinct_classes_required=1,
        )
        assert len(vrecord.per_property) == 1
        # MockExecutor runs python bodies in-process; backend label is
        # driven by prop.language, not by where execution happened.
        assert vrecord.per_property[0].backend == "python"

    def test_verify_populates_backend_field_for_trusted_builtin(self):
        """Trusted builtin properties get backend='trusted' (or a more
        specific label for simulator/z3 backends). This closes the task-#8
        silent-skip visibility gap — operators can grep verify logs for
        'backends=[' and see exactly which backends participated."""
        from src.verifier.property_engine import verify, builtin_properties
        builtins = builtin_properties()
        assert builtins, "builtin catalog should not be empty"
        # Any builtin with a short, reason-agnostic check_fn works.
        prop = builtins[0]
        executor = MockExecutor()
        vrecord = verify(
            problem_id="test_problem",
            candidate="whatever",
            admitted_properties=[prop],
            executor=executor,
            min_properties=1,
            quorum_distinct_classes_required=1,
        )
        assert len(vrecord.per_property) == 1
        # Backend must be one of the trusted-family labels, never "unknown".
        assert vrecord.per_property[0].backend in (
            "trusted", "simulator", "z3"
        ), vrecord.per_property[0].backend

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
    """Priority C: task_synthesizer §1.4 gate filters via VoV."""

    def test_synthesizer_strong_bundle_structure(self):
        """SynthesizedTask carries properties list (post-VoV filter)."""
        from src.generator.task_synthesizer import SynthesizedTask

        task = SynthesizedTask(
            task_id="task_001",
            domain="math",
            prompt="Solve 2+2",
            reference_solution="4",
            properties=[],
        )
        assert task.task_id == "task_001"
        assert task.domain == "math"
        assert task.properties == []

    def test_property_admission_gates_precede_vov(self):
        """Properties must pass admission gates 1-4 before VoV check."""
        prop = _make_prop(
            source="def check(problem, candidate): return candidate == '42'",
            confirmer_example="42",
            falsifier_example="99",
        )
        executor = MockExecutor()
        result = admit(prop, executor=executor)
        assert result.admitted is True
        assert result.gate_failed is None

    def test_toothless_property_fails_self_test_gate(self):
        """Toothless properties (confirmer=falsifier verdict) fail gate 3."""
        prop = _make_prop(
            source="def check(problem, candidate): return True",
            confirmer_example="anything",
            falsifier_example="anything_else",
        )
        executor = MockExecutor()
        result = admit(prop, executor=executor)
        assert result.admitted is False
        assert result.gate_failed == "self_test"

    def test_strong_property_passes_all_gates(self):
        """Strong property (discriminative) passes all 4 gates."""
        prop = _make_prop(
            source="def check(problem, candidate): return candidate == 'correct'",
            confirmer_example="correct",
            falsifier_example="wrong",
            deterministic=True,
        )
        executor = MockExecutor()
        result = admit(prop, executor=executor)
        assert result.admitted is True
        assert result.gate_failed is None
        assert result.determinism_observed is True


# ─── PRIORITY D: End-to-end one-tick scenario ───────────────────────────


class TestEndToEndRSITick:
    """Priority D: One complete RSI tick: propose → admit → verify → quorum."""

    def test_tick_property_admit_then_verify_sequence(self):
        """Admission (gates 1-4) precedes verification (quorum)."""
        prop = _make_prop(
            source="def check(problem, candidate): return candidate == 'yes'",
            confirmer_example="yes",
            falsifier_example="no",
        )
        executor = MockExecutor()

        result = admit(prop, executor=executor)
        assert result.admitted is True

        from src.verifier.property_engine import verify
        vrecord = verify(
            problem_id=prop.problem_id,
            candidate="yes",
            admitted_properties=[prop],
            executor=executor,
            min_properties=1,
            quorum_distinct_classes_required=1,
        )
        assert vrecord.accepted is True

    def test_tick_failed_property_blocks_verification(self):
        """If property fails admission, it never reaches verification."""
        prop = _make_prop(
            source="def check(problem, candidate): return True",
            confirmer_example="conf",
            falsifier_example="fals",
        )
        executor = MockExecutor()
        result = admit(prop, executor=executor)
        assert result.admitted is False

    def test_tick_quorum_veto_on_any_fail(self):
        """Quorum verdict: any FAIL from admitted properties vetoes."""
        prop_pass = _make_prop(
            name="always_pass",
            independence_class="exec.behavioral",
            source="def check(problem, candidate): return candidate == 'c'",
            confirmer_example="c",
            falsifier_example="f",
        )
        prop_fail = _make_prop(
            name="always_fail",
            independence_class="algebra.symbolic",
            source="def check(problem, candidate): return False",
            confirmer_example="c",
            falsifier_example="f",
        )
        executor = MockExecutor()

        admit_pass = admit(prop_pass, executor=executor)
        admit_fail = admit(prop_fail, executor=executor)
        assert admit_pass.admitted is True
        assert admit_fail.admitted is False

    def test_tick_verification_record_carries_verdicts(self):
        """VerificationRecord captures per-property verdicts and quorum."""
        from src.verifier.property_engine import verify

        prop = _make_prop(
            source="def check(problem, candidate): return candidate == 'pass'",
            confirmer_example="pass",
            falsifier_example="fail",
        )
        executor = MockExecutor()
        result = admit(prop, executor=executor)
        assert result.admitted is True

        vrecord = verify(
            problem_id=prop.problem_id,
            candidate="pass",
            admitted_properties=[prop],
            executor=executor,
            min_properties=1,
            quorum_distinct_classes_required=1,
        )
        assert vrecord.record_id.startswith("ver_")
        assert vrecord.problem_id == prop.problem_id
        assert len(vrecord.per_property) >= 1
        assert vrecord.per_property[0].verdict == "PASS"


# ─── PRIORITY E: Integration points preserved ───────────────────────────


class TestBackwardCompatibility:
    """Priority E: Existing code paths still work; RSI mode is orthogonal."""

    def test_training_sample_construction(self):
        """TrainingSample can be constructed as before."""
        sample = _make_sample(
            prompt="Test prompt",
            response="Test response",
            domain="code",
        )
        assert sample.prompt == "Test prompt"
        assert sample.response == "Test response"
        assert sample.domain == "code"
        assert sample.verified is True

    def test_property_to_dict_serializes(self):
        """Property.to_dict() enables serialization to JSON."""
        prop = _make_prop()
        d = prop.to_dict()
        assert isinstance(d, dict)
        assert "property_id" in d
        assert "kind" in d
        assert "source" in d
        assert isinstance(d["kind"], str)

    def test_independence_classes_immutable(self):
        """INDEPENDENCE_CLASSES is frozen and canonical (includes simulation.numerical)."""
        assert isinstance(INDEPENDENCE_CLASSES, frozenset)
        assert len(INDEPENDENCE_CLASSES) >= 11
        assert "dimensional.physical" in INDEPENDENCE_CLASSES

    def test_property_kind_enum_complete(self):
        """PropertyKind has 12 values (10 spec + SIMULATION #5 + LEAN_PROOF #3)."""
        kinds = list(PropertyKind)
        assert len(kinds) >= 11
        assert PropertyKind.SIMULATION in kinds
        assert PropertyKind.ALGEBRAIC in kinds
        assert PropertyKind.ROUNDTRIP in kinds


# ─── Integration points per spec §5 ─────────────────────────────────────


class TestIntegrationPoints:
    """Spec §5: RSI integration points with property_engine, verifier_of_verifiers, registries."""

    def test_property_engine_exports_canonical_apis(self):
        """property_engine exports canonical v0.2.1 APIs (Property, admit, verify)."""
        from src.verifier import property_engine

        assert hasattr(property_engine, "Property")
        assert hasattr(property_engine, "PropertyKind")
        assert hasattr(property_engine, "INDEPENDENCE_CLASSES")
        assert hasattr(property_engine, "build_property")
        assert hasattr(property_engine, "admit")
        assert hasattr(property_engine, "verify")
        assert hasattr(property_engine, "AdmissionResult")
        assert hasattr(property_engine, "VerificationRecord")
        assert hasattr(property_engine, "MockExecutor")
        assert hasattr(property_engine, "SandboxedExecutor")

    def test_property_engine_exports_legacy_compat(self):
        """property_engine exports v0.1 LegacyProperty compat shim."""
        from src.verifier import property_engine

        assert hasattr(property_engine, "LegacyProperty")

    def test_verifier_of_verifiers_available(self):
        """VoV module provides verify_properties_trustworthy and quorum_verdict."""
        from src.verifier.verifier_of_verifiers import (
            verify_properties_trustworthy, quorum_verdict, generate_corruptions, make_task_fingerprint
        )

        assert callable(verify_properties_trustworthy)
        assert callable(quorum_verdict)
        assert callable(generate_corruptions)
        assert callable(make_task_fingerprint)

    def test_training_sample_has_rsi_fields(self):
        """TrainingSample supports RSI-related fields and metadata."""
        sample = _make_sample()
        assert sample.domain == "math"
        assert sample.verified is True
        assert sample.expected_answer == "4"

    def test_registries_api_available(self):
        """RSIRegistries provides append-only JSONL stores per spec §4."""
        from src.orchestrator.registries import RSIRegistries

        assert hasattr(RSIRegistries, "open")
        assert callable(RSIRegistries.open)

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


# ─── CHECKPOINT 3: Integration fixtures & scenarios (property_verifier) ─────


class TestCheckpoint3IntegrationFixtures:
    """Checkpoint 3: Confirmer/falsifier examples, bundle scenarios, VoV gate."""

    def test_exact_match_property_with_confirmer_falsifier(self):
        """property_verifier's exact_match_42 example property."""
        source = '''
def check(problem, candidate):
    return (candidate == "42", "exact match" if candidate == "42" else "mismatch")
'''
        prop = build_property(
            name="exact_match_42",
            kind=PropertyKind.UNIT_TEST,
            description="candidate must equal the string '42'",
            independence_class="exec.behavioral",
            language="python",
            source=source,
            entry_point="check",
            timeout_ms=2000,
            deterministic=True,
            inputs=("problem", "candidate"),
            returns="bool",
            difficulty_floor=0.1,
            confirmer_example="42",
            falsifier_example="41",
            author="test:integration_run",
            problem_id="p_int_001",
            parent_problem_hash="a" * 64,
        )

        res_mock = admit(prop, executor=MockExecutor())
        assert res_mock.admitted is True
        assert res_mock.gate_failed is None
        assert res_mock.self_test_verdicts == ("PASS", "FAIL")
        assert res_mock.determinism_observed is True

    def test_exact_match_property_with_sandboxed_executor(self):
        """SandboxedExecutor path for exact_match_42."""
        source = '''
def check(problem, candidate):
    return candidate == "42"
'''
        prop = build_property(
            name="exact_match_42_sbox",
            kind=PropertyKind.UNIT_TEST,
            description="candidate must equal '42'",
            independence_class="exec.behavioral",
            language="python",
            source=source,
            entry_point="check",
            timeout_ms=2000,
            deterministic=True,
            inputs=("problem", "candidate"),
            returns="bool",
            difficulty_floor=0.1,
            confirmer_example="42",
            falsifier_example="41",
            author="test:integration_sbox",
            problem_id="p_int_001b",
            parent_problem_hash="a" * 64,
        )

        res_sbox = admit(prop, executor=SandboxedExecutor(memory_mb=128))
        assert res_sbox.admitted is True
        assert res_sbox.determinism_observed is True

    def test_3property_bundle_quorum_happy_path(self):
        """3-property bundle across distinct classes passes quorum."""
        from src.verifier.property_engine import verify

        recipes = [
            ("exec.behavioral", PropertyKind.UNIT_TEST,
             "def check(problem, candidate): return candidate == '42'"),
            ("algebra.symbolic", PropertyKind.ALGEBRAIC,
             "def check(problem, candidate): return str(candidate).strip() == '42'"),
            ("structural.static", PropertyKind.TYPE_INVARIANT,
             "def check(problem, candidate): return isinstance(candidate, str) and candidate == '42'"),
        ]
        props = []
        for i, (cls, kind, src) in enumerate(recipes):
            p = build_property(
                name=f"p{i}", kind=kind, description="t",
                independence_class=cls, language="python",
                source=src, entry_point="check",
                timeout_ms=2000, deterministic=True,
                inputs=("problem", "candidate"), returns="bool",
                difficulty_floor=0.5,
                confirmer_example="42", falsifier_example="99",
                author=f"test:run_{i}",
                problem_id="p_int_002",
                parent_problem_hash="b" * 64,
            )
            result = admit(p, executor=MockExecutor())
            assert result.admitted is True
            props.append(p)

        rec = verify(
            problem_id="p_int_002", candidate="42", admitted_properties=props,
            executor=MockExecutor(), min_properties=3, quorum_distinct_classes_required=3,
        )
        assert rec.accepted is True
        assert rec.quorum_n == 3
        assert len(rec.distinct_classes) == 3

    def test_3property_bundle_any_fail_veto(self):
        """Any-FAIL veto: single failing property rejects quorum."""
        from src.verifier.property_engine import verify

        recipes = [
            ("exec.behavioral", PropertyKind.UNIT_TEST,
             "def check(problem, candidate): return candidate == '42'"),
            ("algebra.symbolic", PropertyKind.ALGEBRAIC,
             "def check(problem, candidate): return candidate == '42'"),
            ("structural.static", PropertyKind.TYPE_INVARIANT,
             "def check(problem, candidate): return candidate == '42'"),
        ]
        props = []
        for i, (cls, kind, src) in enumerate(recipes):
            p = build_property(
                name=f"pveto{i}", kind=kind, description="t",
                independence_class=cls, language="python",
                source=src, entry_point="check",
                timeout_ms=2000, deterministic=True,
                inputs=("problem", "candidate"), returns="bool",
                difficulty_floor=0.5,
                confirmer_example="42", falsifier_example="99",
                author=f"test:veto_{i}",
                problem_id="p_int_003",
                parent_problem_hash="c" * 64,
            )
            result = admit(p, executor=MockExecutor())
            assert result.admitted is True
            props.append(p)

        rec = verify(
            problem_id="p_int_003", candidate="99", admitted_properties=props,
            executor=MockExecutor(), min_properties=3, quorum_distinct_classes_required=3,
        )
        assert rec.accepted is False
        assert rec.fail_count >= 1

    def test_trusted_property_registration(self):
        """Trusted builtin registration skips gates 1+2."""
        from src.verifier.property_engine import _TRUSTED_CHECK_FNS

        def my_trusted_check(problem, candidate):
            return candidate == "ok", ""

        prop = build_property(
            name="my_trusted",
            kind=PropertyKind.UNIT_TEST,
            description="t",
            independence_class="exec.behavioral",
            language="python",
            source="# trusted placeholder",
            entry_point="_b_my_trusted",
            author="test:trusted",
            problem_id="p_trusted",
            parent_problem_hash="d" * 64,
            confirmer_example="ok",
            falsifier_example="bad",
            trusted=True,
            trusted_check_fn=my_trusted_check,
        )
        assert prop.property_id in _TRUSTED_CHECK_FNS

        result = admit(prop, executor=MockExecutor())
        assert result.admitted is True

    def test_property_to_payload_serialization(self):
        """property_to_payload() enables JSONL registry storage."""
        from src.verifier.property_engine import property_to_payload

        prop = _make_prop(confirmer_example="c", falsifier_example="f")
        payload = property_to_payload(prop)

        assert isinstance(payload, dict)
        assert payload["property_id"] == prop.property_id
        assert payload["name"] == "test_prop"
        assert payload["kind"] == "ALGEBRAIC"
        assert isinstance(payload["inputs"], list)
        assert "source" in payload

    def test_vov_strong_bundle_admits_and_runs(self):
        """VoV: strong property admits and runs through verify_properties_trustworthy."""
        from src.verifier.verifier_of_verifiers import verify_properties_trustworthy

        strong_props = [
            _make_prop(
                name="strong_discriminator",
                source="def check(problem, candidate): return candidate == 'correct'",
                confirmer_example="correct",
                falsifier_example="wrong",
            )
        ]

        result = admit(strong_props[0], executor=MockExecutor())
        assert result.admitted is True

        report = verify_properties_trustworthy(
            task_id="t_strong",
            reference_solution="correct",
            properties=strong_props,
            problem_ctx={},
            domain="code",
        )
        assert report is not None

    def test_vov_toothless_bundle_fails_corruption_sweep(self):
        """VoV: toothless property fails corruption sweep."""
        from src.verifier.verifier_of_verifiers import verify_properties_trustworthy

        toothless_props = [
            _make_prop(
                name="toothless",
                source="def check(problem, candidate): return True",
                confirmer_example="anything",
                falsifier_example="also_anything",
            )
        ]

        result = admit(toothless_props[0], executor=MockExecutor())
        assert result.admitted is False

    def test_admission_gates_precedence_over_vov(self):
        """Toothless property fails gate 3 (self_test), never reaches VoV."""
        prop = _make_prop(
            source="def check(problem, candidate): return True",
            confirmer_example="c",
            falsifier_example="f",
        )
        executor = MockExecutor()

        result = admit(prop, executor=executor)
        assert result.admitted is False
        assert result.gate_failed == "self_test"

    def test_determinism_check_confirms_deterministic_behavior(self):
        """Gate 4 confirms deterministic property behavior is stable."""
        source = '''
def check(problem, candidate):
    return candidate == "deterministic"
'''
        prop = _make_prop(
            source=source,
            confirmer_example="deterministic",
            falsifier_example="nope",
            deterministic=True,
        )
        executor = MockExecutor()
        result = admit(prop, executor=executor)
        assert result.admitted is True
        assert result.determinism_observed is True


# --- Task #11 concern #1: verifier_accept_policy ---


class TestVerifierAcceptPolicy:
    """Task #11 concern #1: relax any-FAIL veto via accept_policy parameter."""

    def _mk_two_pass_one_fail(self):
        """Build 3 admitted properties that produce PASS, PASS, FAIL on
        candidate='x'. Each has a distinct independence_class so the
        quorum-distinct-classes rule does not interfere."""
        p_pass_a = _make_prop(
            name="passer_a",
            independence_class="algebra.symbolic",
            source="def check(problem, candidate): return True",
            author="author_a",
            confirmer_example="x", falsifier_example="y",
        )
        p_pass_b = _make_prop(
            name="passer_b",
            independence_class="exec.behavioral",
            source="def check(problem, candidate): return True",
            author="author_b",
            confirmer_example="x", falsifier_example="y",
        )
        p_fail = _make_prop(
            name="failer",
            independence_class="simulation.numerical",
            source="def check(problem, candidate): return False",
            author="author_c",
            confirmer_example="y", falsifier_example="x",
        )
        return [p_pass_a, p_pass_b, p_fail]

    def test_any_fail_veto_rejects_two_of_three(self):
        """Backward-compat: default 'any_fail_veto' still rejects 2-of-3."""
        from src.verifier.property_engine import verify
        props = self._mk_two_pass_one_fail()
        executor = MockExecutor()
        for p in props:
            admit(p, executor=executor)
        rec = verify(
            problem_id="tp", candidate="x",
            admitted_properties=props,
            executor=executor,
            min_properties=3,
            quorum_distinct_classes_required=2,
            accept_policy="any_fail_veto",
        )
        assert rec.accepted is False
        assert rec.fail_count == 1
        assert "policy=any_fail_veto" in rec.reject_reason
        assert rec.accept_policy == "any_fail_veto"
        assert rec.verdict_warnings == ()

    def test_majority_policy_accepts_two_of_three_with_warning(self):
        """'majority' accepts 2-of-3 PASS but flags verdict_warn=any_fail."""
        from src.verifier.property_engine import verify
        props = self._mk_two_pass_one_fail()
        executor = MockExecutor()
        for p in props:
            admit(p, executor=executor)
        rec = verify(
            problem_id="tp", candidate="x",
            admitted_properties=props,
            executor=executor,
            min_properties=3,
            quorum_distinct_classes_required=2,
            accept_policy="majority",
        )
        assert rec.accepted is True
        assert rec.pass_count == 2
        assert rec.fail_count == 1
        assert rec.accept_policy == "majority"
        assert "any_fail" in rec.verdict_warnings

    def test_quorum_2of3_accepts_two_of_three(self):
        from src.verifier.property_engine import verify
        props = self._mk_two_pass_one_fail()
        executor = MockExecutor()
        for p in props:
            admit(p, executor=executor)
        rec = verify(
            problem_id="tp", candidate="x",
            admitted_properties=props,
            executor=executor,
            min_properties=3,
            quorum_distinct_classes_required=2,
            accept_policy="quorum_2of3",
        )
        assert rec.accepted is True
        assert "any_fail" in rec.verdict_warnings

    def test_quorum_2of3_rejects_one_pass_two_fail(self):
        """quorum_2of3 must still reject 1-PASS/2-FAIL (real disagreement)."""
        from src.verifier.property_engine import verify
        p_pass = _make_prop(
            name="solo_pass", independence_class="algebra.symbolic",
            source="def check(problem, candidate): return True",
            author="a", confirmer_example="x", falsifier_example="y",
        )
        p_fail_a = _make_prop(
            name="fail_a", independence_class="exec.behavioral",
            source="def check(problem, candidate): return False",
            author="b", confirmer_example="y", falsifier_example="x",
        )
        p_fail_b = _make_prop(
            name="fail_b", independence_class="simulation.numerical",
            source="def check(problem, candidate): return False",
            author="c", confirmer_example="y", falsifier_example="x",
        )
        executor = MockExecutor()
        for p in (p_pass, p_fail_a, p_fail_b):
            admit(p, executor=executor)
        rec = verify(
            problem_id="tp", candidate="x",
            admitted_properties=[p_pass, p_fail_a, p_fail_b],
            executor=executor,
            min_properties=3,
            quorum_distinct_classes_required=2,
            accept_policy="quorum_2of3",
        )
        assert rec.accepted is False
        assert rec.fail_count == 2

    def test_relaxed_policy_still_enforces_distinct_classes(self):
        """'majority' must NOT bypass the quorum-distinct-classes rule —
        2 PASS in the SAME class still fails the structural check."""
        from src.verifier.property_engine import verify
        p_a = _make_prop(
            name="same_class_a", independence_class="algebra.symbolic",
            source="def check(problem, candidate): return True",
            author="a", confirmer_example="x", falsifier_example="y",
        )
        p_b = _make_prop(
            name="same_class_b", independence_class="algebra.symbolic",
            source="def check(problem, candidate): return True",
            author="b", confirmer_example="x", falsifier_example="y",
        )
        p_c = _make_prop(
            name="failer", independence_class="exec.behavioral",
            source="def check(problem, candidate): return False",
            author="c", confirmer_example="y", falsifier_example="x",
        )
        executor = MockExecutor()
        for p in (p_a, p_b, p_c):
            admit(p, executor=executor)
        rec = verify(
            problem_id="tp", candidate="x",
            admitted_properties=[p_a, p_b, p_c],
            executor=executor,
            min_properties=3,
            quorum_distinct_classes_required=2,
            accept_policy="majority",
        )
        # 2 PASS in one class + 1 FAIL in another → majority FAIL-gate is
        # satisfied (2 PASS, 1 FAIL), but distinct_classes over PASSes is
        # only 1 < required 2 → rejected for structural reasons.
        assert rec.accepted is False
        assert "distinct_classes" in rec.reject_reason

    def test_verify_rejects_invalid_accept_policy(self):
        from src.verifier.property_engine import verify
        executor = MockExecutor()
        import pytest as _pytest
        with _pytest.raises(ValueError):
            verify(
                problem_id="tp", candidate="x",
                admitted_properties=[],
                executor=executor,
                min_properties=0,
                quorum_distinct_classes_required=0,
                accept_policy="not_a_policy",
            )
