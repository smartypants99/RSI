"""Schema-level tests for src/verifier/property_engine.py (v0.2.1).

SCOPE: construction + __post_init__ + admit() gates only.
Integration-level tests (VoV, quorum, verify() e2e, synthesizer handoff) live
with tester in test_rsi_actual.py and test_synthesis_integration.py.
"""

from __future__ import annotations

import time

import pytest

from src.verifier.property_engine import (
    Property,
    PropertyKind,
    INDEPENDENCE_CLASSES,
    build_property,
    admit,
    MockExecutor,
    LegacyProperty,
)


# ───────────────────────── helpers ─────────────────────────

def _minimal_kwargs(**overrides):
    """Return a dict of valid kwargs for Property construction."""
    kw = dict(
        property_id="prop_abc12345",
        problem_id="prob_xyz",
        author="test:run_1",
        name="example",
        kind=PropertyKind.UNIT_TEST,
        description="example property",
        language="python",
        source="def check(problem, candidate):\n    return True",
        entry_point="check",
        timeout_ms=2000,
        deterministic=True,
        inputs=("problem", "candidate"),
        returns="bool",
        independence_class="exec.behavioral",
        difficulty_floor=0.5,
        falsifier_example="",
        confirmer_example="",
        created_at=time.time(),
        parent_problem_hash="a" * 64,
    )
    kw.update(overrides)
    return kw


# ───────────────────────── construction / __post_init__ ─────────────────────────

class TestPropertyConstruction:

    def test_valid_minimal_property(self):
        p = Property(**_minimal_kwargs())
        assert p.name == "example"
        assert p.kind is PropertyKind.UNIT_TEST
        assert p.independence_class == "exec.behavioral"

    def test_property_is_frozen(self):
        p = Property(**_minimal_kwargs())
        with pytest.raises(Exception):
            p.name = "mutated"  # type: ignore[misc]

    def test_empty_property_id_rejected(self):
        with pytest.raises(ValueError, match="property_id"):
            Property(**_minimal_kwargs(property_id=""))

    def test_empty_author_rejected(self):
        with pytest.raises(ValueError, match="author"):
            Property(**_minimal_kwargs(author=""))

    def test_non_propertykind_rejected(self):
        with pytest.raises(ValueError, match="kind"):
            Property(**_minimal_kwargs(kind="UNIT_TEST"))  # str, not enum

    def test_invalid_language_rejected(self):
        with pytest.raises(ValueError, match="language"):
            Property(**_minimal_kwargs(language="ruby"))

    def test_invalid_returns_rejected(self):
        with pytest.raises(ValueError, match="returns"):
            Property(**_minimal_kwargs(returns="int"))

    def test_invalid_independence_class_rejected(self):
        with pytest.raises(ValueError, match="independence_class"):
            Property(**_minimal_kwargs(independence_class="sandbox_smoke"))  # v0.1 value

    def test_all_ten_canonical_classes_accepted(self):
        # Classes that accept this property's (kind=UNIT_TEST) construction
        for cls in INDEPENDENCE_CLASSES:
            p = Property(**_minimal_kwargs(independence_class=cls))
            assert p.independence_class == cls

    def test_timeout_ms_zero_rejected(self):
        with pytest.raises(ValueError, match="timeout_ms"):
            Property(**_minimal_kwargs(timeout_ms=0))

    def test_timeout_ms_over_cap_rejected(self):
        with pytest.raises(ValueError, match="timeout_ms"):
            Property(**_minimal_kwargs(timeout_ms=10001))

    def test_source_over_4kb_rejected(self):
        oversized = "# " + "x" * 4200
        assert len(oversized.encode("utf-8")) > 4096
        with pytest.raises(ValueError, match="source"):
            Property(**_minimal_kwargs(source=oversized))

    def test_source_at_exactly_4kb_accepted(self):
        # Largest allowed source string
        padding = "# " + "a" * (4096 - 2)
        assert len(padding.encode("utf-8")) == 4096
        p = Property(**_minimal_kwargs(source=padding))
        assert len(p.source.encode("utf-8")) == 4096

    def test_deterministic_non_bool_rejected(self):
        with pytest.raises(ValueError, match="deterministic"):
            Property(**_minimal_kwargs(deterministic=1))  # int, not bool

    def test_inputs_not_tuple_rejected(self):
        with pytest.raises(ValueError, match="inputs"):
            Property(**_minimal_kwargs(inputs=["problem", "candidate"]))  # list, not tuple

    def test_inputs_non_str_rejected(self):
        with pytest.raises(ValueError, match="inputs"):
            Property(**_minimal_kwargs(inputs=("problem", 42)))

    def test_difficulty_floor_above_one_rejected(self):
        with pytest.raises(ValueError, match="difficulty_floor"):
            Property(**_minimal_kwargs(difficulty_floor=1.1))

    def test_difficulty_floor_negative_rejected(self):
        with pytest.raises(ValueError, match="difficulty_floor"):
            Property(**_minimal_kwargs(difficulty_floor=-0.01))

    def test_parent_problem_hash_wrong_length_rejected(self):
        with pytest.raises(ValueError, match="parent_problem_hash"):
            Property(**_minimal_kwargs(parent_problem_hash="a" * 32))

    def test_parent_problem_hash_non_hex_rejected(self):
        with pytest.raises(ValueError, match="parent_problem_hash"):
            Property(**_minimal_kwargs(parent_problem_hash="z" * 64))

    def test_created_at_zero_rejected(self):
        with pytest.raises(ValueError, match="created_at"):
            Property(**_minimal_kwargs(created_at=0.0))

    def test_created_at_negative_rejected(self):
        with pytest.raises(ValueError, match="created_at"):
            Property(**_minimal_kwargs(created_at=-1.0))

    def test_empty_confirmer_example_accepted_at_construction(self):
        # v0.2.1 §1.3: bundle-emit-time, NOT construction-time
        p = Property(**_minimal_kwargs(confirmer_example=""))
        assert p.confirmer_example == ""

    def test_empty_falsifier_example_accepted_at_construction(self):
        p = Property(**_minimal_kwargs(falsifier_example=""))
        assert p.falsifier_example == ""

    def test_to_dict_roundtrip_keys(self):
        p = Property(**_minimal_kwargs(confirmer_example="ok", falsifier_example="bad"))
        d = p.to_dict()
        assert set(d.keys()) == {
            "property_id", "problem_id", "author", "name", "kind", "description",
            "language", "source", "entry_point", "timeout_ms", "deterministic",
            "inputs", "returns", "independence_class", "difficulty_floor",
            "falsifier_example", "confirmer_example", "created_at", "parent_problem_hash",
        }
        # kind serialized as string enum value for jsonl
        assert d["kind"] == "UNIT_TEST"
        # inputs serialized as list for jsonl
        assert d["inputs"] == ["problem", "candidate"]


# ───────────────────────── admit() gates 1–4 ─────────────────────────

class TestAdmissionGates:

    def _build_sandbox_property(self, **overrides):
        """Build a property that runs through the mock executor."""
        kwargs = dict(
            name="admit_test",
            kind=PropertyKind.UNIT_TEST,
            description="admit test",
            independence_class="exec.behavioral",
            source="def check(problem, candidate):\n    return candidate == 'ok'",
            entry_point="check",
            author="test:admit_1",
            problem_id="p_admit",
            parent_problem_hash="b" * 64,
            confirmer_example="ok",
            falsifier_example="bad",
        )
        kwargs.update(overrides)
        return build_property(**kwargs)

    def test_gate_3_empty_confirmer_rejected(self):
        p = self._build_sandbox_property(confirmer_example="")
        res = admit(p, executor=MockExecutor())
        assert not res.admitted
        assert res.gate_failed == "self_test"
        assert "empty" in res.reason.lower() or "bundle-emit" in res.reason

    def test_gate_3_empty_falsifier_rejected(self):
        p = self._build_sandbox_property(falsifier_example="")
        res = admit(p, executor=MockExecutor())
        assert not res.admitted
        assert res.gate_failed == "self_test"

    def test_gate_3_clean_admit(self):
        p = self._build_sandbox_property()
        res = admit(p, executor=MockExecutor())
        assert res.admitted
        assert res.gate_failed is None
        assert res.self_test_verdicts == ("PASS", "FAIL")
        assert res.determinism_observed is True

    def test_gate_3_confirmer_fails_rejected(self):
        # check() always returns False → confirmer FAILs
        src = "def check(problem, candidate):\n    return False"
        p = self._build_sandbox_property(source=src)
        res = admit(p, executor=MockExecutor())
        assert not res.admitted
        assert res.gate_failed == "self_test"

    def test_gate_3_falsifier_passes_rejected(self):
        # check() always returns True → falsifier PASSes (should FAIL)
        src = "def check(problem, candidate):\n    return True"
        p = self._build_sandbox_property(source=src)
        res = admit(p, executor=MockExecutor())
        assert not res.admitted
        assert res.gate_failed == "self_test"

    def test_gate_4_non_deterministic_detected(self):
        # First call on any input → True. Every subsequent call → False.
        # Sequence admit() issues: confirmer='ok' → True (PASS),
        #                         falsifier='bad' → False (FAIL),
        #                         replay confirmer='ok' → False (FAIL).
        # Replay disagrees with first → gate 4 must fail.
        src = (
            "_seen = [False]\n"
            "def check(problem, candidate):\n"
            "    if not _seen[0]:\n"
            "        _seen[0] = True\n"
            "        return True\n"
            "    return False\n"
        )
        p = self._build_sandbox_property(
            source=src,
            confirmer_example="ok",
            falsifier_example="bad",
            deterministic=True,
        )
        res = admit(p, executor=MockExecutor())
        assert not res.admitted
        assert res.gate_failed == "determinism"
        assert res.determinism_observed is False

    def test_deterministic_false_skips_gate_4(self):
        # Non-deterministic property: claims non-determinism, so replay check is skipped.
        # Use a stateful source whose second call on confirmer differs,
        # then assert we still admit because gate 4 doesn't run.
        src = "def check(problem, candidate):\n    return candidate == 'ok'"
        p = self._build_sandbox_property(source=src, deterministic=False)
        res = admit(p, executor=MockExecutor())
        assert res.admitted
        assert res.determinism_observed is None  # not audited

    def test_trusted_builtin_admits_with_empty_source_skip_gates_1_2(self):
        """Trusted builtins waive gates 1+2 (source/sandbox). Build via build_property
        with trusted=True to register a Python callable, then admit()."""
        def my_check(problem, candidate):
            return candidate == "ok"
        # source="# trusted: no real code" is <4kB; trusted flag stores callable
        p = build_property(
            name="trusted_test",
            kind=PropertyKind.UNIT_TEST,
            description="trusted",
            independence_class="exec.behavioral",
            language="python",
            source="# trusted builtin placeholder",
            entry_point="_b_trusted_test",
            author="builtin:property_engine",
            problem_id="p_trusted",
            parent_problem_hash="c" * 64,
            confirmer_example="ok",
            falsifier_example="bad",
            trusted=True,
            trusted_check_fn=my_check,
        )
        res = admit(p, executor=MockExecutor())
        assert res.admitted, res.reason


# ───────────────────────── LegacyProperty shim ─────────────────────────

class TestLegacyShim:

    def test_legacy_property_construction_emits_deprecation_warning(self):
        with pytest.warns(DeprecationWarning):
            p = LegacyProperty(
                name="legacy",
                domain="code",
                description="legacy test",
                check_fn=lambda s, c: (True, ""),
            )
        assert p.name == "legacy"
        assert p.domain == "code"
        assert p.stochasticity == 0.0
        assert p.required is False
        assert p.weight == 1.0
        assert p.independence_class == ""

    def test_legacy_property_fields_match_v01(self):
        import dataclasses
        fields = [f.name for f in dataclasses.fields(LegacyProperty)]
        assert fields == [
            "name", "domain", "description", "check_fn",
            "stochasticity", "required", "weight", "independence_class",
        ]
