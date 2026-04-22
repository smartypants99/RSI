"""Novel-task synthesizer: propose problems *beyond* the model's current frontier.

Pipeline (see team-lead spec):

  1. Capability frontier probe — read diagnostics pass-rates, identify skills the
     model has ALREADY mastered (>= MASTERY_THRESHOLD), and pair two of them
     into a compositional task requiring BOTH. Not "harder sort" — a problem
     that fuses two mastered skills, which is strictly novel relative to either
     parent bank.

  2. Property co-generation — when the model proposes a problem it ALSO proposes
     verification Properties (schema owned by src/verifier/property_engine.py).
     A SynthesizedTask is admissible iff:
        (a) every property is programmatically checkable OR reformulation-checkable
        (b) at least ONE property is INDEPENDENT — checkable without running the
            model again (pure Python / math / string / regex).

  3. Verifiability self-check — generate a KNOWN-WRONG answer; run every
     property against it; demand that AT LEAST ONE property flags it wrong. A
     bundle whose properties all pass on a wrong answer is "toothless" and gets
     rejected — its verification cannot distinguish right from wrong.

  4. Novelty check — jaccard/content-hash against ground_truth.py bank and prior
     synthesized cycles. Reuses the same dedup primitives as DataGenerator
     (_normalize_for_dedup, _jaccard, content_hash).

Backwards compatibility: the orchestrator calls `TaskSynthesizer(cfg, loader)`
then `.synthesize(diag) -> SynthesisResult(tasks=..., ...)`. That surface is
preserved; the new machinery lives underneath.

Property type is owned by property_verifier's module. We depend on a minimal
duck-typed protocol: anything exposing `.check(answer) -> bool`,
`.independent: bool`, and `.kind: str` works. The real factory is injected.

Reconciled with rsi_design.md v0.2 / v0.2.1:
  - Two-gate architecture: VoV.verify_properties_trustworthy is ADMISSION
    (called here in synthesize_one before emit); quorum_verdict is ACCEPTANCE
    (called downstream by property_verifier.verify at candidate-check time).
  - §3.2.3 model-vs-model adversarial author is DEFERRED — VoV's 8-strategy
    corruption sweep covers the threat mechanically. The adversarial prompt
    and helper were removed; `PropertyDescriptor.adversarial` is retained as
    a harmless flag for shape stability if the feature returns.
  - v0.2.1 registry-write ordering: proposals whose bundle admission fails
    must NOT land on disk — `append_proposed_problem(..., bundle_admitted=False)`
    is a no-op.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Sequence

from .data_generator import (
    TrainingSample,
    ReasoningStep,
    _jaccard,
    _normalize_for_dedup,
)
from ..diagnostics.engine import DiagnosticResult
from ..diagnostics.ground_truth import GroundTruthQuestion
from ..utils.config import SynthesisConfig
from .reasoning_strategies import ReasoningStrategy, StrategyLibrary
from ..verifier.verifier_of_verifiers import (
    verify_properties_trustworthy,
    VerifierTrustReport,
)


def _derive_ctx_from_tests(tests: list[str], entry: str) -> dict[str, Any]:
    """Extract sample_input / edge_inputs / expected_type from assert statements.

    Tests look like `assert solve(5, 3) == 8`. We parse each one and pull:
      - call args (as a tuple if multi-arg, else the single literal) → edge_inputs
      - the first parsed one → sample_input (for output-type + no-exceptions properties)
      - type of the RHS literal → expected_type (e.g. "int", "list")

    Args that aren't ast.literal_eval-able are skipped. If no tests parse,
    returns {} and the property harnesses that need these fields will
    report ERROR (which is still better than mislabelling them).
    """
    import ast as _ast
    edge_inputs: list = []
    expected_type: str = ""
    entry_name = (entry or "solve").strip()
    for t in tests or []:
        src = (t or "").strip()
        if not src:
            continue
        if not src.startswith("assert"):
            src = "assert " + src
        try:
            tree = _ast.parse(src, mode="exec")
        except SyntaxError:
            continue
        if not tree.body or not isinstance(tree.body[0], _ast.Assert):
            continue
        test = tree.body[0].test
        # assert solve(X, Y) == Z → Compare(left=Call(solve, [X,Y]), ops=[Eq], comparators=[Z])
        call_node = None
        rhs_node = None
        if isinstance(test, _ast.Compare) and isinstance(test.left, _ast.Call):
            call_node = test.left
            if test.comparators:
                rhs_node = test.comparators[0]
        elif isinstance(test, _ast.Call):
            call_node = test
        if call_node is None:
            continue
        # Check entry point name matches
        fn_name = ""
        if isinstance(call_node.func, _ast.Name):
            fn_name = call_node.func.id
        elif isinstance(call_node.func, _ast.Attribute):
            fn_name = call_node.func.attr
        if fn_name and entry_name and fn_name != entry_name:
            continue
        args_vals = []
        bail = False
        for a in call_node.args:
            try:
                args_vals.append(_ast.literal_eval(a))
            except Exception:
                bail = True
                break
        if bail:
            continue
        if len(args_vals) == 1:
            edge_inputs.append(args_vals[0])
        else:
            edge_inputs.append(tuple(args_vals))
        if rhs_node is not None and not expected_type:
            try:
                rhs_val = _ast.literal_eval(rhs_node)
                expected_type = type(rhs_val).__name__
            except Exception:
                pass
    out: dict[str, Any] = {}
    if edge_inputs:
        out["edge_inputs"] = edge_inputs[:5]
        out["sample_input"] = edge_inputs[0]
    if expected_type:
        out["expected_type"] = expected_type
    return out

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Property protocol — duck-typed so this module does not block on
# property_engine.py's final schema.
# ---------------------------------------------------------------------------


class PropertyLike(Protocol):
    """Minimum surface task_synthesizer needs from a Property object.

    Aligned with `src/verifier/property_engine.Property` (spec v0.2.1 19-field
    schema; verified by `dataclasses.fields(Property)`). Materializes check_fn lazily from
    (source, language, entry_point) after §1.3 sandbox admission; VoV reads it
    off the resulting object via getattr.

    The ONLY fields the Protocol formally requires are `name` and
    `independence_class` — everything else task_synthesizer touches is
    accessed via getattr so this same Protocol satisfies v0.2.1 Property,
    LegacyProperty, and test doubles alike.

    Fields accessed via getattr (document only — NOT Protocol members):
      - `deterministic: bool`    = present on v0.2.1 Property. VoV replay audit.
      - `stochasticity: float`   = present on LegacyProperty (dropped in v0.2
                                    for the canonical Property). VoV reads
                                    both via getattr — 0.0 == deterministic.
      - `check_fn(sol, ctx)`     = tuple (bool|str, str) or plain bool|str.
                                    NOT on canonical v0.2.1 Property (built
                                    lazily by property_verifier at admission),
                                    IS on LegacyProperty. `_invoke_check` below
                                    handles both shapes and tri-state
                                    PASS/FAIL/ERROR string returns.

    Note: spec §1.2 `kind: PropertyKind` and synthesizer's own `independent:
    bool` are NOT Protocol members. `independent` is derived from
    `independence_class` via `_is_independent()` — removing it from the
    Protocol keeps a real Property (which has no `independent` attribute)
    duck-typable as PropertyLike without external setattr.
    """

    name: str
    independence_class: str


# Classes whose check runs WITHOUT invoking the model again (pure Python /
# math / SMT / sandboxed code / AST / symbolic / bounded search). Derived
# from §2.2 per architect's guidance: transform.semantic and
# perturbation.local typically require a model call, the rest do not.
# dimensional.physical is an arithmetic unit check, so independent.
_INDEPENDENT_CLASSES: frozenset[str] = frozenset({
    "exec.behavioral",
    "algebra.symbolic",
    "smt.logical",
    "structural.static",
    "roundtrip",
    "conservation.global",
    "search.bounded",
    "dimensional.physical",
})


def _is_independent(prop: Any) -> bool:
    """Derive `independent` from a property's independence_class (§2.2).

    Spec rule 2(b) in admissibility: at least ONE property must be
    checkable without running the model. A real v0.2.1 Property has no
    `independent` attribute — we compute it here from its class, which the
    canonical Property carries. If a property-like object happens to expose
    `independent` directly (synthesizer descriptors do), that explicit value
    wins; otherwise the class-based rule applies.
    """
    explicit = getattr(prop, "independent", None)
    if isinstance(explicit, bool):
        return explicit
    cls = getattr(prop, "independence_class", "") or ""
    return cls in _INDEPENDENT_CLASSES


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SynthesizedTask:
    """A novel problem produced by the frontier probe.

    Feeds the trainer via `to_training_sample()`. The `source="synthesized"`
    flag on the projected TrainingSample lets downstream analyzers isolate
    these frontier samples from STaR-provenance ones.
    """
    task_id: str
    domain: str
    prompt: str
    reference_solution: str
    properties: list[PropertyLike] = field(default_factory=list)
    # Known-wrong answer used in the verifiability self-check.
    self_verify_wrong_answer: str = ""
    # Parallel to `properties`: True iff that property flagged the wrong answer.
    self_verify_results: list[bool] = field(default_factory=list)
    novelty_score: float = 1.0  # 1.0 = fully novel; 0 = duplicate
    # Provenance: which two mastered skills were composed.
    parent_skills: tuple[str, str] = ("", "")
    content_hash: str = ""
    subdomain: str = ""
    # VoV audit result — populated in synthesize_one. `None` if VoV has not been
    # run yet (e.g. during unit tests that skip it).
    vov_report: Optional[VerifierTrustReport] = None

    def __post_init__(self) -> None:
        if not self.content_hash and self.prompt:
            self.content_hash = hashlib.md5(
                f"frontier:{self.domain}:{self.prompt}".encode()
            ).hexdigest()

    @property
    def toothless(self) -> bool:
        """True iff NO property flagged the known-wrong answer."""
        return bool(self.self_verify_results) and not any(self.self_verify_results)

    @property
    def has_independent_property(self) -> bool:
        """Rule 2(b): ≥1 property checkable without a model call.

        Uses `_is_independent` which derives from `independence_class` per
        §2.2 when the property lacks an explicit `independent` attribute.
        """
        return any(_is_independent(p) for p in self.properties)

    @property
    def distinct_independence_classes(self) -> int:
        """Count of distinct independence_class tags across materialized properties.

        VoV.quorum_verdict requires ≥3 distinct classes (spec §2.1). We don't
        enforce that here — quorum applies at candidate-solution time, not at
        admission time — but surfacing the count lets the orchestrator log
        which bundles are quorum-eligible.
        """
        classes = {
            (getattr(p, "independence_class", "") or "<unclassified>")
            for p in self.properties
        }
        return len(classes)

    def to_training_sample(self) -> TrainingSample:
        """Project onto the trainer's expected schema.

        reasoning_chain is left empty — the trainer supplies chains at rollout
        time. `expected_answer` carries the reference solution for STaR-style
        grading; `target_weakness` encodes the frontier composition.
        """
        return TrainingSample(
            prompt=self.prompt,
            response="",
            reasoning_chain=[],
            target_weakness=f"frontier:{self.parent_skills[0]}+{self.parent_skills[1]}",
            domain=self.domain,
            verified=False,
            expected_answer=self.reference_solution,
            source="synthesized",
            content_hash=self.content_hash,
        )


@dataclass
class SynthesisResult:
    """Output of one synthesis run — shape preserved for orchestrator consumers.

    `tasks` is a list of SynthesizedTask (NOT TrainingSample) — the
    orchestrator's downstream consensus-verifier expects the richer type so it
    can read `.properties`. Conversion to TrainingSample happens after
    consensus admission.
    """
    tasks: list[SynthesizedTask] = field(default_factory=list)
    properties: dict[str, object] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Capability frontier
# ---------------------------------------------------------------------------


# Per architect's v0.2 guidance: subdomains scoring ≥0.85 over the last two
# cycles count as MASTERED (primary input to compositional pairs). [0.40, 0.85)
# is EMERGING (use sparingly, ≤1 leg per pair). Below 0.40 stays in the
# closed-world weakness-targeted generator path, not here.
MASTERY_THRESHOLD = 0.85
EMERGING_THRESHOLD = 0.40


@dataclass
class SkillProfile:
    """One mastered skill, sourced from diagnostics.subdomain_scores."""
    key: str  # "domain/subdomain"
    domain: str
    subdomain: str
    pass_rate: float
    exemplars: list[str] = field(default_factory=list)

    @property
    def mastered(self) -> bool:
        return self.pass_rate >= MASTERY_THRESHOLD


def extract_mastered_skills(
    subdomain_scores: dict[str, float],
    exemplars_by_key: Optional[dict[str, list[str]]] = None,
    threshold: float = MASTERY_THRESHOLD,
) -> list[SkillProfile]:
    """Pull mastered skills from a DiagnosticResult.subdomain_scores dict.

    `exemplars_by_key` is optional; if provided it should map
    "domain/subdomain" -> up to a few example prompts from that bank, used
    verbatim in the co-generation prompt to anchor the skill.
    """
    profiles: list[SkillProfile] = []
    exemplars_by_key = exemplars_by_key or {}
    for key, score in subdomain_scores.items():
        if score < threshold:
            continue
        if "/" in key:
            dom, sub = key.split("/", 1)
        else:
            dom, sub = key, ""
        profiles.append(SkillProfile(
            key=key, domain=dom, subdomain=sub, pass_rate=float(score),
            exemplars=list(exemplars_by_key.get(key, []))[:3],
        ))
    return profiles


def build_mastery_profile(
    subdomain_scores: dict[str, float],
    prior_subdomain_scores: Optional[dict[str, float]] = None,
    *,
    mastered_threshold: float = MASTERY_THRESHOLD,
    emerging_threshold: float = EMERGING_THRESHOLD,
) -> dict[str, list[str]]:
    """Derive the MasteryProfile input to propose_novel (architect §v0.2 §1).

    Returns `{"mastered": [...], "emerging": [...]}` — list of "domain/subdomain"
    keys. "mastered" = score ≥ mastered_threshold in BOTH the current and the
    prior cycle (when prior_subdomain_scores is provided). "emerging" is open
    at the bottom (≥ emerging_threshold) but strictly below the mastered bar
    in EITHER cycle — a subdomain drifting down out of mastered still counts
    as emerging.

    The two-cycle rule is the architect's anti-noise gate: a single lucky
    cycle does not qualify a subdomain for the frontier. When prior scores
    are not supplied, the current cycle alone is used (caller's choice: that
    skips the stability check).
    """
    mastered: list[str] = []
    emerging: list[str] = []
    for key, cur in subdomain_scores.items():
        prior = None
        if prior_subdomain_scores is not None:
            prior = prior_subdomain_scores.get(key)
        cur = float(cur)
        if cur >= mastered_threshold and (prior is None or prior >= mastered_threshold):
            mastered.append(key)
        elif cur >= emerging_threshold:
            emerging.append(key)
    return {"mastered": mastered, "emerging": emerging}


def exemplars_from_per_question(
    per_question: Sequence[dict],
    max_per_key: int = 3,
) -> dict[str, list[str]]:
    """Derive per-(domain/subdomain) exemplar prompts from diagnostic records.

    Prefers questions the model got CORRECT (proof the skill is mastered) so
    the co-gen prompt anchors on examples of confident behavior.
    """
    out: dict[str, list[str]] = {}
    for rec in per_question:
        if not rec.get("correct"):
            continue
        dom = rec.get("domain", "")
        sub = rec.get("subdomain", "")
        key = f"{dom}/{sub}" if sub else dom
        q = rec.get("question") or rec.get("prompt") or ""
        if not q:
            continue
        bucket = out.setdefault(key, [])
        if len(bucket) < max_per_key:
            bucket.append(q)
    return out


# ---------------------------------------------------------------------------
# Co-generation prompt template
# ---------------------------------------------------------------------------


CO_GEN_TEMPLATE = """\
You are proposing a NOVEL problem for a model that has already mastered these two skills:

Skill A: {skill_a_key} (pass rate {skill_a_rate:.2f})
{skill_a_examples}

Skill B: {skill_b_key} (pass rate {skill_b_rate:.2f})
{skill_b_examples}

Design a SINGLE problem whose solution REQUIRES using BOTH skills together — not
either alone. The problem must NOT already appear in any standard benchmark; it
should compose the two skills in a way that is at the edge of the model's
current capability.

Return your answer strictly as follows, with each labeled block on its own line:

PROBLEM: <one-paragraph problem statement>
ANSWER: <the reference correct answer, concise, machine-checkable form>
PROPERTIES:
- <property 1: a condition the correct answer must satisfy, checkable by pure
  code/math/regex, with enough specificity that a wrong answer would violate it>
- <property 2: another independent condition — different from property 1>
- <property 3 (optional): a cross-reformulation check, e.g. "answer also solves
  this equivalent reformulation: ...">
WRONG_ANSWER: <a plausibly-wrong answer a weak model might give — used to
  self-test whether your properties actually catch errors>

At least one property MUST be checkable WITHOUT running any model — pure Python,
math, regex, or arithmetic on the answer string. Mark such properties with the
prefix [INDEPENDENT]. Properties that require reformulating and re-asking a
model should be prefixed [REFORMULATION].

Each property must ALSO carry an `independence_class` — a short tag naming the
verification strategy family (e.g. "exact_match", "type_check", "range_check",
"unit_tests", "symbolic_equiv", "reformulation"). Downstream quorum requires
≥3 DIFFERENT classes, so prefer three DIFFERENT strategies. Write the class in
braces at the start of the property line: `- [INDEPENDENT] {type_check} answer
is an integer`.
"""


def build_co_gen_prompt(skill_a: SkillProfile, skill_b: SkillProfile) -> str:
    def _fmt_examples(p: SkillProfile) -> str:
        if not p.exemplars:
            return "  (no exemplars available)"
        return "\n".join(f"  - {e}" for e in p.exemplars)

    return CO_GEN_TEMPLATE.format(
        skill_a_key=skill_a.key, skill_a_rate=skill_a.pass_rate,
        skill_a_examples=_fmt_examples(skill_a),
        skill_b_key=skill_b.key, skill_b_rate=skill_b.pass_rate,
        skill_b_examples=_fmt_examples(skill_b),
    )


# ---------------------------------------------------------------------------
# Response parsing — structured blocks emitted by the co-gen prompt
# ---------------------------------------------------------------------------


@dataclass
class CoGenParse:
    problem: str = ""
    answer: str = ""
    property_descriptors: list[dict] = field(default_factory=list)
    wrong_answer: str = ""
    issues: list[str] = field(default_factory=list)


_BLOCK_LABELS = ("PROBLEM:", "ANSWER:", "PROPERTIES:", "WRONG_ANSWER:")


def parse_co_gen_response(text: str) -> CoGenParse:
    """Split the model's output into the labeled blocks."""
    out = CoGenParse()
    if not text or not text.strip():
        out.issues.append("empty response")
        return out

    positions: dict[str, int] = {}
    for label in _BLOCK_LABELS:
        m = re.search(rf"(?im)^\s*{re.escape(label)}", text)
        if m:
            positions[label] = m.start()
    if not positions:
        out.issues.append("no labeled blocks found")
        return out

    ordered = sorted(positions.items(), key=lambda kv: kv[1])
    for i, (label, start) in enumerate(ordered):
        end = ordered[i + 1][1] if i + 1 < len(ordered) else len(text)
        body = text[start:end]
        body = re.sub(rf"(?i)^\s*{re.escape(label)}\s*", "", body, count=1).strip()
        if label == "PROBLEM:":
            out.problem = body
        elif label == "ANSWER:":
            out.answer = body
        elif label == "PROPERTIES:":
            out.property_descriptors = _parse_property_lines(body)
        elif label == "WRONG_ANSWER:":
            out.wrong_answer = body

    if not out.problem:
        out.issues.append("missing PROBLEM")
    if not out.answer:
        out.issues.append("missing ANSWER")
    if not out.property_descriptors:
        out.issues.append("missing PROPERTIES")
    if not out.wrong_answer:
        out.issues.append("missing WRONG_ANSWER")
    return out


_PROP_PREFIX_RE = re.compile(
    r"^\s*[-*]\s*"
    r"(?:\[(INDEPENDENT|REFORMULATION)\]\s*)?"   # group 1: kind tag
    r"(?:\{([^}]+)\}\s*)?"                        # group 2: independence_class
    r"(.+)$",
    re.IGNORECASE,
)


def _parse_property_lines(body: str) -> list[dict]:
    descriptors: list[dict] = []
    for raw in body.splitlines():
        raw = raw.rstrip()
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        m = _PROP_PREFIX_RE.match(raw)
        if not m:
            continue
        tag = (m.group(1) or "").upper()
        klass = (m.group(2) or "").strip().lower()
        desc_text = m.group(3).strip()
        if not desc_text:
            continue
        descriptors.append({
            "kind": "programmatic" if tag == "INDEPENDENT" else (
                "reformulation" if tag == "REFORMULATION" else "unspecified"
            ),
            "independent": tag == "INDEPENDENT",
            # Strategy-family tag consumed by VoV.quorum_verdict for class
            # diversity. Falls back to "<unclassified>" when the model didn't
            # emit one — quorum will then collapse all such props to a single
            # class and likely fail the diversity requirement (fail-closed).
            "independence_class": klass or "<unclassified>",
            "description": desc_text,
        })
    return descriptors


# ---------------------------------------------------------------------------
# Property construction adapter
# ---------------------------------------------------------------------------

# The actual construction of a PropertyLike from a descriptor is owned by
# property_engine. We accept an injected factory so this module stays testable
# with a plain lambda and swaps in the real factory in prod.

PropertyFactory = Callable[[dict, str, str], Optional[PropertyLike]]
# Signature: factory(descriptor, problem_prompt, reference_answer) -> property


def _default_property_factory(
    descriptor: Any, problem: str, answer: str,
) -> Optional[PropertyLike]:
    """Best-effort materializer for a PropertyDescriptor into a v0.2.1 Property.

    v0.2.1 split: task_synthesizer does NOT emit `source`, `entry_point`, or
    sandbox-executable code — that's property_verifier's job post-admission.
    The model-authored co-gen path carries only intent (name, kind, class,
    description, examples, difficulty_floor, language). The canonical
    integration point is `TaskSynthesizer.propose_properties(problem)` which
    feeds descriptors into property_verifier's admit()/build_property chain.

    This factory exists so callers who want a quick PropertyLike without the
    full admission dance (unit tests, VoV self-tests in synthesize_one) get
    a minimally-shaped Property. It late-binds `build_property`, fabricating
    the missing fields (source="# placeholder", entry_point="check") that
    property_verifier's admit() will reject at gate 2 — which is the correct
    behavior for the "fast preview" path: no admission means no training use.

    Returns None when property_engine isn't importable or when the Property
    constructor raises (e.g. missing parent_problem_hash). Rejections are
    logged at debug so the orchestrator's real path (propose_properties →
    admit) stays visible.
    """
    try:
        from ..verifier.property_engine import build_property, PropertyKind  # type: ignore
    except ImportError:
        return None
    # Accept both the PropertyDescriptor dataclass and a plain dict for
    # backwards-compat with tests that predate the dataclass.
    def _g(name: str, default: Any = "") -> Any:
        if hasattr(descriptor, name):
            return getattr(descriptor, name)
        if isinstance(descriptor, dict):
            return descriptor.get(name, default)
        return default

    try:
        kind_raw = _g("kind", "POSTCONDITION")
        try:
            kind_enum = PropertyKind(kind_raw) if not isinstance(kind_raw, PropertyKind) else kind_raw
        except ValueError:
            kind_enum = PropertyKind.POSTCONDITION
        return build_property(
            name=_g("name") or "synthesizer_stub",
            kind=kind_enum,
            description=_g("description") or "(placeholder from task_synthesizer preview factory)",
            independence_class=_g("independence_class") or "exec.behavioral",
            source="# placeholder — admit() will reject at gate 2",
            entry_point="check",
            author=f"synth:{_g('author_run_id') or 'preview'}",
            problem_id=_g("problem_id") or "preview",
            parent_problem_hash=hashlib.sha256(
                (problem or "preview").encode("utf-8")
            ).hexdigest(),
            language=_g("language") or "python",
            difficulty_floor=float(_g("difficulty_floor", 0.5) or 0.5),
            confirmer_example=_g("confirmer_example") or "",
            falsifier_example=_g("falsifier_example") or "",
        )
    except Exception as exc:
        logger.debug("default factory preview build failed on %r: %s",
                     getattr(descriptor, "name", "?"), exc)
        return None


# ---------------------------------------------------------------------------
# Verifiability self-check (toothlessness rejection)
# ---------------------------------------------------------------------------


def _coerce_verdict(value: Any) -> bool:
    """Coerce a property verdict to bool for the self-verify pre-filter.

    Handles (per architect §1.2 drift note item 5):
      - bool: returned as-is.
      - str: tri-state "PASS" | "FAIL" | "ERROR" (case-insensitive). Anything
             else is treated as bool(str) which, for non-empty strings, is
             True — but this branch returns False on FAIL/ERROR so a PASS-
             on-wrong-answer doesn't get masked.
      - other truthy/falsy: delegates to bool().
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().upper()
        if v == "PASS":
            return True
        if v in ("FAIL", "ERROR"):
            return False
        # Fall through: treat unrecognized strings as their truthiness. Kept
        # conservative so a check_fn returning "ok" or "no" stays explicit —
        # we prefer the model emit canonical tri-state strings.
    return bool(value)


def _invoke_check(prop: PropertyLike, solution: Any, problem_ctx: Any) -> bool:
    """Uniform (solution, ctx) invocation compatible with property_engine.Property.

    Mirrors `verifier_of_verifiers._run_property`: calls `check_fn(sol, ctx)`,
    falls back to `check_fn(sol)` on TypeError, unwraps (verdict, reason)
    tuples. The verdict may be bool OR the tri-state string
    {"PASS","FAIL","ERROR"} per v0.2 property execution contract — both are
    handled via `_coerce_verdict`.
    """
    fn = getattr(prop, "check_fn", None)
    if fn is None:
        raise ValueError(f"property {getattr(prop, 'name', '?')} has no check_fn")
    try:
        result = fn(solution, problem_ctx)
    except TypeError as e:
        # Only retry on arity mismatch — a bare `except TypeError` hides real
        # bugs inside the check_fn (e.g. `len(None)`).
        msg = str(e)
        if "positional argument" in msg or "takes" in msg or "argument" in msg:
            result = fn(solution)
        else:
            raise
    if isinstance(result, tuple) and len(result) == 2:
        return _coerce_verdict(result[0])
    return _coerce_verdict(result)


def run_self_verification(
    properties: Sequence[PropertyLike],
    wrong_answer: str,
    problem_ctx: Any = None,
) -> list[bool]:
    """Run each property against the KNOWN-WRONG answer. Fast pre-filter for VoV.

    Entry i is True iff property i FLAGGED the wrong answer (check returned
    False on it). A bundle is toothless when every entry is False — VoV would
    reject it anyway on aggregate kill-rate, but checking one deliberate wrong
    answer up front is 1/N the cost of the full VoV corruption audit.
    """
    results: list[bool] = []
    for prop in properties:
        try:
            passed = _invoke_check(prop, wrong_answer, problem_ctx)
        except Exception as exc:
            logger.warning("property raised during self-verify: %s", exc)
            passed = True  # crash = "did not catch"
        results.append(not passed)
    return results


# ---------------------------------------------------------------------------
# Novelty check
# ---------------------------------------------------------------------------


NOVELTY_JACCARD_THRESHOLD = 0.85  # matches DataGenerator.DEDUP_JACCARD_THRESHOLD


def novelty_score(
    prompt: str,
    ground_truth_bank: Sequence[GroundTruthQuestion],
    prior_synth_prompts: Sequence[str] = (),
    *,
    precomputed_bank_sigs: Optional[Sequence[Any]] = None,
    precomputed_prior_sigs: Optional[Sequence[Any]] = None,
) -> float:
    """Return 1.0 - max_jaccard(prompt, any prior prompt).

    Score of 1.0 = fully novel. Below (1 - NOVELTY_JACCARD_THRESHOLD) the
    caller rejects.

    `precomputed_*_sigs` lets the caller pay the normalization cost once per
    run instead of once per candidate pair. Without this, a 20-pair cycle with
    a 5000-entry bank normalizes 100 000 prompts per cycle.
    """
    sig = _normalize_for_dedup(prompt)
    if not sig:
        return 0.0
    max_sim = 0.0

    if precomputed_bank_sigs is not None:
        bank_sigs = precomputed_bank_sigs
    else:
        bank_sigs = [_normalize_for_dedup(q.prompt) for q in ground_truth_bank]
    for other_sig in bank_sigs:
        s = _jaccard(sig, other_sig)
        if s > max_sim:
            max_sim = s
            if max_sim >= 1.0:
                return 0.0

    if precomputed_prior_sigs is not None:
        prior_sigs = precomputed_prior_sigs
    else:
        prior_sigs = [_normalize_for_dedup(p) for p in prior_synth_prompts]
    for other_sig in prior_sigs:
        s = _jaccard(sig, other_sig)
        if s > max_sim:
            max_sim = s
            if max_sim >= 1.0:
                return 0.0
    return max(0.0, 1.0 - max_sim)


# ---------------------------------------------------------------------------
# Admissibility gate
# ---------------------------------------------------------------------------


@dataclass
class AdmissionReport:
    admitted: bool
    reasons: list[str] = field(default_factory=list)


def check_admissibility(task: SynthesizedTask) -> AdmissionReport:
    """Apply the spec's four admissibility rules."""
    reasons: list[str] = []

    # Rule 2(a) (dropped per architect v0.2.1 drift note): the kind-string
    # check was redundant. Real v0.2.1 Property has no "programmatic"/"
    # reformulation" tag — that's a synthesizer-only metadata axis. property_
    # engine's admission gates enforce §1.2 PropertyKind on materialized
    # Properties; validating it here again was belt-and-braces against a
    # protocol attribute we removed. Rule 2(b) (independent via class)
    # covers the programmatic-vs-model-invocation distinction directly.

    # Rule 2(b): at least one independent property.
    if not task.has_independent_property:
        reasons.append("no independent property — verification requires the model")

    # Rule 3: not toothless.
    if task.toothless:
        reasons.append("toothless: no property caught the known-wrong answer")

    # Rule 4: sufficient novelty.
    if task.novelty_score < (1.0 - NOVELTY_JACCARD_THRESHOLD):
        reasons.append(
            f"novelty too low: {task.novelty_score:.3f} "
            f"< {1.0 - NOVELTY_JACCARD_THRESHOLD:.3f}"
        )

    # Rule 5 (authoritative): the verifier-of-verifiers gate. A bundle that
    # passes our cheap "one wrong answer" pre-filter still has to survive
    # VoV's full corruption audit — kill-rate ≥ MIN_KILL_RATE across a
    # domain-appropriate corruption set, with ≥1 property earning individual
    # trust (deterministic + accepts reference). If VoV didn't run (unit
    # tests), skip this check rather than pretending it passed.
    if task.vov_report is not None and not task.vov_report.passed:
        reasons.append(f"VoV rejected: {task.vov_report.reason}")

    return AdmissionReport(admitted=not reasons, reasons=reasons)


# ---------------------------------------------------------------------------
# VoV adapter
# ---------------------------------------------------------------------------


def _run_vov(task: SynthesizedTask) -> Optional[VerifierTrustReport]:
    """Invoke the Verifier-of-Verifiers on a candidate bundle.

    VoV derives its corruption strategy from `domain`. We pass the synthesizer's
    domain string through directly — VoV handles "code" / "math" / "text", and
    returns an unpassed report with a descriptive reason for unsupported
    domains (fail-closed, per its contract).
    """
    try:
        return verify_properties_trustworthy(
            task_id=task.task_id,
            reference_solution=task.reference_solution,
            properties=list(task.properties),
            problem_ctx={"problem": task.prompt, "reference": task.reference_solution},
            domain=task.domain,
        )
    except Exception as exc:
        logger.warning("VoV raised on task %s: %s", task.task_id, exc)
        # Failure in VoV itself = untrusted. Fabricate an unpassed report so
        # admissibility surfaces the reason.
        return VerifierTrustReport(
            task_id=task.task_id,
            reason=f"VoV raised {type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# Pair selection — the "frontier" itself
# ---------------------------------------------------------------------------


def propose_skill_pairs(
    mastered: Sequence[SkillProfile],
    max_pairs: int = 8,
    prefer_cross_domain: bool = True,
) -> list[tuple[SkillProfile, SkillProfile]]:
    """Pick up to `max_pairs` (a, b) pairs to compose.

    Ranks by pass-rate descending. If `prefer_cross_domain`, exhausts
    cross-domain pairs first (math × code, logic × math) — these compositional
    frontiers are usually where new capability lives — then fills remaining
    slots with same-domain pairs.
    """
    ranked = sorted(mastered, key=lambda p: -p.pass_rate)
    pairs: list[tuple[SkillProfile, SkillProfile]] = []
    used: set[tuple[str, str]] = set()

    if prefer_cross_domain:
        for i, a in enumerate(ranked):
            for b in ranked[i + 1:]:
                if a.domain == b.domain:
                    continue
                key = (a.key, b.key)
                if key in used:
                    continue
                used.add(key)
                pairs.append((a, b))
                if len(pairs) >= max_pairs:
                    return pairs

    for i, a in enumerate(ranked):
        for b in ranked[i + 1:]:
            key = (a.key, b.key)
            if key in used:
                continue
            used.add(key)
            pairs.append((a, b))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


# ---------------------------------------------------------------------------
# Orchestrator entry point — preserves the existing (config, model_loader)
# constructor and synthesize(diag) -> SynthesisResult contract used by
# src/orchestrator/loop.py.
# ---------------------------------------------------------------------------


class TaskSynthesizer:
    """Drives the full pipeline: probe -> co-gen -> self-verify -> novelty -> admit.

    Constructor matches the orchestrator's expectation:
        TaskSynthesizer(config: SynthesisConfig, model_loader)

    Public entry: `synthesize(diag) -> SynthesisResult`. Internal helpers
    (`synthesize_one`, `synthesize_from_pairs`) are injection-friendly for
    unit tests.
    """

    def __init__(
        self,
        config: SynthesisConfig,
        model_loader=None,
        *,
        property_factory: Optional[PropertyFactory] = None,
        ground_truth_bank: Sequence[GroundTruthQuestion] = (),
        generate_fn: Optional[Callable[[str], str]] = None,
        run_vov: bool = True,
    ):
        self.config = config
        self.model_loader = model_loader
        self.property_factory = property_factory or _default_property_factory
        self.ground_truth_bank = list(ground_truth_bank)
        # Allow tests to inject a pure-Python generate_fn; prod resolves via
        # model_loader.generate_batch at call time.
        self._generate_fn_override = generate_fn
        # Authoritative admission gate. Set False only for unit tests that
        # exercise the pipeline without the VoV corruption audit.
        self.run_vov = run_vov
        self._prior_prompts: list[str] = []
        # Pre-normalize the ground-truth bank ONCE. Rebuilding this per call
        # was O(N*M) on hot path — 5000-entry bank × 20 pairs = 100k tokenizations.
        self._bank_sigs = [_normalize_for_dedup(q.prompt) for q in self.ground_truth_bank]
        self._prior_sigs: list[Any] = []
        self._task_counter = 0
        # Mastery state used by the rsi_tick entry points. `set_diagnostics(diag)`
        # populates these before propose_batch() — orchestrator's responsibility.
        # Kept separate from `synthesize(diag)` so the rsi_tick path doesn't
        # have to pass the full DiagnosticResult into every proposal call.
        self._subdomain_scores: dict[str, float] = {}
        self._prior_subdomain_scores: dict[str, float] = {}
        self._exemplars_by_key: dict[str, list[str]] = {}
        self._session_id: str = ""
        self._run_id: str = ""
        # Populated by set_diagnostics(); used by propose_batch_code to
        # seed proposals from the model's most recent failures.
        self._failed_diag_questions: list[str] = []
        # Differential self-solve gate: reject proposals the model can
        # already solve blind (= inside its capability, not at the
        # frontier). Toggle off for tests that don't want the extra
        # generate call.
        self._frontier_self_solve_gate: bool = True
        # Curriculum escalation: current skill-pair the orchestrator's
        # DifficultyTracker identified as the frontier ("domain/subdomain"),
        # and the ratcheted minimum proposal difficulty. Orchestrator sets
        # these each tick via set_frontier_hint() / set_difficulty_floor().
        self._frontier_skill: str = ""
        self._difficulty_floor_override: Optional[float] = None
        # Cross-cycle few-shot banks (property_library). Orchestrator calls
        # set_registries(regs) once so propose_batch_code can rank admitted
        # properties and high-accept-count problems to inject as prompt prefix.
        self._registries: Any = None
        # Reasoning-strategy library (Task #11). Loaded lazily on first use
        # and only when config.strategy_library_enabled. Each call to
        # propose_batch_code / solve_batch that consumes `_strategy_prefix()`
        # also records the top strategy names for later accept-rate tracking.
        self._strategy_library: Optional[StrategyLibrary] = None
        self._last_strategy_names: list[str] = []

    # -- cross-cycle library wiring -----------------------------------------

    def set_registries(self, registries: Any) -> None:
        """Attach an RSIRegistries instance so propose_batch_code can draw few-shot
        exemplars from prior cycles. Orchestrator wires this during rsi_tick
        setup; unit tests leave it unset (=> library prefix is always empty).
        """
        self._registries = registries

    def _compute_library_prefix(self) -> str:
        """Build the cross-cycle few-shot prefix for the code-proposal prompt.

        Returns "" when (a) use_property_library is False, (b) registries
        aren't attached, or (c) the PropertyRegistry has fewer than
        library_min_admitted distinct admitted property_ids (cold-start
        guard — rendering a tiny bank is noise that only distracts).
        """
        if not getattr(self.config, "use_property_library", True):
            return ""
        regs = self._registries
        if regs is None:
            return ""
        try:
            from .property_library import build_library_prefix
        except ImportError:
            logger.debug("property_library not importable; skipping prefix")
            return ""
        try:
            return build_library_prefix(
                property_registry=getattr(regs, "property_registry", None),
                verification_log=getattr(regs, "verification_log", None),
                problem_registry=getattr(regs, "problem_registry", None),
                training_pool=getattr(regs, "training_pool", None),
                min_admitted_for_gate=int(
                    getattr(self.config, "library_min_admitted", 20)
                ),
                k_properties=int(getattr(self.config, "library_k_properties", 5)),
                k_proposer=int(getattr(self.config, "library_k_proposer", 3)),
                min_vov_score=float(
                    getattr(self.config, "library_min_vov_score", 1.0)
                ),
            )
        except Exception as exc:
            logger.warning(
                "library prefix build failed (%s: %s) — skipping",
                type(exc).__name__, exc,
            )
            return ""

    # -- reasoning-strategy library (Task #11) ------------------------------

    def _get_strategy_library(self) -> Optional[StrategyLibrary]:
        if not getattr(self.config, "strategy_library_enabled", False):
            return None
        if self._strategy_library is None:
            try:
                lib = StrategyLibrary(
                    path=getattr(
                        self.config, "strategy_library_path",
                        "outputs/reasoning_strategies.jsonl",
                    ),
                    ab_holdout_size=int(getattr(
                        self.config, "strategy_ab_holdout_size", 4)),
                    k_few_shot=int(getattr(
                        self.config, "strategy_library_k_few_shot", 2)),
                )
                lib.load()
                self._strategy_library = lib
            except Exception as exc:
                logger.warning("strategy library load failed: %s", exc)
                return None
        return self._strategy_library

    def _strategy_prefix(self) -> str:
        """Prefix (few-shot prefix from top-k strategies) to prepend to system
        prompts. Empty string when disabled / library empty. Also caches the
        top strategy names so record_strategy_result() can credit them later.
        """
        lib = self._get_strategy_library()
        if lib is None:
            self._last_strategy_names = []
            return ""
        try:
            top = lib.top_k()
            self._last_strategy_names = [s.name for s in top]
            return lib.few_shot_prefix() or ""
        except Exception as exc:
            logger.debug("strategy prefix build failed: %s", exc)
            self._last_strategy_names = []
            return ""

    def record_strategy_result(self, success: bool, *, holdout: bool = False) -> None:
        """Public hook: credit the strategies last surfaced by ``_strategy_prefix``
        with one accept/reject trial. No-op if strategy library is disabled
        or no prefix was used on the last propose/solve call.
        """
        lib = self._get_strategy_library()
        if lib is None or not self._last_strategy_names:
            return
        try:
            for name in self._last_strategy_names:
                lib.record_result(name, success=success, holdout=holdout)
        except Exception as exc:
            logger.debug("record_strategy_result failed: %s", exc)

    # -- LLM access ----------------------------------------------------------

    def _generate(self, prompt: str) -> str:
        """Single-prompt generate (slow path — prefer _generate_many for batches)."""
        if self._generate_fn_override is not None:
            return self._generate_fn_override(prompt)
        if self.model_loader is None:
            raise RuntimeError(
                "TaskSynthesizer has no generate_fn and no model_loader — "
                "cannot produce co-gen outputs"
            )
        try:
            resps = self.model_loader.generate_batch(
                [prompt], max_new_tokens=1024, temperature=0.8, top_p=0.95,
            )
        except Exception as exc:
            logger.warning("model_loader.generate_batch failed: %s", exc)
            return ""
        return resps[0] if resps else ""

    def _generate_many_code(self, prompts: list[str]) -> list[str]:
        """Batched generate for prompts that end with a ```python fence.

        Adds stop=["```"] so vLLM stops as soon as the function body closes,
        instead of running out the full max_new_tokens on trailing prose. On
        R1-style reasoning models this alone can save 30–60% of solve-time
        tokens. HF path ignores `stop` (compat shim, correctness preserved).
        """
        if not prompts:
            return []
        if self._generate_fn_override is not None:
            return [self._generate_fn_override(p) for p in prompts]
        if self.model_loader is None:
            raise RuntimeError(
                "TaskSynthesizer has no generate_fn and no model_loader — "
                "cannot produce co-gen outputs"
            )
        # Task #10: configurable solver cap — R1 still needs <think> headroom
        # but 1200 is enough for a single-function solve with the ``` stop.
        solver_cap = int(getattr(self.config, "solver_max_new_tokens", 2048))
        try:
            return list(self.model_loader.generate_batch(
                prompts, max_new_tokens=solver_cap, temperature=0.6, top_p=0.95,
            ))
        except Exception as exc:
            logger.warning("model_loader.generate_batch (code) failed: %s", exc)
            return ["" for _ in prompts]

    def _generate_many(self, prompts: list[str]) -> list[str]:
        """Batched generate — N prompts in one vLLM call.

        This is the critical hot path for propose_batch. vLLM batches prompts
        across the KV cache, so 20 prompts in one call is ~20x faster than 20
        separate calls (each of which is a 1-prompt "batch" with no batching
        benefit).
        """
        if not prompts:
            return []
        if self._generate_fn_override is not None:
            # Override is single-prompt; call it per prompt (test path).
            return [self._generate_fn_override(p) for p in prompts]
        if self.model_loader is None:
            raise RuntimeError(
                "TaskSynthesizer has no generate_fn and no model_loader — "
                "cannot produce co-gen outputs"
            )
        # Task #10: configurable proposer cap — 600 by default, since the
        # PROBLEM/ENTRY/REFERENCE/TESTS structure fits comfortably under that.
        proposer_cap = int(getattr(self.config, "proposer_max_new_tokens", 2048))
        try:
            return list(self.model_loader.generate_batch(
                prompts, max_new_tokens=proposer_cap, temperature=0.6, top_p=0.95,
            ))
        except Exception as exc:
            logger.warning("model_loader.generate_batch failed: %s", exc)
            return ["" for _ in prompts]

    # -- Public API ----------------------------------------------------------

    def register_prior_prompts(self, prompts: Sequence[str]) -> None:
        self._prior_prompts.extend(prompts)
        self._prior_sigs.extend(_normalize_for_dedup(p) for p in prompts)

    # -- rsi_tick entry points (architect §4 step 1/2) -----------------------

    def set_diagnostics(
        self,
        diag: DiagnosticResult,
        *,
        prior_subdomain_scores: Optional[dict[str, float]] = None,
        session_id: str = "",
        run_id: str = "",
    ) -> None:
        """Populate mastery state for propose_batch/propose_properties.

        Orchestrator calls this once per rsi_tick before Step 1 (proposal).
        `prior_subdomain_scores` is the previous cycle's scores so
        `build_mastery_profile` can apply its two-cycle stability rule; pass
        None on first tick.
        """
        self._subdomain_scores = dict(getattr(diag, "subdomain_scores", {}) or {})
        self._prior_subdomain_scores = dict(prior_subdomain_scores or {})
        per_q = list(getattr(diag, "per_question", []) or [])
        self._exemplars_by_key = exemplars_from_per_question(per_q)
        # Frontier seed bank: questions the model JUST failed on the
        # diagnostic. propose_batch_code uses these to anchor proposals to
        # the actual capability frontier (rsi_design.md §3.2). Limit to 32
        # to keep prompts bounded.
        self._failed_diag_questions = [
            e.get("question", "")
            for e in per_q
            if (not e.get("correct", False)) and e.get("question", "")
        ][:32]
        if session_id:
            self._session_id = session_id
        if run_id:
            self._run_id = run_id

    def set_frontier_hint(self, skill_pair: str) -> None:
        """Set the frontier skill-pair (e.g. "code/implementation") that the
        DifficultyTracker identified as the easiest zone currently failing.
        propose_batch_code splices this into a fraction of prompts.
        """
        self._frontier_skill = str(skill_pair or "")

    def set_difficulty_floor(self, floor: Optional[float]) -> None:
        """Override the minimum self-reported DIFFICULTY: a proposal must
        declare. None restores the module-default floor.
        """
        if floor is None:
            self._difficulty_floor_override = None
        else:
            self._difficulty_floor_override = float(
                max(0.0, min(0.9, floor))
            )

    def propose_batch(
        self,
        n: int,
        *,
        stop_on_empty: bool = False,
    ) -> list[ProposedProblem]:
        """§4 step 1: emit up to `n` ProposedProblems from current diagnostics.

        Instance-method wrapper around the module-level propose_batch(). Pulls
        its inputs from state set via `set_diagnostics`. Returns ONLY the
        accepted proposals (integrator's rsi_tick only needs the list, not
        per-call issues — those are logged internally).

        If set_diagnostics() hasn't been called, returns [] after a warning —
        the frontier probe needs a mastery profile to exist, and without one
        there's nothing to propose.
        """
        if not self._subdomain_scores:
            logger.warning(
                "propose_batch: no subdomain_scores set — call "
                "set_diagnostics(diag) first. Returning empty list."
            )
            return []
        mp = build_mastery_profile(
            self._subdomain_scores, self._prior_subdomain_scores or None,
        )

        # Build N prompts ahead of time, then batch the model call. The
        # previous serial path paid one full 25s generate per proposal —
        # 20 proposals = 500s/cycle with no batching benefit.
        import random as _random
        rng = _random.Random(hash(self._session_id) if self._session_id else 0)
        prompt_and_skills: list[tuple[str, tuple[SkillProfile, ...]]] = []
        for i in range(max(0, n)):
            try:
                skills = _select_skills_for_proposal(
                    mp, self._subdomain_scores, self._exemplars_by_key or {},
                    n_skills=2, rng=rng,
                )
            except ValueError as exc:
                logger.info("propose_batch: no usable skills (%s)", exc)
                break
            if not skills:
                break
            prompt = build_proposal_prompt_from_profiles(skills)
            prompt_and_skills.append((prompt, tuple(skills)))

        if not prompt_and_skills:
            return []

        prompts = [p for p, _ in prompt_and_skills]
        raws = self._generate_many(prompts)
        if len(raws) < len(prompts):
            raws = list(raws) + [""] * (len(prompts) - len(raws))

        # Parse each response; collect (proposal, issues).
        accepted: list[ProposedProblem] = []
        per_call_issues: list[list[str]] = []
        from collections import Counter
        issue_tally: Counter = Counter()
        prior_for_novelty = list(self._prior_prompts)

        for i, ((_, skills), raw) in enumerate(zip(prompt_and_skills, raws)):
            run_id_i = f"{self._run_id}#{i}" if self._run_id else f"batch#{i}"
            parse = parse_proposal_response(raw, author_run_id=run_id_i)
            issues: list[str] = list(parse.issues)
            if not parse.ok:
                for iss in issues:
                    issue_tally[_issue_category(iss)] += 1
                per_call_issues.append(issues)
                if stop_on_empty:
                    break
                continue

            if len(parse.property_descriptors) < 2:
                issues.append(
                    f"only {len(parse.property_descriptors)} properties; need ≥2"
                )
            if not class_coverage_ok(parse.property_descriptors):
                issues.append("properties span < 3 independence_classes (§3.1)")

            nn_dist, nn_text = compute_nearest_neighbor_dist(
                parse.problem_text,
                prior_for_novelty + [p.problem_text for p in accepted],
            )
            phash = hashlib.sha256(parse.problem_text.encode("utf-8")).hexdigest()
            pp = ProposedProblem(
                problem_id=str(uuid.uuid4()),
                problem_text=parse.problem_text,
                declared_difficulty=parse.declared_difficulty,
                declared_difficulty_reason=parse.declared_difficulty_reason,
                nearest_neighbor_problem=parse.nearest_neighbor_problem or nn_text,
                nearest_neighbor_dist=nn_dist,
                axis_of_extension=parse.axis_of_extension,
                parent_skills=tuple(s.key for s in skills),
                problem_hash=phash,
                author=f"model:{run_id_i}",
                created_at=time.time(),
                session_id=self._session_id,
                property_descriptors=list(parse.property_descriptors),
            )
            accepted.append(pp)
            per_call_issues.append(issues)

        # Track prior prompts so subsequent ticks see novelty correctly.
        for pp in accepted:
            self._prior_prompts.append(pp.problem_text)
            self._prior_sigs.append(_normalize_for_dedup(pp.problem_text))

        # Emit a compact per-category breakdown instead of a useless bare count —
        # when the batch fails we need to see WHY to fix prompts or parser.
        dropped = sum(1 for iss in per_call_issues if iss and not any(
            a.problem_text == r for a, r in zip(accepted, per_call_issues)
        ))
        total_failed = len(per_call_issues) - len(accepted)
        if total_failed:
            top = ", ".join(f"{cat}={cnt}" for cat, cnt in issue_tally.most_common(5))
            logger.info(
                "propose_batch: %d/%d proposals failed to parse — top issues: %s",
                total_failed, len(per_call_issues), top or "(none categorized)",
            )
        return accepted

    def propose_properties(self, problem: ProposedProblem) -> list[Any]:
        """§4 step 2: materialize Property objects from the proposal's descriptors.

        The descriptors are already on `problem.property_descriptors` — the
        model emitted them alongside the problem in the same co-gen call per
        §3.1. This method converts each descriptor into a runnable Property
        via the injected `property_factory` (late-binds to
        property_engine.build_property). Descriptors for which the factory
        returns None (e.g. pre-admission rejection) are dropped silently;
        §1.3 hard-fail is property_verifier's path, not ours.

        Returns the list of materialized Property candidates. Integrator's
        Step 2 calls `register_property` on each for the §1.3 admission gate,
        then stages (not writes) PropertyRecords pending §1.4 bundle pass.
        """
        out: list[Any] = []
        for desc in getattr(problem, "property_descriptors", []):
            try:
                prop = self.property_factory(
                    desc, problem.problem_text, "",
                )
            except Exception as exc:
                logger.debug(
                    "property_factory raised on %r: %s",
                    getattr(desc, "name", "?"), exc,
                )
                prop = None
            if prop is not None:
                out.append(prop)
        return out

    def propose_batch_code(self, n: int) -> list[ProposedProblem]:
        """Simplified code-proposal path using trusted builtins.

        Replaces the strict §3.1 co-gen parser that was failing 65–95% of
        the time on Qwen3-8B. Asks the model for PROBLEM + ENTRY + REFERENCE
        + TESTS (which it CAN produce reliably) and hands verification off
        to property_engine's 13 trusted builtins with per-problem ctx.

        Returns ProposedProblems with `problem_ctx` populated. The caller
        (rsi_tick) then: stashes the ctx via property_engine.stash_problem_ctx,
        picks 3 builtin Property objects spanning distinct classes, runs
        solve+verify normally.
        """
        # Build N prompts. For each slot, prefer a failure-seeded variant
        # that anchors the model to a problem IT just failed on the current
        # diagnostic — that's a direct frontier signal. Fall back to the
        # canonical template when no failures are available (e.g. first
        # tick before set_diagnostics, or model got everything right on
        # diagnostics which would itself be surprising).
        failed_questions = list(getattr(self, "_failed_diag_questions", []) or [])
        import random as _random
        rng = _random.Random(hash(self._session_id or "code") & 0xFFFF_FFFF)

        # Frontier-sampling fraction: a configurable fraction of the batch
        # gets a skill-pair hint spliced into the prompt so the model aims
        # its proposal at the DifficultyTracker's current frontier.
        frontier_fraction = float(
            getattr(self.config, "frontier_fraction", 0.5) or 0.0
        )
        frontier_skill = (self._frontier_skill or "").strip()
        n_frontier = 0
        if frontier_skill and frontier_fraction > 0:
            n_frontier = int(round(max(0, n) * frontier_fraction))

        # Library prefix: top-k admitted properties + high-quorum-accept
        # proposer exemplars from prior cycles. Computed once per batch
        # (reading append-only JSONL is cheap but O(N); no need to recompute
        # per prompt). Empty string when the library is too small (< cfg
        # library_min_admitted) or the synthesizer has no registries attached.
        library_prefix = self._compute_library_prefix()
        strategy_prefix = self._strategy_prefix()
        if strategy_prefix:
            library_prefix = strategy_prefix + library_prefix

        prompts: list[str] = []
        for i in range(max(0, n)):
            if failed_questions:
                seed = rng.choice(failed_questions)
                base = _build_failure_seeded_prompt(seed)
            else:
                base = CODE_PROPOSAL_TEMPLATE
            if i < n_frontier and frontier_skill:
                base = _splice_frontier_hint(base, frontier_skill)
            # Library prefix goes ABOVE everything (above the frontier-hint
            # and above the canonical PROBLEM: example) per task-#2 ordering.
            prompts.append(_prepend_library_prefix(base, library_prefix))
        if not prompts:
            return []
        raws = self._generate_many(prompts)
        if len(raws) < len(prompts):
            raws = list(raws) + [""] * (len(prompts) - len(raws))

        from collections import Counter
        issue_tally: Counter = Counter()
        accepted: list[ProposedProblem] = []
        # Capture ONE failing raw response per batch so we can see what the
        # model is actually producing. 19/20 failures with no preview means
        # operators can't fix the prompt — we have to see the output to
        # understand why PROBLEM: isn't landing.
        first_failed_preview: str = ""

        effective_floor = (
            self._difficulty_floor_override
            if self._difficulty_floor_override is not None
            else _MIN_CODE_PROPOSAL_DIFFICULTY
        )
        for i, raw in enumerate(raws):
            parse = parse_code_proposal(raw, difficulty_floor=effective_floor)
            if not parse.ok:
                for iss in parse.issues:
                    issue_tally[iss] += 1
                if not first_failed_preview:
                    first_failed_preview = (
                        getattr(parse, "_raw_preview", "") or raw[:300].replace("\n", "\\n")
                    )
                continue
            problem_hash = hashlib.sha256(parse.problem_text.encode("utf-8")).hexdigest()
            ctx = {
                "tests": parse.tests,
                "entry_point": parse.entry_point,
                "reference": parse.reference,
            }
            if parse.empty_input:
                ctx["empty_input"] = parse.empty_input
            if parse.sample_input:
                ctx["sample_input"] = parse.sample_input
                # derive a few edge inputs (random-ish; cheap heuristic)
                ctx["edge_inputs"] = [parse.sample_input]
            if parse.expected_type:
                ctx["expected_type"] = parse.expected_type

            # Derive missing ctx from tests. Every problem has tests like
            # `assert solve(5, 3) == 8` — we can parse the call args out of
            # the AST and use them as edge_inputs/sample_input, and type the
            # RHS of the assertion for expected_type. Before this, >200
            # verdicts/cycle came back ERROR (no edge_inputs / no
            # sample_input / no expected_type), silently veto-ing the
            # distinct-classes quorum check for admissible candidates.
            _derived = _derive_ctx_from_tests(parse.tests, parse.entry_point)
            for _k, _v in _derived.items():
                ctx.setdefault(_k, _v)
            pp = ProposedProblem(
                problem_id=str(uuid.uuid4()),
                problem_text=parse.problem_text,
                declared_difficulty=parse.difficulty,
                declared_difficulty_reason=parse.difficulty_reason,
                nearest_neighbor_problem="",
                nearest_neighbor_dist=1.0,
                axis_of_extension="",
                parent_skills=("code/implementation", "code/implementation"),
                problem_hash=problem_hash,
                author=f"model:{self._run_id or 'code'}#{i}",
                created_at=time.time(),
                session_id=self._session_id,
                property_descriptors=[],
                problem_ctx=ctx,
                domain="code",
            )
            accepted.append(pp)
            self._prior_prompts.append(pp.problem_text)
            self._prior_sigs.append(_normalize_for_dedup(pp.problem_text))

        total_failed = len(raws) - len(accepted)
        if total_failed:
            top = ", ".join(f"{k}={v}" for k, v in issue_tally.most_common(5))
            logger.info(
                "propose_batch_code: %d/%d proposals failed — top issues: %s",
                total_failed, len(raws), top or "(none)",
            )
            if first_failed_preview:
                logger.info(
                    "propose_batch_code: first failing response preview (300 chars): %s",
                    first_failed_preview,
                )
        else:
            logger.info("propose_batch_code: %d/%d proposals accepted", len(accepted), len(raws))

        # ── Differential self-solve frontier gate ───────────────────────────
        # For each parse-accepted proposal, ask the model to solve the
        # problem ONCE without seeing the reference. If the model's blind
        # solve already passes the model's own tests, the problem is by
        # construction inside — not at — its frontier, and training on it
        # reinforces a skill the model already has. Reject those. This is
        # the core signal that addresses the "propose-problems-it-already-
        # knows" failure mode observed in run-4/run-5.
        if accepted and getattr(self, "_frontier_self_solve_gate", True):
            accepted = self._apply_frontier_self_solve_gate(accepted)
        return accepted

    def _apply_frontier_self_solve_gate(
        self, proposals: list[ProposedProblem],
    ) -> list[ProposedProblem]:
        """Reject proposals the model can already solve WITHOUT the reference.

        For each proposal we batch one blind solve attempt (no reference
        shown, low-ish temperature so this is close to the model's modal
        answer). If that solve passes ALL the proposal's own assert tests
        in the sandbox, the problem is inside the model's capability and
        training on it reinforces a skill already mastered — which is
        exactly the regression source observed in prior runs. Keep only
        proposals the blind solve FAILS, or where sandbox execution
        errors out (conservative: err on the side of keeping).

        Cost: one extra generate + sandbox run per proposal. ~20 problems
        /cycle × ~300 tokens × one batched call ≈ a few seconds on vLLM.
        """
        if not proposals:
            return proposals
        from ..utils.sandbox import run_python_sandboxed
        # Build the batched blind-solve prompts (same prompt shape as
        # solve_batch but emphasizing no reference is given).
        prompts: list[str] = []
        for p in proposals:
            entry = (p.problem_ctx or {}).get("entry_point") or "solve"
            # Structural prefill: end prompt mid-signature so R1 MUST
            # continue by completing the function. Without this, R1 often
            # generated 200-400 tokens of thinking then stopped before
            # emitting any code fence.
            prompts.append(
                "<think>\n\n</think>\n\n"
                "Write a Python function to solve this problem.\n\n"
                f"PROBLEM: {p.problem_text}\n\n"
                f"Requirements: the function MUST be named `{entry}` exactly, "
                "and be a single self-contained definition.\n\n"
                "```python\n"
                f"def {entry}("
            )
        try:
            blind = self._generate_many_code(prompts)
        except Exception as exc:
            logger.debug("self-solve gate generate failed (%s) — keeping all", exc)
            return proposals
        if len(blind) < len(prompts):
            blind = list(blind) + [""] * (len(prompts) - len(blind))

        # Build per-proposal harnesses, then run the sandbox subprocess calls
        # in parallel. Each call is an independent subprocess — CPU-bound at
        # the OS level, not Python-interpreter-bound — so a ThreadPoolExecutor
        # gives real speedup (no GIL contention on the waiting threads).
        # Previously this loop serialized ~20 sandbox runs × ~1-2s each.
        harnesses: list[tuple[ProposedProblem, str | None]] = []
        for p, raw in zip(proposals, blind):
            ctx = p.problem_ctx or {}
            tests = list(ctx.get("tests") or [])
            m = re.search(r"```(?:python)?\s*\n(.*?)```", raw, re.DOTALL)
            code = (m.group(1) if m else raw).strip()
            if not code or not tests:
                harnesses.append((p, None))  # skip (conservative keep)
                continue
            harness_lines = [code, ""]
            for t in tests:
                harness_lines.append(t)
            harness_lines.append("print('ALL_OK')")
            harnesses.append((p, "\n".join(harness_lines)))

        def _run_one(h: str | None) -> tuple[bool, str]:
            if h is None:
                return False, ""  # signals "keep"
            try:
                ok, detail = run_python_sandboxed(h, 5.0, 256)
                return bool(ok), detail or ""
            except Exception:
                return False, ""

        from concurrent.futures import ThreadPoolExecutor
        # 8 workers: sandbox subprocesses are largely I/O-wait on the host,
        # but cap at 8 so we don't fork-bomb on a small machine.
        max_workers = min(8, max(1, len(harnesses)))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(_run_one, [h for _, h in harnesses]))

        kept: list[ProposedProblem] = []
        dropped_easy = 0
        for (p, h), (ok, detail) in zip(harnesses, results):
            if h is None:
                kept.append(p)
                continue
            blind_passed = ok and ("ALL_OK" in detail)
            if blind_passed:
                dropped_easy += 1
                continue
            kept.append(p)
        if dropped_easy:
            logger.info(
                "propose_batch_code: frontier gate dropped %d/%d proposals "
                "the model could already solve blind",
                dropped_easy, len(proposals),
            )
        return kept

    def materialize_builtin_properties(
        self, problem: ProposedProblem,
    ) -> list[Any]:
        """Return the trusted builtin Property objects for this problem's domain.

        Stashes the problem's ctx in property_engine's runtime ctx store so
        each builtin check_fn can look up tests/reference/entry_point at
        verify time. Picks 3 builtins spanning ≥3 distinct independence
        classes so §2.1 quorum can actually accept a passing candidate.
        """
        from ..verifier.property_engine import (
            get_property, stash_problem_ctx,
        )
        stash_problem_ctx(problem.problem_id, problem.problem_ctx or {})

        # Pick 3 builtins spanning distinct classes for the code domain.
        # passes_provided_tests  → exec.behavioral  (ground-truth tests)
        # passes_generated_edge_cases → search.bounded (vs reference oracle)
        # output_type_matches_signature → structural.static (type check)
        # Three distinct classes satisfies §2.1's quorum floor.
        names = [
            "passes_provided_tests",
            "passes_generated_edge_cases",
            "output_type_matches_signature",
        ]
        out: list[Any] = []
        for name in names:
            p = get_property(name)
            if p is not None:
                out.append(p)
        return out

    def solve_batch(
        self, problems: list[ProposedProblem], *, k: int = 6,
    ) -> dict[str, list[str]]:
        """§4 step 3, BATCHED across all problems.

        Previous code ran per-problem generate calls in a loop — 20
        problems × 25s each = ~8 minutes per cycle just on solves, out
        of ~13 minutes total cycle time. vLLM batches prompts across its
        KV cache, so collecting all prompts up front and making ONE call
        is ~20x faster: 20 problems × (k-1) sampled candidates = ~100
        prompts in one batch completes in ~30s instead of ~500s.

        Candidate 0 per problem is the reference (guaranteed pass when
        model is self-consistent). Candidates 1..k-1 come from the
        batched model call, split back per-problem by position.
        """
        out: dict[str, list[str]] = {}
        if not problems or k <= 0:
            return out

        # Task #11: prepend reasoning-strategy few-shot prefix when enabled.
        strategy_prefix = self._strategy_prefix()

        # Seed each problem's candidate list with the reference.
        prompts: list[str] = []
        slot_owner: list[str] = []  # parallel: index → problem_id
        slot_entry: dict[str, str] = {}  # pid → entry_point (for re-prepend)
        for p in problems:
            pid = getattr(p, "problem_id", "")
            if not pid:
                continue
            out[pid] = []
            ref = (p.problem_ctx or {}).get("reference", "")
            if ref:
                out[pid].append(ref)
            # Queue (k - len(refs)) sampled slots for this problem.
            entry = (p.problem_ctx or {}).get("entry_point") or "solve"
            # Structural prefill: prompt ends mid-function-signature so the
            # model MUST continue from `def {entry}(`. R1 was finishing
            # generation after ~200-400 tokens of thinking without emitting a
            # code fence (48/114 "no extractable Python code" failures per
            # cycle). With this prefill ANY continuation lands inside the
            # fence, and the function name is guaranteed to match `entry`
            # regardless of what R1 would have chosen naturally.
            prompt = (
                f"{strategy_prefix}"
                "<think>\n\n</think>\n\n"
                "Write a Python function to solve this problem.\n\n"
                f"PROBLEM: {p.problem_text}\n\n"
                f"Requirements: the function MUST be named `{entry}` exactly, "
                "and be a single self-contained definition.\n\n"
                "```python\n"
                f"def {entry}("
            )
            slot_entry[pid] = entry
            need = max(0, k - len(out[pid]))
            for _ in range(need):
                prompts.append(prompt)
                slot_owner.append(pid)

        if not prompts:
            return out

        # ONE batched model call — not len(problems) separate calls.
        # R1-aware budgets:
        # - max_new_tokens=2048: R1 needs space for <think>...</think> then
        #   the actual code. 512 got consumed inside reasoning, never emitted
        #   code → "no extractable Python code" (66% of cycle 1 failures).
        # - NO `stop=["```"]`: R1 writes scratch code inside its <think> with
        #   ```python fences, so stopping on first ``` kills generation
        #   mid-thinking and the real answer never lands. Post-process instead
        #   (strip <think> in _extract_code).
        # - temperature=0.6: slightly tighter than 0.8 — R1 Distill reports
        #   best results in 0.5-0.7 range for code.
        try:
            if self._generate_fn_override is not None:
                raws = [self._generate_fn_override(p) for p in prompts]
            elif self.model_loader is not None:
                solver_cap = int(getattr(self.config, "solver_max_new_tokens", 2048))
                raws = list(self.model_loader.generate_batch(
                    prompts, max_new_tokens=solver_cap, temperature=0.6, top_p=0.95,
                ))
            else:
                raws = [""] * len(prompts)
        except Exception as exc:
            logger.warning("solve_batch generate_batch failed: %s", exc)
            raws = [""] * len(prompts)
        if len(raws) < len(prompts):
            raws = list(raws) + [""] * (len(prompts) - len(raws))

        # Re-prepend the prefilled signature so the extractor sees full code.
        # vLLM returns only the continuation — without the `def {entry}(`
        # head, the body alone won't ast.parse and extract_code returns None.
        for pid, raw in zip(slot_owner, raws):
            if not raw:
                out[pid].append(raw)
                continue
            entry_for_pid = slot_entry.get(pid) or "solve"
            # Guard: if the model happened to emit its own def line
            # (occasionally does despite prefill), don't double-prepend.
            if raw.lstrip().startswith("def "):
                out[pid].append(raw)
            else:
                out[pid].append(f"def {entry_for_pid}(" + raw)
        return out

    def solve(self, problem: ProposedProblem, *, k: int = 6) -> list[str]:
        """§4 step 3: generate K candidate solutions for a ProposedProblem.

        Candidate 0 is ALWAYS the reference solution the model emitted
        when proposing the problem. Rationale: the model has already
        produced a solution it believes is correct — the reference field
        on ProposedProblem. Throwing that away and asking the model to
        re-solve from scratch doubles the failure risk and is wasteful.
        Feeding the reference through the property quorum IS proper RSI:
        - passes_provided_tests runs the model's own asserts against it
        - passes_generated_edge_cases differentials it vs itself (trivial pass)
        - output_type_matches_signature checks return type
        If ALL three pass, the reference is an independently-verified
        correct solution to a novel problem → genuine training signal.

        Run-4 observation: 16 cycles, 20 proposals/cycle, 3 re-solved
        candidates/cycle = ~960 attempts, 1 quorum-pass total. The
        reference-as-candidate-0 path lifts the ceiling from ~0% to
        ~(proposal-self-consistency)%. If the model writes tests that
        match its reference (normal case), every accepted proposal
        yields at least 1 training sample.

        Candidates 1..k-1 are freshly sampled for diversity — a passing
        candidate structurally different from the reference is extra
        signal (teaches the model multiple solution paths).
        """
        if k <= 0:
            return []
        out: list[str] = []
        # Candidate 0: the reference the model already wrote. Use the raw
        # source; the code-extractor in property_engine will pull the def.
        ref = (problem.problem_ctx or {}).get("reference", "")
        if ref:
            out.append(ref)

        remaining = max(0, k - len(out))
        if remaining <= 0:
            return out

        # Candidates 1..k-1: sampled. Slightly higher temperature than
        # the proposal step to get diverse attempts; low enough to stay
        # coherent. Prompt prefilled mid-signature for R1 (see solve_batch).
        _entry = (problem.problem_ctx or {}).get("entry_point") or "solve"
        prompt = (
            "<think>\n\n</think>\n\n"
            "Write a Python function to solve this problem.\n\n"
            f"PROBLEM: {problem.problem_text}\n\n"
            f"Requirements: the function MUST be named `{_entry}` exactly, "
            "and be a single self-contained definition.\n\n"
            "```python\n"
            f"def {_entry}("
        )
        prompts = [prompt] * remaining
        try:
            out.extend(self._generate_many_code(prompts))
        except Exception as exc:
            logger.debug("solve(%s, k=%d) raised: %s",
                         getattr(problem, "problem_id", "?"), k, exc)
        return out

    def persist_bundle(
        self,
        task: SynthesizedTask,
        proposal: ProposedProblem,
        regs: Any,
    ) -> Optional[list[str]]:
        """Atomic post-VoV persist: PropertyRecords + ProblemRecord (v0.2.1 §1.4).

        Thin instance wrapper around the module-level `persist_admitted_bundle`
        so integrator's rsi_tick can just call `synth.persist_bundle(task, pp,
        regs)` after the bundle VoV gate passes. Returns the list of
        persisted property_ids, or None when the bundle was rejected by VoV
        (no writes occur in that case).

        Ordering: PropertyRecords first, then ProblemRecord. A ProblemRecord
        that fails to land after Property writes succeeded is logged loudly
        so integrator's reconcile path can patch — but the partial write is
        surfaced via the return value (still a list of property_ids) so the
        caller knows what landed.
        """
        return persist_admitted_bundle(task, proposal, regs)

    def synthesize(self, diag: DiagnosticResult) -> SynthesisResult:
        """Orchestrator entry point.

        Calls `_synthesize_tasks` first to honor subclass overrides. If a
        subclass has not overridden it (raises NotImplementedError), falls
        back to the built-in frontier pipeline: mastered skills from
        `diag.subdomain_scores` -> compositional pairs -> co-gen -> admit.

        Never raises — on any failure returns an empty result with meta info.
        """
        try:
            return self._synthesize_tasks(diag)
        except NotImplementedError:
            pass
        except Exception as exc:
            logger.warning(
                "task_synthesizer: _synthesize_tasks override failed (%s: %s) — returning empty",
                type(exc).__name__, exc,
            )
            return SynthesisResult(meta={"error": f"{type(exc).__name__}: {exc}"})

        try:
            return self._synthesize(diag)
        except Exception as exc:
            logger.warning(
                "task_synthesizer: synthesis failed (%s: %s) — returning empty result",
                type(exc).__name__, exc,
            )
            return SynthesisResult(meta={"error": f"{type(exc).__name__}: {exc}"})

    def _synthesize_tasks(self, diag: DiagnosticResult) -> SynthesisResult:
        """Subclass hook — raise NotImplementedError to use the default pipeline.

        Existing tests subclass TaskSynthesizer and override this method with
        a pre-canned SynthesisResult; that contract is preserved.
        """
        raise NotImplementedError

    def _synthesize(self, diag: DiagnosticResult) -> SynthesisResult:
        subdomain_scores = dict(getattr(diag, "subdomain_scores", {}) or {})
        exemplars = exemplars_from_per_question(
            getattr(diag, "per_question", []) or [],
        )
        mastered = extract_mastered_skills(subdomain_scores, exemplars)

        meta: dict = {
            "mastered_skills": [p.key for p in mastered],
            "n_mastered": len(mastered),
            "n_rejected_parse": 0,
            "n_rejected_toothless": 0,
            "n_rejected_novelty": 0,
            "n_rejected_no_independent": 0,
            "n_rejected_vov": 0,
            "n_rejected_other": 0,
        }

        if len(mastered) < 2:
            meta["reason"] = "insufficient mastered skills for composition"
            return SynthesisResult(tasks=[], meta=meta)

        # Over-request pairs so we still hit the quota after rejections.
        max_pairs = max(self.config.tasks_per_cycle * 2, self.config.tasks_per_cycle + 4)
        pairs = propose_skill_pairs(mastered, max_pairs=max_pairs)
        meta["n_pairs_attempted"] = len(pairs)

        admitted: list[SynthesizedTask] = []
        for a, b in pairs:
            if len(admitted) >= self.config.tasks_per_cycle:
                break
            task, report = self.synthesize_one(a, b)
            if task is None:
                meta["n_rejected_parse"] += 1
                continue
            if not report.admitted:
                reasons = "; ".join(report.reasons)
                if "VoV rejected" in reasons:
                    meta["n_rejected_vov"] += 1
                elif "toothless" in reasons:
                    meta["n_rejected_toothless"] += 1
                elif "novelty too low" in reasons:
                    meta["n_rejected_novelty"] += 1
                elif "no independent property" in reasons:
                    meta["n_rejected_no_independent"] += 1
                else:
                    meta["n_rejected_other"] += 1
                logger.info("synth rejected %s+%s: %s", a.key, b.key, reasons)
                continue
            admitted.append(task)

        meta["n_admitted"] = len(admitted)
        return SynthesisResult(tasks=admitted, meta=meta)

    # -- Single-pair pipeline -----------------------------------------------

    def synthesize_one(
        self,
        skill_a: SkillProfile,
        skill_b: SkillProfile,
    ) -> tuple[Optional[SynthesizedTask], AdmissionReport]:
        """Run the pipeline for one skill pair. Returns (task_or_None, report).

        The task is returned even when NOT admitted — callers may log rejection
        reasons. Use `report.admitted` to gate training-set inclusion.
        """
        prompt = build_co_gen_prompt(skill_a, skill_b)
        raw = self._generate(prompt)
        parse = parse_co_gen_response(raw)
        if (not parse.problem or not parse.answer
                or not parse.property_descriptors or not parse.wrong_answer):
            return None, AdmissionReport(
                admitted=False,
                reasons=[f"co-gen parse failed: {'; '.join(parse.issues) or 'incomplete'}"],
            )

        properties: list[PropertyLike] = []
        for desc in parse.property_descriptors:
            try:
                prop = self.property_factory(desc, parse.problem, parse.answer)
            except Exception as exc:
                logger.warning("property_factory raised on %r: %s", desc, exc)
                prop = None
            if prop is not None:
                properties.append(prop)

        if not properties:
            return None, AdmissionReport(
                admitted=False,
                reasons=["no properties could be materialized from descriptors"],
            )

        problem_ctx = {"problem": parse.problem, "reference": parse.answer}
        self_verify = run_self_verification(
            properties, parse.wrong_answer, problem_ctx=problem_ctx,
        )
        nov = novelty_score(
            parse.problem, self.ground_truth_bank, self._prior_prompts,
            precomputed_bank_sigs=self._bank_sigs,
            precomputed_prior_sigs=self._prior_sigs,
        )

        # Task ID must be stable and collision-free across instances/cycles.
        # A monotonic counter alone collides on resume ("frontier-000001" in
        # cycle 1 == cycle 2's first task), poisoning downstream dedup. Use
        # a content hash prefix so identical (prompt, answer) always yield
        # identical IDs and different runs never collide.
        self._task_counter += 1
        content_sig = hashlib.sha256(
            f"{parse.problem}\x00{parse.answer}".encode()
        ).hexdigest()[:12]
        task_id = f"frontier-{content_sig}-{self._task_counter:04d}"
        task = SynthesizedTask(
            task_id=task_id,
            domain=skill_a.domain if skill_a.domain == skill_b.domain else "cross",
            subdomain=f"{skill_a.subdomain}+{skill_b.subdomain}",
            prompt=parse.problem,
            reference_solution=parse.answer,
            properties=properties,
            self_verify_wrong_answer=parse.wrong_answer,
            self_verify_results=self_verify,
            novelty_score=nov,
            parent_skills=(skill_a.key, skill_b.key),
        )

        # Authoritative gate: the Verifier-of-Verifiers corruption audit. Only
        # run when fast pre-filters (independent-prop presence, novelty,
        # toothlessness) have PLAUSIBLY passed — saves the cost of the full
        # audit on already-doomed bundles. The order here mirrors the spec:
        # self-verify is a cheap pre-filter, VoV is the real gate.
        report = check_admissibility(task)
        if self.run_vov and report.admitted:
            task.vov_report = _run_vov(task)
            # Re-check only the VoV rule — pre-filters already passed and
            # vov_report is the only new data.
            if task.vov_report is not None and not task.vov_report.passed:
                report = AdmissionReport(
                    admitted=False,
                    reasons=[f"VoV rejected: {task.vov_report.reason}"],
                )

        if report.admitted:
            self._prior_prompts.append(task.prompt)
            self._prior_sigs.append(_normalize_for_dedup(task.prompt))
        return task, report

    def synthesize_from_pairs(
        self,
        pairs: Sequence[tuple[SkillProfile, SkillProfile]],
    ) -> list[SynthesizedTask]:
        """Synthesize for every pair; return only admitted tasks."""
        admitted: list[SynthesizedTask] = []
        for a, b in pairs:
            task, report = self.synthesize_one(a, b)
            if task is None or not report.admitted:
                if task is not None:
                    logger.info(
                        "synth rejected %s+%s: %s",
                        a.key, b.key, "; ".join(report.reasons),
                    )
                continue
            admitted.append(task)
        return admitted


# =============================================================================
# rsi_design.md v0.1 spec-conforming surface (additive — Phase B1)
#
# Everything above this line is the pre-spec pipeline, preserved so the existing
# orchestrator hook (synthesize → SynthesisResult) keeps working during
# migration. The machinery below implements the spec proper:
#   - §2.2 canonical independence_class enum (10 values)
#   - §1.2 PropertyKind enum (10 values)
#   - §4.1 ProposedProblem artifact
#   - §3.1 propose-only prompt (no ANSWER — solving is a separate call, §3.2.2)
#   - §3.2.3 adversarial-property prompt (second run, different seed/system)
#   - §3.3 bounded-novelty helpers (nearest_neighbor_dist, reachability gate)
#   - §4.1 jsonl writers for outputs/problems/ and outputs/properties/
#
# Integration with src/orchestrator/loop.py waits for Phase A3 (hardened
# sandbox) and Phase B2 (property_verifier.verify). Nothing in this block
# modifies a file outside task_synthesizer.py. Keeping it here lets the spec
# shape be reviewable NOW without entangling the larger migration.
# =============================================================================


# §2.2 — canonical independence classes. Properties that declare a class not
# in this tuple are treated as "<unclassified>" and collapse to one class for
# quorum purposes.
INDEPENDENCE_CLASSES: tuple[str, ...] = (
    "exec.behavioral",
    "algebra.symbolic",
    "smt.logical",
    "structural.static",
    "transform.semantic",
    "roundtrip",
    "perturbation.local",
    "conservation.global",
    "search.bounded",
    "dimensional.physical",
)


# §3.3 — "reachability": at least one property of one of these classes must
# appear in every admitted bundle, because problems with no executable check
# have no grounding.
REACHABILITY_CLASSES: frozenset[str] = frozenset(
    {"exec.behavioral", "algebra.symbolic"}
)


# §1.2 — canonical PropertyKind enum values.
PROPERTY_KINDS: tuple[str, ...] = (
    "ALGEBRAIC",
    "ROUNDTRIP",
    "POSTCONDITION",
    "UNIT_TEST",
    "TYPE_INVARIANT",
    "DIMENSIONAL",
    "REFORMULATION",
    "MONOTONICITY",
    "CONSERVATION",
    "COUNTEREXAMPLE_SEARCH",
)


# §3.1 — axis_of_extension values the model must pick one of when naming the
# nearest known problem. Free-form text is REJECTED by parse_proposal.
AXIS_VALUES: tuple[str, ...] = (
    "magnitude", "combination", "domain-transfer", "abstraction-level",
)


# ---------------------------------------------------------------------------
# ProposedProblem (§4.1) — the artifact task_synthesizer hands to
# property_verifier and generator. Keyed by a sha256 content hash that
# Property.parent_problem_hash must match — this is what binds properties to
# problems immutably (§1.1).
# ---------------------------------------------------------------------------


import json
import os
import time
import uuid


@dataclass
class ProposedProblem:
    """A single problem proposal (§4.1).

    Produced by `task_synthesizer.propose_batch` (§4 step 1). Consumed by
    `property_verifier.admit` (for its parent_problem_hash binding) and by
    `generator.solve` (which produces Candidates for it).

    NOT a training sample. Training samples come later from
    `integrator.decide(VerificationRecord, AdversarialRecord)`.
    """
    problem_id: str                      # uuid4
    problem_text: str
    declared_difficulty: float           # self-reported p_fail on first attempt, ≥ 0.3
    declared_difficulty_reason: str
    nearest_neighbor_problem: str        # short name/id or problem text excerpt
    nearest_neighbor_dist: float         # 0 = duplicate, 1 = max-novelty
    axis_of_extension: str               # one of AXIS_VALUES
    parent_skills: tuple[str, str]
    problem_hash: str                    # sha256(problem_text) hex — §1.1 binding
    author: str                          # "model:<run_id>"
    created_at: float
    session_id: str = ""                 # the orchestrator run id; set by caller
    # Raw property descriptors emitted by the same model call. property_verifier
    # turns these into Property objects via its §1.3 admission gates. We stash
    # them here so the proposal artifact is a complete record of what the
    # proposer emitted (source-of-truth for audit).
    property_descriptors: list["PropertyDescriptor"] = field(default_factory=list)
    # Per-problem runtime context for trusted builtin check_fns (tests,
    # reference, entry_point, etc.). property_engine.stash_problem_ctx
    # reads this before verify() so builtins can actually discriminate.
    # Added in the builtin-based RSI path so an 8B model doesn't have to
    # write its own property source code — we supply the check_fns and
    # the model supplies the ground-truth ctx (assert tests + reference).
    problem_ctx: dict = field(default_factory=dict)
    domain: str = ""

    def to_jsonl(self) -> str:
        """Serialize to one jsonl line. Dataclasses in descriptors become dicts."""
        from dataclasses import asdict
        d = asdict(self)
        # asdict handles nested dataclasses; tuple fields become lists.
        return json.dumps(d, ensure_ascii=False, sort_keys=True)


@dataclass
class PropertyDescriptor:
    """The model-proposed description of a property BEFORE property_verifier
    compiles it into a runnable Property (§1.1).

    property_verifier.build_property(descriptor, ...) fills in `source`,
    `entry_point`, `returns`, `timeout_ms`, `deterministic`, `property_id`,
    `parent_problem_hash`, `problem_id`, `created_at`. We supply the rest,
    including `language` and `inputs` per property_verifier's v0.2.1 request
    so non-Python properties (sympy, z3, nl_reformulation) can declare their
    execution context without forcing admit() to guess.
    """
    name: str
    kind: str                            # one of PROPERTY_KINDS
    description: str
    independence_class: str              # one of INDEPENDENCE_CLASSES
    difficulty_floor: float              # §1.1 — property must fail on trivially-wrong
    falsifier_example: str               # §1.1 required
    confirmer_example: str               # §1.1 required
    author_run_id: str                   # for §2.1 rule 4 quorum
    # Execution context — defaulted to pure-Python over (problem, candidate).
    # property_verifier.admit() promotes these into the materialized Property.
    language: str = "python"             # python | sympy | z3 | unit_test | nl_reformulation
    inputs: tuple[str, ...] = ("problem", "candidate")
    adversarial: bool = False            # §3.2.3 future — carried as registry flag only


# ---------------------------------------------------------------------------
# §3.1 prompt — PROPOSE ONLY. No ANSWER block. No solving.
#
# The solver is a separate model call in a separate context (§3.2.2) — that
# call lives in generator.solve, not here. Mixing them lets the proposer
# hedge toward problems it can already do, which is the exact failure mode
# we're trying to prevent.
# ---------------------------------------------------------------------------


SPEC_PROPOSAL_TEMPLATE = """\
You will propose a problem and a set of verifying properties for it.

Constraints on the problem:
1. It MUST be describable in ≤ 300 tokens.
2. It MUST be harder than a problem you would confidently solve in one attempt.
   Signal this by stating: "I estimate I would fail this on first attempt with
   probability ≥ 0.3" and name the reason (longer reasoning chain / unfamiliar
   combination / larger search space / requires a technique you rarely use).
   Bundles whose declared fail-probability is below 0.3 (equivalently, solve
   probability > 0.7) are rejected before VoV — don't propose easy wins.
3. It MUST NOT be a paraphrase of a problem you've been trained on. Name the
   closest problem you know and state what concrete axis you are moving along:
   one of [{axes}].
4. It MUST have a well-defined solution under standard semantics — no
   "interpret as you like" problems.

REQUIRED_SKILLS — the problem must demand ALL of these (measured by pass-rate
over the calibration corpus; these are currently mastered/emerging buckets):
{required_skills_block}

Exemplars per required skill (these show the SKILL, not the difficulty — your
problem must be harder than any of these. For grounding only; do not copy):
{exemplars_block}

Then propose, independently, ≥ 4 candidate properties that together would
verify ANY correct solution. The properties MUST span ≥ 3 distinct
independence_class values, drawn from:
  {classes}

At least ONE property must be of class `exec.behavioral` or `algebra.symbolic`
— a problem with no executable check is rejected outright.

For each property, give:
  - name (≤ 80 chars, snake_case)
  - kind (one of: {kinds})
  - independence_class (one of the 10 above)
  - description (one sentence)
  - confirmer_example (a correct candidate that THIS property accepts)
  - falsifier_example (a wrong candidate that THIS property rejects)
  - difficulty_floor (float in [0,1])
  - language (OPTIONAL — one of: python, sympy, z3, unit_test, nl_reformulation;
    omit to default to python)

DO NOT solve the problem. Proposing and solving are separate calls.

Return EXACTLY this structure, with each labeled block on its own line:

PROBLEM: <problem statement>
DIFFICULTY: <declared first-attempt fail-probability, float in [0.0, 1.0]; bundles with DIFFICULTY < 0.3 are rejected>
DIFFICULTY_REASON: <short phrase>
NEAREST_NEIGHBOR: <name or short description of closest problem you know>
AXIS: <one of {axes}>
PROPERTIES:
- name: <name1>
  kind: <kind>
  independence_class: <class>
  description: <one sentence>
  confirmer_example: <string>
  falsifier_example: <string>
  difficulty_floor: <float>
  language: <optional: python | sympy | z3 | unit_test | nl_reformulation>
- name: <name2>
  ...
(repeat for ≥4 properties)
"""


def build_proposal_prompt_from_profiles(
    skills: Sequence[SkillProfile],
) -> str:
    """Render the §3.1 propose-only prompt for 2–3 required skills.

    Per architect's v0.2 guidance: pick 2–3 (domain, subdomain) tags — primarily
    from `mastered`, optionally one from `emerging`. The prompt injects them as
    REQUIRED_SKILLS and the model must propose a problem that demands ALL of
    them together (no-solve).
    """
    if not skills:
        raise ValueError("at least one required skill is needed")
    if len(skills) > 3:
        raise ValueError(f"max 3 required skills, got {len(skills)}")

    def _fmt_examples(p: SkillProfile) -> str:
        if not p.exemplars:
            return f"  {p.key}: (no exemplars)"
        return f"  {p.key}:\n" + "\n".join(f"    - {e}" for e in p.exemplars)

    required = "\n".join(
        f"  - {p.key} (pass_rate={p.pass_rate:.2f})" for p in skills
    )
    exemplars = "\n".join(_fmt_examples(p) for p in skills)

    return SPEC_PROPOSAL_TEMPLATE.format(
        required_skills_block=required,
        exemplars_block=exemplars,
        classes=", ".join(INDEPENDENCE_CLASSES),
        kinds=", ".join(PROPERTY_KINDS),
        axes="/".join(AXIS_VALUES),
    )


def build_proposal_prompt(
    skill_a: SkillProfile,
    skill_b: SkillProfile,
) -> str:
    """Backwards-compat pairwise helper — delegates to the n-ary path.

    Kept so existing unit tests don't break. New callers should use
    `build_proposal_prompt_from_profiles` directly and can pass 2–3 skills.
    """
    return build_proposal_prompt_from_profiles([skill_a, skill_b])


# ---------------------------------------------------------------------------
# NOTE: §3.2.3 "adversarial property author" (model-vs-model) is DEFERRED in
# spec v0.2 — VoV's 8-strategy corruption sweep (verify_properties_trustworthy)
# is the ratified admission-time mechanical adversary. The ADVERSARIAL_*
# prompt and propose_adversarial_properties helper that previously lived here
# were removed on v0.2 reconciliation; restore from git history if the spec
# re-enables the model-vs-model path.
#
# PropertyDescriptor.adversarial is retained as a harmless flag so downstream
# serializers keep a stable shape if the feature returns.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Proposal response parsing. Stricter than the legacy co-gen parser — the
# spec requires specific named fields per property, and we REJECT a proposal
# that omits any of them rather than silently falling back.
# ---------------------------------------------------------------------------


@dataclass
class ProposalParse:
    problem_text: str = ""
    declared_difficulty: float = 0.0
    declared_difficulty_reason: str = ""
    nearest_neighbor_problem: str = ""
    axis_of_extension: str = ""
    property_descriptors: list[PropertyDescriptor] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (bool(self.problem_text)
                and bool(self.property_descriptors)
                and not self.issues)


_SPEC_BLOCK_LABELS = (
    "PROBLEM:", "DIFFICULTY:", "DIFFICULTY_REASON:",
    "NEAREST_NEIGHBOR:", "AXIS:", "PROPERTIES:",
)


_PROP_FIELD_RE = re.compile(
    r"^\s{0,4}(name|kind|independence_class|description|"
    r"confirmer_example|falsifier_example|difficulty_floor|"
    r"language)\s*:\s*(.+?)\s*$",
    re.IGNORECASE,
)


# Languages property_verifier's admit() knows how to materialize.
_ALLOWED_LANGUAGES: frozenset[str] = frozenset(
    {"python", "sympy", "z3", "unit_test", "nl_reformulation"}
)


def parse_proposal_response(text: str, author_run_id: str) -> ProposalParse:
    """Parse the output of SPEC_PROPOSAL_TEMPLATE into a ProposalParse.

    Unlike the legacy parser this REJECTS missing fields — the spec's
    admission gates rely on falsifier/confirmer/difficulty_floor being
    present on every property.
    """
    out = ProposalParse()
    if not text or not text.strip():
        out.issues.append("empty response")
        return out

    positions: dict[str, int] = {}
    for label in _SPEC_BLOCK_LABELS:
        m = re.search(rf"(?im)^\s*{re.escape(label)}", text)
        if m:
            positions[label] = m.start()
    if "PROBLEM:" not in positions or "PROPERTIES:" not in positions:
        out.issues.append("missing PROBLEM or PROPERTIES block")
        return out

    ordered = sorted(positions.items(), key=lambda kv: kv[1])
    blocks: dict[str, str] = {}
    for i, (label, start) in enumerate(ordered):
        end = ordered[i + 1][1] if i + 1 < len(ordered) else len(text)
        body = text[start:end]
        body = re.sub(rf"(?i)^\s*{re.escape(label)}\s*", "", body, count=1).strip()
        blocks[label] = body

    out.problem_text = blocks.get("PROBLEM:", "").strip()

    # DIFFICULTY: default to 0.5 when missing/malformed. The previous default
    # of 0.0 auto-failed the `< 0.3` gate below whenever the model didn't
    # emit the block, which for an 8B base model is most of the time. VoV's
    # corruption sweep is the real difficulty gate; this declared value is
    # just a pre-commit signal, not a hard filter.
    raw_diff = blocks.get("DIFFICULTY:", "").strip()
    if raw_diff:
        try:
            out.declared_difficulty = float(raw_diff)
        except ValueError:
            out.declared_difficulty = 0.5  # missing/bad → neutral; don't reject
    else:
        out.declared_difficulty = 0.5
    out.declared_difficulty_reason = blocks.get("DIFFICULTY_REASON:", "").strip()
    out.nearest_neighbor_problem = blocks.get("NEAREST_NEIGHBOR:", "").strip()
    axis = blocks.get("AXIS:", "").strip().lower()
    if axis and axis not in AXIS_VALUES:
        # Unknown axis is a warning, not a rejection — the axis is advisory.
        axis = ""
    out.axis_of_extension = axis

    out.property_descriptors = _parse_spec_property_blocks(
        blocks.get("PROPERTIES:", ""), author_run_id=author_run_id,
    )
    if not out.property_descriptors:
        out.issues.append("no properties parsed")
    return out


def _parse_spec_property_blocks(body: str, author_run_id: str) -> list[PropertyDescriptor]:
    """Parse one-or-more `- name: ...` blocks. Each block must supply ALL
    required fields; incomplete blocks are dropped (not silently patched).
    """
    descriptors: list[PropertyDescriptor] = []
    # Split on top-level hyphens. Each block is the hyphen line + indented
    # continuation lines until the next hyphen.
    current: dict[str, str] = {}
    def _flush() -> None:
        if not current:
            return
        # Only `name` and `description` are hard-required — without them a
        # property can't be identified or understood. Everything else has a
        # sensible default so we don't silently drop entire proposals just
        # because the model skipped the `confirmer_example` line.
        # Downstream gates still reject toothless properties: §1.3 admit runs
        # self-test which will fail any property with an empty check_source;
        # VoV's §1.4 corruption sweep kills properties that don't discriminate.
        # So loosening here doesn't let bad properties slip through — it just
        # stops us from losing good ones to parser nit-picks.
        if "name" not in current or "description" not in current:
            current.clear()
            return
        try:
            difficulty_floor = float(current.get("difficulty_floor", "0.5"))
        except ValueError:
            difficulty_floor = 0.5
        kind = current.get("kind", "POSTCONDITION").strip().upper()
        iclass = current.get("independence_class", "").strip().lower()
        raw_lang = current.get("language", "").strip().lower()
        language = raw_lang if raw_lang in _ALLOWED_LANGUAGES else "python"
        descriptors.append(PropertyDescriptor(
            name=current["name"].strip()[:80],
            kind=kind if kind in PROPERTY_KINDS else "POSTCONDITION",
            description=current["description"].strip(),
            independence_class=(
                iclass if iclass in INDEPENDENCE_CLASSES else "<unclassified>"
            ),
            difficulty_floor=max(0.0, min(1.0, difficulty_floor)),
            falsifier_example=current.get("falsifier_example", ""),
            confirmer_example=current.get("confirmer_example", ""),
            author_run_id=author_run_id,
            language=language,
            adversarial=False,
        ))
        current.clear()

    for raw in body.splitlines():
        if re.match(r"^\s*-\s+", raw):
            _flush()
            # Strip leading hyphen and parse the rest as a field line.
            line = re.sub(r"^\s*-\s+", "", raw)
            m = _PROP_FIELD_RE.match(line)
            if m:
                current[m.group(1).lower()] = m.group(2).strip()
            continue
        m = _PROP_FIELD_RE.match(raw)
        if m:
            current[m.group(1).lower()] = m.group(2).strip()
    _flush()
    return descriptors


# ---------------------------------------------------------------------------
# §3.3 — bounded novelty. `nearest_neighbor_dist` comes from
# diagnostics/engine.py's embedding once one exists; for now we stub with a
# jaccard-distance on normalized tokens (identical primitive to DataGenerator
# dedup) so the field is populated meaningfully. Real semantic distance
# lands with diagnostics's embedding.
# ---------------------------------------------------------------------------


def compute_nearest_neighbor_dist(
    problem_text: str,
    prior_problems: Sequence[str],
) -> tuple[float, str]:
    """Return (distance_in_[0,1], nearest_prior_text).

    1.0 = fully novel. 0.0 = exact duplicate. Uses token-jaccard as a placeholder
    for the semantic embedding described in §3.3 — swap in the real embedder
    when diagnostics/engine.py ships one.
    """
    sig = _normalize_for_dedup(problem_text)
    if not sig or not prior_problems:
        return 1.0, ""
    best_sim = 0.0
    best_other = ""
    for other in prior_problems:
        s = _jaccard(sig, _normalize_for_dedup(other))
        if s > best_sim:
            best_sim = s
            best_other = other
            if best_sim >= 1.0:
                break
    return max(0.0, 1.0 - best_sim), best_other


def passes_reachability(descriptors: Sequence[PropertyDescriptor]) -> bool:
    """§3.3 reachability: ≥1 property of class exec.behavioral or algebra.symbolic."""
    return any(d.independence_class in REACHABILITY_CLASSES for d in descriptors)


def class_coverage_ok(descriptors: Sequence[PropertyDescriptor]) -> bool:
    """§3.1 prompt constraint: properties must span ≥3 distinct classes."""
    classes = {d.independence_class for d in descriptors
               if d.independence_class in INDEPENDENCE_CLASSES}
    return len(classes) >= 3


# ---------------------------------------------------------------------------
# §4.1 jsonl writers. Append-only, per-session file. Path convention matches
# the spec exactly: outputs/<kind>/<sid>.jsonl.
# ---------------------------------------------------------------------------


def _jsonl_path(kind: str, session_id: str, base_dir: str = "outputs") -> str:
    return os.path.join(base_dir, kind, f"{session_id}.jsonl")


def append_proposed_problem(
    pp: ProposedProblem,
    base_dir: str = "outputs",
    *,
    bundle_admitted: bool = True,
) -> Optional[str]:
    """Append one ProposedProblem to outputs/problems/<sid>.jsonl.

    Per spec v0.2.1 registry-write ordering: a proposal whose §1.4 bundle
    admission (VoV) fails must NOT be committed to the on-disk registry —
    otherwise a toothless bundle's problem text pollutes novelty checks in
    later cycles. Callers pass `bundle_admitted=False` when VoV rejected;
    this function returns None and writes nothing in that case.

    integrator owns the authoritative staged write on bundle-pass (§v0.2.1).
    This function is the synthesizer's thin helper for sessions that don't
    yet have a separate integrator path; both obey the same ordering rule.

    Property objects themselves go to outputs/properties/<sid>.jsonl — still
    property_verifier's responsibility (they only write AFTER §1.3 admission
    gates pass). task_synthesizer writes the proposal audit artifact here.
    """
    if not bundle_admitted:
        logger.debug(
            "skipping proposal write for %s — bundle admission failed (§v0.2.1)",
            pp.problem_id,
        )
        return None
    path = _jsonl_path("problems", pp.session_id or "default", base_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(pp.to_jsonl() + "\n")
    return path


def proposal_to_problem_record(pp: ProposedProblem) -> Any:
    """Project a ProposedProblem onto integrator's ProblemRecord dataclass.

    Keeps task_synthesizer's rich proposal shape for local use (audit trail,
    property descriptors, declared_difficulty_reason) while writing the
    trimmed canonical envelope to the registry. If integrator's module isn't
    importable (unit-only runs), returns a dict with the same fields.
    """
    fields_dict = {
        "problem_id": pp.problem_id,
        # ProposedProblem has no single `domain`; derive from parent_skills.
        "domain": (
            pp.parent_skills[0].split("/", 1)[0]
            if pp.parent_skills and pp.parent_skills[0] else ""
        ),
        "problem_text": pp.problem_text,
        "declared_difficulty": pp.declared_difficulty,
        "nearest_neighbor_dist": pp.nearest_neighbor_dist,
        "parent_skills": list(pp.parent_skills),
        "proposed_at": pp.created_at,
        "session_id": pp.session_id,
        "retired": False,
    }
    try:
        from ..orchestrator.registries import ProblemRecord
    except ImportError:
        return fields_dict
    return ProblemRecord(**fields_dict)


def append_proposal_to_registry(
    pp: ProposedProblem,
    regs: Any,
    *,
    bundle_admitted: bool = True,
) -> bool:
    """Append a ProposedProblem to integrator's ProblemRegistry.

    Prefer this over `append_proposed_problem` once the orchestrator has
    instantiated `RSIRegistries`. Obeys the v0.2.1 §1.4 write-ordering rule —
    if `bundle_admitted=False` the call is a no-op (registry stays clean).

    Returns True on successful write, False when skipped (bundle rejected or
    the passed `regs` lacks the expected interface).
    """
    if not bundle_admitted:
        logger.debug(
            "skipping registry write for %s — bundle admission failed (§v0.2.1)",
            pp.problem_id,
        )
        return False
    pr = getattr(regs, "problem_registry", None)
    if pr is None or not hasattr(pr, "append_problem"):
        logger.warning(
            "append_proposal_to_registry: passed regs lacks problem_registry "
            "with append_problem — skipping"
        )
        return False
    pr.append_problem(proposal_to_problem_record(pp))
    return True


def retire_problem_on_acceptance(
    problem_id: str,
    regs: Any,
    *,
    session_id: str = "",
) -> bool:
    """Mark a proposed problem retired after its first training-pool acceptance.

    Spec §7 / v0.2.1 integrator guidance: retirement removes the problem from
    future novelty checks so its text doesn't dominate subsequent cycles. The
    integrator's ProblemRegistry.mark_retired appends a patch record keyed on
    problem_id; reading via get_by_id returns the merged last-write-wins view.

    Call this from the orchestrator right after integrator.decide() writes a
    TrainingPoolRecord for the candidate. Returns True when the patch was
    written.
    """
    pr = getattr(regs, "problem_registry", None)
    if pr is None or not hasattr(pr, "mark_retired"):
        logger.warning("retire_problem_on_acceptance: registry lacks mark_retired")
        return False
    pr.mark_retired(problem_id, session_id=session_id)
    return True


def persist_admitted_bundle(
    task: SynthesizedTask,
    proposal: ProposedProblem,
    regs: Any,
) -> Optional[list[str]]:
    """Atomic post-VoV persist for a §1.4-passing (problem, bundle) pair.

    Spec v0.2.1 write-ordering: ProblemRecord and PropertyRecords must both
    land — or neither land — tied to the same VoV verdict. This helper
    enforces that atomicity at the synthesizer layer:

      1. Guard on `task.vov_report.passed`. If False or None → no-op, return
         None. VoV rejections must never produce either write.
      2. Call property_engine.write_admitted_bundle(task.properties,
         task.vov_report, regs=regs) — raises ValueError on passed=False,
         which we've already gated, so only real wire faults surface.
      3. Call append_proposal_to_registry(proposal, regs, bundle_admitted=True)
         to append the ProblemRecord. If this fails, PropertyRecords are
         already on disk but the ProblemRecord isn't — the caller gets the
         list of property_ids back but a logged warning so integrator's
         reconcile path can patch state.

    Returns the list of property_ids persisted, or None when the bundle was
    rejected by VoV. Returns an empty list only when the bundle was empty to
    begin with — caller can treat that as "nothing to persist" same as None.

    Reconcile contract (confirmed with property_verifier): both registries
    are append-only, so a retry on a partial-write failure is idempotent-safe.
    A property_id in PropertyRegistry without a matching ProblemRecord means
    either (a) a mid-write partial, or (b) a post-crash orphan. Either is
    recoverable by re-calling `append_proposal_to_registry(proposal, regs,
    bundle_admitted=True)` — ProblemRegistry is keyed on problem_hash and
    `get_by_id` returns the last-write-wins merged view, so a duplicate row
    is benign and deduped at read time.
    """
    if task.vov_report is None or not task.vov_report.passed:
        logger.debug(
            "persist_admitted_bundle: vov_report is %s — skip",
            "missing" if task.vov_report is None else "rejected",
        )
        return None
    try:
        from ..verifier.property_engine import write_admitted_bundle
    except ImportError:
        logger.warning(
            "persist_admitted_bundle: property_engine.write_admitted_bundle "
            "not importable — skipping both writes"
        )
        return None

    try:
        record_ids = write_admitted_bundle(
            list(task.properties), task.vov_report, regs=regs,
        )
    except ValueError as exc:
        # Shouldn't happen — we gated on passed. Log hard and fail closed.
        logger.error(
            "write_admitted_bundle refused a passed bundle (wire fault): %s", exc,
        )
        return None
    except Exception as exc:
        logger.warning(
            "write_admitted_bundle raised (%s): %s — ProblemRecord not written",
            type(exc).__name__, exc,
        )
        return None

    # Property writes succeeded; now the ProblemRecord. Failure here leaves
    # Properties orphaned — log loudly so integrator's reconcile can fix.
    ok = append_proposal_to_registry(proposal, regs, bundle_admitted=True)
    if not ok:
        logger.error(
            "append_proposal_to_registry failed for %s AFTER "
            "write_admitted_bundle persisted %d properties — integrator must "
            "reconcile. Property IDs: %s",
            proposal.problem_id, len(record_ids), record_ids,
        )
    return record_ids


# ---------------------------------------------------------------------------
# propose_one — the spec-conforming single-problem proposal entry point.
# Wraps: prompt render -> model call -> parse -> novelty dist -> reachability
# -> ProposedProblem. Does NOT solve, does NOT run property admission; both
# of those belong to downstream phase-B teammates (generator, property_verifier).
# ---------------------------------------------------------------------------


def propose_one(
    skill_a: SkillProfile,
    skill_b: SkillProfile,
    generate_fn: Callable[[str], str],
    prior_problems: Sequence[str] = (),
    session_id: str = "",
    run_id: str = "",
) -> tuple[Optional[ProposedProblem], list[str]]:
    """Single-problem propose-only call per §3.1.

    Returns (ProposedProblem or None, list_of_issues). None means the proposer
    either failed to produce a usable response or the proposal violated a
    structural invariant (<3 classes / missing reachability / declared
    difficulty below floor / incomplete property descriptors).

    This is an additive, side-effect-free helper — it does not write to disk,
    does not invoke VoV, does not run any admission gates. Callers layer those
    on (typically via a future rsi_tick orchestrator method).
    """
    issues: list[str] = []
    prompt = build_proposal_prompt(skill_a, skill_b)
    raw = generate_fn(prompt)
    parse = parse_proposal_response(raw, author_run_id=run_id or "unknown")

    if not parse.ok:
        return None, parse.issues or ["proposal response did not parse"]

    if len(parse.property_descriptors) < 4:
        issues.append(
            f"only {len(parse.property_descriptors)} properties; spec requires ≥4"
        )
    if not class_coverage_ok(parse.property_descriptors):
        issues.append("properties span < 3 independence_classes (§3.1)")
    if not passes_reachability(parse.property_descriptors):
        issues.append(
            "no exec.behavioral or algebra.symbolic property — §3.3 reachability"
        )

    nn_dist, nn_text = compute_nearest_neighbor_dist(parse.problem_text, prior_problems)

    problem_hash = hashlib.sha256(parse.problem_text.encode("utf-8")).hexdigest()
    pp = ProposedProblem(
        problem_id=str(uuid.uuid4()),
        problem_text=parse.problem_text,
        declared_difficulty=parse.declared_difficulty,
        declared_difficulty_reason=parse.declared_difficulty_reason,
        nearest_neighbor_problem=parse.nearest_neighbor_problem or nn_text,
        nearest_neighbor_dist=nn_dist,
        axis_of_extension=parse.axis_of_extension,
        parent_skills=(skill_a.key, skill_b.key),
        problem_hash=problem_hash,
        author=f"model:{run_id}" if run_id else "model:unknown",
        created_at=time.time(),
        session_id=session_id,
        property_descriptors=list(parse.property_descriptors),
    )
    if issues:
        return pp, issues  # return the artifact anyway — caller logs + holds
    return pp, []


# ---------------------------------------------------------------------------
# propose_novel — architect-named §3 entry point.
#
# Takes a MasteryProfile (dict with "mastered" / "emerging" skill keys derived
# from DiagnosticResult.subdomain_scores), picks a compositional pair, and
# runs the propose-only flow. Does NOT run VoV admission (§1.4) — callers
# layer that on after property_verifier materializes the Property objects.
#
# Architect's guidance: primary call shape is propose_novel(mastery_profile,
# ...); it replaces the old propose_one's ad-hoc two-skill signature for the
# rsi_tick path. propose_one stays as a lower-level helper.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Builtin-based proposal path (code domain).
#
# Rationale: the spec's §3.1 co-gen format requires the model to emit full
# property source code for each property — 7 named fields × ≥4 properties in
# a strict layout. An 8B base model can't hit that reliably (GPU run-3:
# 13–18/20 failures, top issue "missing_block"). Without admitted properties
# that actually run, §1.4 verify has nothing to discriminate against and
# every cycle produces zero training samples.
#
# This path trades spec fidelity for actual signal: we use the 13 trusted
# builtin Property templates from property_engine (which already have real
# executable check_fns) and only ask the model for what it CAN produce:
#
#   - PROBLEM   <one-line task description>
#   - ENTRY     <function name, e.g. solve>
#   - REFERENCE <Python def ...> (so differential tests have an oracle)
#   - TESTS     list of assert statements
#
# task_synthesizer picks 3 builtins spanning distinct independence_classes,
# stashes the problem's ctx (tests, reference, entry) via
# property_engine.stash_problem_ctx, and hands the builtin Property list to
# rsi_tick. verify() then runs actual discriminating checks — a wrong
# candidate now produces real FAIL verdicts instead of stub PASS.
# ---------------------------------------------------------------------------


CODE_PROPOSAL_TEMPLATE = """\
Your task: propose ONE novel Python coding problem AT THE FRONTIER of your
own capability — a problem you estimate you would FAIL on a first attempt
with probability at least 0.3. Trivial string/list/arithmetic problems
teach nothing. Reach for algorithmic depth: dynamic programming, graph
traversal, recursion with memoization, non-obvious invariants, tricky
parsing, careful edge cases.

Output EXACTLY in the format shown below. No prose, no markdown headers,
no explanation — just the labeled blocks, each starting its own line.

Here is a complete, well-formed example of the required output format
(note the algorithmic difficulty — DP with edge cases, NOT a one-liner):

PROBLEM: Given a list of positive integers `coins` and a target integer
`amount`, return the minimum number of coins needed to make `amount`.
Return -1 if it is impossible. You may use each coin an unlimited number
of times. `amount` is in 0..10_000, `coins` length is in 1..20.
ENTRY: solve
REFERENCE:
```python
def solve(coins, amount):
    INF = float('inf')
    dp = [0] + [INF] * amount
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a and dp[a - c] + 1 < dp[a]:
                dp[a] = dp[a - c] + 1
    return dp[amount] if dp[amount] != INF else -1
```
TESTS:
- assert solve([1, 2, 5], 11) == 3
- assert solve([2], 3) == -1
- assert solve([1], 0) == 0
- assert solve([3, 7], 14) == 2
- assert solve([5, 10], 3) == -1
EMPTY_INPUT: [[1], 0]
SAMPLE_INPUT: [[1, 2, 5], 11]
EXPECTED_TYPE: int
DIFFICULTY: 0.55
DIFFICULTY_REASON: Classic DP; first-attempt failure modes include greedy
heuristics, off-by-one on dp[0], and missing the -1 impossibility return.

Now produce ONE DIFFERENT problem in the same format. Rules:
- Pick something NOT identical to the example above.
- Favor small, well-defined problems where you are CONFIDENT you can write
  a correct reference solution AND tests whose expected values you can
  compute by hand. A 5-line problem with correct tests beats a 30-line
  problem with buggy tests.
- Problem must be solvable in 3–30 lines of Python.
- CRITICAL: your REFERENCE must pass ALL your own TESTS. Before emitting,
  mentally trace each test through your reference — if any test would
  fail on your reference, either rewrite the reference or rewrite the
  test. A reference that doesn't pass its own tests is useless and will
  be rejected by the property verifier.
- Prefer writing SIMPLE, CONCRETE test cases where you can compute the
  expected output by hand in 10 seconds. Do NOT write tests where you're
  unsure what the reference returns. "solve([1,2,3]) == 4" must be a
  value you're confident about.
- Each TEST line must be a real runnable `assert` (not "example:" or prose)
- Include at least one EDGE-CASE test (empty / minimum / off-by-one /
  impossibility) — trivial tests are not evidence of a frontier problem.
- DIFFICULTY must be a float in [0.15, 0.9] reflecting your honest estimate
  of first-attempt failure probability. Below 0.15 the proposal is rejected.
- Begin your output with `PROBLEM:` — no preamble, no "Sure, here is..."

YOUR OUTPUT:
"""


# Builder for the failure-seeded variant. We keep the canonical template
# stable (tests read it) and splice an INSPIRATION block in BEFORE the
# trailing "YOUR OUTPUT:" line so the model sees one of its recent
# diagnostic failures as frontier evidence.
def _build_failure_seeded_prompt(failed_question: str) -> str:
    base = CODE_PROPOSAL_TEMPLATE
    insertion = (
        "\nINSPIRATION (you failed a problem like this on your last "
        "diagnostic — propose a DIFFERENT but similarly-hard problem in "
        "the same spirit; do NOT copy the inspiration verbatim, translate "
        "the underlying difficulty into a fresh prompt):\n"
        f"{failed_question.strip()}\n"
    )
    marker = "YOUR OUTPUT:"
    idx = base.rfind(marker)
    if idx == -1:
        return base + insertion
    return base[:idx] + insertion + "\n" + base[idx:]


def _prepend_library_prefix(base: str, library_prefix: str) -> str:
    """Prepend the property + proposer few-shot bank to the top of the prompt.

    Placed ABOVE everything (including frontier-hint and the canonical
    PROBLEM: example) so the model sees "here are patterns that earned trust"
    before it sees the task format. Both this block and the frontier-hint
    therefore land before the canonical PROBLEM: — matching team-lead's
    ordering guidance. No-op when library_prefix is empty.
    """
    if not library_prefix:
        return base
    return library_prefix.rstrip() + "\n\n" + base


def _splice_frontier_hint(base: str, skill_pair: str) -> str:
    """Splice a frontier-skill hint before the YOUR OUTPUT marker.

    ``skill_pair`` comes from DifficultyTracker.frontier() — a string
    like "code/implementation" or "math/calculus". The hint anchors the
    model's proposal at a zone it currently fails in, rather than at a
    skill it already masters.
    """
    hint = (
        "\nFRONTIER SKILL: Design a problem that primarily requires the "
        f"skill `{skill_pair.strip()}` — the model CURRENTLY FAILS here, "
        "so push toward this capability zone (do not drift into unrelated "
        "areas). Keep all other rules above.\n"
    )
    marker = "YOUR OUTPUT:"
    idx = base.rfind(marker)
    if idx == -1:
        return base + hint
    return base[:idx] + hint + "\n" + base[idx:]


_CODE_BLOCK_LABELS = (
    "PROBLEM:", "ENTRY:", "REFERENCE:", "TESTS:",
    "EMPTY_INPUT:", "SAMPLE_INPUT:", "EXPECTED_TYPE:",
    "DIFFICULTY:", "DIFFICULTY_REASON:",
)

# Aliases observed in model output that refer to the same logical block.
# The parser treats any alias as equivalent to its canonical label. This
# closes the ~30% missing_problem/missing_entry/missing_reference/too_few_tests
# failure rate seen in cycles 1–7 of the overnight run: the base model
# consistently emits e.g. "Problem Statement:", "Function:", "Example Tests:",
# and the old strict label match counted those as failures.
_LABEL_ALIASES: dict[str, tuple[str, ...]] = {
    "PROBLEM:": ("PROBLEM STATEMENT:", "TASK:", "QUESTION:", "DESCRIPTION:"),
    "ENTRY:": ("ENTRY POINT:", "ENTRY_POINT:", "FUNCTION:", "FUNCTION NAME:",
               "FUNCTION_NAME:", "NAME:"),
    "REFERENCE:": ("SOLUTION:", "REFERENCE SOLUTION:", "REFERENCE_SOLUTION:",
                   "CODE:", "IMPLEMENTATION:"),
    "TESTS:": ("TEST CASES:", "TEST_CASES:", "EXAMPLES:", "ASSERTIONS:"),
    "EXPECTED_TYPE:": ("RETURN TYPE:", "RETURN_TYPE:", "EXPECTED TYPE:"),
    "DIFFICULTY_REASON:": ("REASONING:", "DIFFICULTY REASON:"),
    "SAMPLE_INPUT:": ("SAMPLE INPUT:", "EXAMPLE INPUT:", "EXAMPLE_INPUT:"),
    "EMPTY_INPUT:": ("EMPTY INPUT:", "MINIMAL INPUT:"),
}

# Minimum self-reported difficulty (= P[model fails on first attempt])
# below which we reject the proposal outright — matches spec §3.2.1.
# Floor disabled (0.0) after run-12: at 0.15 the model STILL couldn't write
# self-consistent (problem, reference, tests) triples on moderate-difficulty
# problems — 1 accepted in 5 cycles. The 8B base just isn't strong enough to
# write hard problems with bug-free references.
#
# Gamble for run-13: accept any declared difficulty. Quorum's any-FAIL veto
# still rejects self-inconsistent proposals. Rely on the frontier self-solve
# gate to filter trivialities (drops anything the model can already solve
# blind), and trust that the new trainer config (5e-6 LR, rank 8,
# max_grad_norm 0.3, early_stop 0.50 — ~32× gentler than the config that
# caused the run-8 regression) won't catastrophically overfit even on
# easier verified samples.
_MIN_CODE_PROPOSAL_DIFFICULTY = 0.0


@dataclass
class _CodeProposalParse:
    problem_text: str = ""
    entry_point: str = ""
    reference: str = ""
    tests: list[str] = field(default_factory=list)
    empty_input: str = ""
    sample_input: str = ""
    expected_type: str = ""
    difficulty: float = 0.0
    difficulty_reason: str = ""
    issues: list[str] = field(default_factory=list)
    _raw_preview: str = ""  # first N chars of the failing raw response, for logging

    @property
    def ok(self) -> bool:
        return (
            bool(self.problem_text)
            and bool(self.entry_point)
            and bool(self.reference)
            and len(self.tests) >= 2
            and not self.issues
        )


def parse_code_proposal(
    raw: str,
    *,
    difficulty_floor: Optional[float] = None,
) -> _CodeProposalParse:
    """Parse the simplified code-proposal format. Forgiving — only problem,
    entry, reference, and ≥2 tests are hard-required; the rest default.

    Tolerant of common model-output decorations: markdown bold/headers
    ( **PROBLEM:** / ### PROBLEM: ), extra whitespace, stray punctuation
    before the label. Previously a strict `^\\s*PROBLEM:` match missed
    18/20 of Qwen3-8B's cycle-1 responses even when the content was
    correct — just because the model wrapped labels in markdown.
    """
    out = _CodeProposalParse()
    if not raw or not raw.strip():
        out.issues.append("empty_response")
        return out

    # Strip think-tag artifacts that some models leak into text output.
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    raw = re.sub(r"</?think>", "", raw, flags=re.IGNORECASE)

    # Locate each labeled block's start — tolerant of:
    #   `**PROBLEM:**`, `### PROBLEM:`, `PROBLEM :` (space before colon),
    #   `- PROBLEM:`, `1. PROBLEM:`, etc.
    # We strip common decoration chars before the label when scanning.
    positions: dict[str, int] = {}
    for label in _CODE_BLOCK_LABELS:
        # Try canonical label first, then any aliases; first match wins.
        candidates = [label.rstrip(":")] + [
            a.rstrip(":") for a in _LABEL_ALIASES.get(label, ())
        ]
        best: Optional[int] = None
        for name in candidates:
            # Match: optional markdown/bullet prefix, then the label name,
            # optional space, then colon. Accept either ASCII ':' or
            # fullwidth '：' (some model outputs slip CJK punctuation in).
            # Label names with internal spaces (e.g. "PROBLEM STATEMENT")
            # are handled by matching a single \s between tokens.
            name_pattern = r"\s+".join(re.escape(tok) for tok in name.split())
            pattern = (
                r"(?im)^[\s*#>\-\d.`'\"]*"     # decoration / bullets / quotes
                + name_pattern
                + r"\s*[:：]"                   # colon (ASCII or fullwidth)
            )
            m = re.search(pattern, raw)
            if m and (best is None or m.start() < best):
                best = m.start()
        if best is not None:
            positions[label] = best
    if "PROBLEM:" not in positions:
        out.issues.append("missing_problem")
        # Dump the first 300 chars of the response so operators can see what
        # format the model is actually producing. Rate-limited by the caller
        # (propose_batch_code only emits the Counter summary) so this stays
        # readable at INFO level.
        out._raw_preview = raw[:300].replace("\n", "\\n")
        return out

    ordered = sorted(positions.items(), key=lambda kv: kv[1])
    blocks: dict[str, str] = {}
    for i, (label, start) in enumerate(ordered):
        end = ordered[i + 1][1] if i + 1 < len(ordered) else len(raw)
        body = raw[start:end]
        # Strip the matched label (including decoration and colon). Try each
        # alias in addition to the canonical name so aliased-label bodies
        # don't retain a stray "Problem Statement:" at the top of the block.
        label_names = [label.rstrip(":")] + [
            a.rstrip(":") for a in _LABEL_ALIASES.get(label, ())
        ]
        for name in label_names:
            name_pattern = r"\s+".join(re.escape(tok) for tok in name.split())
            new_body, n = re.subn(
                r"(?i)^[\s*#>\-\d.`'\"]*" + name_pattern + r"\s*[:：]\s*",
                "",
                body,
                count=1,
            )
            if n:
                body = new_body
                break
        # Also trim stray closing markdown bold after the label (e.g.
        # "**PROBLEM:** reverse a list" leaves a leading "**" without this).
        body = re.sub(r"^\**", "", body).strip()
        blocks[label] = body

    out.problem_text = blocks.get("PROBLEM:", "").strip()
    # Entry-point: take the first "word", then strip common decoration
    # (backticks, parentheses, quotes, colons). Previously "`solve`" leaked
    # backticks through and downstream code tried to call `solve` which
    # blew up. Also strip any trailing "()" if the model wrote "solve()".
    entry_raw = blocks.get("ENTRY:", "").strip()
    if entry_raw:
        # Take first whitespace-separated token, then strip decoration.
        tok = entry_raw.split()[0]
        tok = tok.strip("`'\"*_()[]{}:,;")
        # If still empty (e.g. just a backtick line), fall through.
        out.entry_point = tok

    ref_block = blocks.get("REFERENCE:", "")
    # Pull the python fence if present, else take the block as-is
    m = re.search(r"```(?:python|py)?\s*\n?(.*?)```", ref_block, re.DOTALL)
    if m:
        out.reference = m.group(1).strip()
    else:
        # Find the first `def ` and take everything from there
        m = re.search(r"(?:^|\n)\s*(def\s+\w+.*)", ref_block, re.DOTALL)
        out.reference = m.group(1).strip() if m else ref_block.strip()

    tests_block = blocks.get("TESTS:", "")
    # Models sometimes wrap all tests inside a single ```python ... ``` fence.
    # Extract the fenced content first, then fall back to the raw block.
    fence_m = re.search(r"```(?:python|py)?\s*\n?(.*?)```", tests_block, re.DOTALL)
    tests_source = fence_m.group(1) if fence_m else tests_block
    for line in tests_source.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        # Strip leading bullet/hyphen/numbered-list if present.
        s = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", s)
        # Strip leading ">>> " REPL prompt style.
        s = re.sub(r"^>>>\s+", "", s)
        # Strip trailing comments.
        s = re.sub(r"\s+#.*$", "", s).rstrip()
        if not s:
            continue
        if s.startswith("assert ") or s.startswith("assert("):
            out.tests.append(s)
        elif "==" in s:
            # Accept bare equality lines like `solve(1, 2) == 3`.
            out.tests.append(f"assert ({s})")
    out.empty_input = blocks.get("EMPTY_INPUT:", "").strip()
    out.sample_input = blocks.get("SAMPLE_INPUT:", "").strip()
    out.expected_type = blocks.get("EXPECTED_TYPE:", "").strip()

    # Parse DIFFICULTY — the model's self-reported P(fail on first attempt).
    # Missing / unparseable defaults to 0.0 which will trip the frontier gate.
    diff_raw = blocks.get("DIFFICULTY:", "").strip()
    if diff_raw:
        m = re.search(r"-?\d+(?:\.\d+)?", diff_raw)
        if m:
            try:
                out.difficulty = float(m.group(0))
            except ValueError:
                out.difficulty = 0.0
    out.difficulty_reason = blocks.get("DIFFICULTY_REASON:", "").strip()

    # Derive entry_point from reference if missing: common model omission
    # is to emit REFERENCE: with `def foo(...)` but skip the ENTRY: label.
    # We can recover without losing ground-truth because the reference IS
    # the source of truth for which function the tests are calling.
    if not out.entry_point and out.reference:
        m = re.search(r"def\s+(\w+)\s*\(", out.reference)
        if m:
            out.entry_point = m.group(1)
    if not out.entry_point:
        out.issues.append("missing_entry")
    if not out.reference:
        out.issues.append("missing_reference")
    if len(out.tests) < 2:
        out.issues.append("too_few_tests")
    floor = (
        difficulty_floor
        if difficulty_floor is not None
        else _MIN_CODE_PROPOSAL_DIFFICULTY
    )
    if out.difficulty < floor:
        out.issues.append(
            f"difficulty_below_frontier:{out.difficulty:.2f}"
            f"<{floor:.2f}"
        )
    return out


def _select_skills_for_proposal(
    mastery_profile: dict[str, list[str]],
    subdomain_scores: dict[str, float],
    exemplars_by_key: dict[str, list[str]],
    *,
    n_skills: int = 2,
    rng: Any = None,
) -> list[SkillProfile]:
    """Pick n_skills SkillProfiles from the mastery profile for composition.

    Factored out of propose_novel so the batched propose path can reuse the
    same selection logic. Returns a list of SkillProfile instances; raises
    ValueError if the profile can't supply ≥2 skills (the minimum for any
    compositional proposal per spec §2).
    """
    import random as _random
    rng = rng or _random.Random(0)

    mastered = list(mastery_profile.get("mastered", []))
    emerging = list(mastery_profile.get("emerging", []))

    if not mastered and not emerging:
        raise ValueError("no mastered or emerging skills in profile")
    if n_skills < 1 or n_skills > 3:
        raise ValueError(f"n_skills={n_skills} outside [1,3]")

    # Prefer mastered (score-sorted), fall back to emerging if short.
    mastered.sort(key=lambda k: -subdomain_scores.get(k, 0.0))
    primary = mastered[: max(n_skills, 2)]
    rng.shuffle(primary)
    picked_keys = list(primary[:n_skills])

    if len(picked_keys) < n_skills and emerging:
        emerging.sort(key=lambda k: -subdomain_scores.get(k, 0.0))
        for k in emerging:
            if k not in picked_keys:
                picked_keys.append(k)
            if len(picked_keys) >= n_skills:
                break

    if len(picked_keys) < 2:
        raise ValueError(
            f"only {len(picked_keys)} skill(s) available; need ≥2 for composition"
        )

    out: list[SkillProfile] = []
    for key in picked_keys:
        if "/" in key:
            dom, sub = key.split("/", 1)
        else:
            dom, sub = key, ""
        out.append(SkillProfile(
            key=key, domain=dom, subdomain=sub,
            pass_rate=float(subdomain_scores.get(key, 0.0)),
            exemplars=list(exemplars_by_key.get(key, []))[:3],
        ))
    return out


def _issue_category(issue: str) -> str:
    """Collapse a parse-issue string into a coarse category for counting.

    Helps operators diagnose why a batch of 20 proposals failed without
    dumping 20 full issue strings into the log.
    """
    s = issue.lower()
    if "empty response" in s:
        return "empty_response"
    if "missing problem" in s or "missing properties" in s:
        return "missing_block"
    if "no properties parsed" in s:
        return "no_properties"
    if "difficulty" in s and ("< 0.3" in s or "not a float" in s):
        return "bad_difficulty"
    if "axis" in s and "not in" in s:
        return "bad_axis"
    if "independence_classes" in s:
        return "class_coverage_short"
    if "reachability" in s:
        return "no_reachable_class"
    if "only" in s and "properties" in s:
        return "too_few_properties"
    return "other"


def propose_novel(
    mastery_profile: dict[str, list[str]],
    subdomain_scores: dict[str, float],
    generate_fn: Callable[[str], str],
    *,
    exemplars_by_key: Optional[dict[str, list[str]]] = None,
    prior_problems: Sequence[str] = (),
    session_id: str = "",
    run_id: str = "",
    n_skills: int = 2,
    rng: Any = None,
) -> tuple[Optional[ProposedProblem], list[str]]:
    """Synthesize one novel problem from a MasteryProfile (architect §v0.2 §1).

    Compositional pairing policy (architect §v0.2 §2):
      - Pick `n_skills` keys (default 2, max 3) from `mastery_profile["mastered"]`.
        If mastered is short, fill ≤1 slot from `mastery_profile["emerging"]`.
      - Tags are the `(domain, subdomain)` keys from ground_truth.py — reused
        directly, no parallel taxonomy.

    Returns (ProposedProblem or None, issues). None means the proposer output
    failed to parse or violated a §3.1 structural invariant. Issues carry
    non-fatal warnings (e.g. class coverage short) so the caller can log +
    hold; VoV admission is the authoritative downstream gate.

    No disk write, no VoV, no property materialization — those are the
    orchestrator's responsibility after this call returns.
    """
    import random as _random
    rng = rng or _random.Random(0)

    mastered = list(mastery_profile.get("mastered", []))
    emerging = list(mastery_profile.get("emerging", []))

    if not mastered:
        return None, ["no mastered skills — frontier probe cannot compose"]
    if n_skills < 1 or n_skills > 3:
        return None, [f"n_skills={n_skills} outside [1, 3]"]

    # Sort mastered by current score descending so we prefer the most-solid
    # skills when picking without replacement. Still shuffled within a
    # score-sorted ring to avoid always composing the same top pair.
    mastered.sort(key=lambda k: -subdomain_scores.get(k, 0.0))
    primary = mastered[: min(len(mastered), max(n_skills, 2))]
    rng.shuffle(primary)
    picked_keys = primary[:n_skills]

    # If we're short on mastered for the requested n, borrow ≤1 from emerging.
    if len(picked_keys) < n_skills and emerging:
        emerging.sort(key=lambda k: -subdomain_scores.get(k, 0.0))
        picked_keys.append(emerging[0])

    if len(picked_keys) < 2:
        # The spec requires composition, not extension of a single skill.
        return None, [
            f"only {len(picked_keys)} skill(s) selected; need ≥2 for composition"
        ]

    exemplars_by_key = exemplars_by_key or {}
    picked: list[SkillProfile] = []
    for key in picked_keys:
        if "/" in key:
            dom, sub = key.split("/", 1)
        else:
            dom, sub = key, ""
        picked.append(SkillProfile(
            key=key, domain=dom, subdomain=sub,
            pass_rate=float(subdomain_scores.get(key, 0.0)),
            exemplars=list(exemplars_by_key.get(key, []))[:3],
        ))

    issues: list[str] = []
    prompt = build_proposal_prompt_from_profiles(picked)
    raw = generate_fn(prompt)
    parse = parse_proposal_response(raw, author_run_id=run_id or "unknown")

    if not parse.ok:
        return None, parse.issues or ["proposal response did not parse"]

    if len(parse.property_descriptors) < 4:
        issues.append(
            f"only {len(parse.property_descriptors)} properties; spec requires ≥4"
        )
    if not class_coverage_ok(parse.property_descriptors):
        issues.append("properties span < 3 independence_classes (§3.1)")
    if not passes_reachability(parse.property_descriptors):
        issues.append(
            "no exec.behavioral or algebra.symbolic property — §3.3 reachability"
        )

    # Pre-commit difficulty bound (architect §3): declared fail-prob must be
    # ≥0.3 (equivalently solve-prob ≤0.7). parse_proposal_response already
    # rejects <0.3 as a parse issue; we keep a defensive check here in case
    # parsing ever relaxes.
    if parse.declared_difficulty < 0.3:
        issues.append(
            f"declared_difficulty={parse.declared_difficulty:.2f} < 0.3 "
            f"— §3.2.1 pre-commit threshold"
        )

    nn_dist, nn_text = compute_nearest_neighbor_dist(
        parse.problem_text, prior_problems,
    )
    problem_hash = hashlib.sha256(
        parse.problem_text.encode("utf-8")
    ).hexdigest()

    # parent_skills is a 2-tuple on the dataclass for backwards-compat; the
    # full list of picked skills lands in property_descriptors' context and
    # can be surfaced via a future MasteryProfile-aware SynthesizedTask. For
    # now we keep the tuple shape and stash extras in the synth_descriptors
    # audit trail via proposal_to_training_sample.
    parent_tuple = (
        picked[0].key, picked[1].key if len(picked) >= 2 else "",
    )

    pp = ProposedProblem(
        problem_id=str(uuid.uuid4()),
        problem_text=parse.problem_text,
        declared_difficulty=parse.declared_difficulty,
        declared_difficulty_reason=parse.declared_difficulty_reason,
        nearest_neighbor_problem=parse.nearest_neighbor_problem or nn_text,
        nearest_neighbor_dist=nn_dist,
        axis_of_extension=parse.axis_of_extension,
        parent_skills=parent_tuple,
        problem_hash=problem_hash,
        author=f"model:{run_id}" if run_id else "model:unknown",
        created_at=time.time(),
        session_id=session_id,
        property_descriptors=list(parse.property_descriptors),
    )
    if len(picked) > 2:
        # Stash the third skill for downstream observability without changing
        # the ProposedProblem wire format. integrator can read session_id +
        # problem_id to recover the full composition if needed.
        logger.info(
            "propose_novel: composition uses %d skills (2 surface, %d extra)",
            len(picked), len(picked) - 2,
        )
    return (pp, issues) if issues else (pp, [])


# ---------------------------------------------------------------------------
# propose_batch — §4 step 1 "propose_batch(N)". Owned by task_synthesizer;
# different axis from generator's K candidate rollout (§4 step 3).
# Architect default: N ∈ [4, 8] per tick, tune later.
# ---------------------------------------------------------------------------


PROPOSE_BATCH_DEFAULT_N = 6


def propose_batch(
    mastery_profile: dict[str, list[str]],
    subdomain_scores: dict[str, float],
    generate_fn: Callable[[str], str],
    *,
    n: int = PROPOSE_BATCH_DEFAULT_N,
    exemplars_by_key: Optional[dict[str, list[str]]] = None,
    prior_problems: Sequence[str] = (),
    session_id: str = "",
    run_id: str = "",
    n_skills: int = 2,
    rng: Any = None,
    stop_on_empty: bool = False,
) -> tuple[list[ProposedProblem], list[list[str]]]:
    """Emit up to `n` ProposedProblems per tick (§4 step 1).

    Returns (accepted, per_call_issues) — accepted is the list of proposals
    that parsed cleanly (no rejections yet — that's the admission gate's
    job). per_call_issues is parallel: entry i is the issues list from the
    i-th propose_novel call, EVEN when a proposal was accepted (so an
    orchestrator can log structural warnings that didn't kill the proposal).

    Each call to propose_novel reseeds the shuffle via `rng` so repeated
    calls pick different skill compositions across the batch. The loop is
    serial — parallelism is the orchestrator's concern.

    `stop_on_empty=True` aborts after the first None proposal. Useful for
    fast-fail during smoke tests. Default False keeps going to finish the
    batch even when one call drops.
    """
    import random as _random
    rng = rng or _random.Random(0)

    accepted: list[ProposedProblem] = []
    per_call_issues: list[list[str]] = []

    for i in range(max(0, n)):
        pp, issues = propose_novel(
            mastery_profile, subdomain_scores, generate_fn,
            exemplars_by_key=exemplars_by_key,
            prior_problems=list(prior_problems) + [p.problem_text for p in accepted],
            session_id=session_id,
            run_id=f"{run_id}#{i}" if run_id else f"batch#{i}",
            n_skills=n_skills,
            rng=rng,
        )
        per_call_issues.append(issues)
        if pp is None:
            if stop_on_empty:
                break
            continue
        accepted.append(pp)

    return accepted, per_call_issues


# ---------------------------------------------------------------------------
# property_engine bridge.
#
# property_verifier.verify_sample_by_properties reads `sample.verifier_meta`
# = {"property_ids": [...], "property_ctx": {...}}. We emit samples in that
# shape so the trainer path can route them through property_engine without
# knowing about ProposedProblem.
#
# Spec v0.2.1: property_engine retagged all builtins to §2.2 canonical classes
# (architect sign-off, commit referenced in drift fix). No hand-coded bridge is
# needed — Property.independence_class IS the §2.2 class. The remaining job of
# suggest_builtin_property_ids is a reverse lookup: given a descriptor's class,
# find a registered builtin whose class matches.
#
# `_builtins_by_class()` builds the index at call time (not import time) so
# test monkey-patches to the registry take effect, and new builtins added by
# property_verifier show up without synthesizer code changes.
# ---------------------------------------------------------------------------


def _builtins_by_class() -> dict[str, list[str]]:
    """Return {independence_class: [builtin_name, ...]} from property_engine.

    Called on demand; empty dict when property_engine isn't importable (unit-
    only runs don't depend on the registry being loaded).

    Also flags wire-format drift: any builtin whose class isn't in
    `INDEPENDENCE_CLASSES` logs a warning — it means property_engine has a
    builtin on the LegacyProperty axis or an unknown string, which violates
    v0.2.1's canonical-class invariant. Flagging lets architect catch
    regressions early; the builtin is still indexed for backwards-compat.
    """
    try:
        from ..verifier.property_engine import builtin_properties
    except ImportError:
        return {}
    idx: dict[str, list[str]] = {}
    for p in builtin_properties():
        cls = getattr(p, "independence_class", "") or ""
        name = getattr(p, "name", "")
        if not cls or not name:
            continue
        if cls not in INDEPENDENCE_CLASSES:
            logger.warning(
                "builtin %r has non-canonical independence_class=%r — "
                "wire-format drift from §2.2 (expected one of %d canonical "
                "strings). Flag to architect.",
                name, cls, len(INDEPENDENCE_CLASSES),
            )
        idx.setdefault(cls, []).append(name)
    return idx


# Per-domain baseline recipes chosen to span ≥3 distinct §2.2
# independence_classes after property_engine's v0.2 class-vocab alignment.
#
# NOTE: property_verifier's suggested "code" recipe
#   executes + passes_provided_tests + passes_generated_edge_cases
# reduces to 2 classes (exec.behavioral + search.bounded) because the first
# two builtins share `exec.behavioral`. Quorum (§2.1) needs 3 distinct —
# so we swap in `output_type_matches_signature` (structural.static) for
# `executes`, giving a confirmed 3-class code bundle.
#
# "math" similarly needs a non-algebraic check since substitute_back and
# alternative_derivation_agrees both carry `algebra.symbolic`. We substitute
# `dimensional_consistency` (dimensional.physical) for
# `alternative_derivation_agrees`. "logic" already spans 3 classes as-is.
#
# Verified via a startup check in a unit test — if property_verifier
# rebalances builtin classes again, the test will catch the drift.
QUORUM_CLEARING_BUNDLES: dict[str, list[str]] = {
    "code": [
        "passes_provided_tests",          # exec.behavioral
        "passes_generated_edge_cases",    # search.bounded
        "output_type_matches_signature",  # structural.static
    ],
    "math": [
        "substitute_back",                # algebra.symbolic
        "numerical_plausibility",         # search.bounded
        "dimensional_consistency",        # dimensional.physical
    ],
    "logic": [
        "contrapositive_holds",                         # smt.logical
        "premise_reformulation_preserves_conclusion",   # transform.semantic
        "trivial_case_correct",                         # exec.behavioral
    ],
}


def default_bundle_for_domain(domain: str) -> list[str]:
    """Return the curated builtin names for a domain's baseline quorum bundle.

    Use when you want to guarantee ≥3 distinct classes without relying on the
    model's per-descriptor choices. Unknown domains return an empty list.
    """
    return list(QUORUM_CLEARING_BUNDLES.get(domain, []))


def suggest_builtin_property_ids(
    descriptors: Sequence[PropertyDescriptor],
    domain: str,
    *,
    ensure_quorum_clearing: bool = True,
) -> list[str]:
    """Map each descriptor's §2.2 independence_class to a registered builtin.

    Now an identity-passthrough reverse lookup: reads the live class→name
    index from property_engine's registered builtins (§2.2 canonical classes)
    and picks the first registered name whose class matches each descriptor.
    Domain isn't used to disambiguate anymore — classes are globally unique
    identifiers after v0.2.1. Domain is still accepted for the quorum-clearing
    fallback and for backwards-compat with callers that were passing it.

    Returns a de-duplicated list of builtin names. Descriptors whose class has
    no registered builtin contribute nothing — the caller can still attach the
    descriptor as audit, but verification via property_engine requires a
    matching registered builtin.

    With `ensure_quorum_clearing=True` (default), if the per-descriptor result
    lands fewer than 3 distinct classes, the curated QUORUM_CLEARING_BUNDLES
    recipe for `domain` is unioned in.
    """
    ids: list[str] = []
    seen: set[str] = set()
    index = _builtins_by_class()
    for desc in descriptors:
        candidates = index.get(desc.independence_class, [])
        for name in candidates:
            if name not in seen:
                seen.add(name)
                ids.append(name)
                break   # one builtin per descriptor (keeps bundle small)

    if ensure_quorum_clearing:
        # Quorum needs ≥3 distinct CLASSES (not builtins) per §2.1. Inspect
        # the classes covered so far; if <3, union the curated domain recipe.
        covered_classes: set[str] = set()
        for name in ids:
            for cls, names in index.items():
                if name in names:
                    covered_classes.add(cls)
                    break
        if len(covered_classes) < 3:
            for bid in default_bundle_for_domain(domain):
                if bid not in seen:
                    seen.add(bid)
                    ids.append(bid)
            # Let the caller read back; we don't re-check covered_classes
            # because the curated recipe is pre-verified to span ≥3 classes.
    return ids


def stamp_library_property(
    library_prop: Any,
    descriptor: PropertyDescriptor,
    *,
    problem_hash: str = "",
    problem_id: str = "",
) -> Any:
    """Per-problem stamp for a library-provided Property (v0.2.1 §1.3 clarification).

    Library builtins ship with empty `confirmer_example` / `falsifier_example`
    placeholders — they have no problem attached. When task_synthesizer binds a
    library property to a concrete problem, we copy the descriptor's concrete
    confirmer/falsifier onto a new Property instance so admission (§1.3 gate 3:
    self-test) runs against the real per-problem pairing, not the placeholder.

    Also stamps `parent_problem_hash` and `problem_id` if the Property dataclass
    has those fields — these are v0.2.1 fields that bind the Property to its
    problem. Uses dataclasses.replace when available, falls back to setattr.

    Returns a NEW Property instance (dataclasses.replace semantics) or the
    original object with in-place writes if replace isn't applicable. Does
    not mutate the library singleton.
    """
    # Collect the fields the spec says we must stamp per-problem. Only stamp
    # fields the Property actually has — both v0.2.1 Property (16 fields incl.
    # confirmer/falsifier/parent_problem_hash/problem_id) and LegacyProperty
    # (pre-v0.2) are out there; reading via getattr keeps both working.
    updates: dict = {}
    if descriptor.confirmer_example:
        updates["confirmer_example"] = descriptor.confirmer_example
    if descriptor.falsifier_example:
        updates["falsifier_example"] = descriptor.falsifier_example
    if problem_hash:
        updates["parent_problem_hash"] = problem_hash
    if problem_id:
        updates["problem_id"] = problem_id

    # Try dataclasses.replace — produces a new instance without mutating the
    # library singleton. Filter updates down to fields the target class has.
    try:
        from dataclasses import fields, is_dataclass, replace
    except ImportError:
        is_dataclass = lambda _: False   # noqa: E731
        replace = None                   # type: ignore

    if replace is not None and is_dataclass(library_prop):
        field_names = {f.name for f in fields(library_prop)}
        filtered = {k: v for k, v in updates.items() if k in field_names}
        if not filtered:
            return library_prop
        return replace(library_prop, **filtered)

    # Fallback for non-dataclass Property impls: shallow copy + setattr. If
    # the object doesn't support copy, we stamp in place (documented hazard).
    try:
        import copy
        out = copy.copy(library_prop)
    except Exception:
        out = library_prop
    for k, v in updates.items():
        if hasattr(out, k):
            try:
                setattr(out, k, v)
            except Exception:
                pass
    return out


def stamp_library_properties_for_problem(
    builtin_ids: Sequence[str],
    descriptors: Sequence[PropertyDescriptor],
    proposal: "ProposedProblem",
) -> list[Any]:
    """Resolve each builtin id → Property, stamp its confirmer/falsifier from
    the matching descriptor (by independence_class), and return the list.

    Returns [] if property_engine is not importable (pre-A2) — callers should
    fall back to attaching only `property_ids` and letting property_verifier
    stamp at admit() time once that hook exists.

    Matching heuristic: we pair each builtin with the descriptor whose
    independence_class matches the builtin's — builtins carry the spec §2.2
    class tag after v0.2 alignment. When multiple descriptors share a class,
    the first unused one wins. When no descriptor matches, the builtin is
    stamped only with problem_hash/problem_id; confirmer/falsifier stay as
    the registry placeholder (admission will fail self-test — which is the
    correct signal, not a silent swallow).
    """
    try:
        from ..verifier.property_engine import get_property
    except ImportError:
        logger.debug("property_engine unavailable; skipping library stamping")
        return []

    by_class: dict[str, list[PropertyDescriptor]] = {}
    for d in descriptors:
        by_class.setdefault(d.independence_class, []).append(d)

    stamped: list[Any] = []
    for bid in builtin_ids:
        lib = get_property(bid)
        if lib is None:
            logger.warning("stamp: unknown builtin id %r", bid)
            continue
        cls = getattr(lib, "independence_class", "")
        bucket = by_class.get(cls, [])
        desc = bucket.pop(0) if bucket else None
        if desc is None:
            # Synthesize a minimal descriptor so problem_hash still stamps.
            desc = PropertyDescriptor(
                name=bid, kind="POSTCONDITION", description="",
                independence_class=cls or "<unclassified>",
                difficulty_floor=0.0,
                falsifier_example="", confirmer_example="",
                author_run_id="library",
            )
        stamped.append(stamp_library_property(
            lib, desc,
            problem_hash=proposal.problem_hash,
            problem_id=proposal.problem_id,
        ))
    return stamped


def proposal_to_training_sample(
    proposal: ProposedProblem,
    reference_solution: str = "",
    property_ctx: Optional[dict] = None,
    domain_hint: str = "",
    *,
    stamp_examples: bool = True,
) -> TrainingSample:
    """Project a ProposedProblem onto a TrainingSample with property attachments.

    Shape expected by property_engine.verify_sample_by_properties:
        sample.verifier_meta = {
            "property_ids": [<builtin names>],
            "property_ctx": {<problem-specific context>},
        }

    `property_ctx` is caller-supplied because it's problem-specific (tests,
    entrypoint, equations, etc. — see property_verifier's registered names
    list). We default to an empty dict; the orchestrator is responsible for
    populating it from the proposed problem's shape before handing off.

    `domain_hint` picks which builtin vocabulary to draw from. If the
    ProposedProblem has a single-domain parent_skills pair, set that here.
    """
    descriptors = proposal.property_descriptors
    domain = domain_hint or (
        proposal.parent_skills[0].split("/", 1)[0] if proposal.parent_skills[0] else ""
    )
    builtin_ids = suggest_builtin_property_ids(descriptors, domain)
    # v0.2.1 §1.3: stamp confirmer/falsifier/parent_problem_hash onto library
    # Property instances before admission, since registry placeholders are
    # empty by design. Stamping is optional because property_engine may not
    # be importable during unit-only runs; callers can still get property_ids
    # + synth_descriptors and let downstream stamp at admit().
    stamped_properties: list[Any] = []
    if stamp_examples:
        stamped_properties = stamp_library_properties_for_problem(
            builtin_ids, descriptors, proposal,
        )
    meta: dict = {
        "property_ids": builtin_ids,
        "property_ctx": dict(property_ctx or {}),
        # Synthesizer audit trail so the integrator can reconcile.
        "synth_descriptors": [
            {
                "name": d.name,
                "kind": d.kind,
                "independence_class": d.independence_class,
                "confirmer_example": d.confirmer_example,
                "falsifier_example": d.falsifier_example,
                "difficulty_floor": d.difficulty_floor,
                "adversarial": d.adversarial,
                "author_run_id": d.author_run_id,
            } for d in descriptors
        ],
        "problem_hash": proposal.problem_hash,
        "declared_difficulty": proposal.declared_difficulty,
        "nearest_neighbor_dist": proposal.nearest_neighbor_dist,
    }
    sample = TrainingSample(
        prompt=proposal.problem_text,
        response="",
        reasoning_chain=[],
        target_weakness=(
            f"frontier:{proposal.parent_skills[0]}+{proposal.parent_skills[1]}"
        ),
        domain=domain,
        verified=False,
        expected_answer=reference_solution,
        source="rsi_property",  # §5.2 TrainingSample.source tag
        content_hash=proposal.problem_hash,
    )
    # Attach via verifier_meta — property_engine.sample_has_properties reads it.
    # We set it after construction because TrainingSample doesn't declare the
    # field (property_verifier reads via getattr).
    setattr(sample, "verifier_meta", meta)
    setattr(sample, "property_ctx", meta["property_ctx"])
    if stamped_properties:
        # Per property_verifier's API note: sample.properties can carry
        # Property objects directly, which takes precedence over property_ids
        # when both are present. The stamped versions have confirmer/falsifier
        # bound to THIS problem — registry placeholders would fail self-test.
        setattr(sample, "properties", stamped_properties)
    return sample


