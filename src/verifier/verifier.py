"""Verifier — real multi-layer verification of reasoning chains.

Layers (in order, cheap → expensive):
  1. Structural: chain non-empty, min-step count, conclusion present.
  2. Logical validity: contradictions within a step and across prior steps,
     circular reasoning (step justified only by its own conclusion),
     non-sequitur detection (justification shares no terms with content).
  3. Step completeness: each step builds on prior + premises (lexical +
     symbolic overlap) without introducing hidden premises.
  4. Assumption grounding: declared assumptions are actually referenced in
     the step content; no material claim lacks a declared assumption or
     prior-step derivation.
  5. Domain execution:
       - math   → sympy.simplify(lhs - rhs) == 0 and per-step equation check
       - code   → sandboxed subprocess with timeout + RLIMIT, run embedded
                  tests or sanity-execute the function.
  6. Model-assisted escalation: when calibrated confidence lies in the
     uncertain band, ask the model to adjudicate.

Confidence is a *calibrated weighted score in [0,1]*, not checks_passed /
checks_total. Each signal contributes its weight × its own confidence; the
aggregate is normalized by the sum of weights actually used.
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from ..utils.config import VerifierConfig
from ..utils.sympy_utils import (
    HAS_SYMPY as _HAS_SYMPY,
    sympify_safe as _sympify_safe,
    equation_valid as _sympy_equation_valid,
    symbolic_equiv as _symbolic_equiv,
    numeric_equiv as _numeric_equiv,
    normalize_answer as _normalize_answer,
    solve_single_var as _sympy_solve_single_var,
    gather_equations as _shared_gather_equations,
)
from ..utils.sandbox import run_python_sandboxed as _run_code_sandboxed
from ..generator.data_generator import TrainingSample, ReasoningStep

logger = logging.getLogger(__name__)

if _HAS_SYMPY:
    import sympy  # type: ignore
else:
    sympy = None  # type: ignore


# ──────────────────────────── data classes ────────────────────────────

@dataclass
class CheckOutcome:
    name: str
    weight: float
    confidence: float  # in [0, 1]
    passed: bool
    detail: str = ""


@dataclass
class StepVerification:
    step_number: int
    valid: bool
    confidence: float = 0.0
    checks: list[CheckOutcome] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


@dataclass
class VerificationResult:
    accepted: bool
    step_results: list[StepVerification] = field(default_factory=list)
    chain_checks: list[CheckOutcome] = field(default_factory=list)
    overall_confidence: float = 0.0
    rejection_reasons: list[str] = field(default_factory=list)
    escalated_to_model: bool = False


# ───────────────────────── heuristic constants ─────────────────────────

_RE_STEP_REF = re.compile(r"\bstep\s*\d+\b", re.IGNORECASE)
_RE_PRIOR_REF = re.compile(r"\b(?:above|previous|earlier|prior)\b", re.IGNORECASE)
_RE_MATH_TOKEN = re.compile(r"\d|\^|[+\-*/=<>]")
_RE_EQUATION = re.compile(r"([A-Za-z_][A-Za-z_0-9]*|\([^()=]+\)|[^=]{1,60}?)\s*=\s*([^=]+)")
_RE_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_RE_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")

_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "to", "of", "in", "on", "at", "by", "for", "with",
    "from", "as", "that", "this", "these", "those", "it", "its", "we",
    "our", "you", "your", "they", "their", "i", "he", "she", "him", "her",
    "step", "steps", "then", "so", "if", "thus", "hence", "therefore",
    "because", "since", "given", "have", "has", "had", "will", "can",
    "could", "should", "would", "may", "might", "must", "do", "does",
    "did", "done", "also", "only", "just", "more", "less", "than", "some",
    "any", "all", "no", "not", "yes",
})

_INFERENCE_MARKERS = re.compile(
    r"\b(?:therefore|thus|hence|consequently|because|since|implies|"
    r"substituting|applying|evaluating|computing|simplifying|rearranging|"
    r"expanding|factoring|yields|it follows|by (?:induction|contradiction|"
    r"the (?:chain|product|quotient|power) rule|definition|theorem|lemma|"
    r"assumption|hypothesis)|from the (?:formula|equation|definition|result)|"
    r"we (?:get|obtain|find|know|have|conclude))\b",
    re.IGNORECASE,
)

_NEGATION_PAIRS = [
    ({"is"}, {"is", "not"}),
    ({"are"}, {"are", "not"}),
    ({"can"}, {"cannot"}),
    ({"equals"}, {"does", "not", "equal"}),
    ({"increases"}, {"decreases"}),
    ({"increasing"}, {"decreasing"}),
    ({"positive"}, {"negative"}),
    ({"always"}, {"never"}),
    ({"possible"}, {"impossible"}),
    ({"true"}, {"false"}),
    ({"greater"}, {"less"}),
    ({"convergent"}, {"divergent"}),
    ({"converges"}, {"diverges"}),
    ({"exists"}, {"does", "not", "exist"}),
    ({"even"}, {"odd"}),
    ({"finite"}, {"infinite"}),
]
_NEGATION_PAIRS = [(frozenset(p), frozenset(n)) for p, n in _NEGATION_PAIRS]

# Quantifier conflicts: "for all X" vs "there exists X such that not"
_QUANTIFIER_ALL = re.compile(r"\b(?:for\s+all|every|each|any)\b", re.IGNORECASE)
_QUANTIFIER_EXISTS_NOT = re.compile(
    r"\b(?:there\s+exists?|some)\b.*?\bnot\b", re.IGNORECASE
)


def _words(text: str) -> set[str]:
    return {w.lower() for w in _RE_WORD.findall(text or "")}


def _significant(text: str) -> set[str]:
    return {w for w in _words(text) if len(w) > 3 and w not in _STOPWORDS}


def _cfg(cfg: VerifierConfig, name: str, default):
    return getattr(cfg, name, default)


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ──────────────────────────── the verifier ────────────────────────────

class Verifier:
    def __init__(self, config: VerifierConfig):
        self.config = config
        self._model_verifier = None

        self.code_timeout = _cfg(config, "code_exec_timeout", 5)
        self.code_memory_mb = _cfg(config, "code_exec_memory_mb", 256)
        self.enable_sympy = _cfg(config, "enable_sympy_math_check", True) and _HAS_SYMPY
        self.enable_code_exec = _cfg(config, "enable_code_exec_check", True)
        self.escalate_below = _cfg(config, "escalate_to_model_below", 0.85)
        self.escalate_above = _cfg(config, "escalate_to_model_above", 0.50)
        self.max_prior = _cfg(config, "max_prior_steps_to_compare", 8)
        self.allow_model_override_reject = _cfg(config, "allow_model_override_reject", True)
        self.atomic_mode = _cfg(config, "atomic_mode", False)
        self.weights: dict = _cfg(config, "check_weights", {
            "logical_validity": 1.0,
            "step_completeness": 1.0,
            "assumption_grounding": 1.0,
            "domain_exec": 2.0,
            "consistency": 1.5,
            "structural": 0.75,
        })

    def set_model_verifier(self, model_loader) -> None:
        self._model_verifier = model_loader

    def set_ground_truth_grader(self, grader) -> None:
        """Inject a check_type-dispatched grader, e.g. DiagnosticsEngine._check_answer.

        Signature: grader(response: str, expected: str, check_type: str) -> bool.
        If set, the verifier will use it when a sample carries expected_answer +
        ground_truth_check_type. Otherwise falls back to grade_against_canonical.
        """
        self._gt_grader = grader

    def _ground_truth_gate(self, sample: TrainingSample) -> Optional[bool]:
        """Check canonical-answer gate. Returns True (pass), False (fail), or None (not applicable)."""
        expected = getattr(sample, "expected_answer", "") or ""
        if not expected:
            return None
        # If the producer already verified against ground truth, trust it.
        if getattr(sample, "ground_truth_verified", False):
            return True
        response = sample.response or ""
        if not response and sample.reasoning_chain:
            response = sample.reasoning_chain[-1].content or ""
        if not response:
            return False
        check_type = getattr(sample, "ground_truth_check_type", "") or ""
        grader = getattr(self, "_gt_grader", None)
        if grader is not None and check_type:
            try:
                ok = bool(grader(response, expected, check_type))
            except Exception as e:
                logger.warning(f"ground-truth grader raised {type(e).__name__}: {e}")
                return None
            if ok:
                sample.ground_truth_verified = True
            return ok
        # Fallback: sample-aware grader
        ok, _ = grade_against_canonical(sample, expected, sample.domain or "")
        if ok:
            sample.ground_truth_verified = True
        return ok

    # ───── public API ─────

    def verify_batch(self, samples: list[TrainingSample]) -> list[TrainingSample]:
        """Verify a batch of samples, returning only those that pass."""
        results = [(s, self._verify_heuristic(s)) for s in samples]

        if (_cfg(self.config, "use_model_verification", False)
                and self._model_verifier is not None):
            self._batch_model_escalation(results)

        out: list[TrainingSample] = []
        for sample, res in results:
            sample.verified = res.accepted
            sample.confidence = res.overall_confidence
            sample.verification_notes = self._format_notes(res)
            # Rich per-sample trace stamped on BOTH passing and failing samples so
            # downstream (trainer / diag / auditor) can inspect rejection reasons.
            sample.verifier_meta = {
                "passed": res.accepted,
                "confidence": res.overall_confidence,
                "rejection_trace": "; ".join(res.rejection_reasons) if res.rejection_reasons else "",
                "per_step_results": [
                    {
                        "step_number": sv.step_number,
                        "valid": sv.valid,
                        "confidence": sv.confidence,
                        "issues": list(sv.issues),
                    }
                    for sv in res.step_results
                ],
                "escalated_to_model": res.escalated_to_model,
            }
            if res.accepted:
                out.append(sample)
        return out

    def verify(self, sample: TrainingSample) -> VerificationResult:
        """Verify a single sample with optional model escalation."""
        res = self._verify_heuristic(sample)
        if (_cfg(self.config, "use_model_verification", False)
                and self._model_verifier is not None
                and self._in_escalation_band(res.overall_confidence)):
            self._apply_model_escalation(sample, res)
        return res

    # ───── core pipeline ─────

    def _verify_heuristic(self, sample: TrainingSample) -> VerificationResult:
        # Edge case: empty reasoning chain
        if not sample.reasoning_chain:
            return VerificationResult(
                accepted=False,
                overall_confidence=0.0,
                rejection_reasons=["No reasoning chain present"],
            )

        # Atomic-mode: strict structural gate. Reject BEFORE heuristic scoring.
        if self.atomic_mode:
            atomic_issues = self._atomic_structure_check(sample)
            if atomic_issues:
                return VerificationResult(
                    accepted=False,
                    overall_confidence=0.0,
                    rejection_reasons=[f"[atomic] {m}" for m in atomic_issues],
                )

        # Ground-truth gate: if a canonical answer and check_type are set,
        # enforce them as a hard pass/fail. Reject immediately on fail; accept
        # outright on pass (subject to remaining structural checks for confidence).
        gt_ok = self._ground_truth_gate(sample)
        if gt_ok is False:
            expected = getattr(sample, "expected_answer", "")
            ct = getattr(sample, "ground_truth_check_type", "")
            return VerificationResult(
                accepted=False,
                overall_confidence=0.0,
                rejection_reasons=[
                    f"[ground_truth/{ct or 'grader'}] "
                    f"final answer does not match canonical '{str(expected)[:80]}'"
                ],
            )

        weighted_sum = 0.0
        weight_total = 0.0
        rejection_reasons: list[str] = []

        # Parse-confidence prior from the generator
        pc = getattr(sample, "parse_confidence", 1.0)
        if pc is not None and pc < 1.0:
            weighted_sum += 0.5 * max(0.0, pc)
            weight_total += 0.5
            if pc < 0.5:
                rejection_reasons.append(
                    f"[parse] low generator parse_confidence ({pc:.2f})"
                )
        parse_issues = getattr(sample, "parse_issues", None) or []
        if parse_issues:
            for pi in parse_issues[:2]:
                rejection_reasons.append(f"[parse] {pi}")

        # Structural (chain-level)
        chain_outcomes = self._chain_checks(sample)
        for o in chain_outcomes:
            weighted_sum += o.weight * o.confidence
            weight_total += o.weight
            if not o.passed:
                rejection_reasons.append(f"[chain/{o.name}] {o.detail}")

        # Step-level checks
        step_results: list[StepVerification] = []
        for i, step in enumerate(sample.reasoning_chain):
            prior = sample.reasoning_chain[max(0, i - self.max_prior):i]
            sv = self._verify_step(step, prior, sample)
            step_results.append(sv)
            for o in sv.checks:
                weighted_sum += o.weight * o.confidence
                weight_total += o.weight
                if not o.passed:
                    rejection_reasons.append(
                        f"[step {step.step_number}/{o.name}] {o.detail}"
                    )

        # Domain execution (whole-sample)
        dom_outcomes = self._domain_execution(sample)
        for o in dom_outcomes:
            weighted_sum += o.weight * o.confidence
            weight_total += o.weight
            if not o.passed:
                rejection_reasons.append(f"[domain/{o.name}] {o.detail}")

        # Atomic-mode: per-step sympy/code validation contributes to score
        if self.atomic_mode:
            w_exec = self.weights.get("domain_exec", 2.0)
            if sample.domain == "math" and self.enable_sympy:
                for sid, ok, detail in self._per_step_sympy_validate(sample):
                    weighted_sum += w_exec * (1.0 if ok else 0.0)
                    weight_total += w_exec
                    if not ok:
                        rejection_reasons.append(f"[atomic/step {sid}/sympy] {detail}")
            if sample.domain == "code" and self.enable_code_exec:
                for sid, ok, detail in self._per_step_code_validate(sample):
                    weighted_sum += w_exec * (1.0 if ok else 0.0)
                    weight_total += w_exec
                    if not ok:
                        rejection_reasons.append(f"[atomic/step {sid}/code] {detail}")

        overall = weighted_sum / weight_total if weight_total > 0 else 0.0
        # Clamp to [0, 1]
        overall = max(0.0, min(1.0, overall))
        threshold = _cfg(self.config, "min_confidence_for_accept", 0.85)
        accepted = overall >= threshold

        if _cfg(self.config, "reject_on_any_gap", False):
            if any(not o.passed for o in chain_outcomes + dom_outcomes):
                accepted = False
            if any(not c.passed for sv in step_results for c in sv.checks):
                accepted = False

        if not accepted and not rejection_reasons:
            rejection_reasons.append(
                f"Confidence {overall:.2f} below threshold {threshold:.2f}"
            )

        return VerificationResult(
            accepted=accepted,
            step_results=step_results,
            chain_checks=chain_outcomes + dom_outcomes,
            overall_confidence=overall,
            rejection_reasons=rejection_reasons,
        )

    # ───── chain-level checks ─────

    def _chain_checks(self, sample: TrainingSample) -> list[CheckOutcome]:
        w_struct = self.weights.get("structural", 0.75)
        outcomes: list[CheckOutcome] = []

        has_conc = bool(sample.response and len(sample.response.strip()) >= 3)
        outcomes.append(CheckOutcome(
            "has_conclusion", w_struct, 1.0 if has_conc else 0.0,
            has_conc, "" if has_conc else "missing or trivial conclusion",
        ))

        min_steps = _cfg(self.config, "min_chain_steps", 2)
        n = len(sample.reasoning_chain)
        ok = n >= min_steps
        # Single-step proofs: don't assign 0 confidence, give partial credit
        conf = 1.0 if ok else n / max(min_steps, 1)
        outcomes.append(CheckOutcome(
            "min_steps", w_struct, conf, ok,
            "" if ok else f"only {n} step(s); need {min_steps}",
        ))

        # Conclusion connects to final step
        if sample.reasoning_chain and sample.response:
            last = sample.reasoning_chain[-1]
            last_sig = _significant(f"{last.content} {last.justification}")
            last_nums = set(_RE_NUMBER.findall(last.content + " " + last.justification))
            conc_sig = _significant(sample.response)
            conc_nums = set(_RE_NUMBER.findall(sample.response))
            overlap_w = len(last_sig & conc_sig)
            overlap_n = len(last_nums & conc_nums)
            # For very short conclusions (e.g. just a number), relax word overlap
            connected = overlap_w >= 1 or overlap_n >= 1 or len(conc_sig) <= 1
            # Also check if the conclusion's answer appears anywhere in the chain
            if not connected and conc_nums:
                for step in sample.reasoning_chain:
                    step_nums = set(_RE_NUMBER.findall(step.content))
                    if conc_nums & step_nums:
                        connected = True
                        overlap_n = 1
                        break
            outcomes.append(CheckOutcome(
                "conclusion_connects", self.weights.get("logical_validity", 1.0),
                min(1.0, 0.3 + 0.2 * overlap_w + 0.3 * overlap_n) if connected else 0.2,
                connected,
                "" if connected else "final step shares no terms/numbers with conclusion",
            ))

        # Monotonic progress: steps should not be exact duplicates
        if len(sample.reasoning_chain) >= 2:
            dup_count = 0
            for i in range(1, len(sample.reasoning_chain)):
                prev_c = sample.reasoning_chain[i - 1].content.strip().lower()
                curr_c = sample.reasoning_chain[i].content.strip().lower()
                if prev_c and curr_c and prev_c == curr_c:
                    dup_count += 1
            if dup_count > 0:
                frac = dup_count / (len(sample.reasoning_chain) - 1)
                outcomes.append(CheckOutcome(
                    "no_duplicate_steps", w_struct,
                    max(0.0, 1.0 - frac), dup_count == 0,
                    f"{dup_count} duplicate consecutive step(s)" if dup_count else "",
                ))

        return outcomes

    # ───── step-level checks ─────

    def _verify_step(
        self,
        step: ReasoningStep,
        prior: list[ReasoningStep],
        sample: TrainingSample,
    ) -> StepVerification:
        checks: list[CheckOutcome] = []
        is_first = len(prior) == 0

        # 1. content substance
        w_log = self.weights.get("logical_validity", 1.0)
        content_ok = bool(step.content and len(step.content.strip()) >= 8)
        checks.append(CheckOutcome(
            "content_substance", self.weights.get("structural", 0.75),
            1.0 if content_ok else 0.0, content_ok,
            "" if content_ok else "content empty or trivial",
        ))

        # 2. justification inferential (skip first step)
        if _cfg(self.config, "check_logical_validity", True) and not is_first:
            j = step.justification or ""
            has_j = j and j != "implicit" and len(j.strip()) >= 8
            has_inf = bool(_INFERENCE_MARKERS.search(j)) if has_j else False
            conf = 1.0 if has_inf else (0.5 if has_j else 0.0)
            checks.append(CheckOutcome(
                "justification_inferential", w_log, conf, has_inf,
                "" if has_inf else ("no inferential marker" if has_j else "no justification"),
            ))

            # 2b. non-sequitur: justification must share terms with content
            if has_j:
                j_sig = _significant(j)
                c_sig = _significant(step.content)
                overlap = len(j_sig & c_sig)
                # Also check numeric overlap — math justifications often share
                # numbers rather than words
                j_nums = set(_RE_NUMBER.findall(j))
                c_nums = set(_RE_NUMBER.findall(step.content))
                num_overlap = len(j_nums & c_nums)
                not_sequitur = overlap >= 1 or num_overlap >= 1 or len(j_sig) <= 2
                checks.append(CheckOutcome(
                    "not_non_sequitur", w_log,
                    1.0 if (overlap >= 2 or num_overlap >= 1) else (0.7 if not_sequitur else 0.2),
                    not_sequitur,
                    "" if not_sequitur else "justification shares no terms with content",
                ))

            # 2c. anti-circular: justification must not restate the conclusion
            if has_j:
                circ = self._detect_circular(step.content, j)
                checks.append(CheckOutcome(
                    "non_circular", w_log,
                    0.1 if circ else 1.0,
                    not circ,
                    "justification is a restatement of the conclusion" if circ else "",
                ))

        # 3. step completeness: references prior (non-first)
        if _cfg(self.config, "check_step_completeness", True) and prior:
            refs, ref_conf = self._references_prior(step, prior)
            checks.append(CheckOutcome(
                "references_prior", self.weights.get("step_completeness", 1.0),
                ref_conf, refs,
                "" if refs else "no lexical/symbolic link to any of the last prior steps",
            ))

        # 4. assumption grounding
        if _cfg(self.config, "check_assumption_grounding", True):
            a_ok, a_conf, a_msg = self._assumption_grounding(step, prior, sample, is_first)
            checks.append(CheckOutcome(
                "assumption_grounding",
                self.weights.get("assumption_grounding", 1.0),
                a_conf, a_ok, a_msg,
            ))

        # 5. internal consistency
        ic_ok = self._internal_consistency(step)
        checks.append(CheckOutcome(
            "internal_consistency", self.weights.get("consistency", 1.5),
            1.0 if ic_ok else 0.0, ic_ok,
            "" if ic_ok else "self-contradiction detected within step",
        ))

        # 6. cross-step consistency
        if prior:
            cs_ok = self._cross_step_consistency(step, prior)
            checks.append(CheckOutcome(
                "cross_step_consistency", self.weights.get("consistency", 1.5),
                1.0 if cs_ok else 0.0, cs_ok,
                "" if cs_ok else "contradicts a recent prior step",
            ))

        # 7. per-step math equation check (if sympy available and domain=math)
        if self.enable_sympy and sample.domain == "math":
            math_check = self._math_step_check(step)
            if math_check is not None:
                checks.append(math_check)

        valid = all(c.passed for c in checks)
        total_w = sum(c.weight for c in checks) or 1.0
        conf = sum(c.weight * c.confidence for c in checks) / total_w
        issues = [f"{c.name}: {c.detail}" for c in checks if not c.passed]

        return StepVerification(
            step_number=step.step_number,
            valid=valid,
            confidence=conf,
            checks=checks,
            issues=issues,
        )

    # ───── fine-grained helpers ─────

    @staticmethod
    def _detect_circular(content: str, justification: str) -> bool:
        """Detect if justification is circular (restates the conclusion).

        Uses both substring containment and Jaccard similarity on significant
        words. A high Jaccard with no novel terms in the justification means
        the justification is just rephrasing the content.
        """
        j_norm = re.sub(r"\W+", " ", justification.lower()).strip()
        c_norm = re.sub(r"\W+", " ", content.lower()).strip()
        if not c_norm or not j_norm:
            return False
        # Direct substring containment
        if j_norm in c_norm or c_norm in j_norm:
            return True
        # Jaccard on significant words — if justification adds nothing new
        j_sig = _significant(justification)
        c_sig = _significant(content)
        if not j_sig:
            return False
        jacc = _jaccard(j_sig, c_sig)
        # If >80% overlap AND justification introduces no novel significant terms
        novel = j_sig - c_sig
        if jacc > 0.8 and len(novel) <= 1:
            return True
        return False

    def _references_prior(
        self, step: ReasoningStep, prior: list[ReasoningStep]
    ) -> tuple[bool, float]:
        text = f"{step.content} {step.justification}"
        if _RE_STEP_REF.search(text) or _RE_PRIOR_REF.search(text):
            return True, 1.0

        step_sig = _significant(text)
        step_nums = set(_RE_NUMBER.findall(text))
        step_math = {w for w in _words(text) if _RE_MATH_TOKEN.search(w)}

        best = 0.0
        for p in prior:
            p_text = f"{p.content} {p.justification}"
            p_sig = _significant(p_text)
            p_nums = set(_RE_NUMBER.findall(p_text))
            p_math = {w for w in _words(p_text) if _RE_MATH_TOKEN.search(w)}
            w_overlap = len(step_sig & p_sig)
            n_overlap = len(step_nums & p_nums)
            m_overlap = len(step_math & p_math)
            score = min(1.0, 0.2 * w_overlap + 0.3 * n_overlap + 0.25 * m_overlap)
            if score > best:
                best = score

        return best >= 0.3, best

    def _assumption_grounding(
        self,
        step: ReasoningStep,
        prior: list[ReasoningStep],
        sample: TrainingSample,
        is_first: bool,
    ) -> tuple[bool, float, str]:
        declared = [a for a in (step.assumptions or []) if a and a.lower() not in
                    ("none", "n/a", "implicit", "")]

        if declared:
            content_sig = _significant(f"{step.content} {step.justification}")
            content_words = _words(f"{step.content} {step.justification}")
            used_count = 0
            for a in declared:
                a_sig = _significant(a)
                if a_sig and (a_sig & content_sig):
                    used_count += 1
                elif not a_sig:
                    a_words = _words(a)
                    if a_words & content_words:
                        used_count += 1
            frac = used_count / len(declared)
            if frac >= 0.5:
                return True, 0.5 + 0.5 * frac, ""
            return False, 0.3 * frac, f"{used_count}/{len(declared)} declared assumptions not referenced in step"

        # No declared assumptions.
        if is_first:
            prompt_sig = _significant(sample.prompt)
            prompt_nums = set(_RE_NUMBER.findall(sample.prompt))
            step_sig = _significant(step.content + " " + step.justification)
            step_nums = set(_RE_NUMBER.findall(step.content + " " + step.justification))
            w_overlap = len(prompt_sig & step_sig)
            n_overlap = len(prompt_nums & step_nums)
            if w_overlap >= 2 or n_overlap >= 1:
                return True, min(1.0, 0.4 + 0.15 * w_overlap + 0.2 * n_overlap), ""
            if len(prompt_sig) <= 3 and w_overlap >= 1:
                return True, 0.7, ""
            return False, 0.2, "first step does not ground in the problem prompt"

        refs, conf = self._references_prior(step, prior)
        if refs:
            return True, conf, ""
        return False, 0.3, "no assumptions declared and no prior-step link (possible hidden premise)"

    def _internal_consistency(self, step: ReasoningStep) -> bool:
        text = f"{step.content}. {step.justification}".lower()
        sents = [s.strip() for s in re.split(r"(?<!\d)\.(?!\d)|[!?;]", text) if s.strip()]
        if len(sents) < 2:
            return True
        sent_words = [_words(s) for s in sents]
        for i in range(len(sents)):
            for j in range(i + 1, len(sents)):
                a, b = sent_words[i], sent_words[j]
                for pos, neg in _NEGATION_PAIRS:
                    if pos <= a and neg <= b and len((a - pos) & (b - neg)) >= 3:
                        return False
                    if neg <= a and pos <= b and len((a - neg) & (b - pos)) >= 3:
                        return False
                # Quantifier conflict: "for all X" in one sentence vs
                # "there exists X ... not" in another about the same subject
                if (_QUANTIFIER_ALL.search(sents[i]) and
                        _QUANTIFIER_EXISTS_NOT.search(sents[j])):
                    shared = (a & b) - _STOPWORDS
                    if len(shared) >= 2:
                        return False
                if (_QUANTIFIER_ALL.search(sents[j]) and
                        _QUANTIFIER_EXISTS_NOT.search(sents[i])):
                    shared = (a & b) - _STOPWORDS
                    if len(shared) >= 2:
                        return False

        # Numeric self-contradiction: same variable assigned two different values
        assignments = self._extract_assignments(text)
        seen: dict[str, str] = {}
        for var, val in assignments:
            if var in seen:
                eq = _sympy_equation_valid(seen[var], val)
                if eq is False:
                    return False
            else:
                seen[var] = val

        return True

    def _cross_step_consistency(self, step: ReasoningStep, prior: list[ReasoningStep]) -> bool:
        s_words = _words(step.content + " " + step.justification)
        s_text = (step.content + " " + step.justification).lower()
        for p in prior:
            p_words = _words(p.content + " " + p.justification)
            p_text = (p.content + " " + p.justification).lower()
            for pos, neg in _NEGATION_PAIRS:
                if pos <= p_words and neg <= s_words and len((p_words - pos) & (s_words - neg)) >= 3:
                    return False
                if neg <= p_words and pos <= s_words and len((p_words - neg) & (s_words - pos)) >= 3:
                    return False
            # Quantifier conflicts across steps
            if (_QUANTIFIER_ALL.search(p_text) and _QUANTIFIER_EXISTS_NOT.search(s_text)):
                shared = (p_words & s_words) - _STOPWORDS
                if len(shared) >= 2:
                    return False
            # Numeric contradiction: same variable assigned different values
            for var, val in self._extract_assignments(p.content):
                for var2, val2 in self._extract_assignments(step.content):
                    if var == var2 and val != val2:
                        eq = _sympy_equation_valid(val, val2)
                        if eq is False:
                            return False
        return True

    @staticmethod
    def _extract_assignments(text: str) -> list[tuple[str, str]]:
        out = []
        for m in _RE_EQUATION.finditer(text):
            lhs = m.group(1).strip()
            rhs = m.group(2).strip()
            if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", lhs):
                rhs = re.split(r"[,.;]| and ", rhs, maxsplit=1)[0].strip()
                if rhs:
                    out.append((lhs, rhs))
        return out

    # ───── domain execution ─────

    def _math_step_check(self, step: ReasoningStep) -> Optional[CheckOutcome]:
        w = self.weights.get("domain_exec", 2.0)
        equations = list(_RE_EQUATION.finditer(step.content))
        if not equations:
            return None
        decided = 0
        correct = 0
        for m in equations[:3]:
            lhs = m.group(1).strip()
            rhs = m.group(2).strip()
            rhs = re.split(r"[,.;]| and | so ", rhs, maxsplit=1)[0].strip()
            res = _sympy_equation_valid(lhs, rhs)
            if res is None:
                continue
            decided += 1
            if res:
                correct += 1
        if decided == 0:
            return None
        ok = correct == decided
        return CheckOutcome(
            "math_equation_check", w,
            correct / decided, ok,
            "" if ok else f"{decided - correct}/{decided} equations fail symbolic check",
        )

    def _math_sample_checks(self, sample: TrainingSample, w: float) -> list[CheckOutcome]:
        """Solve equations gathered from prompt + chain and verify conclusion."""
        out: list[CheckOutcome] = []

        # Direct answer equivalence
        expected = getattr(sample, "expected_answer", None)
        if expected and sample.response:
            norm_exp = _normalize_answer(str(expected))
            norm_got = _normalize_answer(sample.response)
            if norm_exp and norm_got:
                eq = _numeric_equiv(norm_exp, norm_got)
                if eq is True:
                    out.append(CheckOutcome(
                        "answer_equivalence", w * 2.0, 1.0, True, "",
                    ))
                elif eq is False:
                    out.append(CheckOutcome(
                        "answer_equivalence", w * 6.0, 0.0, False,
                        f"answer '{norm_got}' != expected '{norm_exp}'",
                    ))

        prompt_eqs = self._gather_equations(sample.prompt)
        chain_eqs: list[tuple[str, str]] = []
        for st in sample.reasoning_chain:
            chain_eqs.extend(self._gather_equations(st.content))
        conclusion_eqs = self._gather_equations(sample.response or "")

        gt = _sympy_solve_single_var(prompt_eqs) if prompt_eqs else {}
        chain_assign = _sympy_solve_single_var(chain_eqs) if chain_eqs else {}
        conc_assign = _sympy_solve_single_var(conclusion_eqs) if conclusion_eqs else {}

        if gt and conc_assign:
            agree = 0
            total = 0
            for var, gt_val in gt.items():
                if var in conc_assign:
                    total += 1
                    try:
                        if sympy.simplify(gt_val - conc_assign[var]) == 0:
                            agree += 1
                    except Exception:
                        pass
            if total:
                frac = agree / total
                ok = frac >= 0.99
                out.append(CheckOutcome(
                    "sympy_conclusion_solves_problem", w if ok else w * 6.0,
                    frac, ok,
                    "" if ok else f"conclusion does not solve the problem's equation(s) ({agree}/{total})",
                ))

        if gt and chain_assign:
            agree = 0
            total = 0
            for var, val in chain_assign.items():
                if var in gt:
                    total += 1
                    try:
                        if sympy.simplify(val - gt[var]) == 0:
                            agree += 1
                    except Exception:
                        pass
            if total:
                frac = agree / total
                ok = frac >= 0.99
                out.append(CheckOutcome(
                    "sympy_chain_consistent", w * 0.75, frac, ok,
                    "" if ok else f"chain derives values inconsistent with the problem ({agree}/{total})",
                ))
        return out

    _gather_equations = staticmethod(_shared_gather_equations)

    def _domain_execution(self, sample: TrainingSample) -> list[CheckOutcome]:
        outcomes: list[CheckOutcome] = []
        w = self.weights.get("domain_exec", 2.0)

        if sample.domain == "math" and self.enable_sympy:
            outcomes.extend(self._math_sample_checks(sample, w))

        if sample.domain == "code" and self.enable_code_exec:
            code = self._extract_code(sample)
            if code:
                tests = self._extract_tests(sample)
                source = code
                if tests:
                    source = code + "\n\n# --- tests ---\n" + "\n".join(tests)
                else:
                    # No explicit tests: try to sanity-execute.
                    # If code prints output, wrap to capture it; if it defines
                    # a function, try calling it with example args from prompt.
                    source = self._build_code_sanity_check(code, sample)
                ok, detail = _run_code_sandboxed(
                    source, self.code_timeout, self.code_memory_mb
                )
                fail_weight = w * 3.0
                outcomes.append(CheckOutcome(
                    "code_sandbox_exec", w if ok else fail_weight,
                    1.0 if ok else 0.0, ok,
                    "" if ok else f"sandbox failed: {detail[:160]}",
                ))
            else:
                outcomes.append(CheckOutcome(
                    "code_sandbox_exec", w * 0.5,
                    0.3, False, "no extractable Python code found in code sample",
                ))
        return outcomes

    @staticmethod
    def _build_code_sanity_check(code: str, sample: TrainingSample) -> str:
        """Build a sanity-check wrapper for code that has no explicit tests.

        Handles two patterns:
        - Code that prints: just run it (a crash = fail).
        - Code that defines a function: try to call it. Look in the prompt
          for example inputs like "f(3) == 9" or "Example: input 5, output 10".
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code
        has_call = any(
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            for node in ast.iter_child_nodes(tree)
        )
        has_print = "print(" in code
        if has_call or has_print:
            return code

        # Find top-level function definitions
        func_names = [
            node.name for node in ast.iter_child_nodes(tree)
            if isinstance(node, ast.FunctionDef)
        ]
        if not func_names:
            return code

        # Try to extract example calls from the prompt
        fn = func_names[0]
        prompt = sample.prompt or ""
        call_pats = re.findall(
            rf"{re.escape(fn)}\s*\(([^)]*)\)\s*(?:==|->|→|returns?|outputs?)\s*(\S+)",
            prompt, re.IGNORECASE,
        )
        lines = [code, ""]
        if call_pats:
            for args_str, expected in call_pats[:3]:
                lines.append(f"_result = {fn}({args_str})")
                lines.append(f"assert str(_result) == str({expected}), "
                             f"f'expected {expected}, got {{_result}}'")
        else:
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.FunctionDef) and node.name == fn:
                    required = len(node.args.args) - len(node.args.defaults)
                    if required == 0:
                        lines.append(f"{fn}()")
                    break
        return "\n".join(lines)

    @staticmethod
    def _extract_code(sample: TrainingSample) -> Optional[str]:
        """Pull Python source out of response / last step."""
        texts = [sample.response or ""]
        if sample.reasoning_chain:
            texts.append(sample.reasoning_chain[-1].content)
        for t in texts:
            # Fenced code block
            m = re.search(r"```(?:python)?\s*\n(.*?)```", t, re.DOTALL)
            if m:
                src = m.group(1).strip()
                if Verifier._is_valid_python(src):
                    return src
            # Raw def/class
            m = re.search(r"((?:def|class)\s+\w+.*)", t, re.DOTALL)
            if m and Verifier._is_valid_python(m.group(1)):
                return m.group(1)
        return None

    @staticmethod
    def _is_valid_python(src: str) -> bool:
        try:
            ast.parse(src)
            return True
        except SyntaxError:
            return False

    @staticmethod
    def _extract_tests(sample: TrainingSample) -> list[str]:
        """Pull 'Test: expr == val' lines from the prompt."""
        out = []
        for m in re.finditer(
            r"(?:^|[\n.;])\s*(?:Test|assert|Example)s?\s*[:]?\s*([^\n.;]+)",
            sample.prompt or "",
            re.IGNORECASE,
        ):
            expr = m.group(1).strip().rstrip(".")
            if "==" in expr or expr.startswith("assert"):
                stmt = expr if expr.startswith("assert") else f"assert {expr}"
                out.append(stmt)
        return out[:10]

    # ───── atomic-step structural verification ─────

    # Rules we recognize. Samples may declare others; we only reject on schema
    # violations (missing field, bad DAG), not on unknown rule strings.
    _ATOMIC_KNOWN_RULES = frozenset({
        "premise", "given",
        "substitute", "simplify", "solve", "factor", "expand",
        "add", "subtract", "multiply", "divide", "rearrange",
        "sympy_check", "numeric_eval", "definition", "theorem", "lemma",
        "modus_ponens", "modus_tollens", "contraposition",
        "and_intro", "and_elim", "or_intro", "or_elim",
        "universal_elim", "universal_intro",
        "existential_intro", "existential_elim",
        "contradiction", "cases",
        "exec", "assert", "state_after", "invariant",
    })

    def _atomic_structure_check(self, sample: TrainingSample) -> list[str]:
        """Strictly validate atomic-step format. Returns list of issues (empty = ok)."""
        issues: list[str] = []
        seen_ids: set[int] = set()
        prev_id = 0
        for idx, st in enumerate(sample.reasoning_chain):
            sid = getattr(st, "step_id", 0) or 0
            rule = (getattr(st, "rule", "") or "").strip()
            deps = list(getattr(st, "depends_on", []) or [])

            if sid <= 0:
                issues.append(f"step #{idx + 1}: missing step_id")
                continue
            if sid in seen_ids:
                issues.append(f"step_id {sid}: duplicate")
            if sid <= prev_id:
                issues.append(f"step_id {sid}: not strictly increasing (prev={prev_id})")
            seen_ids.add(sid)
            prev_id = sid

            if not rule:
                issues.append(f"step_id {sid}: missing rule")
            if not (st.content and st.content.strip()):
                issues.append(f"step_id {sid}: empty claim")

            is_premise = rule in {"premise", "given"} or (deps == [] and idx == 0)
            for d in deps:
                if not isinstance(d, int):
                    issues.append(f"step_id {sid}: non-int dependency {d!r}")
                    continue
                if d == sid:
                    issues.append(f"step_id {sid}: self-dependency")
                elif d not in seen_ids:
                    issues.append(
                        f"step_id {sid}: depends_on {d} which is not a prior step"
                    )
            if not is_premise and not deps:
                issues.append(
                    f"step_id {sid}: non-premise step has no dependencies"
                )

        if sample.reasoning_chain and sample.response:
            final = sample.reasoning_chain[-1]
            f_claim = _normalize_answer(final.content)
            f_resp = _normalize_answer(sample.response)
            if f_claim and f_resp and f_claim != f_resp:
                eq = _symbolic_equiv(f_claim, f_resp) if _HAS_SYMPY else None
                if eq is False:
                    issues.append(
                        f"final conclusion '{f_resp}' != last step claim '{f_claim}'"
                    )
                elif eq is None and f_claim not in f_resp and f_resp not in f_claim:
                    issues.append(
                        f"final conclusion '{f_resp}' not textually linked to last step claim '{f_claim}'"
                    )
        return issues

    def _per_step_sympy_validate(
        self, sample: TrainingSample
    ) -> list[tuple[int, bool, str]]:
        """Per-step equation validation (atomic math). Returns list of (step_id, ok, detail)."""
        out: list[tuple[int, bool, str]] = []
        if not self.enable_sympy:
            return out
        for st in sample.reasoning_chain:
            sid = getattr(st, "step_id", 0) or st.step_number
            rule = (getattr(st, "rule", "") or "").lower()
            if rule in {"premise", "given"}:
                continue
            eqs = _shared_gather_equations(st.content)
            if not eqs:
                # Not every step is an equation — logic rules are fine.
                out.append((sid, True, "no equation to check"))
                continue
            decided = 0
            correct = 0
            failing = ""
            for lhs, rhs in eqs[:3]:
                r = _sympy_equation_valid(lhs, rhs)
                if r is None:
                    continue
                decided += 1
                if r:
                    correct += 1
                elif not failing:
                    failing = f"{lhs} = {rhs}"
            if decided == 0:
                out.append((sid, True, "equation(s) undecidable symbolically"))
            elif correct == decided:
                out.append((sid, True, f"{correct}/{decided} equations pass"))
            else:
                out.append((sid, False, f"{decided - correct}/{decided} fail; e.g. {failing}"))
        return out

    def _per_step_code_validate(
        self, sample: TrainingSample
    ) -> list[tuple[int, bool, str]]:
        """Per-step code assertion/state execution. Returns (step_id, ok, detail) tuples.

        Only validates steps whose rule is one of {exec, assert, state_after}.
        The step's claim is wrapped into a 'assert <claim>' when shaped like
        'after line X, result == Y' or 'x == 1'. Non-code rules are skipped.
        """
        out: list[tuple[int, bool, str]] = []
        if not self.enable_code_exec:
            return out
        code = self._extract_code(sample)
        if not code:
            return out
        for st in sample.reasoning_chain:
            sid = getattr(st, "step_id", 0) or st.step_number
            rule = (getattr(st, "rule", "") or "").lower()
            if rule not in {"exec", "assert", "state_after"}:
                continue
            claim = (st.content or "").strip()
            # Heuristically extract an assertion from the claim
            m = re.search(r"([A-Za-z_]\w*(?:\[[^\]]+\]|\([^)]*\))?)\s*==\s*([^,.;]+)", claim)
            if not m:
                out.append((sid, True, "no assertable expression"))
                continue
            expr = f"{m.group(1).strip()} == {m.group(2).strip()}"
            src = code + f"\n\nassert ({expr}), 'step {sid} assertion failed: {expr}'\n"
            ok, detail = _run_code_sandboxed(src, self.code_timeout, self.code_memory_mb)
            if ok:
                out.append((sid, True, "assertion holds"))
            else:
                out.append((sid, False, (detail or "assertion failed")[:120]))
        return out

    # ───── externalization ─────

    def explain_rejection(self, sample: TrainingSample) -> str:
        """Produce a human-readable trace of which step failed and why.

        Designed so an external verifier (human, other LLM, sympy, unit tests)
        can independently audit the rejection.
        """
        res = self.verify(sample)
        lines: list[str] = []
        lines.append(
            f"VERDICT: {'ACCEPTED' if res.accepted else 'REJECTED'}  "
            f"(confidence={res.overall_confidence:.3f})"
        )
        if res.escalated_to_model:
            lines.append("  [model escalation was applied]")
        lines.append(f"DOMAIN: {sample.domain or '(unspecified)'}")
        lines.append(f"PROMPT: {sample.prompt[:200]}")
        lines.append("")
        lines.append("STEPS:")
        for st, sv in zip(sample.reasoning_chain, res.step_results):
            sid = getattr(st, "step_id", 0) or st.step_number
            rule = getattr(st, "rule", "") or "(no rule)"
            deps = getattr(st, "depends_on", []) or []
            tag = "OK " if sv.valid else "BAD"
            lines.append(
                f"  [{tag}] step {sid} rule={rule} depends_on={deps} "
                f"conf={sv.confidence:.2f}"
            )
            lines.append(f"        claim: {st.content[:160]}")
            for issue in sv.issues:
                lines.append(f"        ! {issue}")
        # Per-step sympy trace (math) for auditor convenience
        if sample.domain == "math" and self.enable_sympy:
            traces = self._per_step_sympy_validate(sample)
            if traces:
                lines.append("")
                lines.append("SYMPY PER-STEP:")
                for sid, ok, detail in traces:
                    lines.append(f"  step {sid}: {'pass' if ok else 'FAIL'} — {detail}")
        if sample.domain == "code" and self.enable_code_exec:
            traces = self._per_step_code_validate(sample)
            if traces:
                lines.append("")
                lines.append("CODE PER-STEP:")
                for sid, ok, detail in traces:
                    lines.append(f"  step {sid}: {'pass' if ok else 'FAIL'} — {detail}")
        if res.rejection_reasons:
            lines.append("")
            lines.append("REJECTION REASONS:")
            for r in res.rejection_reasons[:20]:
                lines.append(f"  - {r}")
        return "\n".join(lines)

    # ───── ground-truth grading ─────

    def grade_against_canonical(
        self,
        sample: TrainingSample,
        canonical_answer: str,
        domain: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Grade a sample against a canonical answer.

        If the sample carries ground_truth_check_type AND a check_type-dispatched
        grader has been injected (via set_ground_truth_grader — typically
        DiagnosticsEngine._check_answer), delegate to it. This keeps diag as the
        single source of truth for rigorous grading. Otherwise fall back to the
        module-level sample-aware grader (sympy / code sandbox / exact).
        """
        ct = getattr(sample, "ground_truth_check_type", "") or ""
        grader = getattr(self, "_gt_grader", None)
        if grader is not None and ct:
            response = sample.response or ""
            if not response and sample.reasoning_chain:
                response = sample.reasoning_chain[-1].content or ""
            try:
                ok = bool(grader(response, canonical_answer, ct))
            except Exception as e:
                return False, f"grader error ({ct}): {type(e).__name__}: {e}"
            return (ok, "" if ok else f"check_type={ct} rejected response")
        return grade_against_canonical(sample, canonical_answer, domain or sample.domain)

    # ───── model escalation ─────

    def _in_escalation_band(self, conf: float) -> bool:
        return self.escalate_above <= conf < self.escalate_below

    def _apply_model_escalation(self, sample: TrainingSample, res: VerificationResult):
        valid, reason = self._model_verify_one(sample)
        res.escalated_to_model = True
        w = self.weights.get("logical_validity", 1.0) * 2.0
        blended = res.overall_confidence + w * (1.0 if valid else 0.0)
        denom_prev = 1.0  # already normalized
        res.overall_confidence = blended / (denom_prev + w)
        threshold = _cfg(self.config, "min_confidence_for_accept", 0.85)
        if valid:
            if res.overall_confidence >= threshold:
                res.accepted = True
        else:
            if self.allow_model_override_reject:
                res.accepted = False
                res.rejection_reasons.append(f"Model: {reason[:200]}")

    def _batch_model_escalation(
        self, results: list[tuple[TrainingSample, VerificationResult]]
    ):
        candidates = [
            (s, r) for s, r in results if self._in_escalation_band(r.overall_confidence)
        ]
        if not candidates:
            return
        prompts = [self._build_verify_prompt(s) for s, _ in candidates]
        try:
            responses: list[str] = self._model_verifier.generate_batch(
                prompts, max_new_tokens=256, temperature=0.0,
            )
            if len(responses) != len(candidates):
                logger.warning("model escalation response count mismatch; skipping")
                return
            for (sample, res), resp in zip(candidates, responses):
                valid, reason = self._parse_model_verdict(resp)
                res.escalated_to_model = True
                w = self.weights.get("logical_validity", 1.0) * 2.0
                res.overall_confidence = (res.overall_confidence + w * (1.0 if valid else 0.0)) / (1.0 + w)
                threshold = _cfg(self.config, "min_confidence_for_accept", 0.85)
                if valid and res.overall_confidence >= threshold:
                    res.accepted = True
                elif not valid and self.allow_model_override_reject:
                    res.accepted = False
                    res.rejection_reasons.append(f"Model: {reason[:200]}")
        except Exception as e:
            logger.warning(f"model escalation failed ({type(e).__name__}): {e}")

    def _model_verify_one(self, sample: TrainingSample) -> tuple[bool, str]:
        resp = self._model_verifier.generate(
            self._build_verify_prompt(sample), max_new_tokens=256, temperature=0.0
        )
        return self._parse_model_verdict(resp)

    @staticmethod
    def _build_verify_prompt(sample: TrainingSample) -> str:
        chain = sample._build_full_response()
        return (
            "<|im_start|>system\n"
            "You are a strict, impartial logic verifier. You check reasoning chains "
            "for correctness. You have no bias toward accepting or rejecting — "
            "report exactly what you find.<|im_end|>\n"
            "<|im_start|>user\n"
            f"PROBLEM:\n{sample.prompt}\n\n"
            f"REASONING CHAIN:\n{chain}\n\n"
            "Verify ALL of the following:\n"
            "1. Does each step follow logically from prior steps and/or the problem premises?\n"
            "2. Are there any contradictions or circular reasoning?\n"
            "3. Does the conclusion follow from the final step?\n"
            "4. Is the final answer correct for the given problem?\n\n"
            "Respond with EXACTLY one of:\n"
            "- VALID\n"
            "- INVALID: <specific reason citing the step number and error>\n\n"
            "Do NOT explain your reasoning. Output only VALID or INVALID: <reason>."
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    @staticmethod
    def _parse_model_verdict(resp: str) -> tuple[bool, str]:
        txt = (resp or "").strip()
        if not txt:
            return False, "no response"
        txt = re.sub(r'<\|im_(?:start|end)\|>.*?\n?', '', txt).strip()
        txt = re.sub(r'^```\w*\s*\n?', '', txt).strip()
        txt = re.sub(r'\n?```\s*$', '', txt).strip()
        normalized = re.sub(r'^[\s*`_#>\-]+', '', txt).upper()
        if not normalized:
            return False, "empty response after cleanup"
        if re.match(r'^VALID\b', normalized) and not normalized.startswith("INVALID"):
            return True, ""
        m = re.match(r'^INVALID\s*[:\-–—]\s*(.*)', normalized, re.DOTALL)
        if m:
            return False, m.group(1).strip()[:200] or "no reason given"
        return False, txt[:200]

    @staticmethod
    def _format_notes(res: VerificationResult) -> str:
        head = f"conf={res.overall_confidence:.2f}"
        if res.escalated_to_model:
            head += " (escalated)"
        if res.rejection_reasons:
            return head + " | " + "; ".join(res.rejection_reasons[:3])
        return head + " | all checks passed"


# ──────────────────────── module-level grading API ────────────────────────

def grade_against_canonical(
    sample: TrainingSample,
    canonical_answer: str,
    domain: str,
) -> tuple[bool, str]:
    """Grade a sample's final answer against a canonical ground-truth answer.

    - math: sympy symbolic_equiv on normalized answers (fallback: numeric).
    - code: treat canonical_answer as assert-style test lines; run them against
            the code extracted from the sample via the sandbox.
    - other: case/whitespace-insensitive exact-match on the normalized final answer.

    Returns (True, "") on success, (False, reason) on failure.
    """
    if sample is None:
        return False, "no sample"
    got = (sample.response or "").strip()
    if not got and sample.reasoning_chain:
        got = (sample.reasoning_chain[-1].content or "").strip()
    if not got:
        return False, "sample has no final answer"

    d = (domain or sample.domain or "").lower()

    if d == "math":
        got_n = _normalize_answer(got)
        can_n = _normalize_answer(canonical_answer or "")
        if not can_n:
            return False, "canonical answer is empty"
        if got_n == can_n:
            return True, ""
        # If both have form "var = expr", compare RHS expressions symbolically.
        def _split_assign(s: str) -> Optional[tuple[str, str]]:
            if s.count("=") == 1 and "==" not in s and "<=" not in s and ">=" not in s:
                lhs, rhs = s.split("=")
                if re.fullmatch(r"\s*[A-Za-z_][A-Za-z_0-9]*\s*", lhs):
                    return lhs.strip(), rhs.strip()
            return None
        g_split = _split_assign(got_n)
        c_split = _split_assign(can_n)
        if g_split and c_split and g_split[0] == c_split[0]:
            got_rhs, can_rhs = g_split[1], c_split[1]
            if _HAS_SYMPY:
                eq = _symbolic_equiv(got_rhs, can_rhs)
                if eq is True:
                    return True, ""
                if eq is False:
                    return False, f"math mismatch: got '{got_n}', expected '{can_n}'"
            if got_rhs == can_rhs:
                return True, ""
            return False, f"math mismatch: got '{got_n}', expected '{can_n}'"
        if _HAS_SYMPY:
            eq = _symbolic_equiv(got_n, can_n)
            if eq is True:
                return True, ""
            if eq is False:
                return False, f"math mismatch: got '{got_n}', expected '{can_n}'"
            eq2 = _numeric_equiv(got_n, can_n)
            if eq2 is True:
                return True, ""
            if eq2 is False:
                return False, f"numeric mismatch: got '{got_n}', expected '{can_n}'"
        # Undecidable symbolically — fall back to string equality
        return (got_n.lower() == can_n.lower(),
                "" if got_n.lower() == can_n.lower()
                else f"math undecidable; strings differ: '{got_n}' vs '{can_n}'")

    if d == "code":
        code = Verifier._extract_code(sample)
        if not code:
            return False, "no extractable code in sample"
        tests = (canonical_answer or "").strip()
        if not tests:
            return False, "canonical answer (tests) is empty"
        # Normalize: each non-empty line that isn't already an assert becomes one
        lines: list[str] = []
        for ln in tests.splitlines():
            s = ln.strip()
            if not s or s.startswith("#"):
                lines.append(ln)
                continue
            if s.startswith("assert ") or s.startswith("assert("):
                lines.append(ln)
            elif "==" in s:
                lines.append(f"assert ({s})")
            else:
                lines.append(ln)
        src = code + "\n\n# --- canonical tests ---\n" + "\n".join(lines) + "\n"
        ok, detail = _run_code_sandboxed(src, 5, 256)
        if ok:
            return True, ""
        return False, f"code failed canonical tests: {(detail or '')[:160]}"

    # Generic domain: exact-match on normalized answer
    g = re.sub(r"\s+", " ", got.strip().lower())
    c = re.sub(r"\s+", " ", (canonical_answer or "").strip().lower())
    if not c:
        return False, "canonical answer is empty"
    if g == c:
        return True, ""
    # Allow substring containment as a softer match
    if c in g or g in c:
        return True, ""
    return False, f"exact-match failed: got '{got[:80]}', expected '{canonical_answer[:80]}'"
