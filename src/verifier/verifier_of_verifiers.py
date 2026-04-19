"""Verifier-of-verifiers: adversarial check that self-proposed properties
actually catch wrong answers.

The core unsolved problem of real RSI: if the model can propose its own
verification properties, what stops it from writing toothless properties
that accept anything? Without a check, the loop degrades to the model
fabricating "tests" it trivially passes, and training data becomes noise.

This module implements the adversarial defense:

  1. Generate N CORRUPTED versions of the reference solution (each a
     mutation that a human would call "wrong" but in a distribution of
     failure modes the model might produce naturally).
  2. Run the proposed properties against every corrupted version.
  3. Measure what fraction of corruptions any property successfully
     flagged as wrong.
  4. If fewer than MIN_KILL_RATE of corruptions get caught by at least
     one property, the property set is TOOTHLESS and the task is rejected.
  5. If a single property catches >UNIVERSAL_RATE of corruptions, it's
     SUSPICIOUSLY UNIVERSAL (might be rejecting valid solutions too) and
     gets flagged for review.

Additional safeguard: properties are stratified by STOCHASTICITY. A
property with stochasticity=0 must give identical results across repeated
runs; we verify this by running it twice on identical inputs. A property
that flips flag under replay is rejected as non-reproducible.

The output is a VerifierTrustReport: properties that earned trust get
a monotonic trust_score in [0,1]; properties that failed get
rejected_reason strings a human can audit.

Why this matters: every previous RSI attempt that relied on self-proposed
verification collapsed into noise because the verification was never
adversarially tested. Task synthesizer's single-wrong-answer check (from
its teammate's module) catches the most egregious cases but misses
systematic blind spots. This module catches them.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ─── data classes ───────────────────────────────────────────────────────


@dataclass
class Corruption:
    """One mutation of the reference solution that SHOULD fail verification."""
    kind: str                    # "off_by_one" | "sign_flip" | "dropped_branch" | ...
    mutated: Any                 # the corrupted solution object
    expected_to_fail: bool = True  # always True for corruptions — set for clarity


@dataclass
class PropertyTrust:
    """Per-property audit after the adversarial sweep."""
    property_name: str
    corruptions_caught: int       # how many corruptions this property flagged
    corruptions_total: int
    kill_rate: float              # corruptions_caught / corruptions_total
    reference_passed: bool        # sanity: property must accept the reference
    stochastic_replay_match: bool # same input → same verdict across reruns
    trust_score: float            # final score ∈ [0, 1]
    independence_class: str = ""  # from spec §2.2 — used by quorum aggregator
    rejected_reason: Optional[str] = None

    @property
    def trusted(self) -> bool:
        return self.rejected_reason is None and self.trust_score >= 0.5


@dataclass
class VerifierTrustReport:
    """Team verdict: are this task's proposed properties strong enough to train on?"""
    task_id: str
    properties: list[PropertyTrust] = field(default_factory=list)
    total_corruptions: int = 0
    corruptions_caught_by_any: int = 0
    aggregate_kill_rate: float = 0.0
    # The pipeline uses `passed` as the single gate. True means the property
    # set as a whole adversarially discriminates between right and wrong
    # solutions well enough to serve as training signal.
    passed: bool = False
    reason: str = ""


# ─── helpers ────────────────────────────────────────────────────────────


def _is_valid_python(src: str) -> bool:
    try:
        ast.parse(src)
        return True
    except (SyntaxError, ValueError):
        return False


def _ast_negate_first_return(src: str) -> str:
    """Rewrite the first `return <expr>` into `return -(<expr>)` via AST."""
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return ""
    mutated = False

    class _Neg(ast.NodeTransformer):
        def visit_Return(self, node):
            nonlocal mutated
            if mutated or node.value is None:
                return node
            mutated = True
            new_val = ast.UnaryOp(op=ast.USub(), operand=node.value)
            ast.copy_location(new_val, node.value)
            return ast.Return(value=new_val)

    new_tree = _Neg().visit(tree)
    if not mutated:
        return ""
    ast.fix_missing_locations(new_tree)
    try:
        return ast.unparse(new_tree)
    except Exception:
        return ""


_BINOP_SWAPS = {
    ast.Add: ast.Sub, ast.Sub: ast.Add,
    ast.Mult: ast.FloorDiv, ast.FloorDiv: ast.Mult,
    ast.Div: ast.Mult,
}


def _ast_swap_first_binop(src: str) -> str:
    """Swap the first arithmetic binop (e.g., + → -, * → //) via AST.

    Targets the common failure mode where the model writes the right shape but
    the wrong operator. Guaranteed valid-syntax output via ast.unparse.
    """
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return ""
    mutated = False

    class _Swap(ast.NodeTransformer):
        def visit_BinOp(self, node):
            nonlocal mutated
            self.generic_visit(node)
            if mutated:
                return node
            repl = _BINOP_SWAPS.get(type(node.op))
            if repl is None:
                return node
            mutated = True
            return ast.BinOp(left=node.left, op=repl(), right=node.right)

    new_tree = _Swap().visit(tree)
    if not mutated:
        return ""
    ast.fix_missing_locations(new_tree)
    try:
        return ast.unparse(new_tree)
    except Exception:
        return ""


def _ast_identity_return(src: str) -> str:
    """Replace the body of the first function with `return <first_arg>`.

    Catches the "right signature, no computation" failure mode. Stronger than
    `return 0` because type-aware properties can't trivially spot it — f(5)
    returns 5, matching the output type signature but not the logic.
    """
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return ""
    mutated = False

    class _Id(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            nonlocal mutated
            if mutated or not node.args.args:
                return node
            mutated = True
            first_arg = node.args.args[0].arg
            node.body = [ast.Return(value=ast.Name(id=first_arg, ctx=ast.Load()))]
            return node

    new_tree = _Id().visit(tree)
    if not mutated:
        return ""
    ast.fix_missing_locations(new_tree)
    try:
        return ast.unparse(new_tree)
    except Exception:
        return ""


def _ast_flip_booleans(src: str) -> str:
    """Flip True↔False and and↔or in the first occurrence.

    Catches boolean-logic inversions that off_by_one/sign_flip miss on code
    that branches on conditions rather than computes arithmetically.
    """
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return ""
    mutated = False

    class _Flip(ast.NodeTransformer):
        def visit_Constant(self, node):
            nonlocal mutated
            if mutated or not isinstance(node.value, bool):
                return node
            mutated = True
            return ast.Constant(value=not node.value)

        def visit_BoolOp(self, node):
            nonlocal mutated
            self.generic_visit(node)
            if mutated:
                return node
            mutated = True
            new_op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
            return ast.BoolOp(op=new_op, values=node.values)

    new_tree = _Flip().visit(tree)
    if not mutated:
        return ""
    ast.fix_missing_locations(new_tree)
    try:
        return ast.unparse(new_tree)
    except Exception:
        return ""


# ─── corruption strategies ──────────────────────────────────────────────


def _corrupt_code(reference: str, rng: random.Random) -> list[Corruption]:
    """Mutate a code string into N plausibly-wrong variants.

    Strategies reflect real failure modes the model produces naturally.
    Each strategy returns a mutation of `reference`; the set is deliberately
    diverse so a property that catches only one kind of error will still
    score low on kill_rate.
    """
    results: list[Corruption] = []

    # Strategy 1: off-by-one — flip comparison operators at a real boundary.
    # No whitespace-only "mutations" (they're no-ops). Prefer ops embedded
    # in expressions over bare substring replacement.
    off_by_one = reference
    for a, b in [("<=", "<"), (">=", ">"), ("<", "<="), (">", ">="),
                 ("range(0,", "range(1,"), ("range(0, ", "range(1, "),
                 (" + 1", ""), (" - 1", "")]:
        if a in off_by_one:
            off_by_one = off_by_one.replace(a, b, 1)
            break
    if off_by_one != reference and _is_valid_python(off_by_one):
        results.append(Corruption(kind="off_by_one", mutated=off_by_one))

    # Strategy 2: sign flip — AST-based negation of the first return value.
    # Falls back to no-op if parsing fails; never emits invalid syntax.
    sign_flipped = _ast_negate_first_return(reference)
    if sign_flipped and sign_flipped != reference:
        results.append(Corruption(kind="sign_flip", mutated=sign_flipped))

    # Strategy 3: dropped branch — delete a random if/elif body line
    lines = reference.split("\n")
    if_indices = [i for i, l in enumerate(lines) if l.strip().startswith(("if ", "elif "))]
    if if_indices and len(lines) > 3:
        drop_idx = rng.choice(if_indices)
        # Delete the line after the if (its body)
        if drop_idx + 1 < len(lines):
            mutated_lines = lines[:drop_idx + 1] + lines[drop_idx + 2:]
            mutated_src = "\n".join(mutated_lines)
            if _is_valid_python(mutated_src):
                results.append(Corruption(
                    kind="dropped_branch",
                    mutated=mutated_src,
                ))

    # Strategy 4: wrong return type — replace every return with `return None`
    # (not just the first — otherwise a multi-branch function still returns
    # the right value from untouched branches and the corruption is invisible).
    lines = reference.split("\n")
    changed = False
    for i, l in enumerate(lines):
        if l.strip().startswith("return ") and l.strip() != "return None":
            indent = len(l) - len(l.lstrip())
            lines[i] = " " * indent + "return None"
            changed = True
    if changed:
        mutated_src = "\n".join(lines)
        if _is_valid_python(mutated_src):
            results.append(Corruption(
                kind="wrong_return_type",
                mutated=mutated_src,
            ))

    # Strategy 5a: arithmetic operator swap (+ ↔ -, * ↔ //, / → *)
    binop_swapped = _ast_swap_first_binop(reference)
    if binop_swapped and binop_swapped != reference:
        results.append(Corruption(kind="operator_swap", mutated=binop_swapped))

    # Strategy 5b: identity return — function body replaced with `return <arg0>`
    identity = _ast_identity_return(reference)
    if identity and identity != reference:
        results.append(Corruption(kind="identity_return", mutated=identity))

    # Strategy 5c: boolean/logic flip — True↔False, and↔or
    bool_flipped = _ast_flip_booleans(reference)
    if bool_flipped and bool_flipped != reference:
        results.append(Corruption(kind="boolean_flip", mutated=bool_flipped))

    # Strategy 6: constant answer — always return 0 / empty list
    if "def " in reference:
        for l in reference.split("\n"):
            if l.strip().startswith("def "):
                indent = len(l) - len(l.lstrip())
                body_indent = " " * (indent + 4)
                stub = l + "\n" + body_indent + "return 0"
                if _is_valid_python(stub):
                    results.append(Corruption(kind="constant_answer", mutated=stub))
                break

    # Final validity filter: only keep syntactically valid corruptions.
    # Invalid Python would make every exec-based property crash — which
    # inflates kill rate regardless of property merit, letting toothless
    # property sets bypass the gate.
    return [c for c in results if _is_valid_python(c.mutated)]


def _corrupt_numeric(reference: Any, rng: random.Random) -> list[Corruption]:
    """Numeric-answer corruptions for math-style tasks."""
    if not isinstance(reference, (int, float)):
        try:
            reference = float(reference)
        except (TypeError, ValueError):
            return []

    return [
        Corruption(kind="off_by_one", mutated=reference + 1),
        Corruption(kind="off_by_one_neg", mutated=reference - 1),
        Corruption(kind="sign_flip", mutated=-reference),
        Corruption(kind="factor_of_two", mutated=reference * 2),
        Corruption(kind="zero", mutated=0),
    ]


def _corrupt_text(reference: str, rng: random.Random) -> list[Corruption]:
    """Text-answer corruptions for reasoning-style tasks.

    The corruptions are deliberately meaning-preserving-looking but
    logically different, to catch shallow text-match verifiers.
    """
    if not isinstance(reference, str):
        return []

    out = []
    # Negate the first assertive word
    words = reference.split()
    if words:
        first = words[0].lower()
        if first in ("yes", "true", "correct"):
            out.append(Corruption(kind="negated", mutated="no " + reference))
        elif first in ("no", "false", "incorrect"):
            out.append(Corruption(kind="negated", mutated="yes " + reference))
        else:
            out.append(Corruption(kind="prefix_negation", mutated="not " + reference))

    # Reverse the sentence
    out.append(Corruption(kind="reversed", mutated=" ".join(words[::-1])))
    # Empty
    out.append(Corruption(kind="empty", mutated=""))
    # Random shuffle
    if len(words) > 3:
        shuffled = words[:]
        rng.shuffle(shuffled)
        out.append(Corruption(kind="shuffled", mutated=" ".join(shuffled)))
    return out


def generate_corruptions(
    reference: Any, domain: str, *, seed: int = 0, max_corruptions: int = 8,
) -> list[Corruption]:
    """Dispatch by domain. Deterministic from `seed`."""
    rng = random.Random(seed)
    if domain == "code" and isinstance(reference, str):
        out = _corrupt_code(reference, rng)
    elif domain in ("math", "numeric") or isinstance(reference, (int, float)):
        out = _corrupt_numeric(reference, rng)
    else:
        out = _corrupt_text(str(reference), rng)
    return out[:max_corruptions]


# ─── the verifier-of-verifiers ──────────────────────────────────────────


# Minimum fraction of corruptions that must be caught by at least one
# property. Below this, the property set is declared toothless.
MIN_KILL_RATE = 0.70

# Above this, a single property is "suspiciously universal" — it's catching
# so much that it likely also rejects valid solutions. Flag for review but
# don't hard-reject (valid universal rules like "must be syntactically
# valid Python" can legitimately score high).
UNIVERSAL_RATE_FLAG = 0.95


def verify_properties_trustworthy(
    task_id: str,
    reference_solution: Any,
    properties: list,  # list[Property] from property_engine — late-import to avoid cycle
    problem_ctx: Any,
    domain: str,
    *,
    seed: int = 0,
    stochastic_replay: bool = True,
) -> VerifierTrustReport:
    """Run the adversarial audit on a set of proposed properties.

    For the properties to earn trust, they must:
      1. Accept the reference solution (sanity).
      2. Collectively reject ≥MIN_KILL_RATE of generated corruptions.
      3. Be deterministic under replay (stochasticity=0 properties only).

    Returns a VerifierTrustReport. `passed=True` means the property set is
    fit to gate a training sample.
    """
    report = VerifierTrustReport(task_id=task_id)

    if not properties:
        report.reason = "no properties provided"
        return report

    # Step 1: generate corruptions
    corruptions = generate_corruptions(
        reference_solution, domain=domain, seed=seed,
    )
    report.total_corruptions = len(corruptions)

    if not corruptions:
        # Domains / reference types we can't corrupt safely — don't pretend
        # we've audited them. Fail closed.
        report.reason = (
            f"no corruption strategies available for domain={domain} "
            f"and reference type {type(reference_solution).__name__}"
        )
        return report

    # Step 2: per-property audit.
    # We track two kill masks:
    #   - `caught_by_trusted`: kills credited only to properties that pass
    #     the reference-accept and replay-determinism gates. This drives the
    #     aggregate verdict — untrusted kills don't count.
    #   - `caught_by_any`: diagnostic only, reported for observability.
    caught_by_trusted = [False] * len(corruptions)
    caught_by_any = [False] * len(corruptions)

    for prop in properties:
        # Sanity: property must accept the reference
        try:
            ref_pass, _ = _run_property(prop, reference_solution, problem_ctx)
        except Exception as e:
            report.properties.append(PropertyTrust(
                property_name=getattr(prop, "name", "<unnamed>"),
                corruptions_caught=0,
                corruptions_total=len(corruptions),
                kill_rate=0.0,
                reference_passed=False,
                stochastic_replay_match=False,
                trust_score=0.0,
                rejected_reason=f"property raised on reference: {type(e).__name__}: {e}",
            ))
            continue

        # Deterministic-replay audit: a property that claims to be
        # deterministic (spec §1.1 `deterministic: bool`) must give the
        # same verdict on two consecutive runs. We fall back to
        # `stochasticity==0.0` for Property dataclasses that don't yet
        # carry the bool field.
        claims_deterministic = (
            getattr(prop, "deterministic", None)
            if hasattr(prop, "deterministic")
            else (getattr(prop, "stochasticity", 0.0) == 0.0)
        )
        replay_match = True
        if stochastic_replay and claims_deterministic:
            try:
                ref_pass_2, _ = _run_property(prop, reference_solution, problem_ctx)
                replay_match = (ref_pass == ref_pass_2)
            except Exception:
                replay_match = False

        # Count corruption kills. An exception on a corrupted input only
        # counts as a kill when the property ACCEPTED the reference — this
        # prevents "universally-crashing" properties from claiming high
        # kill rate. (If the property also crashes on the reference we've
        # already flagged it as untrusted above.)
        caught = 0
        per_prop_kills = [False] * len(corruptions)
        for i, corr in enumerate(corruptions):
            try:
                passed, _ = _run_property(prop, corr.mutated, problem_ctx)
                if not passed:
                    caught += 1
                    per_prop_kills[i] = True
            except Exception:
                if ref_pass:
                    caught += 1
                    per_prop_kills[i] = True

        kill_rate = caught / len(corruptions)
        rejected_reason = None
        if not ref_pass:
            rejected_reason = "rejects the reference solution (would never accept truth)"
        elif not replay_match:
            rejected_reason = "non-reproducible under replay — same input, different verdict"

        # Merge into the two diagnostic masks.
        for i, killed in enumerate(per_prop_kills):
            if killed:
                caught_by_any[i] = True
                if rejected_reason is None:
                    caught_by_trusted[i] = True

        # Trust score: kill_rate × reference_accepted × replay_match, with a
        # soft penalty for suspiciously-universal properties.
        base = kill_rate if ref_pass else 0.0
        base *= (1.0 if replay_match else 0.0)
        if kill_rate > UNIVERSAL_RATE_FLAG:
            base *= 0.8  # flag but don't hard-reject

        report.properties.append(PropertyTrust(
            property_name=getattr(prop, "name", "<unnamed>"),
            corruptions_caught=caught,
            corruptions_total=len(corruptions),
            kill_rate=kill_rate,
            reference_passed=ref_pass,
            stochastic_replay_match=replay_match,
            trust_score=base,
            independence_class=getattr(prop, "independence_class", ""),
            rejected_reason=rejected_reason,
        ))

    # Step 3: aggregate verdict — only trusted-property kills count.
    report.corruptions_caught_by_any = sum(caught_by_any)
    report.aggregate_kill_rate = (
        sum(caught_by_trusted) / max(1, report.total_corruptions)
    )

    trustable = [p for p in report.properties if p.trusted]
    if not trustable:
        report.reason = "no properties earned trust"
        return report

    if report.aggregate_kill_rate < MIN_KILL_RATE:
        report.reason = (
            f"aggregate kill rate {report.aggregate_kill_rate:.2f} < "
            f"MIN_KILL_RATE={MIN_KILL_RATE} — properties are toothless"
        )
        return report

    report.passed = True
    report.reason = (
        f"trusted by {len(trustable)}/{len(report.properties)} properties; "
        f"collective kill rate {report.aggregate_kill_rate:.2f}"
    )
    return report


def _run_property(prop, solution, problem_ctx) -> tuple[bool, str]:
    """Uniform call signature for Property.check_fn — it might be (sol, ctx) or just (sol)."""
    fn = getattr(prop, "check_fn", None)
    if fn is None:
        raise ValueError(f"property {getattr(prop, 'name', '?')} has no check_fn")
    try:
        result = fn(solution, problem_ctx)
    except TypeError as e:
        # Only fall back when the TypeError actually indicates arity mismatch.
        # Bare `except TypeError` masks bugs inside the check_fn.
        msg = str(e)
        if "positional argument" in msg or "takes" in msg or "argument" in msg:
            result = fn(solution)
        else:
            raise
    if isinstance(result, tuple) and len(result) == 2:
        return bool(result[0]), str(result[1])
    return bool(result), ""


def make_task_fingerprint(
    problem_prompt: str, reference_solution: Any, properties: list,
) -> str:
    """Content hash for a proposed task+verification bundle.

    Used by task_synthesizer to dedup and by trainer to trace provenance.
    The hash covers prompt, reference, and property names (NOT their
    closures — check_fn can't be hashed safely).
    """
    parts = [
        problem_prompt,
        repr(reference_solution),
        "|".join(sorted(getattr(p, "name", "") for p in properties)),
    ]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


# ─── quorum aggregator (rsi_design.md §2.1) ─────────────────────────────


@dataclass
class QuorumVerdict:
    """Result of applying the spec's quorum rule to a candidate solution.

    Per rsi_design.md §2.1:
      Accept iff:
        - n ≥ min_properties (default 3)
        - distinct independence_classes ≥ min_classes (default 3)
        - PASS count ≥ ceil(2n/3)
        - FAIL count == 0 (any single FAIL is a veto)
    """
    accepted: bool
    total_properties: int
    pass_count: int
    fail_count: int
    distinct_classes: int
    classes_seen: list[str] = field(default_factory=list)
    reason: str = ""


def quorum_verdict(
    properties_and_results: list[tuple[Any, bool]],
    *,
    min_properties: int = 3,
    min_classes: int = 3,
) -> QuorumVerdict:
    """Apply §2.1 quorum rule to a set of (Property, did_it_pass) pairs.

    This runs AFTER property evaluation on a candidate solution — it does
    not itself run the properties. The caller produces the list by running
    each trusted property against the candidate and collecting pass/fail.

    Any single FAIL is a veto, reflecting the spec's zero-FAIL requirement.
    That's stricter than majority voting; the reasoning is that a property
    that's earned trust AND rejects the candidate has actually found a
    defect — it's evidence the candidate is wrong, and one such piece of
    evidence outweighs many vague "looks ok" passes.
    """
    n = len(properties_and_results)
    if n == 0:
        return QuorumVerdict(
            accepted=False, total_properties=0, pass_count=0,
            fail_count=0, distinct_classes=0,
            reason="no properties provided",
        )

    classes_seen = sorted({
        getattr(p, "independence_class", "") or "<unclassified>"
        for p, _ in properties_and_results
    })
    distinct = len(classes_seen)
    passes = sum(1 for _, ok in properties_and_results if ok)
    fails = n - passes
    import math
    pass_threshold = math.ceil(2 * n / 3)

    reasons = []
    if n < min_properties:
        reasons.append(f"only {n} properties, need ≥{min_properties}")
    if distinct < min_classes:
        reasons.append(
            f"only {distinct} distinct independence classes "
            f"({classes_seen}), need ≥{min_classes}"
        )
    if passes < pass_threshold:
        reasons.append(
            f"{passes}/{n} PASS < threshold ⌈2n/3⌉={pass_threshold}"
        )
    if fails > 0:
        # One FAIL is a veto.
        reasons.append(f"{fails} property FAIL — veto triggered")

    accepted = not reasons
    return QuorumVerdict(
        accepted=accepted,
        total_properties=n,
        pass_count=passes,
        fail_count=fails,
        distinct_classes=distinct,
        classes_seen=classes_seen,
        reason=(
            "accepted by quorum"
            if accepted else
            "; ".join(reasons)
        ),
    )
