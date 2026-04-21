"""Component proposer: the 'pipeline discovers new parts of itself' layer.

Meta-meta-meta. Where meta_meta attributes contribution across a FIXED set of
components, component_proposer emits proposals for entirely NEW components
(new verifier types, new curriculum strategies, new training-data filters,
new reasoning strategies) and decides — via safety-gate + isolated test-bench
+ anchor/diversity/gradient-health deltas — whether to merge.

Pipeline:
  1. `ComponentProposal` is authored (by model, heuristic, or human).
  2. `safety_gate_review` runs static review on the proposal's patch. Any
     proposal that fails is recorded-and-dropped (never runs code).
  3. `run_test_bench` drives N smoke cycles on the mini-eval using a caller-
     supplied isolated runner. We never exec the payload ourselves; the
     runner lives behind a sandbox (worktree / subprocess) owned by
     `src/safety/worktree_sandbox.py`.
  4. `decide_merge` compares mini-eval deltas (anchor, diversity, gradient
     health) against per-metric thresholds. All three must be net-positive
     (or the configured bar) for merge=True.
  5. The decision + full proposal record is appended to a JSONL audit log
     so post-hoc review is always possible. The actual PR-to-orchestrator
     step is OUT OF SCOPE here — we emit a verdict, not a merge.

Nothing in this module imports or executes untrusted code directly. The
runner is injected so tests can drive deterministic behavior and so the
real runtime wires this to the worktree-sandbox.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

from src.safety.review import ReviewResult, review_patch


# Kinds of components the proposer may emit. Keep tight — new kinds require
# a reviewed expansion so the test-bench knows how to exercise them.
ALLOWED_KINDS: frozenset[str] = frozenset({
    "verifier",
    "curriculum",
    "data_filter",
    "reasoning_strategy",
})


# Conservative default allow-list for proposal patches. Matches the spirit of
# meta_meta TIER_0: new components land under generator/ or verifier/ only.
DEFAULT_PROPOSAL_ALLOW_LIST: tuple[str, ...] = (
    "src/generator/",
    "src/verifier/",
)


# Mini-eval thresholds. A merge requires ALL three be met.
DEFAULT_ANCHOR_MIN_DELTA = 0.0
DEFAULT_DIVERSITY_MIN_DELTA = 0.0
DEFAULT_GRADIENT_HEALTH_MIN = 0.0


@dataclass
class ComponentProposal:
    """A proposal for a NEW pipeline component.

    `patch` is a unified diff that, if merged, would land the component.
    `entrypoint` names the symbol the test-bench will exercise (e.g.
    "src.verifier.new_prop:verify"). The test-bench does not import this
    directly; it hands the name to the sandboxed runner.
    """
    name: str
    kind: str
    rationale: str
    patch: str
    entrypoint: str
    proposer: str = "model"
    smoke_cycles: int = 3

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ComponentProposal":
        return cls(
            name=str(d["name"]),
            kind=str(d["kind"]),
            rationale=str(d.get("rationale", "")),
            patch=str(d["patch"]),
            entrypoint=str(d["entrypoint"]),
            proposer=str(d.get("proposer", "model")),
            smoke_cycles=int(d.get("smoke_cycles", 3)),
        )


@dataclass
class BenchResult:
    """Mini-eval deltas from N smoke cycles against a baseline.

    Positive = the proposal improved the metric. gradient_health is a scalar
    in roughly [0, 1] where higher is healthier (low variance, no NaNs).
    `error` is set if the runner failed; in that case deltas are ignored.
    """
    anchor_delta: float = 0.0
    diversity_delta: float = 0.0
    gradient_health: float = 0.0
    cycles_run: int = 0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProposalVerdict:
    proposal: ComponentProposal
    review: ReviewResult
    bench: Optional[BenchResult]
    merge: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "proposal": self.proposal.to_dict(),
            "review_ok": self.review.ok,
            "review_reasons": list(self.review.reasons),
            "bench": self.bench.to_dict() if self.bench else None,
            "merge": self.merge,
            "reasons": list(self.reasons),
        }


# --------------------------------------------------------------------------
# Safety gate
# --------------------------------------------------------------------------


def safety_gate_review(
    proposal: ComponentProposal,
    allow_list: Sequence[str] = DEFAULT_PROPOSAL_ALLOW_LIST,
) -> ReviewResult:
    """Static review. Also rejects unknown kinds and empty entrypoints."""
    if proposal.kind not in ALLOWED_KINDS:
        return ReviewResult(False, [f"unknown component kind: {proposal.kind}"])
    if not proposal.entrypoint or ":" not in proposal.entrypoint:
        return ReviewResult(False, ["entrypoint must be 'module.path:symbol'"])
    if proposal.smoke_cycles <= 0 or proposal.smoke_cycles > 20:
        return ReviewResult(False, [f"smoke_cycles out of range: {proposal.smoke_cycles}"])
    return review_patch(proposal.patch, allow_list)


# --------------------------------------------------------------------------
# Test-bench
# --------------------------------------------------------------------------


# A runner takes (proposal, cycle_index) and returns per-cycle metric deltas
# vs baseline. The runner is responsible for isolation (worktree / sandbox).
# Signature: (proposal, cycle_index) -> dict with keys
# {anchor_delta, diversity_delta, gradient_health}.
BenchRunner = Callable[[ComponentProposal, int], dict]


def run_test_bench(
    proposal: ComponentProposal,
    runner: BenchRunner,
) -> BenchResult:
    """Drive `smoke_cycles` cycles via the injected runner; aggregate.

    The runner is EXPECTED to be sandboxed. Exceptions are caught so a
    misbehaving proposal cannot derail the orchestrator; the error is
    surfaced on the returned BenchResult.
    """
    anchor: list[float] = []
    diversity: list[float] = []
    gradient: list[float] = []
    cycles_run = 0
    try:
        for i in range(proposal.smoke_cycles):
            metrics = runner(proposal, i)
            anchor.append(float(metrics.get("anchor_delta", 0.0)))
            diversity.append(float(metrics.get("diversity_delta", 0.0)))
            gradient.append(float(metrics.get("gradient_health", 0.0)))
            cycles_run += 1
    except Exception as exc:  # runner/sandbox failures land here
        return BenchResult(
            anchor_delta=_mean(anchor),
            diversity_delta=_mean(diversity),
            gradient_health=_mean(gradient),
            cycles_run=cycles_run,
            error=f"{type(exc).__name__}: {exc}",
        )
    return BenchResult(
        anchor_delta=_mean(anchor),
        diversity_delta=_mean(diversity),
        gradient_health=_mean(gradient),
        cycles_run=cycles_run,
        error=None,
    )


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


# --------------------------------------------------------------------------
# Merge decision
# --------------------------------------------------------------------------


def decide_merge(
    bench: BenchResult,
    anchor_min: float = DEFAULT_ANCHOR_MIN_DELTA,
    diversity_min: float = DEFAULT_DIVERSITY_MIN_DELTA,
    gradient_min: float = DEFAULT_GRADIENT_HEALTH_MIN,
) -> tuple[bool, list[str]]:
    """All three metrics must meet their bars and no bench error."""
    reasons: list[str] = []
    if bench.error:
        return False, [f"bench error: {bench.error}"]
    if bench.cycles_run <= 0:
        return False, ["bench ran zero cycles"]
    if bench.anchor_delta < anchor_min:
        reasons.append(f"anchor_delta {bench.anchor_delta:.4f} < {anchor_min}")
    if bench.diversity_delta < diversity_min:
        reasons.append(f"diversity_delta {bench.diversity_delta:.4f} < {diversity_min}")
    if bench.gradient_health < gradient_min:
        reasons.append(f"gradient_health {bench.gradient_health:.4f} < {gradient_min}")
    return (len(reasons) == 0), reasons


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def evaluate_proposal(
    proposal: ComponentProposal,
    runner: BenchRunner,
    allow_list: Sequence[str] = DEFAULT_PROPOSAL_ALLOW_LIST,
    anchor_min: float = DEFAULT_ANCHOR_MIN_DELTA,
    diversity_min: float = DEFAULT_DIVERSITY_MIN_DELTA,
    gradient_min: float = DEFAULT_GRADIENT_HEALTH_MIN,
) -> ProposalVerdict:
    """Full pipeline: safety gate → test-bench → merge verdict."""
    review = safety_gate_review(proposal, allow_list)
    if not review.ok:
        return ProposalVerdict(
            proposal=proposal,
            review=review,
            bench=None,
            merge=False,
            reasons=["blocked by safety gate", *review.reasons],
        )
    bench = run_test_bench(proposal, runner)
    merge, reasons = decide_merge(bench, anchor_min, diversity_min, gradient_min)
    return ProposalVerdict(
        proposal=proposal,
        review=review,
        bench=bench,
        merge=merge,
        reasons=reasons,
    )


def append_verdict_log(path: Path, verdict: ProposalVerdict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(verdict.to_dict()) + "\n")


def load_verdict_log(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def evaluate_batch(
    proposals: Iterable[ComponentProposal],
    runner: BenchRunner,
    log_path: Optional[Path] = None,
    **kwargs,
) -> list[ProposalVerdict]:
    verdicts: list[ProposalVerdict] = []
    for p in proposals:
        v = evaluate_proposal(p, runner, **kwargs)
        verdicts.append(v)
        if log_path is not None:
            append_verdict_log(log_path, v)
    return verdicts
