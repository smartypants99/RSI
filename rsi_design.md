# RSI Design Spec — Property-Based Self-Verification for Open-Ended Capability Extension

Author: `architect` (team actual-rsi)
Date: 2026-04-17
Status: v0.2 — two-gate architecture; VoV adopted as admission backend.

## Changelog v0.1 → v0.2

- **Two-gate architecture ratified.** VoV (`verify_properties_trustworthy` in `src/verifier/verifier_of_verifiers.py`) is the ADMISSION gate run at task-synthesis time on the reference_solution. `quorum_verdict` (same module) is the ACCEPTANCE gate run at candidate-verification time on live generator candidates. Both must pass for a sample to reach the training pool.
- **§1.3 non-triviality backend changed.** The 3-fuzzed-perturbations check is superseded by VoV's 8-strategy corruption sweep at admission.
- **§3.2.3 "adversarial property author" demoted to future-extension.** VoV's mechanical corruption sweep covers the same threat model concretely. Model-vs-model adversarial author remains a valid distinct signal and is deferred to a later milestone.
- **Property schema:** `stochasticity: float` dropped in favor of `deterministic: bool`; `required: bool` dropped (quorum handles acceptance). `check_fn` is NOT a Property schema field; property_verifier materializes it lazily from `source/language/entry_point` after sandbox admission and passes it into VoV as a trusted callable.
- **Canonical homes:** §2.1 quorum rule → `quorum_verdict` in VoV module. Property dataclass + admission gates → `src/verifier/property_engine.py` (new, owned by property_verifier).
- **Stricter quorum than v0.1:** current `quorum_verdict` treats every non-PASS as FAIL (effective rule: unanimous-PASS + class-diversity ≥ 3). The ⌈2n/3⌉ ratio in v0.1 is therefore a non-binding upper bound. Intentional — if empirical false-accept rates stay acceptable we keep it; if they're too high we revisit with a tri-state PASS/FAIL/ERROR.
- **§7 decisions ratified:** K=5 adversarial rollouts default. Old-problem retirement: on first training-pool acceptance, retire parent problem from novelty checks; property record keeps parent_problem_hash for traceability.

---

## 0. What problem this spec solves

The current repo (`src/`) implements a closed-world loop: diagnose weaknesses on a fixed benchmark bank (`diagnostics/ground_truth.py`), synthesize samples targeting those weaknesses (`generator/data_generator.py`), verify them heuristically or against canonical answers (`verifier/verifier.py`), and LoRA-train (`trainer/custom_lora.py`) under the orchestration of `orchestrator/loop.py`. Score goes up on the held-out eval. That is **automated fine-tuning on a fixed benchmark**, not RSI.

RSI requires the loop to *extend* capability — to produce training data at (and slightly beyond) the frontier of the current model, verified in the absence of a canonical answer, and for the verifier itself to remain trustworthy as problem difficulty rises. The unsolved core is:

> **How does a model verify its own work in a domain where no ground truth exists?**

This spec's claim: **a problem is trusted when, and only when, its solution satisfies a set of independently-authored, executable properties whose pass-pattern is unlikely to arise from plausible error modes.** No single property is trusted; agreement across independent properties is the verification signal.

The rest of this document operationalizes that claim into schemas, prompts, protocols, and integration points against `src/`.

---

## 1. The Property object — schema

A **property** is an executable assertion over `(problem, candidate_answer)` that either passes, fails, or errors. It is the atom of verification in this system.

### 1.1 Required fields

Every property must be a dict/dataclass with exactly these fields. The validator (`property_verifier` owns implementation) MUST reject any property missing any field or violating the type contract.

```python
@dataclass
class Property:
    # Identity
    property_id: str            # uuid4, assigned at creation
    problem_id: str             # links to the problem this property verifies
    author: str                 # "model:<run_id>" | "seed:<name>" — never empty

    # Semantics
    name: str                   # human-readable, ≤ 80 chars, e.g. "commutativity_under_swap"
    kind: PropertyKind          # enum, see §1.2
    description: str            # one-sentence English statement of what must hold

    # Execution
    language: str               # "python" | "sympy" | "z3" | "unit_test" | "nl_reformulation"
    source: str                 # the actual code/text to run; ≤ 4 kB
    entry_point: str            # function name the harness invokes, e.g. "check"
    timeout_ms: int             # hard wall-clock bound; default 2000, max 10000
    deterministic: bool         # True iff the property has no RNG / wall-clock dependency

    # Contract with harness
    inputs: list[str]           # names of harness-provided inputs, e.g. ["problem", "candidate"]
    returns: str                # "bool" | "bool_with_witness" | "equivalence_class"

    # Epistemology (these are the RSI-critical fields)
    independence_class: str     # §2.2 — used for cross-consistency quorum
    difficulty_floor: float     # in [0,1]; the property MUST fail on a trivially-wrong answer
    falsifier_example: str      # a wrong candidate that, by construction, this property rejects
    confirmer_example: str      # a right candidate that, by construction, this property accepts

    # Provenance
    created_at: float           # unix ts
    parent_problem_hash: str    # sha256 of problem text — binds property to problem immutably
```

### 1.2 PropertyKind enum

```
ALGEBRAIC        — algebraic identity must hold (sympy simplification to 0)
ROUNDTRIP        — f(f⁻¹(x)) == x or encode/decode agreement
POSTCONDITION    — after execution, a stated invariant holds over outputs
UNIT_TEST        — black-box I/O assertions (HumanEval-style)
TYPE_INVARIANT   — static structural claim (shape, dtype, schema)
DIMENSIONAL      — units/dimensions consistent (for physics-flavoured problems)
REFORMULATION    — problem restated under a semantics-preserving transform; same answer expected
MONOTONICITY     — f(x) relates to f(x') as claimed when x relates to x'
CONSERVATION     — a quantity is preserved across a transformation
COUNTEREXAMPLE_SEARCH — bounded search finds no counterexample within budget
```

Adding a kind requires consensus among `property_verifier` + `architect`.

### 1.3 Validity gates — a property is *admissible* only if

1. **Self-test passes.** `run(property, confirmer_example) == PASS` and `run(property, falsifier_example) == FAIL`. Enforced at registration; any property that fails its own self-test is discarded, never stored.
2. **Determinism matches declared value.** If `deterministic=True`, two runs on the same input yield the same verdict. Verified by two-shot re-execution at admission.
3. **Non-triviality.** The property rejects the `falsifier_example` AND rejects at least one of three *automatically-generated* perturbations of the `confirmer_example` (fuzzed by harness). A property that accepts everything is useless.
4. **Bounded cost.** `timeout_ms ≤ 10000`, `source ≤ 4 kB`, no I/O, no network, no imports outside an allow-list owned by `tester`.
5. **Sandbox-clean.** Executed under `utils/sandbox.py`; any escape attempt = permanent blacklist of author-run.

A property that fails any gate is **never** usable for verification and never counted in quorum.

---

## 2. Cross-consistency protocol

### 2.1 The core decision rule

For a problem `P` with candidate answer `a` and property set `{p_1,…,p_n}`, `a` is **accepted as training data** iff ALL of the following hold:

1. `n ≥ 3` admissible properties exist for `P`.
2. Those properties span `≥ 3` distinct `independence_class` values (see §2.2).
3. `≥ ⌈2n/3⌉` properties return PASS, and **zero** return FAIL. (ERROR counts as neither.)
4. No two PASS-ing properties share the same `author` run_id (prevents a buggy author from dominating quorum).
5. The *same-class agreement rate* across the whole session for these classes is `≥ 0.6` on a held-out sanity set (see §2.4). If any class is currently mis-calibrated, its votes are ignored until re-calibration.

A candidate that gets `n < 3` properties or fewer than 3 distinct classes is **held for further property synthesis**, not accepted. A candidate with *any* FAIL is rejected outright.

### 2.2 Independence classes

Independence is the critical axis — three properties that all reduce to "did sympy simplify this" are one property, not three. We enumerate classes so teammates can tag them deterministically:

| Class                  | Intuition                                              | Example                          |
|------------------------|--------------------------------------------------------|----------------------------------|
| `exec.behavioral`      | Run the candidate and observe I/O                      | unit tests                       |
| `algebra.symbolic`     | Manipulate symbolically, check equivalence             | sympy.simplify(a - b) == 0       |
| `smt.logical`          | Encode claim in SMT, check unsat of negation           | z3 assertion                     |
| `structural.static`    | Parse/inspect without executing                        | AST type-invariant check         |
| `transform.semantic`   | Reformulate problem, resolve independently, compare    | NL→NL restatement, re-solve      |
| `roundtrip`            | Encode-decode / forward-backward                       | serialize/deserialize fixpoint   |
| `perturbation.local`   | Small input change → predicted output change           | monotonicity, continuity         |
| `conservation.global`  | A quantity stays invariant                             | energy, probability mass         |
| `search.bounded`       | Enumerative counterexample hunt                        | random/QuickCheck-style fuzzing  |

A property must declare exactly one class. Two properties in the same class vote as one (min-confidence of the pair) when computing quorum size.

### 2.3 Why ⌈2n/3⌉ with zero FAILs

We want low false-accept rate under correlated-error attack (the dominant failure mode for self-verification). A simple majority (n/2+1) is breakable by two correlated properties; ⌈2n/3⌉ with zero-FAIL requires the attacker to (a) fool two-thirds of *class-diverse* properties and (b) avoid producing any class that disagrees. Combined with class-diversity ≥ 3, this is the tightest gate we can state without ground truth.

This is still not proof of correctness. It is a *filter* whose false-accept rate is empirically measured (§6) and whose threshold can be tightened if drift is detected.

### 2.4 Class calibration — continuous sanity check

A small fixed set of `(problem, known-good answer, known-bad answer)` triples — drawn from the existing `diagnostics/ground_truth.py` bank — is re-verified every cycle. For each class, track:

- true-accept rate on known-good
- true-reject rate on known-bad
- error rate (exceptions)

A class whose true-accept drops below 0.6 OR whose false-accept rises above 0.1 is **suspended from quorum** until its authored properties are re-audited.

This is the one place the existing `ground_truth.py` bank has ongoing use: **as a calibration corpus for the property classes themselves**, not as the training signal.

---

## 3. Novel-task synthesis — how the model proposes problems beyond its current capability

### 3.1 The synthesis prompt (owned by `task_synthesizer`)

The problem the prompt must solve: elicit problems that are **harder than what the model currently handles** without letting the model cheat by proposing problems it secretly knows the answer to in advance.

The structural prompt (to be refined by `task_synthesizer`, but adhering to this shape):

```
You will propose a problem and a set of verifying properties for it.

Constraints on the problem:
1. It MUST be describable in ≤ 300 tokens.
2. It MUST be harder than a problem you would confidently solve in one attempt.
   Signal this by stating: "I estimate I would fail this on first attempt with
   probability ≥ 0.3" and naming the reason (longer reasoning chain / unfamiliar
   combination / larger search space / requires a technique you rarely use).
3. It MUST NOT be a paraphrase of a problem you've been trained on. Name the
   closest problem you know and state what concrete axis you are moving along
   (magnitude / combination / domain-transfer / abstraction-level).
4. It MUST have a well-defined solution under standard semantics — no
   "interpret as you like" problems.

Then propose, independently, ≥ 4 candidate properties (see property schema)
that together would verify any correct solution. The properties MUST span
≥ 3 independence_classes. For each property, state the falsifier_example
and confirmer_example explicitly.

Do NOT solve the problem yet. Proposing and solving must be decoupled.
```

### 3.2 The "don't just solve the easy stuff" guard

Three mechanisms together prevent the trivial-self-verification failure mode where the model proposes problems it can already do and claims success:

1. **Pre-commit difficulty estimate, post-hoc checked.** The model declares "I would fail with probability ≥ 0.3." If, across a window of 50 proposed problems, its actual first-attempt solve rate on them is > 0.8, the synthesizer is flagged (calibration broken) and its output is down-weighted until re-calibration. Owned by `orchestrator` via a new `CalibrationLedger`.

2. **Solve-and-propose are separate model calls with separate contexts.** The proposer does not see its own solve attempt; the solver does not see the declared difficulty. Prevents the proposer from hedging toward things it's confident about.

3. **Adversarial property author.** A *second* run of the model (different seed, different system prompt) is asked to propose properties that would *catch a plausible wrong answer* to the problem. If the primary properties and the adversarial properties span the same classes and the same PASS/FAIL pattern on the candidate, quorum is granted. If the adversarial author finds a property the candidate FAILS, the candidate is rejected. This is the single most important anti-collusion mechanism.

### 3.3 Bounded novelty

Unbounded novelty is a failure mode — the model drifts into nonsense. We bound:

- **Distance from prior distribution.** Each proposed problem is embedded (reusing existing embedding in `diagnostics/engine.py`) and its nearest-neighbor distance to the training corpus is logged. Distances > P99 are held for manual review, not auto-accepted, until the session has ≥ 100 verified samples at lower distances.
- **Reachability.** At least one property of class `exec.behavioral` OR `algebra.symbolic` must be present — problems with no executable check are rejected regardless of other properties, because they have no grounding.

---

## 4. Minimum viable RSI iteration — end-to-end

Every artifact named here is something a teammate must produce or consume. Names are load-bearing; don't rename without broadcasting.

```
                    ┌────────────────────────────────────────────────┐
                    │ 0. SESSION STATE                               │
                    │    - CalibrationLedger (per-class accept rates)│
                    │    - PropertyRegistry  (admissible properties) │
                    │    - ProblemRegistry   (proposed problems)     │
                    │    - VerificationLog   (per-candidate verdicts)│
                    └────────────────────────────────────────────────┘
                                         │
      (per iteration — the "RSI tick"; one tick produces 0–N training samples)

   [STEP 1]  task_synthesizer.propose_batch(N)
             → list[ProposedProblem]           # uses §3.1 prompt
             → each has: problem_text, declared_difficulty, nearest_neighbor_dist

   [STEP 2]  task_synthesizer.propose_properties(problem)
             property_verifier.admit(property)   # gates of §1.3
             → writes to PropertyRegistry        # only admissible ones kept

   [STEP 3]  generator.solve(problem, k=K)       # K independent solve attempts
             → list[Candidate]

   [STEP 4]  property_verifier.verify(problem, candidate, properties)
             → VerificationRecord { per-property verdicts, class coverage, quorum }

   [STEP 5]  ADVERSARIAL pass (§3.2.3)
             task_synthesizer.propose_adversarial_properties(problem, candidate)
             property_verifier.admit + verify
             → AdversarialRecord

   [STEP 6]  integrator.decide(VerificationRecord, AdversarialRecord)
             → ACCEPTED     → append to TrainingPool as TrainingSample
             → REJECTED     → log to VerificationLog with reason
             → INCONCLUSIVE → return to STEP 2 for more properties (bounded retries)

   [STEP 7]  orchestrator.calibration_check()   # §2.4
             → suspends classes whose rates drifted
             → updates CalibrationLedger

   [STEP 8]  trainer.train_if_ready(TrainingPool)
             → triggers when pool ≥ min_batch AND calibration is green
             → LoRA update, then new model_loader revision
             → tick completes; session state persists
```

### 4.1 Intermediate artifacts — named file/table contracts

| Artifact              | Producer          | Consumer(s)                    | Location (proposed)                 |
|-----------------------|-------------------|--------------------------------|-------------------------------------|
| `ProposedProblem`     | task_synthesizer  | property_verifier, generator   | `outputs/problems/<sid>.jsonl`      |
| `Property`            | task_synthesizer  | property_verifier              | `outputs/properties/<sid>.jsonl`    |
| `Candidate`           | generator         | property_verifier              | `outputs/candidates/<sid>.jsonl`    |
| `VerificationRecord`  | property_verifier | integrator, tester             | `outputs/verifications/<sid>.jsonl` |
| `AdversarialRecord`   | property_verifier | integrator                     | same file, tagged `adversarial=true`|
| `CalibrationLedger`   | orchestrator      | property_verifier, integrator  | `outputs/calibration.jsonl`         |
| `TrainingPool`        | integrator        | trainer                        | `outputs/training_pool/<sid>.jsonl` |

`<sid>` = session id (the existing orchestrator run id).

---

## 5. Integration points in existing `src/` — where this plugs in

Do not modify these files yet; these are the interfaces that will change. Broadcast before touching any of them.

### 5.1 `src/verifier/verifier.py` — `Verifier` class (~line 206)

Current: single heuristic/ground-truth check path returning `VerificationResult`.

Change: `Verifier` becomes a thin dispatch layer over a new `PropertyVerifier` component (owned by `property_verifier` teammate). The `VerificationResult` dataclass (line 76) gets a new optional field `property_verdicts: list[PropertyVerdict]` — but is otherwise preserved so the orchestrator keeps working during migration.

Touch point: `Verifier.verify(...)` gains a parameter `properties: list[Property] | None = None`. When non-None, it runs the property path; when None, it falls back to current behavior.

### 5.2 `src/generator/data_generator.py` — `DataGenerator` class (line 610)

Current: generates `TrainingSample` with reasoning chains targeted at weaknesses pulled from diagnostics.

Change: add a sibling method `propose_novel(...)` that issues the §3.1 prompt and returns `ProposedProblem`, bypassing the weakness-targeted path. `TrainingSample` (line 66) gains `source = "rsi_property"` and a new optional field `verification_record_id: str`.

Touch point: the existing `set_grader(...)` hook (called from `orchestrator/loop.py:167`) is replaced by a `set_property_verifier(...)` injection for the novel path. The STaR path is preserved unchanged.

### 5.3 `src/orchestrator/loop.py` — `ImprovementLoop` class (line 139)

Current: per-cycle phases are `diagnose → generate → verify → train → eval`.

Change: a new `rsi_tick` method implementing §4. The existing `run_cycle` stays for closed-world mode; `rsi_tick` is orthogonal and runs when `config.mode == "rsi"`. The `CycleResult` dataclass (line 60) gains `novel_problems_proposed`, `properties_admitted`, `candidates_accepted`, `candidates_rejected_by_adversarial`, `classes_suspended`.

Touch point: `__init__` must construct `PropertyRegistry`, `CalibrationLedger`, and the adversarial synthesizer reference. No existing constructor call sites need to change (the new state is conditional on mode).

### 5.4 `src/diagnostics/ground_truth.py` — `GroundTruthQuestion` (line 32)

Current: benchmark questions with canonical answers, used for diagnosis and STaR filtering.

Change: **reinterpreted as calibration corpus only** (§2.4). No schema change. A new function `calibration_triples() -> list[CalibrationTriple]` converts a sampled subset into `(problem, known_good_answer, known_bad_answer)` triples for class sanity checks. Ground-truth grading (line 1302 `grade_ground_truth`) becomes read-only for calibration; it is no longer a gate on accepted training samples in RSI mode.

### 5.5 `src/utils/sandbox.py`

Current: sandboxed Python execution for code-check graders.

Change: **most critical hardening target**. The property-executor calls sandbox hundreds of times per tick with untrusted code authored by the model itself. `tester` owns:
- CPU/wall limits enforced on every property run, not just best-effort.
- AST allow-list expansion with explicit deny of `__import__`, `open`, `eval`, `exec`, `compile`, attribute access to `__builtins__`, `__globals__`, `__class__`.
- Memory bound (resource.RLIMIT_AS).
- No filesystem, no sockets.

This is non-negotiable: a sandbox escape here turns RSI into arbitrary code execution for every training tick.

---

## 6. How we know it's working (and not degenerate)

The loop can look healthy while actually degenerating (model learns to produce outputs that pass its own properties vacuously). Metrics to compute every tick:

1. **Held-out canonical score.** The existing eval in `orchestrator/loop.py` must not drop. If property-verified training makes the held-out score go DOWN, we are learning artifacts.
2. **Cross-seed consistency.** On the same proposed problem, do two independently-seeded model runs produce candidates that BOTH pass the property set? If only one does, the problem is ambiguous or the properties are too loose.
3. **Class suspension rate.** If more than 2 classes per tick suspend, quorum is collapsing; pause accepting.
4. **Calibration accept/reject rates** per §2.4, plotted per cycle.
5. **Novelty vs acceptance correlation.** Acceptance rate should decrease smoothly with nearest-neighbor distance. A spike of acceptances at high novelty is a red flag (model found a property-passing degeneracy).
6. **Adversarial rejection rate.** Should be nonzero and stable. Zero = adversary is a pushover. >80% = primary synthesis is broken.

Any of these crossing thresholds halts training. The orchestrator must implement these as hard gates, not advisory warnings.

---

## 7. Open questions for teammates

These are things this spec deliberately does NOT decide — raise with `architect` before committing an implementation direction.

- **Property language beyond Python/sympy/z3.** Natural-language reformulation properties (class `transform.semantic`) need a sub-verifier. Who owns it? Probably `property_verifier` + a small model call, but the failure modes need study.
- **K for rollout diversity.** Step 3 samples K candidates per problem. Larger K finds more property-passing candidates but costs compute. Start K=4, tune empirically.
- **Adversarial budget.** How many adversarial properties per candidate? Start at 2, observe rejection rate, tune.
- **When to retire old problems.** A proposed problem accepted at cycle 5 may be trivial by cycle 50. Do we re-verify old training data? Probably yes, at a sampling rate; needs policy.
- **Multi-turn problems.** This spec assumes single-turn problem/answer. Multi-turn (agent trajectories) needs a separate extension.

---

## 8. Runbook — order of landing

This is the sequence agents must ship components in. Earlier items block later ones; parallel items are explicit.

```
PHASE A — Foundations (blockers for everyone)
    A1. architect:      This spec (§1–§5 stable). [DONE on commit]
    A2. property_verifier: Property dataclass + admission gates (§1.3).
                           CalibrationTriple dataclass.
    A3. tester:         Hardened sandbox (§5.5). BLOCKS A2 integration.
    A4. integrator:     Registries (PropertyRegistry, ProblemRegistry,
                           VerificationLog, CalibrationLedger) as append-only
                           jsonl stores with session-scoped files.

PHASE B — Synthesis and verification (can parallelize after A)
    B1. task_synthesizer: Problem-proposal prompt + ProposedProblem schema
                           + adversarial-properties prompt (§3).
    B2. property_verifier: verify(problem, candidate, properties) producing
                           VerificationRecord (§4, step 4); quorum logic (§2).
    B3. integrator:        decide(...) combining primary + adversarial (§4, step 6).

    *** CHECKPOINT 1: team-lead runs smoke test — one problem in, one verdict
        out, all artifacts present. No training yet. ***

PHASE C — Calibration and safety
    C1. property_verifier + integrator: Class calibration loop (§2.4) using
                                         diagnostics/ground_truth.py as corpus.
    C2. task_synthesizer:  Difficulty-calibration ledger (§3.2.1).
    C3. tester:            Red-team the sandbox with a small set of known
                           escape patterns; any escape = stop.

    *** CHECKPOINT 2: calibration green, adversarial rejection rate in
        [0.1, 0.6] on a rehearsal set. ***

PHASE D — Loop wiring (single-agent sequential; don't parallelize)
    D1. integrator:   Glue §4 end-to-end — rsi_tick() in orchestrator/loop.py.
    D2. architect:    Read D1 diff, sign off that artifact names match spec.
    D3. tester:       Run one full tick with training disabled; inspect every
                       artifact file for schema conformance.

    *** CHECKPOINT 3: one dry-run tick passes end-to-end. ***

PHASE E — Training integration
    E1. integrator:   TrainingPool → trainer.train_if_ready hookup.
    E2. tester:       First real tick with training. Monitor §6 metrics.
    E3. architect:    Review §6 dashboard after 10 ticks; decide whether to
                       loosen/tighten quorum threshold based on data.
```

### Coordination checkpoints — who pings whom

- `property_verifier` pings `architect` when Property schema is ratified in code (not before — no freelancing).
- `task_synthesizer` pings `property_verifier` before first proposal test — they must agree on field names in `ProposedProblem`.
- `integrator` broadcasts after each Registry is landed (schema change for everyone).
- `tester` pings `property_verifier` AND `architect` on ANY sandbox finding.
- team-lead calls each of the 3 checkpoints; no one advances past a checkpoint without explicit team-lead ack.

---

## 9. What this spec refuses to do

- It does not prove correctness. Property-based cross-consistency is a filter, not a proof. False accepts are expected; the discipline is measuring their rate (§6) and tightening gates when drift appears.
- It does not promise compute efficiency. Three properties × adversarial pass × K candidates is at least 4× the current verifier cost per sample.
- It does not eliminate the need for a small calibration corpus. `ground_truth.py` stays, repurposed (§2.4). A fully ground-truth-free system is a later milestone, not this one.
- It does not dictate the trainer. Trainer integration is Phase E and reuses the existing LoRA path; nothing in this spec requires a trainer change beyond a new sample source tag.

---

End of spec. Broadcast on commit; treat v0.1 as the stable reference until a labeled v0.2 ships.
