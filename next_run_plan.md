# Next-Run Plan — open-ended RSI on top of commit 24ab247

Author: next_steps_planner (team run2-autopsy)
Date: 2026-04-17
Inputs: run-2 cycle_metrics/ (cycles 1–8), decision_records.jsonl (9 rows, single run), meta_decisions.jsonl, src/ at commit 24ab247.

---

## 1. Predicted behaviour of open-ended RSI on next run

### Baseline trajectory seen in run-2 (before commit 24ab247)

| cycle | pre | post | eval (held-out) |
|------:|----:|----:|----:|
| 1 | 0.413 | 0.350 | 0.089 |
| 2 | 0.375 | 0.538 | 0.363 |
| 3 | 0.450 | 0.750 | **0.867** |
| 4 | 0.663 | 0.663 | 0.859 |
| 5 | 0.700 | 0.700 | 0.859 |
| 6 | 0.275 | 0.438 | 0.474 |
| 7 | 0.350 | 0.413 | null (eval not run) |
| 8 | 0.363 | 0.313 | null (eval not run) |

Interpretation: the jump at cycle 3 is real against the *code* domain bank; the 0.859 plateau at cycles 4-5 is saturation against the fixed bank; the cycle 6 collapse is probable revert or config churn; cycles 7-8 did not produce held-out evaluations.

### With commit 24ab247 in place, prediction for a fresh run

Assumes the same `run.sh` (single domain = code, plateau_patience = 8, heldout_reps = 3), starting from Qwen3-8B base with default `confidence_threshold=0.7` and `difficulty_mix={easy:0.25, medium:0.40, hard:0.25, expert:0.10}`.

- **Cycles 1–3** will mirror run-2 almost exactly — first-weakness fixes on the code bank, held-out score arcs 0.08 → 0.35 → ~0.85.
- **Cycle 4** will still see `diagnostics.weaknesses == []` (saturation). `_should_stop` now raises threshold 0.70 → 0.75 and bumps difficulty mix toward `hard≈0.28, expert≈0.13`, returns False. Held-out probably stays 0.80–0.85 because the new probes are generated at the same moment the threshold changes (no training yet on the harder mix).
- **Cycles 5–6** — first real hard-regime cycles. Expect held-out to **dip** (harder probes, not re-trained) to something like 0.55–0.70, then recover as the generator proposes weaknesses tagged `code/prediction@hard`.
- **Cycles 7–9** — second saturation: threshold bumps 0.75 → 0.80 → 0.85. Held-out drifts back up to ~0.80 on the harder mix. This is the genuine ceiling test.
- **Cycles 10–12** — threshold reaches 0.90 → 0.95. Most likely outcome: the model plateaus here, because the ONLY expert-tier code probe generator is `gen_code_arithmetic` (see §2). Permutation-fair held-out cannot distinguish 0.85 from 0.95 on that narrow distribution.
- **Genuine ceiling: cycle 10–13** for this configuration.

### Numbers to watch

1. `outputs/cycle_metrics/cycle_N.json::scores.eval_mean` trajectory — want monotonic non-decrease across each threshold regime.
2. `difficulty_mix` in config_snapshot — confirm the bump is being applied (check decision_records after each saturation).
3. `confidence_threshold` — should step 0.70 → 0.75 → 0.80 → ...
4. `diagnostics.weaknesses` count — should transition empty → non-empty after each bump.
5. Stop reason: `saturated at confidence_threshold=0.95` means we hit the real ceiling. Anything else is a false termination.

---

## 2. Remaining blockers

### 2a. Curriculum difficulty-mix bump vs expert problem depth

`src/diagnostics/ground_truth.py` contains **0 statically-tagged expert-tier problems**. Breakdown of static `"difficulty"` tags in curated banks:

- `easy`: 9
- `medium`: 8
- `hard`: 0 static
- `expert`: 0 static

All hard/expert problems come from programmatic generators which *accept* a difficulty argument and widen their random ranges. Generator inventory by domain:

- math: 8 generators (arithmetic, linear_system, gcd, modular, combinatorics, coin_probability, derivative, percentage)
- logic: 2 generators (syllogism, truth_table)
- reasoning: 3 generators (sequence, ordering, arithmetic)
- **code: 1 generator only — `gen_code_arithmetic`**

Because run.sh passes `--domains code`, the entire expert-tier curriculum for the next run is a single generator that parametrically widens integer ranges. **That is not sufficient probing surface area for a threshold of 0.95.** The model can over-fit to this one template in a handful of cycles; raising the threshold won't detect it.

Blocker verdict: the difficulty-mix bump is NOT enough. We need either (a) more code generators tagged expert, (b) curated expert-tier code problems, or (c) cross-domain training (code + math + logic + reasoning, which has 14 generators collectively).

**Held-out for code has 13 curated problems** (HumanEval-style). With a 20% holdout fraction that's ~2–3 held-out items — statistically very noisy as the threshold rises. 0.867 in run-2 likely means "13/15 on a tiny bank".

### 2b. LR bandit sufficiency

CausalTracker requires `2 × MIN_PAIRS_FOR_DECISION = 6` usable records, plus `≥3 high / ≥3 low` around the LR median (decision_log.py:87, :176).

After ASK 2 (always-eval) landed, the latest run-2 decision_records still has all 9 entries tagged `tracker=insufficient_data`. The highest `n=4` reported at cycle 9. Why it hasn't crossed 6 usable:

- Cycle 1 has `eval_delta=None` by definition (no prev).
- Cycles 7 and 8 have `eval_score=None` (`holdout not run` — see cycle_metrics) → `eval_delta` also null.
- Net: 9 records, 3 null, 6 usable — but the LR variance hasn't crossed the median in 3/3 split yet, because the bandit pushes LR in the same direction for several cycles before flipping.

Estimate: **bandit needs 2–3 more cycles past cycle 9 with non-null eval_delta** to hit `≥3 high, ≥3 low`. On a clean run that would be ~cycle 11. If cycles 7-8's `eval=None` bug recurs, it'll take ~cycle 14. **Fix the null eval path first (see §4 instrumentation).**

### 2c. Verifier weight + prompt template proposals (null in every decision_record)

Not a tracker limitation — it's a **loop-registration bug**. In `src/orchestrator/meta.py`:

- `self.verifier_weights` starts as `{}` (meta.py:154).
- `self.prompt_variants` starts as `[]` (meta.py:152).
- Both are populated by `add_prompt_variant()` / assigning to `self.verifier_weights`.
- Grep of loop.py shows **zero callsites** that register variants or populate the weights dict.

So meta.py line 328 `if ... and self.verifier_weights:` is always False → no verifier reweight proposal ever.
And meta.py line 343 `paired_effect("generator_template_id")` always returns `insufficient_data` because no cycle ever logged a template_id (and there's nothing to rotate to at line 347 since `self.prompt_variants` is empty).

Fix required in `src/orchestrator/loop.py`: at loop construction, register at least 2 prompt variants (current + one perturbation) and seed `meta.verifier_weights` with the default weights from config.

---

## 3. Recommended next-run configuration

### Restart or resume?

**Recommend restart fresh.** Reasons:

1. Cycle 6's 0.474 collapse and cycles 7-8's null evals indicate a dirty trainer/vLLM state downstream of cycle_3's peak. The commit 24ab247 open-ended RSI logic was never exercised against a clean run.
2. Cycle_3 is only a LoRA adapter checkpoint, not a merged base. Resuming compounds the cycle-4-onwards divergence.
3. We need the saturation handler to fire at its natural point (~cycle 4) for the decision_records to be cleanly comparable; resuming from cycle_3 means cycle 4 of the new run starts already-saturated, triggering the bump before the bandit has any prior-cycle signal. That starves the bandit for the whole saturation regime.

Exception: if GPU time is tight and the user accepts a weaker diagnostic signal, `bash run.sh --resume outputs/checkpoints/cycle_3` is acceptable. Expect ~50% of the full cycle count.

### Exact command

```bash
# Pre-flight (do BEFORE the GPU run):
# 1. Archive run-2 artifacts so the tracker starts empty:
mv outputs outputs_run2_archive_2026-04-17
mkdir -p outputs
# 2. Apply the prompt-variant + verifier-weight registration fix (see §2c).
# 3. (Optional but recommended) patch max_cycles from 100 → 25 to bound the
#    experiment; the saturation handler is what we're validating, not a
#    long-horizon grind.

# Command:
bash run.sh
```

No flag overrides beyond what `run.sh` already bakes in. `--domains code` stays (scope per team-lead); plateau_patience=8 gives the saturation handler room to breathe.

### max_cycles setting

- Current: `orchestrator.max_cycles=100`
- Recommend: **25 for this validation run**. We care about reaching the 0.95-threshold ceiling, not maxing wall-clock. 25 cycles covers: 3 baseline + 6 saturation regimes × ~3 cycles each = well past ceiling.

---

## 4. New instrumentation needed (propose, do not yet implement)

### 4a. Per-problem held-out outcome log

At threshold 0.95, the held-out eval returns a scalar mean (e.g. 0.87) and per-domain means — **but we cannot see WHICH specific expert problems are nailed vs missed**. Propose a new file written each cycle:

```
outputs/cycle_heldout_breakdown/cycle_N.jsonl
# one row per (question_id, repetition) with:
# {question_id, prompt_hash, domain, subdomain, difficulty, is_correct, model_answer, expected}
```

This is cheap (held-out is ~15 problems × 3 reps = 45 rows/cycle). Lives alongside `cycle_samples/`.

### 4b. Decision record: include threshold + difficulty_mix

Currently `config_snapshot` carries `diagnostics.confidence_threshold` as a scalar but not the full `difficulty_mix`. Add `diagnostics.difficulty_mix.{easy,medium,hard,expert}` as flat fields so the CausalTracker can detect "did raising expert weight correlate with eval delta?". This becomes the signal for the NEXT meta-improver upgrade: adapt the mix based on tracker verdict.

### 4c. Eval-skipped reason code

Cycles 7 and 8 had `eval_mean=None` — root cause unknown. Add an `errors` field (already present, always `[]` in run-2) population for `holdout_skipped_reason` when eval is bypassed.

### 4d. Expert-problem-id persistence across cycles

Programmatic generators use a seeded RNG but we don't currently pin a *fixed* expert-held-out set across cycles. Without that, the "nail vs miss" question at 4a is undefined. Propose: serialize a 30-problem expert held-out bank at run start (written to `outputs/expert_holdout_bank.jsonl`) and reuse identically every cycle.

---

## 5. Long-horizon — what "infinitely better" actually requires

Impact ranking (1 = highest):

### 1. Cross-domain expansion (drop `--domains code`)

Impact: very high. Cheapest intervention (one flag).

- Current: 1 generator tagged expert (code_arithmetic). 13 curated problems.
- Full domains: 14 generators and 60+ curated problems spanning math/logic/reasoning/code.
- Also exercises the sub-domain rebalance fix (commit 2928ad0) which is currently dead because there's only one sub-domain.
- Verifier reweighting logic needs multi-domain to work (weights are per-domain).

Recommend this even before the next GPU run if user scope allows it.

### 2. Generative expert-problem pipeline

Impact: high, but later. The ground_truth.py generators cap out because they're integer-range widenings, not structurally harder problems. A true expert pipeline would:

- Compose multiple generators (2-step problem: solve `gen_gcd` then feed to `gen_modular`).
- Use model-generated problems gated by ground-truth execution (HumanEval-style but with a code executor as oracle).
- Potentially use the trainee itself as a problem-generator when it's confident — a form of self-play.

Blocker: need a second independent grader to avoid self-referential eval. Not cheap.

### 3. External benchmark eval (GSM8K, HumanEval, MMLU subsets)

Impact: medium. Useful primarily as a **sanity signal** that in-system scores aren't diverging from real benchmarks. Not load-bearing for the improvement loop — the loop doesn't need HumanEval to generate weaknesses, and mixing external benchmarks into training-signal scoring would leak test data. Recommended as an **audit pass** every N cycles, logged separately, never fed into meta or trainer.

Implementation sketch: lm-eval-harness compatible runner, 100 problems/benchmark, takes ~5 min at vLLM speed. Run every 5 cycles, write to `outputs/external_eval/`.

### Synthesis

For the NEXT run (the one we're planning), #1 (cross-domain) gives the most slack for the open-ended RSI handler to run against. It costs one flag. Do it.

For the run AFTER that, build #2 (generative pipeline) to sustain improvement beyond what the static generators can probe.

Use #3 as an audit track, not a signal track, whenever ready.

---

## Appendix: quick facts

- Commit under test: `24ab247`
- Open-ended RSI entry point: `src/orchestrator/loop.py::ImprovementLoop._should_stop` (lines ~1295-1345 of post-commit file)
- Threshold ceiling: 0.95 (hardcoded)
- Per-saturation threshold step: +0.05
- Per-saturation mix bump: easy -0.05, medium -0.03, hard +0.04, expert +0.04 (then renormalized)
- LR bandit min_pairs: 3 per side, alpha 0.10, 1000 permutations (decision_log.py:87-96)
- Meta verifier-reweight gate: `MIN_CYCLES_FOR_REWEIGHT = 4` (meta.py:134) AND `self.verifier_weights` truthy (currently always {})
- Prompt-variant gate: `self.prompt_variants` non-empty (currently always [])
