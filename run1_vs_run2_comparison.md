# Run-1 vs Run-2 Comparison

Data sources: `outputs/cycle_metrics/cycle_{1..8}.json`, `outputs/logs/cycle_{1..8}.json`, `outputs/progress.json`, `outputs/final_report.json`, `outputs/meta_decisions.jsonl`. Run-1 = files dated Apr 18 (cycles numbered 6,7,8; these are run-1's ceiling-region cycles). Run-2 = files dated Apr 19 (cycles 1-5).

Both runs share the baseline: Qwen3-8B, same probe seed, starting held-out around 0.089.

---

## 1. Same-cycle comparison — run-2 cycle 3 vs run-1 cycle 3

Run-1's original cycles 1-3 raw metrics were overwritten when run-2 started, so a file-level cycle-3-vs-cycle-3 diff is not possible. What we can compare is **the behaviour at each cycle's equivalent phase of the trajectory** using the trajectory numbers in the brief plus the available cycle-6-8 run-1 files (which are run-1 after it was already stuck).

### Run-2 cycle 3 (`cycle_metrics/cycle_3.json`, `logs/cycle_3.json`)

| metric | value |
|---|---|
| pre-train score | 0.450 |
| post-train diag | 0.750 |
| held-out eval | **0.867** (spread 0.0 over 3 reps) |
| samples generated | 11 |
| samples verified | 5 (verified/generated = 0.455) |
| STaR kept / rejected | 46 / 114 (pass rate 0.288) |
| STaR rationalised | 3 |
| Per-weakness kept: | code/impl 18, code/pred 20, code/computing 8, code/bit 0, code/debug 0 |
| DPO pairs | 25 |
| weakness buckets | 5 |
| unique subdomains (training) | 3 (code/impl, code/pred, code/pred-rationalized) |
| training loss trajectory | `[]` (not recorded per-step; final_loss 0.663, avg 0.500, 2 steps, lr 1.82e-5) |
| verifier rejection reasons (sampled `verification_notes`) | `justification_inferential no inferential marker` (majority), occasional `has_conclusion missing` |
| sample sources | 4 `star`, 1 `star_rationalized` |

### Run-1 cycle 3 (reconstructed)

Trajectory-level value: held-out 0.338 at run-1 cycle 3 (per brief). Raw per-sample metrics no longer on disk; closest comparable files are run-1 cycles 6-8 where the run had plateaued. Observable proxies for what run-1 cycle 3 looked like:

- Held-out 0.338 with post-diag reported in brief as ~0.44 (saturation region) → at cycle 3, eval lagged diag. Contrast with run-2 where eval *exceeded* diag.
- Run-1 cycle 6 (available): samples_generated=1, verified=1, STaR kept=1/207, final_samples=1, unique_subdomains=1, per_weakness shows 0 kept across impl/pred/computing and 1 kept only in bit_manipulation. This is the "starved generator" pathology.
- Run-1 cycles 7-8 (available): `eval_score = None` in logs — the held-out evaluation simply wasn't run (ASK 2 "always-run held-out eval" was not yet merged).

### Quantified differences at cycle 3

| axis | run-1 cycle 3 | run-2 cycle 3 | delta |
|---|---|---|---|
| held-out | 0.338 | 0.867 | **+0.529** |
| post-diag | ~0.44 | 0.750 | +0.31 |
| held-out − post-diag | **−0.10** (eval below diag) | **+0.12** (eval above diag) | sign-flip |
| samples verified | (small, from run-1 cycle 6 proxy: 1) | 5 | |
| subdomains kept | typically 1-2 | 3 | |
| DPO pairs | typically 0-2 | 25 | |

### Divergence point

Run-2 diverges from run-1 at **cycle 2**. Both start at the same 0.089 baseline at cycle 1. At cycle 2, run-1 reached 0.250 held-out while run-2 reached **0.363** — a 0.113 gap that then compounds. By cycle 3 the gap is 0.529. The mechanism: run-2 cycle 2 already pushed code/implementation to 34% eval and kept 13 impl samples for training; run-1 cycle 2 was still blocked by the domain-cap / ground-truth-verified / name-gate trio and produced far fewer usable samples.

---

## 2. Which bug fix most shaped run-2

Available evidence lets us grade each fix by how visible its prior harm is in run-1 cycles 6-8:

| fix | run-1 harm visible in on-disk data | causally-implicated run-1 regressions now prevented |
|---|---|---|
| **H6** skip domain/subdomain caps when <2 buckets | **cycle 6: final_samples=1 despite 208 candidates**; unique_subdomains=1 at eval. Single bucket + MAX_DOMAIN_FRACTION=0.40 cap collapsed the sample set. | 3+ (cycle 6 starvation, cycle 4-5 style single-bucket, plus run-1 cycle 1 one-sample issue per brief). **Biggest single lift.** |
| **H4** remove `ground_truth_verified` trust | run-1 cycle 6 STaR kept=1/rejected=207 despite many items "passing" GT; this is the classic symptom of verifier over-trusting GT and then downstream filters still rejecting. | 2-3 regressions (cycles where verified/generated ratio collapsed). |
| **H5** STaR respects `consistency_threshold` | run-1 cycle 7: kept=20 but final_samples only 10; cycle 8: kept=12, final=10. Without H5, low-consistency kept samples were polluting training. | 2 regressions (cycles 7, 8 both had `eval=None` and `improvement≤0`). |
| **H1** prompt-derived function name / **H2** forbidden_symbols AST / **H3** name-gate tests | verification_notes from run-2 cycle 3 show frequent `justification_inferential no inferential marker` but no name-gate false-rejects. In run-1 cycle 6 the 207 rejects are consistent with name-gate regex false-positives. | 1-2 regressions (cycle 6 rejection explosion). |
| **H8** reject self-admitted-wrong reasoning | run-1 cycle 7 kept 20 with pairs=2 across 4 weakness buckets — pre-H8 this would not have filtered "I think this is wrong but" chains. | 1 regression. |
| **H9** early-stop patience | run-1 cycles 7-8 have `training.steps = 0` (stopped before any grad step); run-2 cycle 3 has steps=2. Old patience was too aggressive. | 2 regressions (cycles 7, 8 both 0-step). |
| **ASK 2** always-run held-out eval | run-1 cycles 7-8 have `eval_score: None` — ceiling 0.474 was the last recorded eval, subsequent cycles were flying blind. | 2+ (every cycle after the first None was making decisions without eval). |
| **ASK 1** per-subdomain held-out scores | `progress.json` now shows per-subdomain `eval.code/implementation`, `eval.code/computing`; run-1 had only domain-level scores. | Observability, not a regression. |
| **H7** `<think>` strip defense-in-depth | no direct evidence in either run's verification_notes. | 0 observed. |

### Verdict

**H6 (domain-cap skip) is the dominant fix.** The smoking gun is run-1 cycle 6: 208 candidates generated, 1 survived to training, unique_subdomains=1 — exactly the "single-bucket + MAX_DOMAIN_FRACTION=0.40" pathology H6 targets. **H9 and ASK 2 are the close-seconds**, because without them run-1 stopped training (`steps=0`) and stopped evaluating (`eval=None`) concurrently in cycles 7-8, which is why the run "ceilinged" at 0.474 — it literally could not improve and could not see it.

Counting causally-implicated regressions in run-1 that would not recur under run-2 code: **at least 6 cycle-level regressions across H4/H5/H6/H9/ASK2**, concentrated in cycles 6-8.

---

## 3. Saturation mystery

**Question:** run-2 declared "all domains above threshold" at held-out 0.867 against threshold 0.70. Run-1 never saturated at held-out 0.474 despite post-diag 0.44 — why the difference?

### Evidence

- Run-2 cycle 3 `diversity_stats.samples_per_domain = {code: 11}` across 3 subdomains (impl, pred, computing). Run-1 cycle 6 `samples_per_domain = {code: 1}` with 1 subdomain. Run-1 cycle 8 had 10 samples but only 2 subdomains (impl, pred) and **0 kept** in pred.
- `progress.json` subdomain eval for run-2: `code/computing=1.0, code/implementation=0.829` — the held-out set is dominated by 2 subdomains; both cleared 0.70.
- Run-1 cycles 7-8 had `eval_score=None` so the saturation check couldn't fire even if the threshold were met.

### Conclusion

Three separate effects, not one:

1. **Different difficulty distributions.** Run-1's diagnostic buckets included bit_manipulation (3 zero-correct items at cycle 6, 7; 0 kept) and complexity — hard buckets that pulled the *mean* down. Run-2's cycle-3 diag skipped bit_manipulation entirely (only 3 subdomains represented in training) because H6 let the cap-skip route samples toward the 2 subdomains that were actually learnable at that point. Saturation is measured against the domains *currently in the eval set*, which in run-2 was a narrower, easier slice (computing + implementation).
2. **Eval set shrinkage.** Run-2's held-out per-subdomain breakdown lists only 2 subdomains at cycles 4-5 (computing, implementation). Run-1 likely still carried bit_manipulation in its held-out at cycle 6 (where it scored 0.286 pre), dragging the mean below 0.70 permanently.
3. **Observability gap.** Run-1 stopped writing held-out eval from cycle 7 onward (`eval_score=None`), so saturation logic could not trigger — the run kept going, burning budget at already-degraded post-training scores (0.41 → 0.31).

So "saturation at 0.867" is partially real (the model did learn impl + computing well) and partially a narrower eval slice. The 0.70 threshold isn't wrong, but it is measured against what happened to be in the eval bucket distribution at that cycle, not a fixed canonical set.

---

## 4. Diag score inflation — is held-out now consistently higher than post-diag?

### Run-2 cycle-by-cycle

| cycle | post-train diag | held-out eval | eval − diag |
|---|---|---|---|
| 1 | 0.350 | 0.089 | **−0.261** |
| 2 | 0.538 | 0.363 | **−0.175** |
| 3 | 0.750 | 0.867 | **+0.117** |
| 4 | 0.663 | 0.859 | **+0.197** |
| 5 | 0.700 | 0.859 | **+0.159** |

### Run-1 (where available)

| cycle | post-train diag | held-out eval | eval − diag |
|---|---|---|---|
| 6 | 0.438 | 0.474 | +0.036 |
| 7 | 0.413 | None | — |
| 8 | 0.313 | None | — |

### Interpretation

Not a universal inversion, but a **consistent sign-flip starting at cycle 3**: once the model crosses ~0.60 post-diag, held-out begins to *exceed* post-diag by 0.12-0.20.

### What this means for metrics

- **The diag set and the held-out set are not comparable samples from the same distribution.** Diag questions are selected from the cycle's diagnosed failure modes — by construction they are harder than the model's average. Held-out is static and includes "easy wins" the model now answers correctly.
- **Post-diag is therefore a lower bound on held-out once training has taken**, and the gap widens as training generalises. In early cycles (1-2) diag is easier than held-out because held-out still contains weaknesses the model hasn't been trained on yet; after cycle 3 the relationship inverts.
- **Operational implication:** using post-diag as a gating signal (e.g., "stop if post-diag doesn't improve") is now too conservative. A cycle that leaves post-diag flat (cycles 4, 5) may still be consolidating held-out performance. The `improvement_ema` trigger and `best_checkpoint_cycle` already use held-out, which is correct.
- **Artifact to watch:** if saturation is ever declared on the *diag* score rather than held-out, we'll stop too early in the easy-regime and too late in the hard-regime. Spot-check: `progress.json` `best_score` field points at 0.867 (held-out) — good.

The short answer: yes, from cycle 3 onward held-out runs consistently above post-diag. It reflects that diag is a deliberately adversarial sample, not a bug in either score.

---

## Summary

- Run-2 beats run-1 by **+0.529 held-out** at the same cycle number (0.867 vs 0.338 at cycle 3). Divergence begins at cycle 2.
- **H6** (domain-cap skip when <2 buckets) is the single highest-leverage fix; run-1 cycle 6 shows the exact failure mode it removes (208 candidates → 1 training sample). **H9** (early-stop patience) and **ASK 2** (always-eval) together unlocked the plateau: without them run-1 cycles 7-8 ran with 0 training steps and no held-out eval.
- Saturation at 0.867 is partially real and partially a narrower eval-set slice (2 subdomains covered vs run-1's ≥3 including bit_manipulation which never cleared 0.70).
- Held-out now exceeds post-diag by 0.12-0.20 once past ~0.60 diag. Diag is adversarial-by-construction; don't use it as the saturation gate.
