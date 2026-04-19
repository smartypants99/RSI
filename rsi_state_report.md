# RSI State Report — n=8 Meta-Analysis

Author: meta_analyst. Date: 2026-04-17. Inputs: `outputs/logs/cycle_{1..8}.json`, `cycle_metrics/*`, `cycle_samples/*`, `meta_decisions.jsonl`, `progress.json`, `final_report.json`, plus prior reports (analysis.md, hypothesis_verdicts.md, correlations.md, patterns_report.md, samples_report.md, training_dynamics.md).

---

## 0. The n=8 scoreboard (authoritative)

| cycle | pre | post | Δ | held-out | trained? | steps | final_loss | kept/gen | samples_used | subdomain(s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.463 | 0.313 | **−0.150** | 0.089 | yes | 1 | 1.18 | 2/4 | 2 | implementation |
| 2 | 0.413 | 0.263 | **−0.150** | 0.089 | yes | 1 | 0.95 | 3/6 | 3 | implementation |
| **3** | 0.263 | 0.438 | **+0.175** | **0.281** | yes | **2** | **0.65** | 5/10 | 5 | impl+prediction |
| 4 | 0.363 | 0.400 | +0.038 | null | **no** (early-stop) | 0 | 0.23 | 4/9 | 4 | — |
| 5 | 0.425 | 0.275 | **−0.150** | 0.141 | yes | 1 | **0.14** | 7/10 | 7 | impl+prediction |
| **6** | 0.275 | 0.438 | **+0.163** | **0.474** | yes | 1 | 0.35 | 1/1 | 1 | **bit_manipulation** |
| 7 | 0.350 | 0.413 | +0.063 | null | **no** | 0 | 0.26 | 7/10 | 7 | — |
| 8 | 0.363 | 0.313 | −0.050 | null | **no** | 0 | 0.23 | 8/10 | 8 | — |

Final held-out measurement is C6 @ 0.474. Orchestrator marks C6 as `best_checkpoint`. `improvement_ema = +0.009` (~flat). Three of last five cycles skipped held-out eval entirely — so the "did it improve" question is answered by C6 alone.

---

## 1. What we KNOW at n=8 that was uncertain at n=3

Walking hypothesis_verdicts.md verdicts, promoted/demoted by n=8 evidence:

### H1 (cycle-2 gain ≥70% noise) — **SUPERSEDED by a stronger claim**
At n=3 the "cycle-2 gain" was 0.089→0.089→0.281 (a C3 jump was misattributed to C2 earlier). At n=8, C1 and C2 held-out are **identical** (0.089 each). The "cycle-2 gain" never existed in the held-out series; it was a pre/post diag artifact. Binomial noise story remains correct; the object it was explaining was mis-identified. **Confidence on noise-floor mechanism: 0.95** (binomial σ at n=45, p=0.2 is 0.060 → 0.089 vs 0.089 is physically identical). Demoted the specific "C2" framing; promoted the general claim.

### H2 (true delta ≤ 0.08 pp per cycle) — **PARTIAL CONFIRM, PARTIAL REFUTE**
Across 8 cycles the cumulative held-out improvement is 0.089→0.474 = **+0.385 pp**, carried entirely by two cycles (C3: +0.192, C6: +0.193 vs their predecessor). Per-cycle average across trained-and-evaluated cycles: +0.077 pp. **H2 holds on average** (median cycle contributes ~0). **H2 is violated in tail:** when a cycle hits a novel subdomain with the right dose, it delivers ~0.19 pp, ~2.5× the H2 ceiling. Confidence updated: **0.55 (median) / 0.1 (tail)**. Reframed: capability deltas are heavy-tailed, not Gaussian.

### H3 (curriculum drift explains replay variance) — **CONFIRMED, promoted 0.80→0.85**
Held-out seeds are fixed; curriculum still mutates across cycles. C1/C2 scoring identically (0.089) on different weights argues the floor is question-set-bounded, not weight-bounded. Mechanism unchanged.

### H4 (class-level overlap between training and held-out) — **CONFIRMED as a design feature, promoted 0.90→0.95**
samples_report.md establishes every training sample is `code.*`, every held-out item is `code.*`. The "overlap" is structural — it's how the system was built. The question is now not *whether* overlap exists but *which subdomain*. C6's win is the cleanest case: one `bit_manipulation` sample produced a held-out jump specifically on base/bitwise held-out items.

### H5 (one lucky subdomain carries each gain) — **CONFIRMED at n=8, promoted 0.5→0.85**
patterns_report.md identifies C3 ∩ C6 moved-wrong-to-right IDs = 3 of 25 (12% overlap) — each winner flips a different cluster. This is the single biggest finding promoted by n=8: **capability gain is subdomain-local, not global**.

### N1 (cycle-2 gain was cycle-1 regression unwinding) — **REFUTED**
C1 and C2 held-out are 0.089 each. No unwinding occurred. Original story was based on a briefing error. Confidence 0.7 → 0.05. Drop.

### N2 (cycle-1 regression from 1-sample × LR 2e-5 × rank-64) — **CANNOT BE TESTED**, the rank-64 claim in the brief was wrong; all 8 cycles used rank=16. The "regression from tiny training set at too-high LR" class of claim is supported by C1/C2/C5 all showing −0.15 diag improvement when trained on pathological sample sets.

### N3 (C3 overfit from 25 steps) — **REFUTED by n=8 data**
Real C3 ran **2 steps**, not 25 (the "25 steps" figure came from a resumed-C3 run that is not the original C3). Original C3 was a success, not an overfit. Confidence 0.80 → 0.05. Drop.

### N4 (held-out and diag are not independent) — **CONFIRMED, 0.85 → 0.9**
All held-out items are in `code` domain, curriculum never samples math/reasoning even at n=8. Still an untested-outside-code system.

### N5 (LR bandit is broken: n_obs=0) — **CONFIRMED at n=8, promoted 0.9 → 0.98**
Reading meta_decisions.jsonl: every cycle reports `tracker=insufficient_data (n=k)` with k ∈ {0,0,1,2,2,3,4,?}. Bandit has made 8 proposals with essentially no feedback. LR wandered 2e-5 → 1.4e-5 → 1.82e-5 → 1.27e-5 → 8.9e-6 → 1.16e-5 → 8.1e-6 → 5.7e-6 — a random walk, not closed-loop. **This is a concrete code bug, not a learning-signal problem.**

### N6 (STaR pass-rate as capability proxy) — **REFUTED at n=8**
C6 passed 0.5% and won; C8 passed 80% and regressed. Pass rate is actually **anti-correlated** with held-out in the trained cycles. Confidence 0.65 → 0.05. Drop as proxy.

### Newly KNOWN at n=8 (not in the n=3 list)

- **K1. Early-stop gate fires pre-training in 3/8 cycles (C4, C7, C8).** All three had STaR kept samples below a threshold that makes the optimizer skip. All three are wasted compute on the same `implementation` bucket — no weight update, cycle contributes nothing. Confidence: 0.99 (direct read from `training.steps`).
- **K2. The system has no held-out measurement for its last 3 cycles.** Not "bad held-out" — absent. Post_diag ≠ held-out; relying on post_diag for "is C7/C8 good?" is treating a bias-known training-proxy as an eval. Confidence: 0.99.
- **K3. Winning regime is "1 sample, 1 step, final_loss 0.3–0.7, novel subdomain."** Derived from the only two wins in 8 tries. Confidence: 0.7 (n=2).
- **K4. Losing regime is "1 step, ≥3 implementation samples, final_loss ≤ 0.2 or ≥0.9."** Derived from C1, C2, C5. Confidence: 0.7.

---

## 2. Ordinal predictors of "will this cycle improve held-out?"

Trained cycles with measured held-out: C1(−), C2(−), C3(+), C5(−), C6(+). Ranking candidate predictors by ordinal concordance with the binary success label:

| rank | signal | C1 | C2 | C3 | C5 | C6 | concordant? |
|---|---|---|---|---|---|---|---|
| 1 | **has novel subdomain in training** | no | no | **yes** (prediction) | no | **yes** (bit_manip) | 5/5 ✓ |
| 2 | **final_loss in [0.3, 0.7]** | 1.18 | 0.95 | **0.65** | 0.14 | **0.35** | 5/5 ✓ |
| 3 | **samples_used ≤ 5** | 2 | 3 | 5 | 7 | 1 | 4/5 (C1,C2 fail) |
| 4 | **avg_chain_length = 3.0** | 4.0 | 3.3 | 3.0 | 3.0 | 3.0 | 4/5 |
| 5 | samples_verified / generated | 0.50 | 0.50 | 0.50 | 0.70 | **1.00** | 3/5 |
| 6 | weaknesses_found | 2 | 3 | 5 | 3 | 5 | 3/5 |
| 7 | learning_rate | 2e-5 | 1.4e-5 | 1.8e-5 | 8.9e-6 | 1.2e-5 | 2/5 |
| 8 | unique_subdomains in training | 2 | 1 | 2 | 2 | 1 | 2/5 |
| 9 | STaR pass rate | 0.16 | 0.12 | 0.20 | 0.16 | 0.005 | **anti** |
| 10 | pre_score | 0.46 | 0.41 | 0.26 | 0.42 | 0.27 | 3/5 (noisy) |

**Top two predictors (novel subdomain + final_loss in sweet spot) jointly classify all 5 trained/evaluated cycles correctly.** Either alone classifies 5/5. This is the **ordinal model to use going forward**: if either is false, do not train — generate more samples first.

The verifier pass rate, which the brief's shipped fixes *explicitly tuned*, is the single worst predictor at n=8. Raising verifier strictness was solving a non-problem.

---

## 3. Expected lift from the shipped fixes (ranked by data-backed impact)

| fix | addresses | n=8 evidence strength | my expected held-out lift next cycle |
|---|---|---|---|
| **Subdomain rebalance** | H5 + K3: winning requires novel subdomain | **very strong** — it's the #1 predictor | **+0.10 to +0.15 pp** if it forces non-impl buckets; **0** if verifier still drops them |
| **Pre-backward early-stop** | K4 (final_loss ≤ 0.2 failures: C5 collapsed to 0.14 and regressed) | **strong** — C5 is the direct failure case | **+0.05 to +0.10 pp** expected by *preventing* a C5-type regression; gain shows up as avoided loss, not new gain |
| **Think-tag stripping** | samples_report #1 (`</think>` leaked in C1, C3, C5 training targets) | **medium** — plausibly corrupts grads; no direct measurement | **+0.02 to +0.05 pp** (cleaner grads); low confidence |
| **Function-name enforcement** | samples_report #2 (`mergesorted` vs `merge_sorted` passes verify) | **medium** — fixes a verifier loophole, but *verifier strictness is predictor #9* (anti-signal) | **−0.02 to +0.03 pp** — may *reduce* sample counts further and trigger more C4/C7/C8-style training skips |
| **Top-k filter** | generator-side quality | **weak** — unclear what was filtered and why at k | **−0.05 to +0.05 pp** — risks the same problem as function-name check |

**Ranked by expected positive impact next cycle:** subdomain rebalance ≫ pre-backward early-stop > tag stripping > top-k filter ≥ function-name enforcement. The last two risk making the C4/C7/C8 early-skip problem worse by rejecting more samples without addressing *which* subdomains they come from.

**Biggest single win available** is subdomain rebalance. C6 demonstrates a single, structurally correct, novel-subdomain sample can do what 30 implementation samples couldn't.

---

## 4. Predicted next-10 trajectory

Baseline (no further code changes, just run): expect median cycle to be C7/C8-like — early-skip, no training, no held-out. Maybe 1 of 10 lucks into another C6-like unlock → end state ~0.50 held-out.

**With the five shipped fixes applied correctly**, modeling each of 10 cycles as an independent draw from the winning regime (K3/K4), with subdomain rebalance forcing ~1 novel subdomain per 2 cycles:

| cycle | predicted held-out | driver | confidence |
|---|---|---|---|
| 9  | 0.48 | first rebalanced cycle, likely same subdomain re-probe; small gain if any | medium |
| 10 | **0.54** | novel subdomain unlocked (e.g. `debugging` or `complexity`) | medium |
| 11 | 0.54 | consolidation / impl refresh, no held-out gain | medium |
| 12 | **0.60** | another novel subdomain (e.g. `computing`) | medium-low |
| 13 | 0.59 | regression risk — early-stop catches it, so ~flat | low |
| 14 | **0.64** | novel `complexity` or `debugging` | low |
| 15 | 0.63 | flat / minor regress | low |
| 16 | **0.67** | diminishing returns on code-only, each unlock smaller | low |
| 17 | 0.67 | plateau starts | low |
| 18 | 0.68 | plateau — code domain saturated | low |

**Expected end-of-cycle-18 held-out: ~0.67** (vs current 0.47), with the trajectory dominated by 4 subdomain-unlock events of diminishing size (+0.06, +0.06, +0.05, +0.03). Between unlocks the held-out is flat-or-noisy.

**Confidence interval on cycle-18 held-out: [0.52, 0.78]** — wide because we have n=2 winning cycles to calibrate from.

**Failure mode** (likely if function-name + top-k fixes drop keep-rate too much): C4/C7/C8-style skips become the norm, new-subdomain rate drops to <1 per 5, and by cycle 18 held-out is still 0.48–0.52. This is roughly 35% probable given current evidence.

---

## 5. Single biggest remaining gap

**It is not another code fix — it is a measurement fix.**

Held-out is skipped in 3 of the last 5 cycles. The orchestrator is gating on cost; the cost should be accepted. **Every cycle must produce a held-out score**, or we are navigating blind and will not detect whether the shipped fixes work for another 5+ cycles.

Secondarily — and the single line of engineering I would spend on a "next cycle" budget — **add a per-subdomain held-out breakdown to `eval_domain_scores`**. Currently `eval_domain_scores = {"code": 0.47}`. The entire "subdomain unlock" story is inferred from `moved_wrong_to_right` question IDs, which is an indirect proxy. With per-subdomain held-out (bit_manipulation: 0.8, implementation: 0.3, computing: 0.1, ...), we could:

1. Directly measure that C6 moved `bit_manipulation` and nothing else, confirming H5 quantitatively.
2. Measure whether the shipped subdomain-rebalance fix actually moves the unlocked subdomain or just re-samples the already-saturated ones.
3. Detect per-subdomain regression (fixing the next C5 before it ships).
4. Retire ~3 of the indirect signals (w2r counting, curriculum-solve-rate mining, STaR bucket accounting) because the direct measurement would dominate.

Engineering cost: extend the held-out evaluator to partition its n=45 items by `item.subdomain` and emit a dict. One file (`src/diagnostics/engine.py` eval path), ≤30 LOC, no new concepts.

**Why this beats every other fix:** every other question is currently rate-limited by held-out measurement quality. The C6 "miracle" is structurally interpretable only because we can read jsonl; the *size* of the miracle is uncertain because held-out n=45 is ~±0.08 at p=0.5. If per-subdomain resolves which 5 items moved, we can pin down whether the gain is +0.19 (real) or noise + a subdomain flip in a 9-item sub-bucket.

---

## Appendix — what the five prior reports should be updated with

- **hypothesis_verdicts.md**: N1, N3, N6 refuted. H5 promoted to 0.85.
- **correlations.md**: STaR pass-rate vs eval flipped sign at n=8 (was monotonic up; now anti-correlated). Recompute.
- **patterns_report.md**: holds up — n=8 confirms the "novel subdomain" framing.
- **samples_report.md**: `</think>` leakage now has shipped mitigation; verification loopholes (function-name) still to be validated post-fix.
- **training_dynamics.md**: the "58× parameter movement" and "25 steps" claims apply only to the resumed-C3 run, not original C3. The core "samples-per-step" rule and "loss floor 0.15" rule are still supported.
