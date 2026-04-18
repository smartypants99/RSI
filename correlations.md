# Pairwise Correlation Candidates — cycles 1, 2, 3

**Hard caveat up front:** n=3. No Pearson r is stable at n=3 — any two points define a line and a third either lies on it or not. Every "correlation" below is a directional sketch, not inference. Rank is by *plausibility given known mechanism*, not by r.

Abbreviations: C1 / C2 / C3 refer to the three cycles. "Original" = C1, C2. "Resumed" = C3 (lora_rank 64→16, num_epochs 3→5, grad_accum 16→1).

---

## Configuration confounds — read these first

Between C2 and C3 **four** things change at once: starting weights (base→C1-ckpt→C2-ckpt), lora_rank (64→16), num_epochs (3→5), grad_accum (16→1), plus curriculum state and bandit-picked LR. Any C2→C3 correlation is overdetermined. Only within-original (C1 vs C2) changes are approximately clean.

Cycle-3 has **25 steps** vs 1 step for C1/C2. That single change (grad_accum 16→1 with more verified samples) dominates training-intensity signals below.

---

## HIGH plausibility (mechanism exists in code, and data points line up)

### 0. `samples_per_optimizer_step` ↔ `improvement` (STRONGEST — from team-lead's training_dynamics.md)
- C1: 1 / 1  = **1.0 samples/step** → improvement −0.325
- C2: 5 / 1  = **5.0 samples/step** → improvement **+0.150** (only positive cycle)
- C3: 9 / 25 = **0.36 samples/step** (each sample traversed 2.8× — memorization regime) → improvement −0.125
- Monotonic in the "goldilocks" sense: too few samples/step (C1) or too many step-traversals (C3) both lose; C2 was in the safe band.
- Team-lead decomposes C3's effective parameter movement vs C2 as **~58× larger** (rsLoRA α/√r doubled 2.0× × LR 1.16× × steps 25×). That single scalar collapses 3 of the 4 confounded knobs (rank, epochs, grad_accum) into a physical quantity.
- Proposed guards: samples/step ≥ 3, early-stop at final_loss < 0.15 absolute (C3 hit 0.044 → memorized).
- **Verdict: highest-plausibility single-scalar explanation of the C2→C3 regression.** Supersedes my earlier "uncontrolled 4-variable swap" framing.

### 1. `steps × samples_verified` ↔ `final_loss` (strong negative, high plausibility)
- C1: 1 step,  1 verified → final_loss 1.120
- C2: 1 step,  5 verified → final_loss 0.638
- C3: 25 steps, 9 verified → final_loss 0.044
- Pearson(steps, final_loss) ≈ −0.95. Plausible — more optimizer updates on a tiny sample pool drives loss toward zero. This is mechanical, not a capability signal.
- **Verdict: plausible but tautological.** `final_loss/avg_loss` ratio drops 1.0 → 0.86 → 0.30, confirming within-cycle convergence grows with step count. Meaningless as a capability proxy.

### 2. `final_loss` ↔ `eval_score` (negative, high plausibility *but misleading*)
- C1: loss 1.120, eval 0.0625
- C2: loss 0.638, eval 0.250
- C3: loss 0.044, eval 0.338
- Monotonic. r ≈ −0.93. Looks like "lower loss → better held-out."
- But C3's loss is 14× lower than C2's and eval only grew by +0.088. Diminishing (or disappearing) return. And the analyzer's claim in `analysis.md` — that C2→C3 eval gain is within binomial noise — would make this correlation largely noise.
- **Verdict: directional match is real; the *magnitude* relationship is weak and likely non-causal beyond C1→C2.**

### 3. `verified / generated` (STaR pass-rate) ↔ `eval_score` (positive, high plausibility)
- C1: 1/3  = 0.33, eval 0.0625
- C2: 5/14 = 0.36, eval 0.250
- C3: 9/10 = 0.90, eval 0.338
- Monotonic. Pass-rate jumping to 0.9 on C3 coincides with highest eval.
- Confound: C3 uses resumed-from-C2 model (already better), so its generations pass its own verifier more easily. Self-consistency amplifies as capability grows.
- **Verdict: plausible positive link; cannot separate "better model → higher pass-rate" from "higher pass-rate → better training data → better model."**

### 4. `train_time` ↔ `eval_score` (positive, high plausibility)
- C1: 24.5 s,  eval 0.0625
- C2: 101.4 s, eval 0.250
- C3: 294.5 s, eval 0.338
- Monotonic. r ≈ +0.97.
- Explained trivially by step count (see #1). Not independent of #1.
- **Verdict: plausible but redundant with #1.**

### 5. `samples_verified` ↔ `eval_score` (positive, high plausibility)
- C1: 1,  eval 0.0625
- C2: 5,  eval 0.250
- C3: 9,  eval 0.338
- Monotonic. r ≈ +0.98.
- More data → more training signal. Standard ML intuition.
- **Verdict: plausible.** Probably the most defensible positive correlation in this dataset.

---

## MEDIUM plausibility (directionally consistent, but mechanism weak or confounded)

### 6. `pre_score` ↔ `improvement` (strong negative, plausible as regression-to-mean)
- C1: pre 0.4625, improvement −0.325
- C2: pre 0.075,  improvement +0.150
- C3: pre 0.3375, improvement −0.125
- r ≈ −0.99 across all three.
- Mechanism: diagnostic questions are re-sampled per cycle; when pre happens to land high, post has nowhere to go but down on a n=16 set. Pure regression-to-mean with noise floor σ≈0.11.
- **Verdict: plausible *as noise artefact*, not as capability signal.** Matches `analysis.md` §1.4–1.5.

### 7. `chain_length_spread` ↔ `post_score` (inverse, medium)
- C1: spread 0.69, post 0.1375
- C2: spread 0.93, post 0.225
- C3: spread 0.00, post 0.2125
- Non-monotonic (C3 spread collapsed to 0). Yet C3 post ≈ C2 post. Suggests chain-length diversity does *not* drive post-training score much.
- **Verdict: medium — the C3 zero-spread is striking and deserves its own investigation, but doesn't correlate with post_score.**

### 8. `avg_chain_length` ↔ `improvement` (weak positive over originals, broken by C3)
- C1: 4.3, impr −0.325
- C2: 3.2, impr +0.150
- C3: 3.0, impr −0.125
- Not monotonic. Longer chains in C1 coincide with worst improvement. Ambiguous.
- **Verdict: medium-low — no clean signal.**

### 9. `weaknesses_found` ↔ `samples_generated` (positive, mechanistic)
- C1: 2 w, 3  gen
- C2: 6 w, 14 gen
- C3: 5 w, 10 gen
- r ≈ +0.996. Matches code: `samples-per-weakness` bucketing.
- **Verdict: plausible and expected from design. Not interesting.**

### 10. `generate_time` ↔ `samples_generated` (positive, mechanistic, violated C3)
- C1: 61  s / 3 samples  → 20.5 s/sample
- C2: 470 s / 14 samples → 33.6 s/sample
- C3: 220 s / 10 samples → 22.0 s/sample
- Should be roughly constant per-sample. C2 is 1.5× slower per sample than C1/C3 — worth flagging (log says C2 had 122 STaR rejections vs C3 had fewer before cap). STaR rejection loop inflates C2's generate time.
- **Verdict: plausible; confounded by STaR rejection count which isn't logged per cycle.**

---

## LOW plausibility / likely coincidence or noise

### 11. `learning_rate` ↔ `eval_score`
- C1: 2e-5,       eval 0.0625
- C2: 9.9e-6,     eval 0.250
- C3: 1.15e-5,    eval 0.338
- LR dropped C1→C2 (bandit first pull, `last_pulled: 5e-06` in history). Eval rose. LR rose C2→C3 slightly, eval rose more. No monotonic relationship.
- `analysis.md` §2.1 item 5 explicitly rules LR out as a driver.
- **Verdict: likely coincidence.**

### 12. `avg_rank` (LoRA rank) ↔ `eval_score`
- C1: 64, 0.0625
- C2: 64, 0.250
- C3: 16, 0.338
- Rank dropped 4× between C2 and C3, yet eval rose. Either (a) rank-16 is sufficient, (b) other concurrent changes dominated, or (c) noise.
- Cannot attribute with n=1 rank change.
- **Verdict: coincidence given confounds — but interesting enough to warrant a controlled ablation.**

### 13. `lora_layers` ↔ anything
- Constant 252 across all cycles. Zero variance → zero correlation. Drop.

### 14. `topic_coverage` ↔ anything
- Constant 0.125 across all cycles. Zero variance. Drop.

### 15. `unique_domains` / `unique_subdomains`
- All cycles: 1 / 2. Zero variance. Drop.

### 16. `errors_count` ↔ anything
- All cycles: 0. Zero variance. Drop. (Note: the update-log contains a vLLM OOM fallback on C3-resumed eval but it was not written to `errors` in the JSON — this is a logging gap.)

### 17. `escalation_events` ↔ anything
- All cycles empty. Zero variance. Drop.

### 18. `phase_times.diagnose` ↔ anything
- C1 18.49, C2 18.69, C3 18.14. Essentially constant (~18 s). No signal.

### 19. `phase_times.verify` ↔ `verified/generated`
- C1 0.038 s, C2 0.055 s, C3 0.230 s. Rises with samples_verified (+ consistency-samples=3 on resumed run). Mechanistic, not interesting.

### 20. `post_score - pre_score` (improvement) vs `eval_score`
- C1: impr −0.325, eval 0.0625
- C2: impr +0.150, eval 0.250
- C3: impr −0.125, eval 0.338
- Sign of improvement does NOT predict eval direction. C3 has negative live improvement yet higher held-out than C2. Supports `analysis.md` H3 (curriculum drift + binomial noise decouples pre/post from held-out).
- **Verdict: weak/anti-correlated. Supports the "single-cycle improvement is noise" narrative.**

### 21. `post_over_pre` ratio ↔ `eval_score`
- C1: 0.297, eval 0.0625
- C2: 3.000, eval 0.250
- C3: 0.630, eval 0.338
- Not monotonic. Ratio is unstable when pre is small (C2 pre=0.075). Drop.

### 22. `eval_delta` (held-out delta) ↔ `improvement`
- C2: eval_delta +0.188, improvement +0.150 (both positive)
- C3: eval_delta +0.088, improvement −0.125 (opposite signs)
- n=2 for this pair. One match, one mismatch. Inconclusive and consistent with noise.

---

## Cross-reference against `analysis.md` hypotheses

The analyzer wrote 5 hypotheses (H1–H5). Verdicts using the now-loaded numbers:

### H1 — "Cycle-2 gain is ≥70% measurement noise (curriculum drift + binomial floor)"
**Numbers available:** C2 live held-out 0.250; resumed replay noted in analysis as 0.338. C1 held-out 0.0625. Binomial σ at n=16, p=0.25 ≈ 0.108. The 0.188 C1→C2 swing is ~1.7σ.
- **Verdict: CONFIRMED as plausible.** Consistent with analyzer. Also C3's post=0.2125 vs pre=0.3375 on the *same* held-out seed (0xE7A1) shows 12.5 pp drop from training — i.e. within-cycle noise of the order the analyzer claimed. Supports H1.

### H2 — "True capability delta from C2 training is ≤0.08 held-out points"
**Numbers available:** eval_trajectory [0.0625, 0.250, 0.3375]. The C1→C2 step is 0.188, the C2→C3 step is 0.088. C3 brought lora_rank down and still gained.
- **Verdict: CANNOT CONFIRM from data alone.** H2 requires pairwise same-set comparison which this run lacks. But: `improvement_ema` ended at −0.054 after 3 cycles, meaning the EMA of post−pre is *negative*. No evidence contradicting H2, and the EMA is consistent with "small-or-no real capability delta."

### H3 — "Curriculum drift is the specific mechanism flipping 0.250 → 0.338 on replay"
**Numbers available:** curriculum `solve_rate` changed between checkpoint C2 (history.json) and checkpoint C3. E.g., `code.predict_output[5]`: C2 snapshot has 16 attempts/6 solved; C3 has 21 attempts/7 solved. `code.base_conversion[7]` appears only in C3 (22 attempts/14 solved) — wasn't in C2 at all. Curriculum state drifted substantively.
- **Verdict: CONFIRMED plausibility.** The drift is real and large enough that `pick_frontier` would select different classes on replay, as the analyzer predicted.

### H4 — "Cycle-2 training samples overlapped held-out question classes"
**Numbers available:** `samples_per_domain` is `{code: 14}` for C2 and `{code: 10}` for C3. No per-question logs persisted. `pending_regressions` in C3 history shows held-out contains implementation questions (`multiply`, `is_even`, `tail`, `sum_digits`, `length`), MCQ computing, and prediction. C2 samples were all domain=code but we can't see subdomain composition.
- **Verdict: CANNOT TEST without per-sample logs.** Matches `missing_metrics.md` #2 (per-item logs missing). Open.

### H5 — "Signal dominated by one 'lucky' subdomain in C2"
**Numbers available:** all `eval_domain_scores` only have `code` key — no subdomain breakdown persisted. Only 1 domain, 2 subdomains per `diversity_stats`. Cannot decompose.
- **Verdict: CANNOT TEST.** Matches `missing_metrics.md` #2, #8. Open.

---

## Synthesis

- The **strongest** single-scalar explanation of the C2→C3 regression is **samples_per_optimizer_step** (team-lead): 1.0 / 5.0 / 0.36 across cycles, with only C2 in the safe band. The companion "effective parameter-movement factor" of ~58× (C3 vs C2) gives the mechanism a quantitative basis.
- The **only other** correlation that (a) has a physical mechanism, (b) is monotonic across all 3 cycles, and (c) is not trivially tautological is `samples_verified ↔ eval_score` (and its near-duplicate `verified/generated ↔ eval_score`). Even that is confounded by resume-from-ckpt.
- Every other monotonic correlation at n=3 is either mechanical (steps↔loss, gen_time↔samples), a constant (lora_layers, topic_coverage), or best explained as **regression-to-mean + binomial noise** — which is exactly what `analysis.md` argued.
- **Directly supports** analyzer's H1 (noise-dominant) and H3 (curriculum drift). **Cannot falsify** H2, H4, H5 without the logging gaps in `missing_metrics.md` being closed.
- The C3 config change (rank 64→16, epochs 3→5, grad_accum 16→1) is an uncontrolled multi-variable swap; do not treat any C2→C3 delta as attributable to a single knob.

## Recommendations for team-lead

1. The strongest single recommendation: close `missing_metrics.md` items #2 (per-item diag logs) and #3 (k-seed held-out bootstrap) before any more cycles. Everything else is noise arguments.
2. Second: run one **controlled** resume (same config as original) before running more resumed cycles with C3's altered config — otherwise correlations past C3 will compound confounds.
3. The single best within-dataset evidence for capability gain is `verified/generated` jumping 0.36 → 0.90 at C3, which is a model-internal signal not subject to binomial-noise criticism of the held-out set. Worth preserving as a secondary metric.
