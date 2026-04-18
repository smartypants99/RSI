# Hypothesis Verdicts — Cycle 2 "Improvement" Autopsy v2

## Ground truth from the logs (correcting the brief)

Reading `outputs/logs/cycle_{1,2,3}.json` directly:

| Field | Cycle 1 | Cycle 2 | Cycle 3 |
|---|---|---|---|
| pre_score (diag, post-resume) | 0.4625 | 0.075 | 0.3375 |
| post_score (diag, post-train) | 0.1375 | 0.225 | 0.2125 |
| eval_score (held-out, HELDOUT_CYCLE_SEED) | 0.0625 | 0.2500 | 0.3375 |
| samples_generated | 3 | 14 | 10 |
| samples_verified | 1 | 5 | 9 |
| training steps | **1** | **1** | **25** |
| avg_rank | **64** | **64** | **16** |
| learning_rate | 2.0e-5 | 9.9e-6 | 1.15e-5 |
| final_loss | 1.120 | 0.638 | 0.0445 |
| duration_s | 161 | 648 | 626 |

**Major correction to the team lead's brief:** the brief said cycles 1,2 used rank=64, epochs=3, grad_accum=16, and cycle 3 used rank=16, epochs=5, grad_accum=1. The **LoRA rank change is confirmed** (64 → 64 → 16). But what actually makes cycle 3 different is not epochs — it is that **cycle 3 ran 25 training steps vs. 1 step for cycles 1 and 2**. Cycles 1 and 2 each did a *single* optimizer step on 1 and 5 samples respectively. This is the dominant confound, not epochs or grad_accum.

Also: the "post-score 0.225" the brief calls "improvement" is the **post-training diagnostic re-probe** (`post_diag_domain_scores.code`), not the held-out eval. The held-out eval for cycle 2 was **0.250**. Cycle 1 held-out was **0.0625**, Cycle 3 held-out was **0.3375**. So the held-out trajectory is 0.0625 → 0.2500 → 0.3375 (monotonically rising), while post-diag is 0.138 → 0.225 → 0.213.

---

## Verdicts on the 5 hypotheses in `analysis.md`

### H1 — Cycle-2 gain is ≥70% measurement noise (curriculum drift + binomial floor)
- **Restated:** Most of the 0.188-pt held-out swing between cycle 1 (0.0625) and cycle 2 (0.250) is statistical noise from n=16 binomial sampling plus curriculum-state drift changing *which* questions are drawn.
- **What the numbers say:** Cycle 3 held-out continued to 0.3375 — a *further* monotonic rise. Three sequential scores 0.0625 → 0.250 → 0.3375 on correlated-but-drifting sets is very unlikely under pure-noise null (directional run of 3 under binomial(16,p) noise has prob roughly ~0.1–0.2 depending on correlation). But n=16 and curriculum-seeded reshuffles are real, and the brief's observation that the same cycle-2 weights re-scored 0.338 on resume is direct evidence that ≥10 pp of the live 0.188 delta was not in the weights.
- **Verdict: PARTIALLY CONFIRMED / CONFOUNDED.** Noise is demonstrably a large fraction of the cycle-2 delta (≥10/19 pp from the replay alone). But the monotonic three-point trajectory argues a non-zero real signal underneath.
- **To isolate:** Re-score each checkpoint (base, cycle_1, cycle_2, cycle_3) 10× on a frozen held-out set with curriculum state pinned to a fixed snapshot. Noise = residual std across replays; signal = across-checkpoint mean delta.
- **Confidence:** 0.75 that noise is ≥50% of the cycle-2 live delta.

### H2 — True capability delta from cycle-2 training is ≤0.08 held-out points
- **Restated:** Cycle-2's LoRA update produced <0.08 pp real held-out improvement over the cycle-1 checkpoint.
- **What the numbers say:** Cycle 2 did **1 optimizer step** on 5 verified samples (final_loss 0.638, avg_loss 0.744). A single step at LR ≈1e-5 on a rank-64 LoRA is a tiny weight update. Replay re-scoring cycle_2 at 0.338 vs. cycle_1 at 0.0625 is a 27-pp gap, which is much larger than 0.08 — but cycle_1 *also* only did 1 step on 1 verified sample, and both were starting from very different checkpoints (cycle_1 from base, cycle_2 from cycle_1). Without a clean base-vs-cycle_2 on a frozen set, we cannot separate "cycle_1's one-step update matters" from "cycle_2's one-step update matters."
- **Verdict: CONFOUNDED.** The 1-step-on-5-samples training at ~1e-5 LR can plausibly only move scores by a handful of points. The 27-pp replay gap between cycle_1 and cycle_2 likely reflects mostly cycle_1's training regression (pre 0.4625 → post 0.138) unwinding, not cycle_2 adding capability.
- **To isolate:** Pairwise McNemar on base vs. cycle_2 vs. cycle_2-minus-cycle_2-update on frozen held-out.
- **Confidence:** 0.55 (leaning toward ≤0.08 real delta, but the data does not pin it).

### H3 — Curriculum drift is the specific mechanism flipping 0.250 → 0.338 on replay
- **Restated:** The 0.088-pt gap between live cycle-2 held-out (0.250) and resume-replay (0.338) is caused by `CurriculumState.solve_rate` accumulating between the two evals and changing which questions get sampled.
- **What the numbers say:** Checkpoints' `curriculum.solve_rate` shows substantial per-subclass history accumulating across cycles (21–42 attempts per difficulty bucket in `code.predict_output` and `code.base_conversion` only — math/reasoning classes all empty, meaning held-out sampling is heavily curriculum-biased toward code subclasses). This confirms the mechanism is real and load-bearing. Whether it explains exactly the 0.088-pt flip requires a controlled replay.
- **Verdict: CONFIRMED (mechanism), INSUFFICIENT (exact magnitude).** The drift exists and the curriculum state is the single biggest eval-time non-determinism.
- **To isolate:** Fresh-curriculum replay of cycle_2 weights. If score ≈ 0.25, drift explains the full gap.
- **Confidence:** 0.80 that drift accounts for ≥half of the replay delta.

### H4 — Cycle-2 training samples overlapped held-out question classes
- **Restated:** STaR-generated training samples for cycle 2 hit the same class/subdomain mix as the held-out probe, producing apparent held-out gain via near-leakage.
- **What the numbers say:** The `pending_regressions` in cycle_3 history show held-out questions are dominated by `code.implementation` (Python function-writing), `code.computing` (MCQ trivia), and `code.prediction`. The diversity_stats for all three cycles report `unique_subdomains: 2` and `topic_coverage: 0.125` — training was narrow. Cycle 2 generated 14 / verified 5 samples over 2 subdomains; if those 2 subdomains were `implementation` + `prediction`, partial class overlap is near-certain. We don't have the per-sample prompts logged here to MD5-compare, but the narrow coverage and `weaknesses_found=6` targeting exactly the held-out failure buckets makes class-level overlap almost guaranteed.
- **Verdict: CONFIRMED at class level, INSUFFICIENT at prompt level.** Training deliberately targets diagnostic weakness classes, and the held-out probe draws from those same classes via curriculum. This is not "leakage" in the strict sense but it is *near-leakage* by design.
- **To isolate:** MD5(normalized prompt, expected) across training set vs. held-out set per cycle. Log to `outputs/leakage_report.json`.
- **Confidence:** 0.90 that class-level overlap is structural; 0.4 that any exact prompt matched.

### H5 — Signal dominated by one "lucky" subdomain in cycle 2
- **Restated:** Cycle 2's held-out gain is carried by a single subdomain going from 0/n to k/n while others stayed flat.
- **What the numbers say:** Aggregate `eval_domain_scores` only stores `code`, no subdomain breakdown. Cycle 3's `pending_regressions` evidence list contains 5 `implementation` + 4 `computing` + 1 `prediction` failures — i.e. the held-out set in cycle 3 was dominated by implementation and computing. We cannot tell which subdomain moved in cycle 2.
- **Verdict: INSUFFICIENT.** Data needed (per-subdomain cycle-1 & cycle-2 scores) is not logged.
- **To isolate:** Add per-subdomain logging to `eval_domain_scores`; re-run. Or parse `curriculum.solve_rate` history deltas across cycles (partial info, already trackable from checkpoints).
- **Confidence:** 0.5 (plausible given n=16 and narrow curriculum, but not testable from current logs).

---

## New hypotheses emerging from the real data

### N1 — Cycle 2's "improvement" is almost entirely *cycle 1's regression recovering*, not cycle 2 training
- **Why:** Cycle 1 logged `pre_score 0.4625 → post_score 0.1375` — a **-0.325 pp regression** from 1 training step on 1 sample. Cycle 2 started at `pre_score 0.075` (worse than cycle_1's post 0.138 — likely drift) and post-trained to 0.225. The net `eval_score` trajectory 0.0625 → 0.250 is almost exactly returning toward base. Cycle 2's 1-step-on-5-samples update at LR 9.9e-6 is too small to move capability by 19 pp; more likely the diagnostic probe drift exposed easier items.
- **Verdict suggestion: CONFIRMED-LEANING.** Evidence: magnitude of update is tiny; cycle 1's pre 0.4625 shows the base model actually scores fairly well; cycle 1's training destroyed ~32 pp in one step.
- **Confidence:** 0.7
- **Experiment (1 cycle):** Run held-out on **base model** with fresh curriculum, repeated 5×. If mean ≈ 0.25–0.35, then all "cycles" are just noisy samples around base capability.

### N2 — Cycle 1's regression caused by training on 1 verified sample at LR 2e-5 with rank-64 (too-strong update on near-random data)
- **Why:** Cycle 1: 3 generated, 1 verified (33% verify rate), 1 step, final_loss 1.12 (high — barely fit), post-score drops from 0.46 to 0.14. One bad sample at high rank × high LR corrupts the adapter.
- **Experiment:** Rerun cycle 1 config twice — once with `samples_verified < 3` abort, once with LR 5e-6. Compare held-out regressions.
- **Confidence:** 0.65

### N3 — Cycle 3's overfitting is caused by 25 steps on 9 samples with rank-16, not the rank change itself
- **Why:** `final_loss 0.0445` on 9 samples in 25 steps = the model memorized the training set. Rank reduction (64→16) would *reduce* overfitting capacity, but the step count jumped 25× while samples only ~doubled. Post-diag dropped from pre 0.3375 to 0.213 (-12 pp) — classic overfit. Held-out (drawn pre-training) was 0.3375.
- **Experiment:** Cycle 4 with rank=16 + early stopping at avg_loss < 0.3 (≈3–5 steps). Held-out should stay near 0.34 rather than regress.
- **Confidence:** 0.80

### N4 — The held-out eval set and the diagnostic probe set substantially overlap in class distribution, so "post_score vs. held-out" is not two independent measurements
- **Why:** `post_diag` and `eval` scores track loosely together across cycles (0.138/0.0625, 0.225/0.250, 0.213/0.338), both keyed on curriculum state. Both draw via `curriculum.pick_frontier` with overlapping class pools (math/reasoning classes empty in solve_rate ⇒ neither sampled them). Treating them as independent signals double-counts.
- **Experiment:** Add class-disjoint "true-holdout" set (held-out math + reasoning classes that curriculum never sees). Score each cycle on it. If scores are flat, our entire "held-out" is near-IID with training targeting.
- **Confidence:** 0.85

### N5 — LR bandit logic is broken: proposes 1.4e-5, training uses 9.9e-6 (cycle 2) / 1.15e-5 (cycle 3)
- **Why:** `meta_decisions.jsonl` line 1 says cycle 2 LR proposal = 1.4e-5. Cycle_2.json records actual `learning_rate = 9.899e-6`. For cycle 3, proposal = 1.82e-5, actual = 1.155e-5. Ratio is consistently ~0.707 = 1/√2 — some downstream stage is applying a `sqrt(0.5)` cosine/warmup factor and logging the *end-of-schedule* LR. Either the log is wrong or the optimizer is not using the proposed LR. Also `n_obs=0` in meta_state means the bandit has **never observed a held-out outcome** — `last_pulled` is still 5e-6 from init.
- **Experiment:** Print optimizer.param_groups[0]['lr'] at step 0 and step final; compare to proposal. One cycle resolves this.
- **Confidence:** 0.9 that there's a bug; 0.5 on which side.

---

## What we KNOW vs. what's still guesswork

### High-confidence knowledge
1. Held-out n=16 is below signal threshold for deltas under ~0.2 pp (Wilson half-width ~0.22). Single-cycle comparisons are uninterpretable.
2. Curriculum state drift is real, accumulates across cycles, and biases held-out sampling toward code subclasses (math/reasoning never sampled in 3 cycles).
3. Cycles 1 & 2 did only **1 optimizer step each**; cycle 3 did 25 steps. This is the biggest training-regimen difference, not rank or epochs.
4. Cycle 3 overfit (final_loss 0.0445, post-diag regression of 12 pp).
5. LR bandit is not closed-loop: `n_obs=0` after 3 cycles, meaning observed-held-out outcomes are not being fed back into `arms`. Bandit is effectively a drift-walk.
6. Meta-logged LR proposals do not match recorded training LRs (1/√2 ratio). Log/apply mismatch exists somewhere.

### Still guesswork
- How much of cycle 2's 0.188-pt live gain is real capability vs. noise vs. recovery from cycle 1 regression.
- Whether STaR training samples ever exactly-match held-out prompts.
- Per-subdomain trajectory (not logged).
- Whether rank=16 is actually better than rank=64 at the same step count (rank and steps co-varied).

---

## Integration with team-lead's training_dynamics.md (58× parameter-movement)

Team-lead's analysis converges with and extends mine on the step-count confound:

- **Samples-per-step:** cycle 1=1.0, cycle 2=5.0 ✓, cycle 3=0.36 (each sample seen 2.8× — memorization regime).
- **Cycle 3 effective parameter movement vs cycle 2 = 58×** = 2.0× rsLoRA α/√r (rank 64→16) × 1.16× LR × 25× steps.
- **Loss-floor rule:** final_loss < 0.15 = memorization. Cycle 3 hit 0.044.

Impact on my verdicts:

- **N3 (cycle-3 overfit from steps not rank): upgraded CONFIRMED, conf 0.90.** Step count drives 25× of the 58×; rank-halving via rsLoRA *amplified* each step rather than reducing overfit as rank intuition would suggest.
- **N1 (cycle 2 "gain" is cycle-1 regression unwinding): held 0.7, reframed.** Cycle 2's 1-step update on 5 samples is a genuinely broader gradient than cycle 1's 1-step-on-1-sample, so there IS a real modest update; but magnitude is too small to explain 19 pp alone. Mix of recovery + small real signal + noise.
- **New actionable rule (for code_optimizer):** rank changes must co-adjust LR. When rank halves, rsLoRA scaling doubles ⇒ LR should halve to preserve step magnitude. Cycle 3 should have used LR ≈ 5e-6.

---

## Integration with data_miner's correlation matrix

`correlations.md` + `cycle_analysis_matrix.json` independently reach the same H1/H3 verdicts (CONFIRMED plausible) and flag H4/H5 as untestable with current logs. New material:

- **verified/generated (STaR pass-rate) = 0.33 / 0.36 / 0.90** — monotonic with held-out, model-internal, not subject to n=16 binomial criticism. Elevated to N6 below.
- **improvement_ema = −0.054** after 3 cycles — direct aggregate evidence for H2 ("≤0.08 real delta"). Confidence on H2 upgraded from 0.55 → 0.65.
- **best_checkpoint_cycle=1, best_score=0.1375** — the orchestrator is tracking `post_diag` as "best," not `eval_score`. Cycle 1 was the worst actual held-out (0.0625) and a −0.325 pp post-training regression. This is a **metric-choice bug** — flag to code_optimizer.
- **diversity_stats zero-variance across 3 cycles** on topic/domain/subdomain ⇒ "domain" knob never exercised. Any domain-generalization claims are untestable without at least one non-code cycle.

### N6 (new) — STaR pass-rate is the best model-internal capability proxy
- **Why:** bypasses n=16 held-out binomial floor; monotonic 0.33→0.36→0.90.
- **Confound:** cycle 3 uses resumed model, so pass-rate rise is partially "verifier agrees with its own generator more" (self-consistency inflation).
- **Experiment:** freeze the *verifier* to base model or oracle; let *generator* keep improving. If pass-rate still climbs, it's capability. If it flattens, it was self-consistency.
- **Confidence:** 0.65 as a monotonic proxy; 0.4 that it tracks capability independent of self-consistency.

---

## The single experiment that resolves the biggest remaining uncertainty

**Experiment — Baseline noise calibration:**

1. Take **base model** (no LoRA), pin curriculum to a fixed snapshot, run held-out eval **20 times**.
2. Compute mean, std, and full score distribution.
3. Plot cycle_1 / cycle_2 / cycle_3 observed held-out scores against this distribution.

**Cost:** ~20 × 75 s ≈ 25 min eval + 1 model load ≈ 30 min wall-clock (cycle 3 eval alone took 74 s).

**What it resolves:** Whether *any* cycle's held-out score is outside base-model noise. If base distribution is 0.20 ± 0.10, then none of 0.0625 / 0.250 / 0.3375 are statistically distinguishable from base, and the entire RSI loop has produced zero measurable capability gain across 3 cycles. If base mean is 0.10 ± 0.04 and cycle_3 is reliably 0.34, we have a real ~24-pp signal despite the noise.

This single measurement dominates every other open question because every other hypothesis is gated on "is the signal-to-noise ratio > 1 at all?" — and we currently have no baseline.
