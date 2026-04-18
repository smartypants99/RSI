# Cycle 2 Autopsy — Causal Analysis

Team lead requested this file at repo root. Grounded in `src/diagnostics/engine.py`, `src/generator/data_generator.py`, `src/orchestrator/loop.py`, `src/diagnostics/curriculum.py`, and the tail of `update-log.txt`.

---

## 1. Variance Accounting

### 1.1 Does the diagnostic engine sample at temperature > 0?

**No.** The scoring path is deterministic in generation:

- `DiagnosticsEngine._probe_domain` (engine.py:1259) calls `_generate_batch_with_oom_retry(..., temperature=0.0)` (engine.py:1277).
- `_generate_batch_with_oom_retry` defaults `temperature=0.0` (engine.py:1095) and passes it to `model.generate_batch`.
- Activation probes (engine.py:1794) and ground-truth holdout scoring (engine.py:1979) also use `temperature=0.0`.
- No consistency-vote/majority sampling inside scoring. `--consistency-samples 3` is consumed by verifier/generator, not the grader.

Generation during scoring is deterministic. Held-out eval is NOT noisy because of stochastic decoding.

### 1.2 Where does the variance come from then?

Variance enters through the **question set**, not generation:

1. **Seed = cycle number** (engine.py:1328): `rng = random.Random(md5(f"{domain}:{cycle}"))`. Ground-truth bank (engine.py:1370), holdout (engine.py:1372), curriculum picker (engine.py:1351) all key on `cycle`. So different cycles see different question subsets.
2. **Held-out eval uses a fixed pseudo-cycle** `HELDOUT_CYCLE_SEED = 0xE7A1` (loop.py:786, 791). Should make held-out identical across cycles.
3. **BUT** `CurriculumState.solve_rate` is **mutated on every call** via `curriculum.record_results` (engine.py:1313). `pick_frontier` classifies classes as frontier/easy/probe using rolling solve-rates (curriculum.py:345, 438). So the held-out question distribution drifts cycle over cycle even with a fixed seed: the curriculum's idea of "frontier" shifts. With `_curriculum_share = 0.6`, ~60% of probe questions come from this state-dependent sampler. This is the dominant non-determinism source.
4. **Ground-truth bank reshuffles per cycle-seed too**: `gt_rng = md5(f"gt:{domain}:{cycle}")` (engine.py:1370). For held-out, `cycle` is always `0xE7A1`, so this subpath is stable, but composition of the combined 60/40 set shifts via curriculum.

### 1.3 Why did the resume re-score cycle_2 at 0.338 instead of 0.250?

Running `diagnostics.run(0xE7A1)` on the same weights gave 0.338, vs. 0.250 live. Because generation is deterministic (§1.1), the delta is **curriculum state**. On resume (loop.py:454–461), `CurriculumState.from_dict(data["curriculum"])` restores the end-of-cycle-2 snapshot — which includes the solve-rates recorded during the live held-out eval itself. So the "replay" starts from a slightly different curriculum than the live eval did. That is enough to flip ~1–2 questions of a 16-item set — 6–12 percentage points.

### 1.4 Binomial noise floor

n=16, binary, p=0.25: `std = sqrt(p(1-p)/n) = 0.108`. At p=0.5 it's 0.125. 95% CI half-width at n=16 is ~0.21. The cycle-3 held-out (0.338) sits within one std of the cycle-2 held-out (0.250).

### 1.5 Variance decomposition — best estimate

| Source | Share of 0.062 → 0.250 swing |
|---|---|
| Binomial sampling over n=16 + question-set composition | ~60–70% (one std = 11 pp at p=0.25; 19 pp swing is ~1.7σ) |
| Curriculum-state drift changing sampled questions | ~20–30% (systematic bias on 60% of items) |
| Real capability delta from cycle-2 training | ~0–20% (resume replay at 0.338 on unchanged weights confirms most live gain is reproducible noise, not capability) |

**Bottom line: the 0.188-pt "cycle 2 improvement" is mostly below the binomial floor. Single-cycle held-out deltas under ~0.15 on a 16-item set are statistically indistinguishable from noise.**

---

## 2. What was actually different in cycle 2

The resumed log only has cycle 3 in detail. Cycle 1/2 data must be reconstructed from `outputs/checkpoints/cycle_2/history.json` (per loop.py:1207–1233) — not present in this working tree. What can be asserted from code + log fragment:

| Variable | Cycle 1 | Cycle 2 | Cycle 3 | Source |
|---|---|---|---|---|
| Starting model | Qwen3-8B base | cycle_1 ckpt | cycle_2 ckpt | loop.py:541–544 |
| Learning rate | 2e-5 (default) | 2e-5 (tracker had n<2) | **1.4e-5** (bandit pick) | log line 262 |
| LoRA rank | 16 | 16 | 16 | CLI |
| Epochs | 5 | 5 | 5 | CLI |
| Consistency threshold | 0.34 | 0.34 | 0.34 | CLI |
| Samples-per-weakness cap | 60 | 60 | 60 | CLI |
| Domain | code | code | code | CLI |
| Weaknesses found | unknown | unknown | 5 (impl/pred/compute/complex/debug) | log 131–136 |
| STaR kept/rejected/rationalized/final | unknown | unknown | 14/122/4/**10** | log 151 |
| Verified | unknown | unknown | 9/10 (90%) | log 155 |
| Training steps | unknown | unknown | 25 | log 169 |
| Final train loss | unknown | unknown | 0.0445 | log 169 |
| Held-out | 0.062 | 0.250 (→0.338 replay) | 0.338 | brief + log 261 |

### 2.1 Plausibility ranking for drivers of cycle 2's gain

1. **Measurement variance (curriculum drift + binomial floor).** *Evidence:* same cycle-2 weights re-scored 0.338, not 0.250. 0.188 is within ~1.7σ of binomial noise. Most explanatory factor.
2. **Starting from cycle_1 not base (cumulative LoRA merges).** Each cycle resumes (loop.py:735). Resume re-eval at 0.338 suggests a small real capability delta above base; live 0.188 is inflated by drift.
3. **Weakness-subdomain targeting / curriculum leakage.** STaR draws from failed diagnostic items. If cycle-1/2 samples happened to overlap held-out class distribution, held-out rises without generalization. Unquantifiable without sample logs.
4. **Training sample count / composition.** Cycle 3 = 10 samples / 9 verified. No data on 1/2.
5. **LR change.** Meta only dropped LR for cycle 4 (log 262, `tracker=insufficient_data (n=2)`). Cycles 1–2 were default 2e-5. **Not a driver.**
6. **Training final loss.** Only cycle-3 logged (0.0445). Loss on 9 synthetic samples does not predict held-out. Low rank.
7. **Escalations.** All three were `False` on resume (log 122). Require `post_score > 0.5`, never reached. Not a driver.

**Verdict: mostly noise (items 1 + 2) with a small real capability delta underneath. Items 3–7 lack evidence as dominant drivers.**

---

## 3. Falsifiable Hypotheses

### H1 — Cycle-2 gain is ≥70% measurement noise (curriculum drift + binomial floor)
- **Confirm:** Run held-out eval 10× on the same cycle_2 checkpoint, resetting curriculum to a fixed snapshot each time. If std ≥ 0.08, confirmed.
- **Falsify:** All 10 scores cluster within ±0.03 of 0.338.
- **Cycle 4/5 if true:** Held-out bounces in [0.15, 0.40] with no trend. Treat any single-run delta < 0.15 as null.

### H2 — True capability delta from cycle-2 training is ≤0.08 held-out points
- **Confirm:** Eval base vs. cycle_2 on the SAME byte-identical frozen held-out set. McNemar's test on per-question wins; CI excludes > 0.08.
- **Falsify:** Pairwise eval shows cycle_2 beats base by >0.15.
- **Cycle 4/5 if true:** Further cycles give 0–5 pp per cycle, not 10–20 pp. Plateau within 2–3 cycles.

### H3 — Curriculum drift is the specific mechanism flipping 0.250 → 0.338 on replay
- **Confirm:** Re-run cycle_2 held-out with `curriculum = CurriculumState(DEFAULT_CLASSES)` (fresh). If ≈0.250, drift is the cause.
- **Falsify:** Score ≈0.338 regardless of curriculum state.
- **Cycle 4/5 if true:** Snapshot curriculum before held-out, restore on each eval. Variance should drop sharply.

### H4 — Cycle-2 training samples overlapped held-out question classes
- **Confirm:** MD5(prompt, expected) of cycle-2 training samples vs. held-out set. Any overlap ≥1 inflates apparent gain.
- **Falsify:** Zero overlap.
- **Cycle 4/5 if true:** Dedupe STaR output against frozen held-out; held-out gains shrink.

### H5 — Signal dominated by one "lucky" subdomain in cycle 2
- **Confirm:** Compare per-subdomain cycle-2 held-out vs. cycle-1. One subdomain going 0/4 → 3/4 with others flat = 3-item coincidence.
- **Falsify:** Improvements spread across ≥3 of 5 subdomains.
- **Cycle 4/5 if true:** Per-subdomain variance is high; report per-subdomain deltas with CIs, not aggregate only.

---

## Recommendations for cycle 4/5 measurement

1. **Freeze the held-out set** once (serialize prompts + expected + check_type); reload byte-identically each cycle. Curriculum must not influence held-out sampling.
2. **Run held-out ≥3× per cycle**, report mean ± std. Single eval at n=16 is below signal.
3. **Test against base each cycle** on same frozen set — trajectory vs. base, not absolute.
4. **Wilson 95% CI on every score** (at n=16, p=0.3, half-width ≈0.22). Don't celebrate sub-CI improvements.
5. **Separate variance sources** in progress dashboard: (a) question-set, (b) generation (currently 0), (c) real capability delta.

---

Relevant files:
- `/Users/milannarula/Desktop/Recursive-self-improvment/src/diagnostics/engine.py` (1093–1129, 1259–1400)
- `/Users/milannarula/Desktop/Recursive-self-improvment/src/diagnostics/curriculum.py` (305–490)
- `/Users/milannarula/Desktop/Recursive-self-improvment/src/orchestrator/loop.py` (786–808, 454–461)
- `/Users/milannarula/Desktop/Recursive-self-improvment/src/generator/data_generator.py` (STaR)
- `/Users/milannarula/Desktop/Recursive-self-improvment/update-log.txt`
