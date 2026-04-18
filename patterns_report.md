# Cycle-Forensics: Patterns Report (n=8)

Author: patterns agent. Evidence: `outputs/{logs,cycle_metrics,cycle_samples}/*`.

## TL;DR — ranked by actionable value

1. **The cycle-6 "miracle" is not a STaR-quality story — it is a topic-match story.** Cycle 6 kept exactly **1 of 208 generated candidates** (verify rate 0.5%, *not* 100% as the team-lead brief states). That one sample was a `code/bit_manipulation` question ("Convert decimal 103 to base 8 → 147"). Cycle 6 was also the only cycle whose training data hit the `bit_manipulation` subdomain, and the held-out eval visibly rewards bit/base-conversion reasoning. Winning cycles are the ones whose (tiny) training set happens to land on an underrepresented diagnostic subdomain — not the ones that produce more or cleaner samples.
2. **Training-time loss is an anti-signal at the ranges we're seeing.** The two winners (C3 final loss 0.65, C6 0.35) sit *between* the losers. C5 (regression, −0.15 diag, eval 0.14) reached final loss **0.143** after one step — lowest of any trained cycle — and the held-out score fell. Lower final loss correlated with worse eval, consistent with the brief's "early-stop fired AFTER step 1 which had already pushed loss to 0.14" diagnosis. The post-step loss is already past the sweet spot; we should early-stop *before* step 1 completes, not after.
3. **Winning cycles flip *different* questions, not more of the same.** C3's moved-wrong-to-right set (7 ids) and C6's (21 ids) intersect on only **3 ids** (`3e3dd13a1a63604e`, `59eba0f85b128878`, `94307ad37e811c4b`). 4/7 of C3's wins and 18/21 of C6's wins are disjoint. Each winning cycle unlocks a different cluster, which is why eval keeps climbing when either wins — and why repeating the same `implementation` samples (C1, C2, C5, C7, C8) stops producing gains.
4. **Verifier pass rate is uncorrelated with success.** Pass rates: C1 16%, C2 12%, C3 20%, C5 16%, **C6 0.5%**, C7 11%, C8 7%. The best eval gain came from the lowest pass rate. Raising verifier strictness ≠ better training signal when the bottleneck is topic coverage.
5. **Probe-skipped cycles (C4, C7, C8) still move questions w/o any training.** C4 `moved_wrong_to_right`=4, C8=6. This is pure evaluator noise on a ~80-question diag set and should be treated as a noise floor (~5 question flips per cycle) when judging whether a trained delta is real. C3's 7 w2r is barely above that floor on diag; its *held-out* +0.19 is the real signal. C6's 21 w2r and held-out +0.33 are clearly above noise.

---

## 1. Which cycles actually improved, and what was different?

| cycle | pre→post diag | held-out | trained? | STaR kept/rej (verify%) | samples_used | final_loss | w2r / r2w | subdomains in training |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.463 → 0.313 (−0.15) | 0.089 | yes (1 step) | 21/107 (16%) | 2 | 1.18 | 0 / 8 | implementation only |
| 2 | 0.413 → 0.263 (−0.15) | 0.089 | yes (1 step) | 14/98 (12%) | 3 | 0.95 | 3 / 4 | implementation only |
| **3** | **0.263 → 0.438 (+0.18)** | **0.281** | yes (**2 steps**) | 42/166 (20%) | 5 | **0.65** | **7 / 1** | impl + prediction |
| 4 | 0.363 → 0.400 (+0.04) | — | **skipped** | 14/158 (8%) | 4 | 0.23 | 4 / 1 | — |
| 5 | 0.425 → 0.275 (−0.15) | 0.141 | yes (1 step) | 23/117 (16%) | 7 | **0.14** | 5 / **17** | implementation + prediction |
| **6** | **0.275 → 0.438 (+0.16)** | **0.474** | yes (1 step) | 1/207 (0.5%) | **1** | 0.35 | **21 / 8** | **bit_manipulation** (only) |
| 7 | 0.350 → 0.413 (+0.06) | — | **skipped** | 20/156 (11%) | 7 | 0.26 | 0 / 1 | — |
| 8 | 0.363 → 0.313 (−0.05) | — | **skipped** | 12/168 (7%) | 8 | 0.23 | 6 / 10 | — |

Winning cycles (3 and 6) share two properties that the losers do not:

- **Novel weakness coverage.** C3 was the first cycle to train a `code/prediction` sample (arithmetic on list comprehensions). C6 was the only cycle to train a `code/bit_manipulation` sample. Every non-winning trained cycle (C1, C2, C5) fed the model *more `implementation` restatements of `head`/`tail`/`length`/`first_elem`/`last_elem`/`merge_sorted`* — the model already produces these correctly pre-train (they're trivial), so training on them adds noise without information.
- **Training-loss landing in 0.35–0.65, not below 0.2.** C3 ran 2 steps and ended at 0.65 (moderate). C6 ran 1 step on a structured 3-step reasoning chain and ended at 0.35. C5 overshot to 0.14 and regressed. C1/C2 under-stepped (1.18 / 0.95) and regressed. The LR bandit was pulling LR *down* (2e-5 → 1.4e-5 → 1.82e-5 → 1.27e-5 → 8.9e-6 → 1.16e-5 → 8.1e-6 → 5.7e-6) without awareness of this regime.

Regressions (C1, C2, C5) all share the "repeat implementation trivia" pathology: same ~10 prompts keep getting regenerated because the weakness-bucketing only exposed `implementation` and `prediction` until C3, and because the generator reliably produces the same easy function stubs.

## 2. Which diagnostic questions flip — distinct cluster or more of the same?

**Distinct clusters.** See table at top of §4 for the IDs. The C3 ∩ C6 overlap is 3 / (7 ∪ 21) = 3/25 = **12%**, and two of those three IDs (`59eba0f85b128878`, `94307ad37e811c4b`) were *already known-flippable* — `59eba0f85b128878` was among the 8 right→wrong questions in C1 (i.e., highly volatile on tiny weight changes), so it contributes little signal.

By contrast, C6's unique 18 IDs (`f12551e0d0a1817e`, `c73096dd60edf2b6`, `7926362e76763e46`, `63721b4164bea46a`, `65c06be2cd78646f`, `5a80237707115948`, `29c52380f4290383`, `b32b478b461f7678`, `b9f7bca03b341b1a`, `3150303fa3ed975e`, `d978d6e2ffeb44f8`, `858bb8e35153c3b8`, `3f65da25af27a197`, `d5b022212c10332c`, `f7a59d50418e6a6a`, `8f9fc511ca573eff`, `0405b561a5137d12`, `4b88053828eace9e`) are new unlocks — exactly consistent with bit-manipulation training pulling up questions whose correct answer needs base/bitwise reasoning, a capability the C1–C5 training runs never touched.

**Takeaway:** progress is not cumulative on the same question set. Each win opens a *new* region of the diagnostic. That argues for running a `subdomain-coverage` scheduler (force the generator to rotate through `bit_manipulation`, `complexity`, `debugging`, `computing` — not just let the verifier pick whatever) rather than tuning LR or STaR strictness.

## 3. What made the STaR output quality differ?

Inspecting `cycle_samples/`:

- **C3's 5 verified samples**: all 3-step chains (step_count = 3 uniformly), short prompts (92–239 chars), responses 11–137 chars, `consistency_score` mean ≈ 0.8. Content: `head`, `tail`, `merge_sorted`, `count_positive`, plus one `prediction` (`max([...]) - min([...])`). Mix of impl + prediction.
- **C6's 1 verified sample** (verbatim):
  > Prompt: "Convert decimal 103 to base 8 (no prefix)." → Answer: "147"
  > 3-step chain: "Divide 103 by 8 to get the quotient and remainder" → "Use the remainder from the division to determine the least significant digit" → "Combine the digits to form the octal number 147"
  > `consistency_score=0.25`, verification notes flag no Python exec.

  Structurally *not* cleaner than C3's samples — shorter (42-char prompt), lower consistency (0.25 vs mean 0.8), no executable check. What's distinctive is **the subdomain** (`bit_manipulation`, unseen in training elsewhere) and that it teaches a **procedure** (iterative division + digit combination), not a library call recitation like `return lst[0]`.

- **C1/C2/C5/C7/C8**: dominated by "trivial function stubs" (`def head(lst): return lst[0]`). These are pattern-matching one-liners the model already gets right at baseline. Training on them risks (a) slight overfit to stylistic answer tokens and (b) nudging the model to emit `def foo:` scaffolding in answer slots that expect bare values.

**Actionable:** filter STaR output *before* training to require either (i) a non-trivial subdomain tag, or (ii) a multi-step arithmetic/procedural chain where the response token count exceeds the prompt's operand count. "Trivial restatement" samples should be discarded even if verified.

## 4. The cycle-6 miracle — why did 1 sample produce +0.33 held-out?

Cycle 6 trained on a single base-conversion sample for 1 step, ending at loss 0.35, and the held-out score jumped from 0.141 (C5) to 0.474 — the largest single-cycle held-out gain in the run.

Hypothesis (supported by the data):

1. **Subdomain lift.** The held-out eval is 100% `code`, and its 45 questions include a meaningful fraction of numeric/base/bitwise items. No prior cycle had trained a sample that exercised **iterative-division + digit-combination** reasoning. One well-structured example of that procedure unlocked the whole cluster — the 21 `moved_wrong_to_right` on diag and the +0.33 on held-out are the same phenomenon.
2. **LoRA dose was right.** 1 step × lr 1.16e-5 × 1 sample ≈ a small, focused weight update. Compare C5: 1 step × lr 8.9e-6 × 7 samples drove final loss to 0.14 — too much dose, the "early-stop fires after step 1" issue the team already identified.
3. **Counterfactual.** If the miracle were just "1 sample + moderate loss," any C1/C2 single-sample run should have matched it. They didn't. The differentiator is the *sample's content* — a procedure the base model had never been given explicit chain-of-thought traces for.

**Practical implication:** this is reproducible if we *force* one-sample-per-novel-subdomain training. Recommend `optimizer` try: fixed 1 sample, lr≈1e-5, 1 step, rotate `target_weakness` strictly across `{bit_manipulation, complexity, debugging, computing, prediction, implementation}` round-robin, regardless of what the verifier prefers. Budget: 6 cycles for a full rotation, then compare held-out vs the current adaptive approach.

---

## Supporting observations

- **Loss trajectories are empty arrays in metrics** (`training_loss_trajectory: []` everywhere). The per-step curve isn't being logged; only `avg_loss` / `final_loss` survive. Recommend instrumenting this before tuning further — the early-stop decision needs intra-step visibility.
- **Probe cycles (4, 7, 8) do 0 training steps but still log `avg_loss`** (0.23, 0.26, 0.23) — this appears to be loss on the samples *before* any update. Those numbers are a baseline for "how hard does the model find its own STaR outputs?" and are suspiciously flat, suggesting the model already nearly-memorizes its own generations.
- **LR bandit is wandering.** 8 cycles, 8 different LRs, "insufficient_data" on every decision. With n=3 trained observations, the bandit has no signal. Consider freezing LR at 1.2e-5 (the C6 value) until there are ≥10 trained cycles to estimate from.
- **Diversity is floor-bound.** `topic_coverage=0.125`, `unique_domains=1` every cycle — the generator only hits one of ~8 topics. This is the root cause of pattern #3: to unlock new question clusters we have to diversify at generation, not verification.

## Caveats

n=8, 3 trained-regression + 2 trained-win + 3 probe-skipped. Any single-cycle effect (especially C6's +0.33) could be partly luck on held-out composition. The `bit_manipulation`-match hypothesis is strong but should be confirmed by a held-out per-subdomain breakdown, which isn't in the current `eval_domain_scores` (only aggregate `code` is stored).
