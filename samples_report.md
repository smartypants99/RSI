# Samples Deep Dive — cycle-forensics

Scope: 8 cycles of code-only RSI (every cycle is `domain: code`). 37 samples total across cycle_1..cycle_8.jsonl. Numbers in this report are pulled from the jsonl files and `cycle_metrics/cycle_N.json` verbatim.

## Key score numbers (from cycle_metrics)

| cycle | pre   | post  | Δ      | eval_mean | kept/rej (STaR) | final_samples | jsonl rows |
|-------|-------|-------|--------|-----------|-----------------|---------------|------------|
| 1     | 0.462 | 0.312 | -0.150 | 0.089     | 21 / 107        | 4             | 2          |
| 2     | 0.412 | 0.262 | -0.150 | 0.089     | 14 / 98         | 6             | 3          |
| 3     | 0.262 | 0.438 | +0.175 | 0.281     | 42 / 166        | 10            | 5          |
| 4     | 0.362 | 0.400 | +0.038 | null      | 14 / 158        | 9             | 4          |
| 5     | 0.425 | 0.275 | -0.150 | 0.141     | 23 / 117        | 10            | 7          |
| 6     | 0.275 | 0.438 | +0.163 | 0.474     | 1 / 207         | 1             | 1          |
| 7     | 0.350 | 0.412 | +0.063 | null      | 20 / 156        | 10            | 7          |
| 8     | 0.362 | 0.312 | -0.050 | null      | 12 / 168        | 10            | 8          |

Note — the team-lead brief says "+0.333 held-out" for cycle 6; the JSON shows `improvement: +0.1625` and `eval_mean: 0.474`. Cycle 6 has the highest eval_mean of any cycle by a wide margin (next best is cycle 3 at 0.281). Cycle 3's pre→post gain is real but its held-out mean is still ~0.28.

## Part 1 — What winning-cycle samples look like

### Cycle 3 (5 kept, all verified)

All five reasoning chains are exactly 3 steps. Every step has `confidence: -1.0` and `rule: ""` — those fields are never populated anywhere in the corpus.

1. `head(lst) → lst[0]` — response: `def head(lst): return lst[0]`. Steps: "Identify the problem requirements…", "Determine the correct implementation…", "Define the head function with the correct syntax". consistency=1.0. target_weakness=code/implementation.
2. `tail(lst) → lst[1:]` — response: `def tail(lst): return lst[1:] if lst else []  # … </think>`. Consistency=0.75. The `</think>` leaking into the final response is a tokenizer/strip bug — this is in the *training target*, not just the reasoning.
3. `merge_sorted(a, b)` — response: `def mergesorted(a: list, b: list) -> list: return [x for x in sorted(a + b)]  # noqa: E501 </think>`. **Wrong answer in every structural way**: (a) function name is `mergesorted` not `merge_sorted` — will fail any exact-name test; (b) the prompt says "without using built-in sort" and the response uses `sorted()`; (c) reasoning chain says "iterate through both lists using a while loop to compare elements" which doesn't describe what the response actually does. Consistency=0.25. **Verified=true despite being wrong.**
4. `count_positive(lst)` — response: `def countpositive(lst): return len([x for x in lst if x > 0])`. Name mismatch (`countpositive` vs `count_positive`). Consistency=1.0.
5. `max(...) - min(...) = 21` — actual answer is 13 - (-8) = 21. ✓ correct. Consistency=1.0. target_weakness=code/prediction.

Format: all code-gen prompts are "Write a Python function `name` … Provide the function in a ```python``` block." The model's responses are one-liners without the requested code block. Expected_answer is just the bare function name (e.g. `"head"`, `"tail"`, `"merge_sorted"`) — not a test case, not runnable ground truth, just the identifier the grader greps for.

### Cycle 6 (1 kept)

Sole sample verbatim:
- prompt: `"Convert decimal 103 to base 8 (no prefix)."`
- response: `"Combine the digits to form the octal number 147"`
- expected_answer: `"147"` ✓ (103 = 1·64 + 4·8 + 7 = 147₈)
- reasoning chain (3 steps): "Divide 103 by 8 to get the quotient and remainder" → "Use the remainder from the division to determine the least significant digit" → "Combine the digits to form the octal number 147"
- target_weakness: `code/bit_manipulation`; consistency=0.25; severity_at_generation=0.554.

### Structural contrast vs. no-gain cycles

Cycles 1, 2, 5 (all −0.15 improvement) differ in a few ways:

- **Cycle 1 sample 1** (`last_elem`) has `n_reasoning_steps: 6` — the chain is literally duplicated (same three steps appear twice verbatim). The `assumptions` field contains `"</think>"` as a list element — the special token is being captured as structured data. This is the only sample in the entire corpus with chain_length > 3.
- **Cycle 2** has 3 samples, all `code/implementation`, zero subdomain diversity (`topic_coverage: 0.125`, `unique_subdomains: 1`).
- **Cycle 5** has 7 jsonl rows (final_samples: 10, but jsonl file has 7) — includes a `contains(x, lst)` sample whose reasoning says to "Use the list method index to find the position" but the response uses `x in lst`. Reasoning and code disagree.
- **Cycle 5** also has a `sum(x for x in [...] if x % 2 != 0)` sample whose step 3 literally says `"Summing the filtered list [-1, -2, 7] gives -1 + (-2) + 7 = 4, which is incorrect, but the correct answer is 6 due to a miscalculation in the steps."` — the chain explicitly admits it's wrong, yet `verified: true, ground_truth_verified: true`.

There is no structural difference in *length* or *format* between winning and losing cycles — everything is 3 steps, prose-only, no sympy, no test cases, one-liner responses. The differences are:
- Cycle 6 has just 1 sample and it's a **clean arithmetic problem with a correct numeric answer** and no function-name mismatch.
- Cycle 3's gain is driven by samples where `expected_answer == prompt-specified function name`, so even when the response is wrong (sample 3) the grader's substring match over the identifier still counts it correct.
- Losing cycles include self-contradicting chains, duplicated chains with `</think>` tokens, and reasoning/response mismatches.

## Part 2 — Rejected samples

**Rejected chains are not persisted anywhere.** `outputs/` has `cycle_samples/`, `cycle_metrics/`, `checkpoints/`, `logs/`, `meta_decisions.jsonl`, `progress.json`. None of these retain rejected reasoning chains — only counts in `star.per_weakness.rejected` and `zero_correct_items`.

What can be inferred from counts:

- Cycle 6: 1 kept, 207 rejected (99.5% rejection rate). Per-weakness: `code/implementation 0/88`, `code/prediction 0/100`, `code/bit_manipulation 1/15`, `code/computing 0/4`. The diagnostic targeted 4 weaknesses but only one produced any survivor. 52 `failed_items_processed` is consistent with ~4 candidates × 52 items ≈ 208 generations.
- Cycle 3: 42 kept / 166 rejected (20% keep). Implementation 23/65, prediction 15/85, complexity 4/4, debugging 0/8, bit_manipulation 0/4. Debugging and bit_manip were fully rejected; the 10 final_samples are drawn only from implementation + prediction + complexity.
- Cycle 1: `code/implementation` kept 1 of 80 attempts; `code/prediction` kept 20 of 48. Implementation was near-impossible to keep yet the 2 final-selected samples are both implementation (via rationalization for one).

Without the chains themselves we cannot say *why* each was rejected. This is a concrete instrumentation gap — I have pinged `optimizer` about it.

## Part 3 — Sample → weakness targeting

`target_weakness` on kept samples vs what the per-cycle diagnostic surfaced (= the weakness buckets STaR attempted):

| cycle | buckets attempted                                                    | kept sample weaknesses                        | covered? |
|-------|----------------------------------------------------------------------|-----------------------------------------------|----------|
| 1     | implementation, prediction                                           | impl, impl                                    | partial — prediction had 20 kept but none reached final |
| 2     | implementation, debugging, prediction                                | impl × 3                                      | no — debugging, prediction zero in final     |
| 3     | impl, prediction, complexity, debugging, bit_manipulation           | impl × 4, prediction × 1                      | no — complexity (4 kept in STaR), debugging, bit_manip missing from final |
| 4     | impl, prediction, complexity, debugging, bit_manipulation           | impl × 3, prediction × 1                      | no — same pattern |
| 5     | impl, prediction, complexity                                         | impl × 6, prediction × 1                      | partial — complexity had 2 STaR keeps but 0 in final |
| 6     | impl, prediction, bit_manipulation, computing                       | bit_manipulation × 1                          | yes, but only 1 of 4 |
| 7     | prediction, impl, bit_manipulation, complexity                      | prediction × 2, impl × 5                      | no — bit_manip, complexity missing |
| 8     | prediction, impl, bit_manipulation, debugging, complexity           | prediction × 4, impl × 4                      | no — 3 of 5 missing |

Pattern: **the final training set consistently drops the rarer weaknesses (bit_manipulation, debugging, complexity, computing) even when STaR kept samples in those buckets.** Only cycle 6 trained on a rare bucket, and that's because *every other bucket failed to produce a keeper*.

The diagnostic-targeted list of weaknesses is only weakly connected to what actually gets trained on. The selection funnel (STaR → final_samples) collapses toward `code/implementation` every time.

## Part 4 — The cycle 6 singleton

Re-quoting verbatim:

```json
{"prompt": "Convert decimal 103 to base 8 (no prefix).",
 "response": "Combine the digits to form the octal number 147",
 "reasoning_chain": [
   {"content": "Divide 103 by 8 to get the quotient and remainder", ...},
   {"content": "Use the remainder from the division to determine the least significant digit", ...},
   {"content": "Combine the digits to form the octal number 147", ...}],
 "expected_answer": "147",
 "target_weakness": "code/bit_manipulation",
 "consistency_score": 0.25,
 "verified": true}
```

What's structurally special:

1. **Not a code-generation prompt.** It's a base-conversion word problem. Unlike every other cycle's training data, the response is not `def foo(...): ...` — it's prose ending in the numeric answer. The grader sees `"147"` in the response and in expected_answer, and matches.
2. **Expected_answer is a concrete numeric string**, and 147 *is* actually correct (103₁₀ = 147₈). Most other cycles' expected_answer is a function *name* — the grader matches the identifier regardless of correctness. Here the match mechanism and the semantic correctness coincide.
3. **Reasoning chain is a clean algorithm** (divide-by-base remainder collection) stated at the right abstraction level. No `</think>` token leakage, no contradictions, no "iterate through both lists using a while loop" for a problem that doesn't have two lists.
4. **Tiny training set prevents overfitting to bad data.** Cycle 6 did 1 epoch over 1 sample (+ 2 DPO pairs). With final_samples=10 in cycles 3–8, the network sees several reasoning chains with known defects (response/reasoning mismatches, `</think>` tokens, wrong function names). Cycle 6 dodges all of that by accident — only the 99.5% rejection rate let one clean needle through.
5. **severity_at_generation=0.554, consistency=0.25** — on the metric sheet it looks like a low-quality sample. The held-out eval tells a different story.

The takeaway: **+0.163 held-out came from a single high-quality arithmetic demonstration**, not from the bulk-implementation samples that dominate every other cycle. The "1 kept out of 207" pipeline was accidentally a quality filter rather than a quantity bottleneck.

## Signals worth surfacing to optimizer

Already pinged. Summary:

1. **`</think>` tokens leak into training data** (cycle 1 `last_elem` assumptions, cycle 3 `tail`/`merge_sorted` responses, cycle 5 row 1). Strip before save.
2. **Function-name mismatches pass verification** (`countpositive` vs `count_positive`, `mergesorted` vs `merge_sorted`, `firstelem` vs `first_elem`, `lastelem` vs `last_elem`, `binarysearch` vs `binary_search`). Verification appears to only check that expected_answer substring appears in response; `"merge_sorted"` does not appear in `"def mergesorted(...): ..."`. Either the check is loose or it's matching against the prompt, not the response.
3. **Prompt-constraint violations pass verification** (cycle 3 `merge_sorted` says "without using built-in sort" and response is `return [x for x in sorted(a + b)]`). No semantic check against prompt constraints.
4. **Explicit self-contradicting chains pass** (cycle 5 row 7: "gives 4, which is incorrect, but the correct answer is 6").
5. **Rejected chains are not logged.** Can't audit 207 rejected vs 1 kept in cycle 6. Add a truncated rejected-chain sink (even sampling 5% would help).
6. **Weakness targeting is illusory** — diagnostic surfaces 4-5 weaknesses, final training data is almost always 100% `code/implementation`. Either the bucket-balancing in final_samples selection is broken, or STaR can't generate survivors for rare weaknesses and the system doesn't escalate.
7. **Chain length is frozen at 3.** `avg_chain_length: 3.0` in cycles 2,3,4,6,7; spread of 0 in cycles 3,4,6,8. Either the generator is templated, or the filter rejects anything else.

## Absolute paths

- /Users/milannarula/Desktop/Recursive-self-improvment/outputs/cycle_samples/cycle_{1..8}.jsonl
- /Users/milannarula/Desktop/Recursive-self-improvment/outputs/cycle_metrics/cycle_{1..8}.json
- /Users/milannarula/Desktop/Recursive-self-improvment/samples_report.md (this file)
