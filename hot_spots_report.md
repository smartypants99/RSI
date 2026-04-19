# hot_spots_report

Surveyor: **hot_spots** | Data: `outputs/` (8 cycles) + `src/`
Priority ordering: **impact on held-out score**, highest first.

Held-out eval trajectory (progress.json): 0.089 → 0.089 → 0.281 → null → 0.141 → **0.474 (peak, cycle 6)** → null → null. System has degraded since cycle 6; `degradation_count=1` in progress.json.

---

## HIGH impact

### H1. `_gen_code_implementation` sets `expected="def"` — the function-name gate is a no-op
- **File/line**: `src/diagnostics/engine.py:739` (`"expected": "def"` for every code/implementation question from the model-generated path).
- **Consequence chain**: When `_check_answer` routes to `_check_code_executes` (`src/diagnostics/engine.py:2096` → `:2254-2256`), the identifier gate `re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", expected)` is True for `"def"`, and `re.search(r"(?m)^\s*def\s+def\s*\(", code)` never matches any real function. The hard name-check fails silently for 100 % of model-generated implementation questions.
- **Data evidence**: cycle_2 sample `"expected_answer": "def"` paired with `"prompt": "Write a Python function that merges two sorted lists into one sorted list without using built-in sort."`, `"response": "def merge(list1, list2): return [x for x in sorted(list1 + list2)]"` — passed verification.
- **Fix sketch**: parse the backticked identifier out of the prompt and store it in `"expected"`, or use `""` to disable the gate explicitly; additionally exclude Python keywords from the identifier-gate in `_check_code_executes`.

### H2. Prompt constraints ("without built-in sort", "binary search") are never verified
- **File/line**: `src/diagnostics/ground_truth.py:255-276` — unit tests only assert I/O behavior.
- **Consequence**: Training signal rewards cheating. Any `return sorted(a+b)` passes `merge_sorted`. Any `arr.index(target)` passes `binary_search`. This bakes anti-algorithmic behavior into the LoRA.
- **Data evidence** (count across cycles): `sorted(a+b)` in `merge_sorted` responses — cycles 2, 3 (×2), 4 (×2), 5, 7, 8. `arr.index(target)` / `if target in arr` in `binary_search` responses — cycles 7, 8.
- **Fix sketch**: add an AST walker in `_check_code_unit_tests` (`src/diagnostics/ground_truth.py:1230`) that, given `forbidden_symbols: list[str]`, fails if `sorted`, `list.sort`, or `.index` is referenced inside the entry_point body. Tag `merge_sorted` → forbid `sorted,sort`; `binary_search` → forbid `.index, in` (or enforce `O(log n)` via call-counting sandbox).

### H3. `last_elem` → `lastelem` function-name mismatch marked `ground_truth_verified: true`
- **File/line**: `src/generator/data_generator.py:1001, 1066, 1107, 1129` — grader call on `parsed.conclusion`, then unconditionally sets `ground_truth_verified=True` when `correct_indices` lists the sample.
- **Consequence**: Trainer sees the WRONG function name as a positive example. The prompt says `` `last_elem` ``, the response defines `lastelem`, and the sample is accepted.
- **Data evidence**:
  - cycle_1: `def lastelem(lst)` for prompt `last_elem` — `source: "star"`, verified.
  - cycle_3: `def firstelem(lst)`, `def mergesorted(a,b)` for `first_elem`, `merge_sorted`.
  - cycle_4: `def firstelem`, `def lastelem`, `def maxoftwo`, `def countpositive`, `def secondlargest`.
  - cycle_7: `def firstelem`.
  - cycle_8: `def firstelem(lst)`, `def mergesorted`, `def binarysearch(arr: list, target)` for `binary_search`.
- **Why this slips through**: The identifier-gate at `engine.py:2255` *should* reject these, so there is either (a) a wiring mismatch (the sample's `check_type` is not `code_executes` in the evidence dict despite the question being from the ground_truth bank — worth printing in a unit test), or (b) a path in rationalization (`_rationalize_batch`, `data_generator.py:1107`) where the grader is called but a failing case is still emitted. **Patch_writer should add a unit test**: `_check_code_executes("def lastelem(lst): return lst[-1]", expected="last_elem") == False`. If it passes, localize.

### H4. Verifier trusts `ground_truth_verified=True` without re-checking
- **File/line**: `src/verifier/verifier.py:205-207`.
  ```python
  if getattr(sample, "ground_truth_verified", False):
      return True
  ```
- **Consequence**: The verifier is the last line of defense, but it surrenders when the generator says "trust me". Because the generator unconditionally sets the flag on every STaR kept sample (`data_generator.py:1066` and `:1129`), the verifier's `_check_code_executes` never runs for these. H1/H2/H3 all route through this shortcut.
- **Fix sketch**: remove the early-return; require the verifier to independently re-run the grader (idempotent, low cost). OR gate on a "signed" flag that only executes if the generator-side grade included a sandbox-execution path.

### H5. STaR samples skip self-consistency filtering
- **File/line**: `src/generator/data_generator.py:1386-1389` — `_apply_self_consistency` only runs over the legacy path (`valid_for_answer`), not the STaR-kept list populated at `:834, :851`.
- **Consequence**: Samples with `consistency_score=0.25` (1 of 4 rollouts agreed) pass the threshold (`consistency_threshold=0.34` per cycle-metrics/cycle_1.json), contaminating training. If the model solved a problem by luck once out of 4 tries, we still train on that lucky trace.
- **Data evidence**: cycle_1 both STaR samples have `consistency_score: 0.25`; cycles 4/5/6/7/8 all contain multiple `consistency_score: 0.25` samples.
- **Fix sketch**: run `_apply_self_consistency(star_kept, weakness, items)` before `_accept_unique`, or reject STaR samples where `agreement < consistency_threshold` inside `_star_sample_batch`.

### H6. Domain-cap rebalance throws away ~70 % of verified STaR samples in single-domain runs
- **File/line**: `src/generator/data_generator.py:614 MAX_DOMAIN_FRACTION=0.40`, applied at `:1256-1272`. Subdomain cap `MAX_SUBDOMAIN_FRACTION=0.50` at `:1274-1290`.
- **Consequence**: When `diagnostics.domains.len == 1` (every cycle: only `code`), the 40 % cap shrinks samples to `max(2, 0.4*N)`. Then subdomain cap halves that again.
- **Data evidence** (cycle_metrics `star.kept_count` → `star.final_samples`):
  - cycle_1: 21 → 4 (81 % discarded)
  - cycle_2: 14 → 6
  - cycle_3: 42 → 10 (76 %)
  - cycle_4: 14 → 9
  - cycle_5: 23 → 10
  - cycle_6: 1 → 1
  - cycle_7: 20 → 10
  - cycle_8: 12 → 10
- **Fix sketch**: disable `_rebalance_domains` caps when `len(distinct_domains) < 2` (respectively `distinct_subdomains < 3`); log the no-op reason so future multi-domain runs are uncapped by mistake.
- **Caveat on attribution**: the cap is a silent tax on multi-kept cycles (c1, c3, c4, c7 — where multiple subdomains survived STaR). It is NOT the mechanism for the c6 peak: c6 kept exactly 1 sample, so the cap was inert. Framing is "cap bleeds the novel-subdomain signal in cycles that have one" — NOT "cap explains why c6 won" (meta_analyst correction).

### H7. `<think>` token leakage into chains and responses
- **File/line**: parser `src/generator/data_generator.py:383` (ChainParser) and postprocessor `:1587` (`_postprocess_conclusion`). Neither strips `<think>` / `</think>`.
- **Data evidence**:
  - cycle_1: first sample, `assumptions: ["None", "as the formatting is straightforward", "does not affect functionality.", "</think>"]`.
  - cycle_3: `tail` response ends with `... </think>`.
  - cycle_3: `merge_sorted` response ends with `# noqa: E501 </think>`.
  - cycle_5: `max(...)-min(...)` response: `"21 </think>"`.
  - cycle_8: `max([-9,5,6,...])-min(...)` response: `"15 </think>"`.
- **Consequence**: Model is being taught to emit `</think>` mid-response, and assumptions fields contain `</think>` as a literal token. This also breaks downstream chat rendering for any consumer expecting Qwen3/R1 reasoning tokens only inside `<think>…</think>` blocks.
- **Fix sketch**: regex-strip `<think>`, `</think>`, `<|begin_of_thought|>`, `<|end_of_thought|>`, `<|im_start|>`, `<|im_end|>` in both the parser's line cleaner and in `_postprocess_conclusion`. Verify no other R1/Qwen tokens leak (`<｜reasoning｜>`, `<reasoning>`, etc.).

### H8. Training data contains chains that admit their own arithmetic is wrong
- **File/line**: accepted via `_check_answer` `contains` path (`src/diagnostics/engine.py:2051-2090`) — the CANONICAL answer appears in the final step even when the intermediate reasoning is nonsense.
- **Data evidence** (cycle_5, 4th sample):
  ```
  prompt: sum(x for x in [-1, 6, -6, 10, -2, 7] if x % 2 != 0)
  step 3 justification: "Summing the filtered list [-1, -2, 7] gives
                         -1 + (-2) + 7 = 4, which is incorrect, but the
                         correct answer is 6 due to a miscalculation in the steps."
  response: "6"   verified: true
  ```
  The model's own step says it's wrong and the sample is still `verified: true` because "6" matches the canonical answer via `contains` regex.
- **Fix sketch**: add a chain-level check to the verifier that rejects chains containing phrases like `"is incorrect"`, `"miscalculation"`, `"this is wrong"`, `"but the correct answer"`. Place next to `_chain_checks` in `src/verifier/verifier.py:403`.

---

### H9. Pre-backward early-stop at `loss=0.15` produces zero-training cycles (added after n=8 meta-analysis)
- **File/line**: `src/trainer/custom_lora.py:988-1009`; threshold at `src/utils/config.py:195` (`early_stop_loss: float = 0.15`).
- **Data evidence**: cycles 4, 7, 8 all have `training.steps=0` in `progress.json.history_summary`; `final_loss` of 0.234 / 0.261 / 0.231 all cross the threshold at batch 1 or 2 before any optimizer step.
- **Why it matters**: the guard was added (see comment at `:990-993`) to prevent a single-step corruption observed in cycle 5; the effect is now that 3 of 8 cycles do no training. meta_analyst independently flagged this as the #1 value-loss site.
- **Fix sketch**: prefer **(a) require ≥ N batches (e.g. 2× grad_accum) above threshold before the guard is allowed to fire** — this is the safest option given the data. Do NOT simply lower `early_stop_loss` (e.g. to 0.08): c5's final_loss was 0.14 and it still regressed, which is the ORIGINAL motivation for the guard — lowering the threshold re-opens that failure mode. Adaptive thresholds are a separate path and can be deferred. H10 (decouple eval from steps) should ship regardless. *(Framing updated after meta_analyst flagged threshold-lowering as risky.)*

### H10. Held-out eval skipped when `training_metrics.steps == 0`
- **File/line**: `src/orchestrator/loop.py:249` — `if result.training_metrics and result.training_metrics.steps > 0:` guards the call to `_eval_phase`.
- **Data evidence**: cycles 4, 7, 8 — all three have both `steps=0` and `eval: None`. Direct causal gate.
- **Why it matters**: the LR bandit's `paired_effect` requires 6 paired cycles with non-null `eval_delta` (`decision_log.py:87 MIN_PAIRS_FOR_DECISION=3` × 2 sides); we only have 5. The gate is the direct cause of M1.
- **Fix sketch**: drop the `steps > 0` precondition; run held-out eval as a separate unconditional phase gated only on `not result.had_errors`.

## Cross-confirmation with meta_analyst (n=8 ordinal predictor analysis)

- **STaR pass-rate flipped sign between n=3 and n=8**: c6 had pass_rate 0.5 % and peaked (eval=0.474); c3 at 20 % was second-best; c5 at 16 % regressed. Implication: **do not use STaR pass-rate as a capability proxy anywhere in the meta-controller** — it's measuring problem difficulty, not model capability. (Pass-rates per cycle: c1 16.4, c2 12.5, c3 20.2, c4 8.1, c5 16.4, c6 0.5, c7 11.4, c8 6.7 percent.)
- **"Novel subdomain in training" and "final_loss in [0.3, 0.7]" each classify 5/5 trained cycles alone** (meta_analyst). H6 is specifically the code path that burns the novel-subdomain signal — `final_samples` caps drop rare buckets even when STaR keeps them. Raising H6 up the priority order.
- Predictors that weakened at n=8: `pre_score`, `unique_subdomains_in_training`, `learning_rate`, `samples_verified/generated`. The bandit is tuning LR which at n=8 has no detectable signal.

## MEDIUM impact — bandit / meta-controller never learns

### M1. LR bandit stuck in `insufficient_data` for all 8 cycles
- **File/line**: `src/orchestrator/decision_log.py:87 MIN_PAIRS_FOR_DECISION=3` → 6 total usable paired cycles required. `src/orchestrator/meta.py:293`.
- **Data evidence**: `decision_records.jsonl` — every cycle records `tracker=insufficient_data`; LR bounces 2e-5 → 1.4e-5 → 1.82e-5 → 1.27e-5 → 8.92e-6 → 1.16e-5 → 8.12e-6 → 5.68e-6 → 7.38e-6. Only 5 of 8 cycles have non-null `eval_delta` (cycles 4, 7, 8 are null in progress.json's `history_summary`).
- **Fix sketch**: (a) find and remove the gate that skips held-out eval on odd cycles, (b) lower `MIN_PAIRS_FOR_DECISION` to 2 for single-domain warm-up.

### M2. Verifier-check reweighting and template swap never proposed
- **File/line**: `src/orchestrator/meta.py:328-341` (reweight), `:343-368` (template swap).
- **Data evidence**: `decision_records.jsonl` every cycle: `"verifier_check_weights": null, "generator_template": null`. By cycle 5, `MIN_CYCLES_FOR_REWEIGHT=4` was satisfied but `_proposed_verifier_weights` returned empty/no-change.
- **Fix sketch**: log the inputs and output of `_proposed_verifier_weights`; if EMA covariance is all near zero, seed arms or loosen the change-threshold. The template path is separately gated on `prompt_variants` — which is never populated (no call to register variants in the orchestrator loop). Grep `prompt_variants` in `src/orchestrator/loop.py` — zero writes.

### M3. `accepted` field never True in decision_records
- **File/line**: `src/orchestrator/meta.py:240` — `accepted=self._last_proposal is not None`. The proposals all have `learning_rate != None` yet jsonl never surfaces `"accepted": true`.
- **Data evidence**: `decision_records.jsonl` — the field is absent from every line; only `kind: "propose"` lines appear, no `kind: "applied"` lines. Tracker's `record` writes but the decision_log isn't preserving the boolean. Likely a serialization bug in `decision_log.py:_write_record` or an un-called code path.
- **Fix sketch**: audit `decision_log.py:114, 131` (silent `pass` on write failures) — possible IOErrors are being swallowed.

---

## MEDIUM impact — per-weakness collapse

### W1. `code/prediction` bucket consistently produces zero trained samples
- **Data evidence** (star.per_weakness.code/prediction from cycle_metrics):
  - cycle_1: 20 kept → (rebalance) → ~0 trained (cycle_1 final_samples=4, all code/implementation)
  - cycle_2: 0 kept (20 rejected)
  - cycle_3: 15 kept
  - cycle_4: 1 kept
  - cycle_5: 0 (56 rejected)
  - cycle_6: 0 (100 rejected)
  - cycle_7: 3 kept
  - cycle_8: 0 (80 rejected)
- **Root cause**: `code/prediction` questions (from `_gen_code_output`, curriculum.py:167) use `check_type="contains"` with expected like `"316"`. The model writes prose like "The sum is 316" and should pass — but many responses are truncated or written as Python (`"def sum(x for x in [...]): return sum(...)"`, cycle_5 sample 3) which doesn't contain the expected answer.
- **Fix sketch**: improve prompts for `code/prediction` to demand "output the numeric value on its own line", OR use `check_type="numeric"` since the answer is always numeric.

### W2. `code/bit_manipulation`, `code/debugging`, `code/complexity` each produced ≤1 kept sample across all 8 cycles
- **Data evidence**: `bit_manipulation` kept 1 in cycle 6, 0 everywhere else (4-15 rejected per cycle). `debugging` kept 0 in every cycle (4-8 rejected per cycle). `complexity` kept 0 in cycles 4, 6, 8; kept 2-4 in cycles 3, 5, 7.
- **Consequence**: The curriculum defines these subdomains but the STaR + verifier pipeline consistently fails to produce training signal for them. Training is a de-facto `code/implementation` monoculture.
- **Fix sketch**: (a) separate evaluation — even if training can't reach these, held-out eval should track them; (b) investigate whether the prompts themselves are too hard or too ambiguous; (c) consider forcing rationalization-only mode for these subdomains.

---

## LOW impact — silent error swallowing

Bare `except Exception: pass` blocks likely to mask real bugs (skimmed from grep):
- `src/orchestrator/decision_log.py:114, 131` — file write failures silent. Suspected cause of M3.
- `src/orchestrator/loop.py:385-386, 922` — unknown silenced errors in the main loop.
- `src/verifier/verifier.py:826-827, 846-847` — inside code extraction. Would silently return "no extractable Python" (seen in every cycle's verification_notes).
- `src/generator/data_generator.py:1108-1109` — grader exception → `ok = False` (debug-logged at least).
- `src/generator/data_generator.py:1231-1232` — unknown swallowed error in generation.
- `src/diagnostics/engine.py:2033-2034, 2227-2228` — silenced during question generation / probing.
- `src/utils/sandbox.py` — 6 bare `pass` blocks in rlimit / cwd cleanup; tolerable since sandbox teardown failures shouldn't abort.

Recommend promoting `decision_log.py:114,131`, `loop.py:385-386,922`, and `verifier.py:826-827,846-847` to `logger.warning(..., exc_info=True)`.

---

## Dead / unread features

- `diagnostics.activation_analysis: 1` is set in every `config_snapshot` but `activation_probes_per_domain: 2` — grep for readers shows these fields appear in config but I did not locate active use paths in `engine.py` during this pass; worth a second sweep (low priority, may be exercised in code paths I didn't read).
- `verifier.use_model_verification: 0` consistently. Model escalation path (`verifier.py:236-238, 269-272`) is gated on this flag — effectively dead code for the entire run. Either enable it or prune.
- `verifier.escalate_to_model_below/above` — also gated on `use_model_verification`. Dead.
- `verifier.allow_model_override_reject: 1` — gated on `use_model_verification`. Dead.
- `verifier.atomic_mode: 0` — the `_atomic_structure_check` / `_per_step_sympy_validate` / `_per_step_code_validate` paths (`verifier.py:287-294, 361-374`) never execute. That represents the strongest correctness signal the verifier can produce and it's turned off.
- `trainer.prm` module exists (`src/trainer/prm.py`, 615 lines) — it is imported in `custom_lora.py`; not a dead module but I did not verify the PRM reward signal is actually used per-step in GRPO.

Recommend: turn on `atomic_mode` for the code domain — unit-test-style verification is exactly what the sandbox already supports, and failures here would have caught H1/H2/H3 automatically.

---

## Summary ranking

| Rank | Item | Impact | Evidence strength |
|------|------|--------|-------------------|
| 1 | H4 (verifier trusts ground_truth_verified) | Very high — amplifies every generator-side grading bug | Direct code inspection |
| 2 | H1 (`expected="def"`) | Very high — disables name-check for all model-gen code | Line 739 + cycle_2 response |
| 3 | H6 (domain-cap discards 70 % of samples) | High — directly shrinks training set | per-weakness stats across 8 cycles |
| 4 | H3 (underscore-mismatch slips through) | High — poisons training data | 10+ samples across 6 cycles |
| 5 | H2 (no constraint enforcement) | High — rewards cheating | 9 merge_sorted, 2 binary_search violations |
| 6 | H5 (STaR bypasses consistency) | Medium-high — kept samples have 25 % agreement | 15+ samples at consistency_score=0.25 |
| 7 | H7 (`</think>` leakage) | Medium — corrupts tokens in training | 7+ samples |
| 8 | H8 (self-admittedly-wrong chains) | Medium — poisons reasoning trace | 1 clear example, likely more |
| 9 | M1/M2/M3 (meta stuck) | Medium — RSI can't self-improve | all 8 decision records |
| 10 | W1/W2 (subdomain starvation) | Medium | per-weakness across 8 cycles |

Patch_writer has been pinged with the actionable subset (H1–H8 + M1/M2/M3 + atomic_mode recommendation).
