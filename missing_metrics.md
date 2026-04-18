# Missing Metrics — what would help answer "why did cycle 2 help?"

Each entry: what's missing, where in src/ it could be captured, why it matters for the cycle 2 question, priority.

## HIGH priority

### 1. Per-sample / per-step training loss trajectory
- **Currently:** `TrainingMetrics` (src/trainer/custom_lora.py:472) only stores `avg_loss` and `final_loss`. Per-batch `unweighted_loss` is computed at line 858/863 but discarded after summing into `total_loss`.
- **Why:** A cycle with final_loss 0.04 could be fitting the 9 samples cleanly or catastrophically memorizing a narrow pattern. Without the curve we can't tell cycle 2's training dynamics from cycle 3's.
- **Capture:** append `(step, loss)` to a list on TrainingMetrics; write alongside history.json.

### 2. Per-diagnostic-item correctness, pre vs post
- **Currently:** `DiagnosticResult` keeps aggregate `overall_score` / `domain_scores`. Individual item outcomes exist during `diagnostics.run()` (engine.py) but are reduced before return — no per-question id/result matrix is persisted.
- **Why:** The "+0.188 then -0.188" swing is exactly the kind of thing a question-level diff would explain (e.g. 3 specific items flipped, rest noise). Currently we cannot attribute held-out deltas to specific questions.
- **Capture:** persist `[{question_id, prompt_hash, expected, got, correct, confidence}]` per diag run to `outputs/diag/cycle_{n}_{phase}.jsonl`.

### 3. Held-out eval seed variance / bootstrap CI
- **Currently:** `HELDOUT_CYCLE_SEED = 0xE7A1` (loop.py:786) — single fixed seed, single point estimate per cycle. No repetition, no CI.
- **Why:** The same checkpoint now scores 0.338 after re-eval; clearly the per-seed variance is large. Without N-sample bootstraps we cannot separate signal from noise.
- **Capture:** run held-out eval with k seeds (e.g. 3-5), log mean ± stderr.

### 4. Which weaknesses each training sample targeted (and whether those weaknesses moved)
- **Currently:** `TrainingSample.target_weakness` exists (data_generator.py:70). Not carried through to TrainingMetrics, not correlated post-hoc with weakness score deltas.
- **Why:** Cycle 2's +0.188 might be driven entirely by one weakness bucket. Without the mapping we can't say.
- **Capture:** persist `{target_weakness: [sample_ids]}` and compute delta-per-bucket after post-diag.

### 5. Gradient norms per step (pre-clip)
- **Currently:** `clip_grad_norm_` is called at lines 871/883/1113/1123/1585/1593 — the return value (pre-clip norm) is ignored.
- **Why:** Blown-up grads on cycle 3 vs stable grads on cycle 2 would be a smoking gun for training instability driving regression.
- **Capture:** record the scalar returned by `clip_grad_norm_`; log mean/max/95p per cycle.

## MEDIUM priority

### 6. Per-sample reasoning chain length / structure
- **Currently:** `num_reasoning_steps` is in sample.to_dict() (data_generator.py:124) but never aggregated into cycle metrics.
- **Why:** If cycle 2 kept longer, more structured chains and cycle 3 got short hacks, that shows up here.
- **Capture:** add chain-length histogram to `diversity_stats`.

### 7. STaR filter breakdown by weakness
- **Currently:** logged only as global counts: `kept=14, rejected=122, rationalized=4` (log line 151).
- **Why:** 122 rejected across 5 buckets — is the rejection uniform or concentrated? A bucket with 0 kept samples means the weakness wasn't trained on at all.
- **Capture:** break `kept/rejected/rationalized` down per weakness bucket in data_generator's STaR path.

### 8. Verifier check-by-check pass rates
- **Currently:** `self.verifier._last_check_mean_scores` exists (loop.py:837) but only fed to MetaController — not persisted in cycle_result.
- **Why:** If cycle 2 passed consistency checks at a different rate than cycle 3, it tells us whether "9/10 passed" means the same thing across cycles.
- **Capture:** include per-check dict in progress.json / history.json.

### 9. DPO reward margins per pair
- **Currently:** `avg_reward_margin` aggregated in TrainingMetrics. Per-pair margin discarded.
- **Why:** Distribution of margins distinguishes "9 weak pairs" from "3 strong + 6 noise".
- **Capture:** log histogram of final-pass margins.

### 10. LoRA weight-delta magnitude per cycle
- **Currently:** weights saved (loop.py:711), but no summary statistic computed (‖ΔW‖ vs base).
- **Why:** A tiny delta that somehow flips +0.188 is suspicious; a large delta that gains nothing is also diagnostic.
- **Capture:** compute Frobenius norm of LoRA B·A per layer at save time.

### 11. Per-phase wall-clock timing (already partially captured)
- **Currently:** `result.phase_times` is populated in loop.py, but not always serialized in a consistent way; `held_out_eval` in particular lives outside `phase_times` (loop.py:246 writes it under key "eval").
- **Why:** Helps diagnose whether a cycle spent time in the right places.
- **Capture:** unify timing keys and serialize all of them in history.json.

## LOW priority

### 12. Activation norms pre/post training on fixed probe inputs
- **Why:** Detects representation drift beyond loss. Useful for catching "model forgot something" beyond what weaknesses cover.
- **Capture:** once per cycle, feed a fixed 5-prompt probe, log hidden-state norms at selected layers.

### 13. Token-level confidence on diagnostic outputs
- **Currently:** per-step `[C:x]` markers feed calibration ECE/Brier, but not persisted per question.
- **Why:** Drop in confidence on specific items pre-vs-post could distinguish "forgot" from "never knew".

### 14. Sample deduplication / novelty vs prior cycles
- **Currently:** `diversity_stats` captured but not compared cross-cycle.
- **Why:** Cycle 3 may be regenerating cycle 2's samples with minor perturbations; overlap ratio would show this.

### 15. Bandit/MetaController decision audit trail per proposal
- **Currently:** `meta_decisions.jsonl` logs decisions; counterfactual ("what would the score have been without this LR change?") is never estimated.
- **Why:** Hard to assess meta layer's contribution without a baseline arm.

### 16. Which layers the LoRA was injected into (layer_health at inject time)
- **Currently:** `lora_layers_injected` count only. The `weak_layers` dict passed to `inject_lora` (loop.py:656) is not persisted.
- **Why:** If cycle 2 hit different layers than cycle 3, capacity allocation may explain the divergence.

### 17. Random seed state at each phase
- **Why:** The resumed-cycle-3 producing a different score than original-cycle-3 is partially a seed-variance story. Logging the exact RNG state (torch, numpy, python, vllm sampling) per phase would make this reproducible.
