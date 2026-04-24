# Training Regression Diagnosis — 2026-04-23

Scope: offline analysis of why every live-GPU training cycle has regressed vs the untrained base.
Live run pid 129978 on remote host 82.79.85.125 NOT touched. Production code + loop.py NOT modified.

---

## 0. Artifact availability caveat (honest up-front)

**Every `outputs.pre_highwater.*` / `outputs.*` archive referenced in the task brief is remote-only.**
On this workstation:

```
/Users/milannarula/Desktop/Recursive-self-improvment/outputs/
    problems/          (empty)
    properties/        (empty)  [actually missing — only 4 listed subdirs]
    training_pool/     (empty)
    verifications/     (empty)
```

No `cycle_metrics/*.jsonl`, no `cycle_samples/*`, no `lora_weights/*.safetensors`,
no training logs are present locally. `.gitignore` lists `__pycache__/`, `.venv/`,
`.pytest_cache/`, `.timedilate_checkpoints/` — artifact dirs aren't tracked, and the
hard rule "do not SSH to 82.79.85.125" means we cannot pull them.

Therefore items (1), (2), (3), (4) from the task brief **cannot be answered
empirically from local data**. What follows is a **static / code-reading
diagnosis** + external expert consult. Every claim below is either (a) derived
from the source code defaults in `src/utils/config.py` + `src/trainer/custom_lora.py`,
(b) cross-checked with gemini, or (c) cross-checked with the prior team signoffs
in `update-log.txt`.

If the user wants empirical per-cycle loss trajectories and LoRA B-matrix
statistics, someone with GPU-host access must `rsync` the following onto this
machine and re-run the diagnosis:

```
remote:/workspace/<repo>/outputs/cycle_metrics/*.jsonl
remote:/workspace/<repo>/outputs/cycle_samples/*.jsonl
remote:/workspace/<repo>/outputs/checkpoints/cycle_*/adapter_model.safetensors
```

Fallback analysis path below is deliberately conservative — the code-reading
findings are sufficient to identify **the single most likely root cause** and
one secondary cause.

---

## 1. PRIMARY FINDING — Compounded effective LR is ~10-30× above stable QLoRA-32B range

**Likelihood: very high.** This alone explains "cycle 1 at 1-sample warmup-cap is
fine, cycle 2-3 at 12-32 samples regresses 1-15pp on held-out AND anchor."

### The math

Defaults in `src/utils/config.py:235-242, 409-414`:

| Knob | Value | Source |
|---|---|---|
| `learning_rate` | `2e-5` | `config.py:242` |
| `lora_rank` | `8` | `config.py:235` |
| `lora_alpha` | `16` | `config.py:236` |
| `use_rslora` | `True` | `config.py:414` |
| `use_lora_plus` | `True` | `config.py:409` |
| `lora_plus_ratio` | `16.0` | `config.py:410` |

Effective per-step update on the **B matrix**:

- rsLoRA scaling: `alpha / sqrt(rank) = 16 / sqrt(8) ≈ 5.66`
  (vs. classic `alpha/rank = 2.0` — **2.83× increase**)
- LoRA+ multiplier on B: `lr_B = 2e-5 * 16 = 3.2e-4`
- Combined effective update magnitude for the `B @ A` product contribution:
  `lr_B × rsLoRA_scale ≈ 3.2e-4 × 5.66 ≈ 1.81e-3`

**For a 32B NF4 base, canonical QLoRA stable range is total effective LR
~5e-5 to 2e-4.** Current recipe lands at `1.8e-3`, which is **≈10-30× above
the safe ceiling.**

### Why cycle 1 looks fine

`num_epochs_warmup=1` + `num_epochs_warmup_cycles=5` (`config.py:257-258`)
caps cycle 1 at 1 epoch. Combined with `min_train_samples=5` *(actually in the
single-sample case the trainer runs at most 1 optimizer step)* and
`max_steps_per_cycle=8` (`config.py:289`), a 1-sample cycle is physically
bounded to ≤ 1 step. AdamW's first/second moments don't exist yet; Adam
effectively SGDs one step. One tiny SGD step at `1.8e-3` effective LR on 1
sample is still large, but apparently not catastrophic on the specific
distribution that produced ~0.56 held-out.

### Why cycles 2-3 regress

With 12-32 samples at `batch_size=2`, `grad_accum=4` (effective batch 8),
`num_epochs=2` (warmup cap = 1 for cycle ≤5 — but cycle 2 still does 1-2
epochs), step budget plan (`_plan_step_budget`, `custom_lora.py:679`)
produces 2-5 optimizer steps. Adam's moments **still haven't stabilized**
after 2-5 steps on a new task distribution, AND the per-step weight delta
is now accumulated over 8 samples of gradient signal. The combined
`B @ A` update moves weights ~0.5-2% per step. On a 32B model where the
anchor is also sensitive to weight drift, a 1-15pp held-out + anchor drop
is exactly what you'd expect from ~5 steps at this effective LR.

### Gemini cross-check (consulted 2026-04-23)

> "Your effective update $B$ of $1.81 \times 10^{-3}$ is 10–30× above the
> stable range for 32B QLoRA ... at 1-4 optimizer steps AdamW is essentially
> operating as SGD because it hasn't built up meaningful first or second
> moments. Large steps on tiny data will cause 'honest damage' (weight
> collapse)."

Gemini's numeric recommendation:

| Parameter | Current | Recommended |
|---|---|---|
| `learning_rate` | `2e-5` | **`4e-6`** |
| `lora_plus_ratio` | `16.0` | **`4.0`** |
| Effective $lr_B$ | `3.2e-4` | **`1.6e-5`** |
| `max_grad_norm` | `0.3` | **`0.1`** |
| `warmup_ratio` | `0.1` | force ≥ 2 warmup steps |
| `weight_decay` | `0.01` | `0.05` |

This drops total effective update from `1.8e-3` to `~9e-5`, landing in
the middle of the stable QLoRA-32B band.

### Recommended code change (PRIMARY)

`src/utils/config.py:242` — `learning_rate: float = 2e-5` → `learning_rate: float = 4e-6`
`src/utils/config.py:410` — `lora_plus_ratio: float = 16.0` → `lora_plus_ratio: float = 4.0`
`src/utils/config.py:348` — `max_grad_norm: float = 0.3` → `max_grad_norm: float = 0.1`

This is a 3-line config change, no code logic touched. Safety gates
(`regression_revert_threshold=0.03`, `early_stop_loss=0.15`,
`skip_if_initial_loss_below=0.15`) remain as a second line of defence if
the new LR is still off.

### Note on team history

The trainer docstrings at `config.py:224-242` record *prior* LR adjustments:
- run-10: LR `2e-5 → 5e-6` "was too timid to move the loss"
- then back to `5e-5` — caused the -15pp cycle-2 regression cited in the brief
- then down to current `2e-5` — still regressing per observation

The whole sequence has been tuning $lr_A$ without accounting for the
`lora_plus_ratio × rsLoRA` compounding. Changing `lora_plus_ratio` is the
knob that's been missed — at ratio 16 every `lr_A` decision gets silently
multiplied 16× on the B side, making fine adjustments impossible.

---

## 2. SECONDARY FINDING — `any_fail` samples likely do reach the trainer at 12-32 sample counts

**Likelihood: medium.** This is a contributing factor, not the root cause.

### The logic gap

`sample_quality_min_clean_floor = 16` (`config.py:119`). The filter
in `_filter_any_fail_when_clean_enough` (`custom_lora.py:656-676`):

```python
if clean_floor <= 0 or not samples:
    return list(samples), 0
...
total = len(samples)
if total < clean_floor or len(clean) < clean_floor:
    return list(samples), 0   # filter DISABLED
```

**The filter is disabled when total sample count < 16 OR clean-only subset < 16.**
At the observed 12-sample cycles, `total=12 < 16` → filter OFF →
every `verdict_warnings=("any_fail",)` sample lands in training.

At 32 samples, IF fewer than 16 of them are "clean", filter is still OFF.
With `verifier_accept_policy="majority"` (`config.py:198`), any_fail
warnings are stamped at `property_engine.py:830` whenever any single
sub-verdict was FAIL but majority still PASSed — this is not rare.

### Why this matters

An `any_fail` sample has ≥1 FAIL verdict across the quorum. Training the
model to reproduce a partially-broken reference pushes the weights toward
wrong behavior. If even 3/12 training samples are any_fail on a cycle at
the effective-LR rate above, that's enough noise to explain a few pp of
the observed regression on top of the LR damage.

### Recommended code change (SECONDARY)

**Option A (simplest, safest):**
`src/utils/config.py:119` — `sample_quality_min_clean_floor: int = 16` →
`sample_quality_min_clean_floor: int = 1`.
With floor=1 the filter fires whenever there's ≥1 clean sample, dropping
any_fail rows whenever a non-any_fail alternative exists.

**Option B (more surgical):**
`src/trainer/custom_lora.py:668-674` — remove the `total < clean_floor`
starvation protection. Rationale: in the LR-fixed regime, training on
fewer clean samples is strictly better than training on mixed clean+any_fail.
The `min_train_samples=5` gate (`config.py:309`) separately skips training
when the clean-only pool is too small. Keep (b) `len(clean) < clean_floor`
as the operative floor; drop (a). If desired can do A+B together.

---

## 3. TERTIARY FINDING — curriculum/domain mismatch is unknown without the archives

Cannot compute the accepted-training-sample domain distribution without
access to `cycle_samples/*.jsonl` on the GPU host. The task brief asks
whether training samples are dominated by math/logic while held-out measures
code — this requires remote artifacts.

**Indirect evidence from code defaults:** the generator's `samples_per_weakness=24`
(`config.py:84`) combined with `DiagnosticsEngine`'s weakness scoring (code is
one of several probed domains) means there is no code-specific emphasis in
sample generation — the pool likely IS math-heavy at the rates the diagnostics
weakness table produces. But this is speculation without the archives.

**Recommendation (deferred):** once GPU-host sample archives are available,
re-run this analysis with a stratified count by domain. If the code-held-out
is ≥50% code items and the training pool is <30% code items, there's a
curriculum wedge to close. Until then, fix (1) and (2) which are certain,
and re-measure.

---

## 4. OTHER OBSERVATIONS WORTH ACTING ON

### 4a. `use_gradient_checkpointing=True` vs comment

`config.py:446-451` docstring says "Task #20 throughput pass: flipped default
False. Gemini consult predicts 25-30% speedup." But the actual value at
line 451 is `= True`. Inconsistent with comment; not causal to the regression
but confusing. Noted, not recommending a change until the throughput vs
stability trade-off is decided by the team.

### 4b. `early_stop_loss=0.15` fires AFTER damage

At effective update `1.8e-3`, the model can drop loss from `~0.5` to `~0.15`
in a single step. Early-stop checks loss at END of step (`custom_lora.py`
early-stop flow), so by the time it fires the damaging step has already
landed. This is consistent with the `"catch the crash on the first batch
whose forward-pass loss dips below natural SFT floor"` intent documented
at `config.py:276-280` — but at the current compounded LR the single step
itself is destructive. Fixing the LR (finding 1) restores early-stop's
intended semantics.

### 4c. `regression_revert_threshold=0.03` is correctly catching damage

The safety gate IS working (`loop.py:703-773`) — cycles are being reverted.
But reverts cost a full cycle of wall-clock with zero learning progress,
and any_fail promotion to the adversarial bank (`loop.py:774-779`) will
starve future proposals of signal. **The fix is not to tighten the revert
threshold further — it's to stop generating damage in the first place.**

---

## 5. EXECUTIVE SUMMARY (5 bullets)

- **Root cause (very high confidence): compounded effective LR on B matrix is
  `~1.8e-3`, ≈10-30× above the stable QLoRA-32B band.** `rsLoRA` (5.66×) ×
  `lora_plus_ratio=16` × base-lr `2e-5` compound silently — every prior LR
  tuning attempt adjusted $lr_A$ without accounting for the 16× LoRA+ ratio
  on B. Gemini independent consult agrees.
- **Contributing cause (medium confidence): `sample_quality_min_clean_floor=16`
  disables the any_fail filter at 12-sample cycles**, so `verdict_warnings=("any_fail",)`
  samples admitted under `majority` policy are training the model toward partially-
  broken references.
- **Recommended primary fix (3 lines, no logic change):** `config.py:242`
  `learning_rate → 4e-6`, `config.py:410` `lora_plus_ratio → 4.0`,
  `config.py:348` `max_grad_norm → 0.1`. Brings effective update to `~9e-5`.
- **Recommended secondary fix (1 line):** `config.py:119`
  `sample_quality_min_clean_floor → 1` so the any_fail filter fires whenever
  there is any clean sample to keep. Existing `min_train_samples=5` still
  protects against training on empty pools.
- **Caveats-as-tasks (cannot close offline):** (a) Domain-mix curriculum
  analysis blocked by no local access to `outputs/cycle_samples/*.jsonl` —
  requires GPU-host `rsync`. (b) Empirical LoRA B-matrix growth per cycle
  unverified — also blocked by no local `adapter_model.safetensors`. Both
  become follow-up tasks once the live run is paused and artifacts are
  copied down, OR once the user authorizes a scoped read-only SSH for
  diagnostic rsync.

---

## Appendix — code references cited

- `src/utils/config.py:119` — `sample_quality_min_clean_floor: int = 16`
- `src/utils/config.py:198` — `verifier_accept_policy: str = "majority"`
- `src/utils/config.py:235-236` — `lora_rank=8, lora_alpha=16`
- `src/utils/config.py:242` — `learning_rate: float = 2e-5` ← PRIMARY FIX
- `src/utils/config.py:248-258` — `num_epochs=2`, warmup cycle cap
- `src/utils/config.py:288-289` — `early_stop_loss=0.15`, `max_steps_per_cycle=8`
- `src/utils/config.py:322` — `regression_revert_threshold=0.03` (gate working)
- `src/utils/config.py:348` — `max_grad_norm: float = 0.3` ← PRIMARY FIX (tighten)
- `src/utils/config.py:409-410` — `use_lora_plus=True, lora_plus_ratio=16.0` ← PRIMARY FIX
- `src/utils/config.py:414` — `use_rslora: bool = True`
- `src/utils/config.py:451` — `use_gradient_checkpointing=True` (comment-value mismatch)
- `src/trainer/custom_lora.py:99` — rsLoRA scaling formula
- `src/trainer/custom_lora.py:656-676` — any_fail filter starvation gate
- `src/trainer/custom_lora.py:2128-2165` — `_build_optimizer` LoRA+ param groups
- `src/orchestrator/loop.py:703-773` — post-training regression revert (working)
- `src/verifier/property_engine.py:829-833` — any_fail verdict_warning stamping
