# Training dynamics: why cycle 2 worked and cycle 3 didn't

Raw numbers from `outputs/logs/cycle_*.json`:

|  | cycle 1 | cycle 2 | cycle 3 |
|---|---|---|---|
| samples used | 1 | 5 | 9 |
| optimizer steps | 1 | 1 | **25** |
| final loss | 1.12 | 0.64 | **0.044** |
| avg loss | 1.12 | 0.74 | 0.15 |
| lora rank | 64 | 64 | **16** |
| rsLoRA scaling (α/√r) | 16 | 16 | **32** |
| learning rate | 2e-5 | 9.9e-6 | 1.15e-5 |
| post-training Δ | −0.325 | **+0.150** | −0.125 |
| held-out Δ | — | **+0.188** | (drift) |

## The ratio that matters: samples per optimizer step

- Cycle 1: 1 / 1 = **1.0 sample per step**
- Cycle 2: 5 / 1 = **5.0 samples per step** ✓
- Cycle 3: 9 / 25 = **0.36 samples per step** (each sample traversed 2.8× during training)

Each gradient step averages over its batch. More samples per step → update reflects
a broader signal. Cycle 3's 0.36 samples/step means each sample was effectively
"trained on" 2.8 times — classic memorization territory.

**Derived rule:** target ≥3 samples per optimizer step. Below 1.0 is dangerous.

## Effective parameter-movement factor

Cycle 3's weight update magnitude relative to cycle 2, decomposed:

| factor | cycle 2 | cycle 3 | ratio |
|---|---|---|---|
| rsLoRA scaling | 16 | 32 | 2.0× |
| learning rate | 9.9e-6 | 1.15e-5 | 1.16× |
| optimizer steps | 1 | 25 | 25× |

Combined: **~58× more effective parameter movement in cycle 3.** With the same model
and similar training data, 58× movement is the difference between a nudge and overwriting
pretrained representations with 9 specific problems.

## Loss trajectory as overfit signal

- Final loss > 0.5: model is still generalizing (cycle 1: 1.12, cycle 2: 0.64)
- Final loss 0.1–0.3: training has converged on the specific training set
- Final loss < 0.1: memorization — model has found shortcut features of the exact training samples (cycle 3: 0.044)

**Proposed threshold:** early-stop if training loss drops below **0.15** on any batch.
At that point the model has learned the training set faster than it can generalize
from it.

## What would have made cycle 3 succeed

Targeting cycle-2 regime (5 samples/step, loss ~0.65) from cycle 3's 9 samples:

- Need ~2 optimizer steps (9 samples ÷ 4-5 samples/step)
- With batch_size=2 and 2 steps: 4 batches total → grad_accum=2, num_epochs=1, OR grad_accum=4, num_epochs=2
- Lower LR: 5e-6 instead of 1.15e-5 to offset the 2× rank-16 scaling boost
- Early stop at loss 0.15

Expected outcome: 2-3 optimizer steps, final loss 0.4-0.7, moderate weight update,
likely held-out improvement similar to cycle 2.

## Key knobs the code should self-adjust

1. **Expected steps per cycle = samples × epochs / (batch × grad_accum)**.
   Target this in [2, 8]; auto-scale `grad_accum` if out of range.

2. **Loss floor guard**: abort training when `final_loss < 0.15 * initial_loss` or
   just `< 0.15` absolute.

3. **Rank-adjusted LR**: effective step size scales with α/√r, so when rank
   changes, the effective LR should compensate — specifically, when rank halves,
   scaling doubles, so LR should halve to keep the update magnitude constant.
