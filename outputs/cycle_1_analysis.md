# Cycle analysis — cycle=1

- cycle_dir: `outputs/cycle_1`

## Training health
- Steps: **8**
- Loss: `N/A` → `N/A`
- max(grad_norm_B): `N/A`
- Fraction of steps where B moved (>1e-05): `100.00%`
- Mean applied LR_B: `N/A`

## Training damage probe (per-domain pre→post score delta)
| domain | n_heldout | pre_mean | post_mean | Δ | trained_in_cycle |
|---|---:|---:|---:|---:|---:|

## Verifier noise
- Accepted samples: **1**
- ...with `verdict_warnings` containing `any_fail`: **0** (0.00%)
- Mean heldout Δ on domains touched by *warned* training samples: `N/A`
- Mean heldout Δ on *clean-only* domains: `N/A`

## ρ decomposition
| domain | n | ρ(pre,post) |
|---|---:|---:|

## Proposer bottleneck
- Total attempts: **15**, succeeded: **0**, total time: `0.0s`
| failure_reason | count | total_time_s |
|---|---:|---:|
| missing_problem | 9 | 0.0 |
| unknown | 3 | 0.0 |
| missing_entry;missing_reference;too_few_tests | 2 | 0.0 |
| too_few_tests | 1 | 0.0 |

## Bottom line — 3-bullet TL;DR
1. Training-health signals missing — cannot attribute.
2. Damage-probe signals missing.
3. ρ/verifier within acceptable ranges (or data missing).
