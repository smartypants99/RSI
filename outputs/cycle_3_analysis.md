# Cycle analysis — cycle=3

- cycle_dir: `outputs/cycle_3`

## Training health
- Steps: **14**
- Loss: `N/A` → `N/A`
- max(grad_norm_B): `N/A`
- Fraction of steps where B moved (>1e-05): `100.00%`
- Mean applied LR_B: `N/A`

## Training damage probe (per-domain pre→post score delta)
| domain | n_heldout | pre_mean | post_mean | Δ | trained_in_cycle |
|---|---:|---:|---:|---:|---:|

## Verifier noise
- Accepted samples: **15**
- ...with `verdict_warnings` containing `any_fail`: **14** (93.33%)
- Mean heldout Δ on domains touched by *warned* training samples: `N/A`
- Mean heldout Δ on *clean-only* domains: `N/A`

## ρ decomposition
| domain | n | ρ(pre,post) |
|---|---:|---:|

## Proposer bottleneck
- Total attempts: **55**, succeeded: **0**, total time: `0.0s`
| failure_reason | count | total_time_s |
|---|---:|---:|
| missing_problem | 25 | 0.0 |
| unknown | 18 | 0.0 |
| missing_entry;missing_reference;too_few_tests | 4 | 0.0 |
| missing_entry;missing_reference;too_few_tests;difficulty_below_frontier:0.00<0.05 | 4 | 0.0 |
| too_few_tests | 2 | 0.0 |
| missing_reference;too_few_tests | 1 | 0.0 |
| difficulty_below_frontier:0.00<0.05 | 1 | 0.0 |

## Bottom line — 3-bullet TL;DR
1. Training-health signals missing — cannot attribute.
2. Damage-probe signals missing.
3. Verifier noisy: 93.3% of accepted samples carry `any_fail` warnings.
