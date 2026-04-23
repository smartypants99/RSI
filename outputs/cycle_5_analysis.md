# Cycle analysis — cycle=5

- cycle_dir: `/Users/milannarula/Desktop/Recursive-self-improvment/outputs/cycle_5`

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
- Accepted samples: **25**
- ...with `verdict_warnings` containing `any_fail`: **20** (80.00%)
- Mean heldout Δ on domains touched by *warned* training samples: `N/A`
- Mean heldout Δ on *clean-only* domains: `N/A`

## ρ decomposition
| domain | n | ρ(pre,post) |
|---|---:|---:|

## Proposer bottleneck
- Total attempts: **135**, succeeded: **0**, total time: `0.0s`
| failure_reason | count | total_time_s |
|---|---:|---:|
| missing_problem | 58 | 0.0 |
| unknown | 44 | 0.0 |
| missing_entry;missing_reference;too_few_tests;difficulty_below_frontier:0.00<0.05 | 12 | 0.0 |
| missing_entry;missing_reference;too_few_tests | 11 | 0.0 |
| too_few_tests | 3 | 0.0 |
| missing_reference;too_few_tests | 3 | 0.0 |
| too_few_tests;difficulty_below_frontier:0.00<0.05 | 3 | 0.0 |
| difficulty_below_frontier:0.00<0.05 | 1 | 0.0 |

## Bottom line — 3-bullet TL;DR
1. Training-health signals missing — cannot attribute.
2. Damage-probe signals missing.
3. Verifier noisy: 80.0% of accepted samples carry `any_fail` warnings.
