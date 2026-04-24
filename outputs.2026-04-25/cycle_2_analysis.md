# Cycle analysis — cycle=2

- cycle_dir: `outputs/cycle_2`

## Training health
- Steps: **12**
- Loss: `N/A` → `N/A`
- max(grad_norm_B): `N/A`
- Fraction of steps where B moved (>1e-05): `100.00%`
- Mean applied LR_B: `N/A`

## Training damage probe (per-domain pre→post score delta)
| domain | n_heldout | pre_mean | post_mean | Δ | trained_in_cycle |
|---|---:|---:|---:|---:|---:|

## Verifier noise
- Accepted samples: **30**
- ...with `verdict_warnings` containing `any_fail`: **29** (96.67%)
- ...with ≥2 non-PASS verdicts (real disagreement): **0** (0.00%)
- Mean heldout Δ on domains touched by *warned* training samples: `N/A`
- Mean heldout Δ on *clean-only* domains: `N/A`

## ρ decomposition
| domain | n | ρ(pre,post) |
|---|---:|---:|

## Proposer bottleneck
- Total attempts: **35**, succeeded: **0**, total time: `0.0s`
| failure_reason | count | total_time_s |
|---|---:|---:|
| unknown | 35 | 0.0 |

## Bottom line — 3-bullet TL;DR
1. Training-health signals missing — cannot attribute.
2. Damage-probe signals missing.
3. Verifier noisy: 96.7% of accepted samples carry `any_fail` (above structural quorum floor ~66%).
