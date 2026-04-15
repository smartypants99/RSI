# timedilate — AI Time Dilation Runtime

Give an AI model more subjective thinking time. A dilation factor of *N* runs
roughly *N* reasoning cycles (score → critique → refine → repeat), optionally
bounded by a wall-clock `--time-budget`.

## Install

```bash
pip install -e .
```

## Quick start

```bash
timedilate run "Write a quicksort in Python" --factor 100
timedilate run "Prove the halting problem is undecidable" --factor 1000000 --time-budget 5
timedilate explain --factor 1000000 --time-budget 5
```

## Key flags

- `--factor` — dilation multiplier (e.g. 10, 1000, 1000000)
- `--time-budget` — real wall-clock seconds
- `--branch-factor`, `--patience`, `--early-stop` — controller tuning
- `--gpu-mem-util`, `--max-model-len`, `--dtype`, `--enforce-eager`, `--swap-space` — vLLM hardware
- `--temperature`, `--seed` — sampling
- `--no-critique`, `--no-cot` — disable self-critique / chain-of-thought
- `--stream-progress` — live score sparkline
- `--report` — write a JSON report next to the run

Short alias: `td` is equivalent to `timedilate`.
