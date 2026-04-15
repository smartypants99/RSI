# GPU Runbook — Time Dilation Runtime

Target host: rented GPU, workspace `/workspace/TD.org`. Package: `timedilate` (lives in `Time-dialation/` subdir of repo).

All commands below are copy-paste ready for the GPU host shell. CLI entry points: `timedilate` and short alias `td` (interchangeable — this runbook uses `timedilate`).

---

## 1. First-time setup

```bash
cd /workspace/TD.org/Time-dialation
pip install -e .
python3 -c 'import timedilate; print(timedilate.__file__)'
timedilate --version
```

If `import timedilate` errors on `vllm`, install torch+vllm matching the CUDA toolkit first:

```bash
pip install --upgrade pip
pip install "vllm>=0.6.0"
pip install -e .
```

---

## 2. Pull latest

```bash
cd /workspace/TD.org
git fetch origin
git status                       # see any local drift
# If local files diverged (edits from a prior run), discard them:
git checkout -- .                # drops ALL local changes — only run if you know nothing is worth keeping
# OR target a single file:
# git checkout -- path/to/file
git pull --ff-only origin main
```

If `git pull` still refuses due to untracked files blocking a checkout:

```bash
git clean -fd                    # removes untracked files/dirs — inspect `git status` first
git pull --ff-only origin main
```

---

## 3. Clear caches

```bash
# Python bytecode caches (safe to remove)
find /workspace/TD.org -type d -name __pycache__ -prune -exec rm -rf {} +

# HuggingFace download locks (only if a prior download was killed mid-flight)
find ~/.cache/huggingface -name '*.lock' -delete 2>/dev/null
find ~/.cache/huggingface -name '*.incomplete' -delete 2>/dev/null

# vLLM compiled-graph cache (clear on vLLM upgrade or OOM storms)
rm -rf ~/.cache/vllm 2>/dev/null
```

Do NOT blow away `~/.cache/huggingface/hub` unless you want to re-download the 8B model (~16 GiB).

---

## 4. Check GPU free memory

```bash
nvidia-smi --query-gpu=memory.free,memory.total,name --format=csv
```

Re-run this right before launching — a stale process can hold VRAM. If another `python` is camping the card:

```bash
nvidia-smi                       # find the PID in the bottom table
kill -9 <PID>                    # only kill PIDs you own
```

---

## 5. VRAM decision tree

Read the free memory from section 4, then pick a row.

| Free VRAM      | Recommended flags                                             |
|----------------|----------------------------------------------------------------|
| > 40 GiB       | (defaults — no extra flags)                                    |
| 25–40 GiB      | `--gpu-mem-util 0.45`                                          |
| < 25 GiB       | `--gpu-mem-util 0.30 --max-model-len 4096 --enforce-eager`     |

Other useful tuning flags (exposed by `timedilate run`):
`--dtype`, `--swap-space` (GiB), `--seed`, `--branch-factor`, `--patience`, `--early-stop`, `--temperature`, `--stream-progress`, `--no-critique`, `--no-cot`.

Engine already has an internal OOM fallback chain (`engine.py` retries lower utilization on CUDA OOM), but starting at the right value avoids wasted init time.

---

## 6. Baseline run (factor 5)

```bash
cd /workspace/TD.org/Time-dialation
timedilate run "Write a Python function that returns the nth Fibonacci number." \
    --factor 5 \
    --report \
    --verbose \
    2>&1 | tee /workspace/TD.org/logs/baseline_$(date +%Y%m%d_%H%M%S).log
```

Append the VRAM flags from section 5 if free memory is tight.

---

## 7. Full dilation run (factor 1,000,000, 30s budget)

```bash
cd /workspace/TD.org/Time-dialation
mkdir -p /workspace/TD.org/logs
timedilate run "Design and implement a production-grade LRU cache with TTL in Python." \
    --factor 1000000 \
    --time-budget 30 \
    --report \
    --output /workspace/TD.org/logs/dilation_$(date +%Y%m%d_%H%M%S).out \
    --verbose \
    2>&1 | tee /workspace/TD.org/logs/dilation_$(date +%Y%m%d_%H%M%S).log
```

For a constrained GPU, append section 5 flags, e.g.:

```bash
timedilate run "..." --factor 1000000 --time-budget 30 \
    --gpu-mem-util 0.30 --max-model-len 4096 --enforce-eager \
    --report --verbose
```

---

## 8. Troubleshooting

### CUDA OOM at model init
- Lower `--gpu-mem-util` one step (0.60 → 0.45 → 0.30).
- Add `--max-model-len 4096` (cuts KV cache).
- Add `--enforce-eager` (skips CUDA graph capture — saves ~1–2 GiB).
- Confirm no other process holds VRAM: `nvidia-smi`.

### Model not found / download fails
- Check HF auth: `huggingface-cli whoami` (Qwen3 is public but proxied hosts sometimes throttle).
- Force re-download: `rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B` then re-run.
- Manual pre-pull: `huggingface-cli download Qwen/Qwen3-8B`.

### Qwen3 tokenizer errors
- Usually `trust_remote_code` or tokenizer version. Upgrade: `pip install -U transformers tokenizers`.
- If vLLM complains about chat template: pin `transformers>=4.45` (Qwen3 chat template requires recent transformers).
- `AttributeError` on tokenizer import usually means a stale `tokenizer_config.json` from an older Qwen variant in the HF cache. Remedy: `rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B` and re-pull.

### dtype / NaN logits
- Default `--dtype auto` picks bfloat16 on Ampere/Hopper (good for Qwen3) and float16 on older cards (T4/V100).
- If you see NaN logits on T4/V100, force `--dtype float16` explicitly.

### Pairing max-model-len with enforce-eager
- Qwen3-8B native context is 32768. With `--enforce-eager` there's no CUDA graph reuse, so per-token KV cost goes up.
- On <25 GiB free, keep `--max-model-len` at 4096; do not exceed 8192 when also using `--gpu-mem-util 0.30`.

### Slow disk / network-mounted HF cache
- Default `--swap-space 4` can thrash if disk is slow (e.g., network-mounted cache). For low-VRAM + slow disk, set `--swap-space 0` and accept a smaller effective batch.

### Health check failures
- `health_check()` raising `HealthCheckError` (ImportError/AttributeError/TypeError) is a wiring bug, NOT OOM — check `vllm` install and Python version, not the GPU.
- True OOM manifests as `health_check()` returning False with `_last_health_status="oom"`.

### Empty / truncated responses
- Raise `--max-tokens` (default 4096). For long-horizon runs bump to 8192–16384.
- Check the report JSON (`timedilate_report_*.json`) — if `cycles_completed` is low, the time-budget was too tight for init.
- Verify the model actually loaded: look for `Initializing model:` in the log.

### Cache-miss tokenization slow-down
- First cycle re-tokenizes the full system prompt. Subsequent cycles reuse the prefix — expect 5–10x speedup after cycle 1.
- If every cycle is slow, the KV cache is being evicted: lower `--max-model-len` (less likely) or raise `--gpu-mem-util` (more KV cache room). Bumping `--swap-space` (default 4 GiB) gives KV more room on host RAM.

---

## 9. Grab & share logs

```bash
# Tail the newest log
ls -t /workspace/TD.org/logs | head
tail -n 200 /workspace/TD.org/logs/<logfile>

# Bundle everything from the last run for sharing
cd /workspace/TD.org
tar -czf td_run_$(date +%Y%m%d_%H%M%S).tgz logs/ timedilate_report_*.json 2>/dev/null
ls -lh td_run_*.tgz
```

Share the `.tgz` via whatever channel the team uses. Reports include scores, cycle timings, and final output — that is usually enough to diagnose.
