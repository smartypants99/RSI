#!/usr/bin/env bash
# One-shot setup for a fresh vast.ai pod. Idempotent — safe to re-run.
#
# Installs the dependency set the RSI loop needs: transformers, accelerate,
# bitsandbytes, peft, datasets, vllm, z3-solver, sympy. Skips flash-attn
# (vLLM ships its own flash-attn wheel; building from source spawns 300+
# compiler procs and takes 30-60 min — not worth it).
#
# Usage on pod:
#     cd /workspace/RSI && bash scripts/pod_setup.sh
#
# After setup completes:
#     bash run_deepseek.sh
set -euo pipefail

PIP="/venv/main/bin/pip"
PY="/venv/main/bin/python"

if [ ! -x "$PIP" ]; then
    echo "[pod_setup] /venv/main/bin/pip not found; this script targets vast.ai's standard pod image."
    exit 1
fi

echo "[pod_setup] Python: $($PY --version)"
echo "[pod_setup] Installing core deps…"
"$PIP" install --quiet \
    "transformers>=4.30" \
    "accelerate>=0.20" \
    "sympy>=1.12" \
    "bitsandbytes>=0.41" \
    "z3-solver>=4.12" \
    peft \
    datasets

echo "[pod_setup] Installing vllm (binary wheel; ~2-3 min)…"
"$PIP" install --quiet vllm

echo "[pod_setup] Verifying imports…"
"$PY" - <<'PYEOF'
import torch, vllm, transformers, bitsandbytes, peft, datasets
print(f"  torch        {torch.__version__}")
print(f"  vllm         {vllm.__version__}")
print(f"  transformers {transformers.__version__}")
print(f"  bitsandbytes {bitsandbytes.__version__}")
print(f"  peft         {peft.__version__}")
print(f"  datasets     {datasets.__version__}")
PYEOF

echo "[pod_setup] Done. Launch with:"
echo "    cd /workspace/RSI && bash run_deepseek.sh"
