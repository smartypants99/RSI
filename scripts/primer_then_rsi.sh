#!/usr/bin/env bash
# Blocking wait for primer training to finish, validate, then launch RSI.
# Meant to be run on the pod after scripts/train_format_primer.py was kicked
# off in the background.
set -euo pipefail

ADAPTER=${1:-outputs/format_primer_adapter}
LOG=outputs/format_primer_train.log

echo "[primer_then_rsi] Waiting for $ADAPTER/lora_weights.pt…"
until [ -f "$ADAPTER/lora_weights.pt" ] && [ -f "$ADAPTER/manifest.json" ]; do
    if ! pgrep -f train_format_primer.py > /dev/null; then
        echo "[primer_then_rsi] FATAL: train_format_primer.py no longer running and adapter not produced."
        tail -50 "$LOG" || true
        exit 1
    fi
    sleep 20
done
echo "[primer_then_rsi] Adapter present."
cat "$ADAPTER/manifest.json"

echo "[primer_then_rsi] Validating parse pass rate…"
/venv/main/bin/python scripts/validate_format_primer.py \
    --model unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit \
    --load-in-4bit \
    --adapter "$ADAPTER" \
    --n 20 --pass-threshold 0.8

echo "[primer_then_rsi] Validation passed. Launching RSI…"
exec bash run_deepseek.sh
