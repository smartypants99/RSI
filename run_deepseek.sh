#!/usr/bin/env bash
# RSI on DeepSeek-R1-Distill-Qwen-32B via bitsandbytes 4-bit (QLoRA).
# Pre-quantized weights published by unsloth — ~18 GB on disk vs ~64 GB
# for raw bf16, so both download and VRAM footprint stay inside the
# 48 GB GPU budget. Training path: 4-bit frozen base + custom-LoRA
# adapters in fp16. Inference: vLLM with load_format=bitsandbytes.
set -euo pipefail

export PYTORCH_ALLOC_CONF=expandable_segments:True

RESUME_ARG=""
if [ "$#" -ge 1 ]; then
    RESUME_ARG="$*"
fi

exec python main.py \
    --model unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit \
    --load-in-4bit \
    --use-vllm \
    --gpu-memory-utilization 0.75 \
    --domains code,math,logic \
    --use-dora \
    --mode rsi \
    --enable-task-synthesis \
    --synthesis-tasks-per-cycle 30 \
    --property-consensus-threshold 0.7 \
    --consistency-samples 3 \
    --samples-per-weakness 60 \
    --consistency-threshold 0.34 \
    --lora-rank 16 \
    --plateau-patience 8 \
    --heldout-repetitions 2 \
    --max-cycles 40 \
    --write-cycle-metrics \
    --write-cycle-samples \
    ${RESUME_ARG}
