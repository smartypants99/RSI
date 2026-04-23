#!/usr/bin/env bash
# RSI on DeepSeek-R1-Distill-Qwen-32B via bitsandbytes 4-bit (QLoRA).
# Pre-quantized weights published by unsloth — ~18 GB on disk vs ~64 GB
# for raw bf16, so both download and VRAM footprint stay inside the
# 48 GB GPU budget. Training path: 4-bit frozen base + custom-LoRA
# adapters in fp16. Inference: vLLM with load_format=bitsandbytes.
set -euo pipefail

export PYTORCH_ALLOC_CONF=expandable_segments:True

# Preflight: kill stale VLLM::EngineCore / multiprocessing.resource_tracker
# orphans from a prior crashed main.py. They get reparented to init when
# main.py dies abnormally and are invisible to nvidia-smi's compute-app
# list but still pin KV-cache VRAM. Confirmed cause of 36GB "leaked" VRAM
# blocking run on 2026-04-23 (unblocker task #1 diagnosis).
for _proc in $(pgrep -f 'VLLM::EngineCore|multiprocessing.resource_tracker' 2>/dev/null); do
    kill -9 "$_proc" 2>/dev/null || true
done

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
    --mode rsi \
    --enable-task-synthesis \
    --synthesis-tasks-per-cycle 20 \
    --property-consensus-threshold 0.7 \
    --consistency-samples 3 \
    --samples-per-weakness 60 \
    --consistency-threshold 0.34 \
    --lora-rank 16 \
    --plateau-patience 8 \
    --heldout-repetitions 1 \
    --max-cycles 40 \
    --write-cycle-metrics \
    --write-cycle-samples \
    ${RESUME_ARG}
