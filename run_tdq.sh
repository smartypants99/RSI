#!/usr/bin/env bash
# TDQ-backend runner: uses DeepSeek-R1-Distill-Qwen-32B.tdq (~16 GB)
# decompressed into an HF model via TDQModelHF. No vLLM — all inference
# runs through HF .generate(). Trades throughput for a much stronger
# reasoning base that can actually write self-consistent (problem,
# reference, tests) triples (the failure mode that stuck us at 0.5
# accepted/cycle on Qwen3-8B base).
set -euo pipefail

export PYTORCH_ALLOC_CONF=expandable_segments:True
# Make sure the TDQ tooling is importable. Override by exporting
# TDQ_INFERENCE_DIR before running.
export TDQ_INFERENCE_DIR="${TDQ_INFERENCE_DIR:-/workspace}"

RESUME_ARG=""
if [ "$#" -ge 1 ]; then
    RESUME_ARG="$*"
fi

exec python main.py \
    --backend tdq \
    --model /workspace/DeepSeek-R1-Distill-Qwen-32B.tdq \
    --dtype float16 \
    --max-seq-length 4096 \
    --domains code,math,logic \
    --use-dora \
    --mode rsi \
    --enable-task-synthesis \
    --synthesis-tasks-per-cycle 20 \
    --property-consensus-threshold 0.7 \
    --consistency-samples 3 \
    --samples-per-weakness 60 \
    --consistency-threshold 0.34 \
    --lora-rank 8 \
    --plateau-patience 8 \
    --heldout-repetitions 2 \
    --max-cycles 25 \
    --write-cycle-metrics \
    --write-cycle-samples \
    ${RESUME_ARG}
