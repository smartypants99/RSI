#!/usr/bin/env bash
# One-shot runner for the code-focused RSI loop.
# All flags baked in so a single `bash run.sh` works from a clean shell
# without the terminal wrapping that corrupts long paste commands.
#
# Pass `--resume outputs/checkpoints/cycle_N` as $1 to continue from a
# known-good checkpoint; omit to start fresh from the base model.
set -euo pipefail

RESUME_ARG=""
if [ "$#" -ge 1 ]; then
    RESUME_ARG="$*"
fi

# To opt in to synthesis mode (task_synthesizer + property_engine pipeline), add:
#   --enable-task-synthesis --synthesis-tasks-per-cycle 20 --property-consensus-threshold 0.7
#
# To run the full RSI tick loop (spec §4, steps 1-8), add:
#   --mode rsi --enable-task-synthesis --synthesis-tasks-per-cycle 20 --property-consensus-threshold 0.7

exec python main.py \
    --model Qwen/Qwen3-8B \
    --use-vllm \
    --gpu-memory-utilization 0.60 \
    --domains code,math,logic \
    --use-dora \
    --consistency-samples 3 \
    --samples-per-weakness 60 \
    --consistency-threshold 0.34 \
    --lora-rank 16 \
    --plateau-patience 8 \
    --heldout-repetitions 3 \
    --max-cycles 25 \
    --write-cycle-metrics \
    --write-cycle-samples \
    ${RESUME_ARG}
