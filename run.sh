#!/usr/bin/env bash
# One-shot runner for the full v0.2.1 RSI tick loop (spec §4, steps 1–8).
# Uses property-based self-verification: task_synthesizer proposes novel
# problems from mastered skill pairs, property_engine runs the §1.3 per-property
# admission gates and §2.1 quorum acceptance, and VoV (verifier-of-verifiers)
# runs the §1.4 adversarial bundle audit before any task enters training.
#
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

exec python main.py \
    --model Qwen/Qwen3-8B \
    --use-vllm \
    --gpu-memory-utilization 0.60 \
    --domains code,math,logic \
    --use-dora \
    --mode rsi \
    --enable-task-synthesis \
    --synthesis-tasks-per-cycle 20 \
    --property-consensus-threshold 0.7 \
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
