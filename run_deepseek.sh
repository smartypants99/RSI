#!/usr/bin/env bash
# RSI on Qwen2.5-Coder-32B-Instruct (AWQ 4-bit).
#
# Pivot on 2026-04-24: DeepSeek-R1-Distill-Qwen-32B-bnb-4bit emits token
# streams without BPE space-prefix convention when fed our CODE_PROPOSAL
# prompt — output has no whitespace between words, generated Python refs
# are uncompilable, parser failed 100%. Swapped to Qwen2.5-Coder-Instruct
# AWQ which: (a) has a correctly-configured tokenizer, (b) is chat-tuned
# so apply_chat_template switches it to clean-English distribution,
# (c) is already code-tuned (better proposer+solver baseline), (d) 18GB
# VRAM via AWQ marlin kernel. 5/5 parse pass on first proposal test.
#
# Training path: AWQ frozen base + custom-LoRA adapters in fp16.
# Inference: vLLM with awq_marlin quantization.
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

# Format-primer adapter: DISABLED after Qwen2.5-Coder pivot (2026-04-24).
# The primer was trained on DeepSeek-R1-Distill LoRA shapes and won't load
# into Qwen — and isn't needed anyway since Qwen produces parse-compliant
# proposals from the chat-template path alone. Resurrect by passing
# `--format-primer-adapter PATH` explicitly if a Qwen-shaped primer is
# trained later.
PRIMER_ARG=""

# Foom resilience: auto-restart main.py up to AUTORESTART_MAX times on
# crash (transient OOM, vLLM engine died, network blip downloading HF
# datasets, etc.). With multi-cycle LoRA persistence + .reverted markers
# + adapter dirs on disk, restarted runs RESUME from the last good cycle
# automatically via _lora_resume_path. Disable by setting
# AUTORESTART_MAX=0 in the env.
AUTORESTART_MAX="${AUTORESTART_MAX:-5}"
_attempt=0
while true; do
    python main.py \
        --model unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit \
        --load-in-4bit \
        --use-vllm \
        --gpu-memory-utilization 0.75 \
        --domains code,math,logic \
        --mode rsi \
        --enable-task-synthesis \
        --synthesis-tasks-per-cycle 20 \
        --property-consensus-threshold 0.7 \
        --consistency-samples 3 \
        --samples-per-weakness 100 \
        --consistency-threshold 0.34 \
        --lora-rank 128 \
        --learning-rate 8e-6 \
        --plateau-patience 8 \
        --heldout-repetitions 1 \
        --max-cycles 200 \
        --write-cycle-metrics \
        --write-cycle-samples \
        ${PRIMER_ARG} \
        ${RESUME_ARG}
    _rc=$?
    # Clean exit (max-cycles reached, user Ctrl-C, or graceful end) → stop.
    if [ "$_rc" = "0" ]; then
        echo "[run_deepseek] main.py exited cleanly (rc=0); stopping."
        break
    fi
    _attempt=$((_attempt + 1))
    if [ "$_attempt" -ge "$AUTORESTART_MAX" ]; then
        echo "[run_deepseek] main.py failed $_attempt times; exhausted AUTORESTART_MAX=$AUTORESTART_MAX, giving up."
        exit "$_rc"
    fi
    # Kill any orphan vLLM/EngineCore that pinned VRAM, then re-launch.
    for _proc in $(pgrep -f 'VLLM::EngineCore|multiprocessing.resource_tracker' 2>/dev/null); do
        kill -9 "$_proc" 2>/dev/null || true
    done
    _wait=$((10 * _attempt))
    echo "[run_deepseek] main.py crashed (rc=$_rc); restart $_attempt/$AUTORESTART_MAX in ${_wait}s — _lora_resume_path picks up from latest non-reverted adapter."
    sleep "$_wait"
done
