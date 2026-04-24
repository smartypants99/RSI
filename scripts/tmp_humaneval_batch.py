"""Run 10 HumanEval items via the SAME path production uses (model_loader
.generate_batch with auto chat-template wrap), grade them, save full per-item
report including raw predictions.
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.utils.vllm_backend import VLLMModelLoader
from src.utils.external_benchmarks import _grade_humaneval, BenchmarkItem, _extract_code, load_benchmark

MODEL = "unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit"

loader = VLLMModelLoader(
    model_path=MODEL, dtype="bfloat16",
    quantization_config={"load_in_4bit": True, "bnb_4bit_compute_dtype": "bfloat16"},
    max_model_len=4096, gpu_memory_utilization=0.75, allow_remote_code=True,
    enforce_eager=True,
)
loader.load()

items = load_benchmark("humaneval", cache_dir="/workspace/RSI/outputs/external_benchmarks")
print(f"Loaded {len(items)} HumanEval items")
sample = items[:10]

prompts = [it.prompt for it in sample]
preds = loader.generate_batch(prompts, max_new_tokens=1024, temperature=0.0, top_p=1.0)

for i, (it, pred) in enumerate(zip(sample, preds)):
    extracted = _extract_code(pred)
    has_def = f"def {it.meta['entry_point']}" in extracted
    graded = _grade_humaneval(it, pred)
    print(f"\n====== [{i}] {it.item_id} entry={it.meta['entry_point']!r} has_def={has_def} PASS={graded} ======")
    print(f"[prompt tail 100]: {it.prompt[-100:]!r}")
    print(f"[raw pred first 400]: {pred[:400]!r}")
    print(f"[raw pred LAST 200]: {pred[-200:]!r}")
    print(f"[extracted code first 300]: {extracted[:300]!r}")
    print(f"[extracted code LAST 200]: {extracted[-200:]!r}")
