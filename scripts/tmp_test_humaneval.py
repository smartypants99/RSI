"""Debug why anchor HumanEval scores 0/50 on Qwen2.5-Coder-32B-Instruct when
published is ~92%. Runs 3 HumanEval problems end-to-end and shows raw output
+ grading decision.

Run on pod: /venv/main/bin/python scripts/tmp_test_humaneval.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from src.utils.external_benchmarks import _grade_humaneval, BenchmarkItem, _extract_code
from src.utils.sandbox import run_python_sandboxed

MODEL = "unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
llm = LLM(
    model=MODEL, quantization="bitsandbytes", load_format="bitsandbytes",
    gpu_memory_utilization=0.75, max_model_len=4096, dtype="bfloat16",
    enforce_eager=True, trust_remote_code=True,
)
sp = SamplingParams(max_tokens=800, temperature=0.0, top_p=1.0)

ds = load_dataset("openai_humaneval", split="test")
print(f"Total HumanEval: {len(ds)}")

for i, r in enumerate(ds):
    if i >= 3:
        break
    item = BenchmarkItem(
        benchmark="humaneval",
        item_id=str(r["task_id"]),
        prompt=str(r["prompt"]),
        answer=str(r["canonical_solution"]),
        domain="code",
        meta={"test": r["test"], "entry_point": r["entry_point"]},
    )
    wrapped = tok.apply_chat_template(
        [{"role": "user", "content": item.prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    print(f"\n{'='*80}\n[{i}] {item.item_id} entry={item.meta['entry_point']!r}")
    print(f"[prompt first 200]: {item.prompt[:200]!r}")
    out = llm.generate([wrapped], sp)
    pred = out[0].outputs[0].text
    print(f"[raw pred first 600]: {pred[:600]!r}")
    extracted = _extract_code(pred)
    print(f"[extracted code first 400]: {extracted[:400]!r}")
    graded = _grade_humaneval(item, pred)
    print(f"[graded (our scorer)]: {graded}")

    # Manual direct grade: full program = extracted + test + check(entry)
    if f"def {item.meta['entry_point']}" in extracted:
        program = extracted
    else:
        program = item.prompt + extracted
    source = program + "\n\n" + item.meta["test"] + f"\n\ncheck({item.meta['entry_point']})\n"
    ok, tail = run_python_sandboxed(source, timeout_s=5, memory_mb=256)
    print(f"[manual sandbox ok={ok}] tail[:400]: {tail[:400]!r}")
