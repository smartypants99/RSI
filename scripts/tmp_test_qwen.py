"""Test Qwen2.5-Coder-32B-Instruct-AWQ as the base. If output is clean and
parses, we pivot the RSI to this model. R1-Distill tokenizer behavior is
producing no-space fused content that no LoRA can fix.

Prints 3 chat-templated proposals and reports parse pass/fail counts.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from src.generator.task_synthesizer import CODE_PROPOSAL_TEMPLATE, parse_code_proposal

MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"

print(f"Loading tokenizer for {MODEL}")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

print(f"Loading vLLM with AWQ quant")
llm = LLM(
    model=MODEL,
    quantization="awq_marlin",
    gpu_memory_utilization=0.75,
    max_model_len=4096,
    dtype="float16",
    enforce_eager=True,
)
sp = SamplingParams(max_tokens=1500, temperature=0.6, top_p=0.95)

wrapped = tok.apply_chat_template(
    [{"role": "user", "content": CODE_PROPOSAL_TEMPLATE}],
    tokenize=False,
    add_generation_prompt=True,
)
print(f"\n[prompt tail 200 chars]: {wrapped[-200:]!r}\n")

n = 5
outs = llm.generate([wrapped] * n, sp)
passed = 0
for i, o in enumerate(outs):
    text = o.outputs[0].text
    parsed = parse_code_proposal(text)
    status = "PASS" if parsed.ok else f"FAIL {parsed.issues}"
    if parsed.ok:
        passed += 1
    print(f"\n========== [{i}] {status} ==========")
    print(f"first 500 chars: {text[:500]!r}")
    if parsed.ok:
        print(f"entry={parsed.entry_point!r} tests={len(parsed.tests)} difficulty={parsed.difficulty}")

print(f"\n\nSUMMARY: {passed}/{n} parses OK")
