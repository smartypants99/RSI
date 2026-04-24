"""Quick diagnostic: does applying DeepSeek's chat template fix the mangled
proposer output (no-spaces content, literal Ġ/Ċ tokens)?

Generates 3 completions with raw prompt vs chat-template-wrapped prompt.
If B produces clean English and A produces garbage, prompt template is the
root cause and primer is a red herring.

Run on pod:
    /venv/main/bin/python scripts/tmp_test_chat_template.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from src.generator.task_synthesizer import CODE_PROPOSAL_TEMPLATE

MODEL = "unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

llm = LLM(
    model=MODEL,
    quantization="bitsandbytes",
    load_format="bitsandbytes",
    gpu_memory_utilization=0.75,
    max_model_len=4096,
    dtype="bfloat16",
    enforce_eager=True,
)
sp = SamplingParams(max_tokens=800, temperature=0.6, top_p=0.95)

print("=" * 80)
print("MODE A: raw prompt (current production behavior)")
print("=" * 80)
outs = llm.generate([CODE_PROPOSAL_TEMPLATE] * 3, sp)
for i, o in enumerate(outs):
    print(f"\n--- A[{i}] first 500 chars ---\n{o.outputs[0].text[:500]!r}")

print("\n" + "=" * 80)
print("MODE B: chat template wrapped")
print("=" * 80)
wrapped = tok.apply_chat_template(
    [{"role": "user", "content": CODE_PROPOSAL_TEMPLATE}],
    tokenize=False,
    add_generation_prompt=True,
)
print(f"[wrapped prompt tail]: {wrapped[-200:]!r}")
outs = llm.generate([wrapped] * 3, sp)
for i, o in enumerate(outs):
    print(f"\n--- B[{i}] first 500 chars ---\n{o.outputs[0].text[:500]!r}")
