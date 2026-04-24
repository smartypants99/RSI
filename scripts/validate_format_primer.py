"""Post-primer validation: generate N proposals with primer adapter loaded
and report parse pass rate.

Pre-primer baseline on fresh pod: 0/20 (100% parse-fail). Target: ≥80% pass.

Run after scripts/train_format_primer.py finishes:
    /venv/main/bin/python scripts/validate_format_primer.py \
        --model unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit \
        --load-in-4bit \
        --adapter outputs/format_primer_adapter
"""
from __future__ import annotations
import argparse, logging, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from src.utils.config import SystemConfig, ModelConfig
from src.utils.model_loader import ModelLoader
from src.generator.task_synthesizer import (
    CODE_PROPOSAL_TEMPLATE, parse_code_proposal,
)
from src.trainer.custom_lora import CustomLoRATrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("primer_validate")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--pass-threshold", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.6)
    args = ap.parse_args()

    config = SystemConfig()
    quant = {"load_in_4bit": True, "bnb_4bit_compute_dtype": "bfloat16"} if args.load_in_4bit else None
    config.model = ModelConfig(model_path=args.model, quantization_config=quant)

    loader = ModelLoader(config.model)
    logger.info(f"Loading base model: {args.model}")
    loader.load()

    trainer = CustomLoRATrainer(config.trainer, loader)
    trainer.load_lora_weights(args.adapter)
    logger.info(f"Primer adapter loaded from {args.adapter} ({len(trainer._lora_layers)} layers)")

    model = loader._model
    tok = loader._tokenizer
    model.eval()

    prompt = CODE_PROPOSAL_TEMPLATE
    passed = 0
    failures: list[str] = []
    for i in range(args.n):
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=0.95,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        parsed = parse_code_proposal(raw)
        if parsed.ok:
            passed += 1
            logger.info(f"[{i+1:02d}] PASS: entry={parsed.entry_point!r} "
                        f"problem={parsed.problem_text[:60]!r}")
        else:
            failures.append(raw[:200])
            logger.info(f"[{i+1:02d}] FAIL issues={parsed.issues} "
                        f"first-200-chars={raw[:200]!r}")

    rate = passed / max(args.n, 1)
    logger.info(f"\nParse pass rate: {passed}/{args.n} = {rate:.1%}")
    if rate < args.pass_threshold:
        logger.error(
            f"FAIL: {rate:.1%} < threshold {args.pass_threshold:.0%}. "
            f"Increase primer epochs/pairs before launching RSI."
        )
        sys.exit(2)
    logger.info(
        f"PASS: {rate:.1%} >= {args.pass_threshold:.0%}. Primer ready — "
        f"launch RSI with `bash run_deepseek.sh`."
    )


if __name__ == "__main__":
    main()
