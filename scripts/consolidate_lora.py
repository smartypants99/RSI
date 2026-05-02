"""Consolidate accumulated LoRA into a new base checkpoint, resetting
capacity for the next phase of training.

When LoRA rank caps out (e.g. cycle 200 with rank-256 LoRA accumulated
across all training data), per-cycle gain hits the rank ceiling — the
floor enforcer rotates without producing 1%. This script breaks out of
that ceiling: dequantize the bnb-4bit base, merge the latest LoRA into
it, requantize, and save as a new base directory. Re-launching RSI with
`--model outputs/consolidated_base_NN` resumes with the merged weights
as the new starting point + fresh LoRA on top → full rank capacity again.

Effective compute cost: ~5-10 min on A6000 (one-shot dequant + merge
+ requantize). Run between phases, not every cycle.

Usage:
    /venv/main/bin/python scripts/consolidate_lora.py \\
        --base unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit \\
        --adapter outputs/lora_weights/lora_cycle_200 \\
        --out outputs/consolidated_base_1

After consolidation, edit run_deepseek.sh `--model` to point at the new
base, optionally bump `--lora-rank` back to start (rank-128 fresh on
merged base = effectively rank-(prior + 128) total capacity), and
relaunch.
"""
from __future__ import annotations
import argparse, gc, logging, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("consolidate_lora")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True,
                    help="HF repo or path to base model (bnb-4bit)")
    ap.add_argument("--adapter", required=True,
                    help="Path to LoRA adapter dir (PEFT format)")
    ap.add_argument("--out", required=True,
                    help="Output dir for consolidated base")
    ap.add_argument("--quantize-out", action="store_true", default=True,
                    help="Save consolidated base as bnb-4bit (default True)")
    ap.add_argument("--save-bf16-temp", action="store_true",
                    help="ALSO save bf16 temp (~64GB) — debugging only")
    args = ap.parse_args()

    out_path = pathlib.Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading tokenizer from {args.base}")
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    tok.save_pretrained(out_path)

    logger.info(f"Loading base in bf16 (dequantized) — needs ~64GB RAM/swap")
    # We MUST load in bf16 not bnb-4bit because PEFT merge doesn't work
    # on packed bnb weights. Once merged we'll requantize back.
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        dtype=torch.bfloat16,
        device_map="cpu",  # CPU offload to avoid OOM on 48GB GPU during merge
        trust_remote_code=True,
    )
    logger.info(f"Loading LoRA adapter from {args.adapter}")
    base = PeftModel.from_pretrained(base, args.adapter)
    logger.info("Merging LoRA into base weights…")
    merged = base.merge_and_unload()
    if args.save_bf16_temp:
        bf16_dir = out_path / "_bf16_temp"
        bf16_dir.mkdir(exist_ok=True)
        merged.save_pretrained(bf16_dir)
        logger.info(f"  bf16 saved to {bf16_dir} (debugging — large)")
    if args.quantize_out:
        logger.info("Requantizing merged model to bnb-4bit for next-phase training")
        # Save merged in bf16 first as scratch (PEFT's save_pretrained doesn't
        # quantize directly), then re-load with quant config and re-save.
        scratch = out_path / "_merged_bf16_scratch"
        scratch.mkdir(exist_ok=True)
        merged.save_pretrained(scratch, safe_serialization=True)
        del merged, base
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Re-load with bnb 4-bit config and save canonical layout.
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        )
        requant = AutoModelForCausalLM.from_pretrained(
            scratch, quantization_config=bnb, device_map="auto",
            trust_remote_code=True,
        )
        requant.save_pretrained(out_path, safe_serialization=True)
        logger.info(f"Consolidated bnb-4bit base saved to {out_path}")
        # Clean up scratch
        import shutil
        shutil.rmtree(scratch, ignore_errors=True)
    else:
        merged.save_pretrained(out_path, safe_serialization=True)

    # Drop a manifest so RSI knows what's in here.
    manifest = out_path / "consolidation_manifest.txt"
    manifest.write_text(
        f"base: {args.base}\n"
        f"adapter: {args.adapter}\n"
        f"quantized_out: {args.quantize_out}\n"
    )
    logger.info("Done. Update run_deepseek.sh:")
    logger.info(f"    --model {out_path}")
    logger.info("    --load-in-4bit  (still required — base is bnb-4bit)")
    logger.info("    --lora-rank 128  (or any starting rank — fresh capacity)")


if __name__ == "__main__":
    main()
