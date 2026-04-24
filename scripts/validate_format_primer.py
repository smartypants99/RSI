"""Post-primer validation: run 20 proposer_batch_code calls with the
format-primer adapter loaded and report parse pass rate.

Pre-primer baseline on a fresh pod: 0/20 (100% parse-fail, missing ENTRY /
REFERENCE / TESTS everywhere). Target: ≥16/20 (80%) pass.

Run on the pod AFTER scripts/train_format_primer.py finishes:
    python scripts/validate_format_primer.py \
        --model unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit \
        --load-in-4bit \
        --adapter outputs/format_primer_adapter
"""
from __future__ import annotations
import argparse, logging, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.utils.config import SystemConfig, ModelConfig, VLLMConfig
from src.utils.vllm_backend import VLLMModelLoader
from src.generator.task_synthesizer import TaskSynthesizer
from src.trainer.custom_lora import CustomLoRATrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("primer_validate")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True,
                    help="Path to outputs/format_primer_adapter/ (the dir, not the .pt file).")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--pass-threshold", type=float, default=0.8)
    args = ap.parse_args()

    config = SystemConfig()
    quant = {"load_in_4bit": True, "bnb_4bit_compute_dtype": "bfloat16"} if args.load_in_4bit else None
    config.model = ModelConfig(model_path=args.model, quantization_config=quant)

    # Use the same vLLM-backed loader path the orchestrator uses so timing and
    # detokenization match the production propose path.
    loader = VLLMModelLoader(config.model, VLLMConfig(gpu_memory_utilization=0.75))
    loader.load()

    trainer = CustomLoRATrainer(config.trainer, loader)
    trainer.load_lora_weights(args.adapter)
    logger.info(f"Primer adapter loaded from {args.adapter}")

    synth = TaskSynthesizer(config.generator, loader, session_id="primer_validate")
    logger.info(f"Running {args.n} proposer_batch_code calls…")
    proposals = synth.propose_batch_code(args.n)

    passed = len(proposals)
    total = args.n
    rate = passed / max(total, 1)
    logger.info(f"Parse pass rate: {passed}/{total} = {rate:.1%}")
    if rate < args.pass_threshold:
        logger.error(
            f"FAIL: {rate:.1%} < threshold {args.pass_threshold:.0%}. "
            f"Primer is NOT production-ready. Investigate training loss, "
            f"pair diversity, or epoch count before launching RSI."
        )
        sys.exit(2)
    logger.info(
        f"PASS: {rate:.1%} >= {args.pass_threshold:.0%}. Primer is ready — "
        f"launch RSI with `bash run_deepseek.sh`."
    )


if __name__ == "__main__":
    main()
