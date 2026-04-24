"""Train a format-primer LoRA adapter on the proposer output schema.

Standalone pre-RSI warmup: 3 epochs of SFT on data/format_primer_pairs.jsonl
teaches R1 the PROBLEM:/ENTRY:/REFERENCE:/TESTS:/... schema so that cycle-1
proposer batches don't parse-fail 100%.

Run on the pod:
    python scripts/train_format_primer.py \
        --model unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit \
        --load-in-4bit \
        --output outputs/format_primer_adapter

Then launch RSI with:
    bash run_deepseek.sh --format-primer-adapter outputs/format_primer_adapter
"""
from __future__ import annotations
import argparse, json, logging, pathlib, sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.utils.config import SystemConfig, ModelConfig
from src.utils.model_loader import ModelLoader
from src.trainer.custom_lora import CustomLoRATrainer
from src.generator.data_generator import TrainingSample

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("format_primer")


@dataclass
class _PrimerSample(TrainingSample):
    """TrainingSample that emits raw (prompt, completion) with NO 'Conclusion:'
    wrapper, so the model learns to produce schema-compliant output verbatim."""
    raw_completion: str = ""

    def to_training_format(self) -> dict:
        return {
            "prompt": self.prompt,
            "completion": self.raw_completion,
            "metadata": {"domain": "code", "verified": True},
        }


def _load_pairs(path: Path) -> list[_PrimerSample]:
    samples: list[_PrimerSample] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            samples.append(_PrimerSample(
                prompt=rec["prompt"],
                response=rec["completion"],
                raw_completion=rec["completion"],
                domain="code",
                verified=True,
                confidence=1.0,
                ground_truth_verified=True,
                source="format_primer",
            ))
    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--load-in-8bit", action="store_true")
    ap.add_argument("--pairs", default="data/format_primer_pairs.jsonl")
    ap.add_argument("--output", default="outputs/format_primer_adapter")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5,
                    help="Higher than RSI LR — format absorption, not delicate capability shaping.")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--max-seq-length", type=int, default=4096)
    args = ap.parse_args()

    if args.load_in_4bit and args.load_in_8bit:
        ap.error("--load-in-4bit and --load-in-8bit are mutually exclusive")

    pairs_path = Path(args.pairs)
    if not pairs_path.exists():
        ap.error(f"Pairs file not found: {pairs_path}. Run scripts/build_format_primer_data.py first.")

    samples = _load_pairs(pairs_path)
    logger.info(f"Loaded {len(samples)} primer pairs from {pairs_path}")
    if not samples:
        ap.error("No primer pairs loaded — refusing to train.")

    config = SystemConfig()
    quant = None
    if args.load_in_4bit:
        quant = {"load_in_4bit": True, "bnb_4bit_compute_dtype": "bfloat16"}
    elif args.load_in_8bit:
        quant = {"load_in_8bit": True}
    config.model = ModelConfig(
        model_path=args.model,
        dtype=args.dtype,
        quantization_config=quant,
        max_seq_length=args.max_seq_length,
    )
    config.trainer.training_mode = "sft"
    config.trainer.learning_rate = args.lr
    config.trainer.num_epochs = args.epochs
    config.trainer.num_epochs_warmup_cycles = 0
    config.trainer.lora_rank = args.lora_rank
    config.trainer.max_grad_norm = 1.0  # format task is simple; no need for aggressive clip
    config.trainer.lora_plus_ratio = 1.0  # uniform A/B LR — we want fast surface memorization

    loader = ModelLoader(config.model)
    logger.info(f"Loading base model: {args.model}")
    loader.load()

    trainer = CustomLoRATrainer(config.trainer, loader)
    logger.info(f"Injecting LoRA (rank={args.lora_rank})")
    trainer.inject_lora(weak_layers={})

    logger.info(f"Training {args.epochs} epochs on {len(samples)} samples, lr={args.lr}")
    metrics = trainer.train(samples, cycle=1)
    logger.info(
        f"Training done: avg_loss={metrics.avg_loss:.4f} final_loss={metrics.final_loss:.4f} "
        f"steps={metrics.steps}"
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = trainer.save_lora_weights(out_dir, cycle=1)
    if saved is None:
        logger.error("save_lora_weights returned None — adapter may not have been written")
        sys.exit(1)
    logger.info(f"Saved format-primer adapter to: {saved}")

    # Drop a small manifest so RSI preload code can sanity-check shape.
    manifest = {
        "base_model": args.model,
        "lora_rank": args.lora_rank,
        "epochs": args.epochs,
        "lr": args.lr,
        "num_pairs": len(samples),
        "final_loss": float(metrics.final_loss),
        "saved_path": str(saved),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Primer adapter ready. Launch RSI with --format-primer-adapter " + str(out_dir))


if __name__ == "__main__":
    main()
