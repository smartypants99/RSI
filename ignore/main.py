"""Entry point for the recursive self-improvement system."""

import argparse
import logging
from pathlib import Path

from src.utils.config import SystemConfig, ModelConfig, VLLMConfig
from src.orchestrator.loop import ImprovementLoop


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Recursive Self-Improvement System")
    parser.add_argument("--model", required=True, help="Path to the base model")
    parser.add_argument("--max-cycles", type=int, default=100, help="Maximum improvement cycles")
    parser.add_argument("--output-dir", default="./outputs", help="Output directory")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                        help="Model dtype (use float16 for GPUs without bf16 support)")
    parser.add_argument("--lora-rank", type=int, default=None, help="LoRA rank (default: 64)")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate (default: 2e-5)")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size (default: 4)")
    parser.add_argument("--num-epochs", type=int, default=None, help="Training epochs per cycle (default: 3)")
    parser.add_argument("--questions-per-domain", type=int, default=None, help="Diagnostic questions per domain (default: 200)")
    parser.add_argument("--samples-per-weakness", type=int, default=None, help="Training samples per weakness (default: 100)")
    parser.add_argument("--use-vllm", action="store_true", help="Use vLLM for 5-10x faster inference (pip install vllm)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85, help="vLLM GPU memory fraction (default: 0.85)")
    parser.add_argument("--load-in-8bit", action="store_true", help="Load model in 8-bit quantization (halves memory, needs bitsandbytes)")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit quantization (quarters memory, needs bitsandbytes)")
    args = parser.parse_args()

    # Create output dir BEFORE setting up file logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "run.log"),
        ],
    )

    if args.load_in_8bit and args.load_in_4bit:
        parser.error("--load-in-8bit and --load-in-4bit are mutually exclusive")
    config = SystemConfig()
    quant_config = None
    if args.load_in_8bit:
        quant_config = {"load_in_8bit": True}
    elif args.load_in_4bit:
        quant_config = {"load_in_4bit": True, "bnb_4bit_compute_dtype": "bfloat16"}
    config.model = ModelConfig(model_path=args.model, dtype=args.dtype, quantization_config=quant_config)
    config.orchestrator.max_cycles = args.max_cycles
    config.orchestrator.output_dir = output_dir
    config.orchestrator.log_dir = output_dir / "logs"
    if args.resume:
        config.orchestrator.resume_from = args.resume
    if args.lora_rank is not None:
        config.trainer.lora_rank = args.lora_rank
    if args.learning_rate is not None:
        config.trainer.learning_rate = args.learning_rate
    if args.batch_size is not None:
        config.trainer.batch_size = args.batch_size
    if args.num_epochs is not None:
        config.trainer.num_epochs = args.num_epochs
    if args.questions_per_domain is not None:
        config.diagnostics.questions_per_domain = args.questions_per_domain
    if args.samples_per_weakness is not None:
        config.generator.samples_per_weakness = args.samples_per_weakness

    # vLLM mode: use VLLMModelLoader instead of default ModelLoader
    if args.use_vllm:
        config.use_vllm = True
        config.vllm = VLLMConfig(
            model_path=args.model,
            dtype=args.dtype,
            max_model_len=config.model.max_seq_length,
            gpu_memory_utilization=args.gpu_memory_utilization,
            quantization_config=quant_config,
        )

    loop = ImprovementLoop(config)

    try:
        loop.run()
    except KeyboardInterrupt:
        logger.info("\nInterrupted — cleaning up and saving checkpoint + report")
        # Ignore a second Ctrl-C during critical cleanup — aborting strip_lora
        # mid-way leaves a mix of LoRALayer/Linear modules that would be saved.
        import signal
        prev_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            try:
                loop.trainer.strip_lora()
            except Exception as e:
                logger.debug(f"strip_lora during cleanup failed: {e}")
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception as e:
                logger.debug(f"empty_cache during cleanup failed: {e}")
            if loop.history:
                try:
                    loop._save_checkpoint(loop.history[-1].cycle)
                except Exception as e:
                    logger.warning(f"Checkpoint save during cleanup failed: {e}")
            try:
                loop._save_final_report()
            except Exception as e:
                logger.warning(f"Final report save during cleanup failed: {e}")
        finally:
            signal.signal(signal.SIGINT, prev_handler)


if __name__ == "__main__":
    main()
