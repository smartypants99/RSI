"""Entry point for the recursive self-improvement system."""

import argparse
import logging
from pathlib import Path

from src.utils.config import SystemConfig, ModelConfig, VLLMConfig
from src.orchestrator.loop import ImprovementLoop


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Recursive Self-Improvement System")

    # Model
    parser.add_argument("--model", required=True, help="Path to the base model")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                        help="Model dtype (default: bfloat16)")
    parser.add_argument("--max-seq-length", type=int, default=None, help="Max sequence length (default: 4096)")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Load model in 8-bit quantization (halves memory, needs bitsandbytes)")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load model in 4-bit quantization (quarters memory, needs bitsandbytes)")

    # Orchestrator
    parser.add_argument("--max-cycles", type=int, default=100, help="Maximum improvement cycles (default: 100)")
    parser.add_argument("--output-dir", default="./outputs", help="Output directory (default: ./outputs)")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path")
    parser.add_argument("--checkpoint-every", type=int, default=None,
                        help="Save checkpoint every N cycles (default: 1)")
    parser.add_argument("--plateau-patience", type=int, default=None,
                        help="Cycles without improvement before stopping (default: 3)")
    parser.add_argument("--min-improvement-threshold", type=float, default=None,
                        help="Minimum improvement to count as progress (default: 0.01)")

    # Diagnostics
    parser.add_argument("--questions-per-domain", type=int, default=None,
                        help="Diagnostic questions per domain (default: 300)")
    parser.add_argument("--diagnostics-batch-size", type=int, default=None,
                        help="Diagnostics batch size (default: 16)")
    parser.add_argument("--confidence-threshold", type=float, default=None,
                        help="Confidence threshold for weakness detection (default: 0.7)")

    # Generator
    parser.add_argument("--samples-per-weakness", type=int, default=None,
                        help="Training samples per weakness (default: 100)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Generation temperature (default: 0.7)")
    parser.add_argument("--top-p", type=float, default=None,
                        help="Generation top-p (default: 0.9)")
    parser.add_argument("--consistency-samples", type=int, default=None,
                        help="Self-consistency: N independent solutions per problem. "
                             "Samples below --consistency-threshold agreement are rejected; "
                             "survivors are downweighted by agreement fraction. "
                             "N=1 disables. Recommended: 3 for real RSI. "
                             "Cost: N× generation time (default: 1)")
    parser.add_argument("--consistency-threshold", type=float, default=None,
                        help="Min agreement fraction to keep a sample (default: 0.5)")

    # Trainer
    parser.add_argument("--lora-rank", type=int, default=None, help="LoRA rank (default: 64)")
    parser.add_argument("--lora-alpha", type=int, default=None, help="LoRA alpha (default: 128)")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate (default: 2e-5)")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size (default: 2)")
    parser.add_argument("--num-epochs", type=int, default=None, help="Training epochs per cycle (default: 3)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None,
                        help="Gradient accumulation steps (default: 16)")
    parser.add_argument("--warmup-ratio", type=float, default=None,
                        help="Warmup ratio (default: 0.1)")
    parser.add_argument("--training-mode", default=None,
                        choices=["sft", "dpo", "mixed", "grpo"],
                        help="Trainer objective: sft (positives only, default), "
                             "dpo (preference pairs), mixed (SFT then DPO per cycle), "
                             "grpo (group-relative policy optimization, DeepSeek-R1)")
    parser.add_argument("--dpo-beta", type=float, default=None,
                        help="DPO KL regularization strength (default: 0.1)")
    parser.add_argument("--grpo-group-size", type=int, default=None,
                        help="GRPO: completions sampled per prompt (default: 8)")
    parser.add_argument("--grpo-clip-eps", type=float, default=None,
                        help="GRPO: importance-ratio clip epsilon (default: 0.2)")
    parser.add_argument("--grpo-rollout-refresh-steps", type=int, default=None,
                        help="GRPO: refresh cached rollouts every N optimizer steps (default: 64)")
    parser.add_argument("--grpo-max-new-tokens", type=int, default=None,
                        help="GRPO: max tokens per sampled completion (default: 512)")
    parser.add_argument("--grpo-rollout-temperature", type=float, default=None,
                        help="GRPO: sampling temperature for rollouts (default: 1.0)")
    parser.add_argument("--grpo-rollout-top-p", type=float, default=None,
                        help="GRPO: sampling top-p for rollouts (default: 0.95)")

    # PRM (Process Reward Model) — dense per-step rewards for GRPO
    parser.add_argument("--use-prm", action="store_true",
                        help="Train a PRM each cycle and use it as GRPO reward_fn (requires --training-mode grpo)")
    parser.add_argument("--prm-lr", type=float, default=None,
                        help="PRM head learning rate (default: 1e-4)")
    parser.add_argument("--prm-epochs", type=int, default=None,
                        help="PRM training epochs per cycle (default: 1)")

    # Metacognitive calibration
    parser.add_argument("--enable-calibration-loss", action="store_true",
                        help="Weight training loss by per-step Brier score (rewards calibrated confidences)")
    parser.add_argument("--calibration-loss-weight", type=float, default=None,
                        help="Lambda on the calibration auxiliary term (default: 0.1)")

    # vLLM
    parser.add_argument("--use-vllm", action="store_true",
                        help="Use vLLM for 5-10x faster inference (pip install vllm)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                        help="vLLM GPU memory fraction (default: 0.85)")

    # Logging
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

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

    # Model config
    quant_config = None
    if args.load_in_8bit:
        quant_config = {"load_in_8bit": True}
    elif args.load_in_4bit:
        quant_config = {"load_in_4bit": True, "bnb_4bit_compute_dtype": "bfloat16"}
    model_kwargs = {"model_path": args.model, "dtype": args.dtype, "quantization_config": quant_config}
    if args.max_seq_length is not None:
        model_kwargs["max_seq_length"] = args.max_seq_length
    config.model = ModelConfig(**model_kwargs)

    # Orchestrator config
    config.orchestrator.max_cycles = args.max_cycles
    config.orchestrator.output_dir = output_dir
    config.orchestrator.log_dir = output_dir / "logs"
    if args.resume:
        config.orchestrator.resume_from = args.resume
    if args.checkpoint_every is not None:
        config.orchestrator.checkpoint_every = args.checkpoint_every
    if args.plateau_patience is not None:
        config.orchestrator.plateau_patience = args.plateau_patience
    if args.min_improvement_threshold is not None:
        config.orchestrator.min_improvement_threshold = args.min_improvement_threshold

    # Diagnostics config
    if args.questions_per_domain is not None:
        config.diagnostics.questions_per_domain = args.questions_per_domain
    if args.diagnostics_batch_size is not None:
        config.diagnostics.batch_size = args.diagnostics_batch_size
    if args.confidence_threshold is not None:
        config.diagnostics.confidence_threshold = args.confidence_threshold

    # Generator config
    if args.samples_per_weakness is not None:
        config.generator.samples_per_weakness = args.samples_per_weakness
    if args.temperature is not None:
        config.generator.temperature = args.temperature
    if args.top_p is not None:
        config.generator.top_p = args.top_p
    if args.consistency_samples is not None:
        config.generator.consistency_samples = args.consistency_samples
    if args.consistency_threshold is not None:
        config.generator.consistency_threshold = args.consistency_threshold

    # Trainer config
    if args.lora_rank is not None:
        config.trainer.lora_rank = args.lora_rank
    if args.lora_alpha is not None:
        config.trainer.lora_alpha = args.lora_alpha
    if args.learning_rate is not None:
        config.trainer.learning_rate = args.learning_rate
    if args.batch_size is not None:
        config.trainer.batch_size = args.batch_size
    if args.num_epochs is not None:
        config.trainer.num_epochs = args.num_epochs
    if args.gradient_accumulation_steps is not None:
        config.trainer.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.warmup_ratio is not None:
        config.trainer.warmup_ratio = args.warmup_ratio
    if args.training_mode is not None:
        config.trainer.training_mode = args.training_mode
    if args.dpo_beta is not None:
        config.trainer.dpo_beta = args.dpo_beta
    if args.grpo_group_size is not None:
        config.trainer.grpo_group_size = args.grpo_group_size
    if args.grpo_clip_eps is not None:
        config.trainer.grpo_clip_eps = args.grpo_clip_eps
    if args.grpo_rollout_refresh_steps is not None:
        config.trainer.grpo_rollout_refresh_steps = args.grpo_rollout_refresh_steps
    if args.grpo_max_new_tokens is not None:
        config.trainer.grpo_max_new_tokens = args.grpo_max_new_tokens
    if args.grpo_rollout_temperature is not None:
        config.trainer.grpo_rollout_temperature = args.grpo_rollout_temperature
    if args.grpo_rollout_top_p is not None:
        config.trainer.grpo_rollout_top_p = args.grpo_rollout_top_p
    if args.use_prm:
        config.trainer.use_prm = True
    if args.prm_lr is not None:
        config.trainer.prm_lr = args.prm_lr
    if args.prm_epochs is not None:
        config.trainer.prm_epochs = args.prm_epochs
    if args.enable_calibration_loss:
        config.trainer.enable_calibration_loss = True
    if args.calibration_loss_weight is not None:
        config.trainer.calibration_loss_weight = args.calibration_loss_weight
    config.trainer.__post_init__()

    # vLLM mode
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
