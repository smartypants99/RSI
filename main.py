"""Entry point for the recursive self-improvement system."""

import argparse
import logging
from pathlib import Path

from src.utils.config import SystemConfig, ModelConfig, VLLMConfig, SynthesisConfig
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
    parser.add_argument("--heldout-repetitions", type=int, default=None,
                        help="Run held-out eval N times per cycle. Spread between reps "
                             "is a direct measurement of evaluation noise (default: 1)")
    parser.add_argument("--write-cycle-metrics", action="store_true",
                        help="Dump outputs/cycle_metrics/cycle_N.json with per-sample, "
                             "STaR, per-rep, and pre/post-diff data for forensics")
    parser.add_argument("--write-cycle-samples", action="store_true",
                        help="Dump outputs/cycle_samples/cycle_N.jsonl — full training "
                             "samples per cycle for manual inspection / diffing")
    parser.add_argument("--collect-training-loss-trajectory", action="store_true",
                        help="Capture per-batch SFT loss for cycle_metrics forensics")

    # Diagnostics
    parser.add_argument("--questions-per-domain", type=int, default=None,
                        help="Diagnostic questions per domain (default: 80)")
    parser.add_argument("--diagnostics-batch-size", type=int, default=None,
                        help="Diagnostics batch size (default: 16)")
    parser.add_argument("--confidence-threshold", type=float, default=None,
                        help="Confidence threshold for weakness detection (default: 0.7)")
    parser.add_argument("--domains", default=None,
                        help="Comma-separated domain subset, e.g. 'code' for coding-only RSI, "
                             "or 'code,math' for both. Skips probes/training for unlisted domains. "
                             "Default: all 8 (reasoning,math,code,science,logic,common_sense,"
                             "language_understanding,abstraction)")
    parser.add_argument("--skip-vllm-reload", action="store_true",
                        help="After training, stay in HF mode instead of reloading vLLM. "
                             "Saves ~3-5 min per cycle at the cost of slower post-training "
                             "diagnostic + held-out eval. Net win for small domain subsets.")

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

    # rsLoRA & PiSSA
    rslora_group = parser.add_mutually_exclusive_group()
    rslora_group.add_argument("--use-rslora", dest="use_rslora", action="store_true", default=None,
                              help="Use rsLoRA scaling = alpha/sqrt(rank) (default: on)")
    rslora_group.add_argument("--no-rslora", dest="use_rslora", action="store_false",
                              help="Use classic LoRA scaling = alpha/rank")
    parser.add_argument("--init-method", default=None, choices=["kaiming", "pissa"],
                        help="LoRA init: kaiming (default) or pissa (SVD-based, faster convergence)")

    # DoRA (Liu et al. 2024): weight-decomposed LoRA — separates magnitude
    # from direction, better quality at the same rank.
    dora_group = parser.add_mutually_exclusive_group()
    dora_group.add_argument("--use-dora", dest="use_dora", action="store_true", default=None,
                            help="Enable DoRA — magnitude/direction decomposition (better quality, "
                                 "+5-10%% VRAM)")
    dora_group.add_argument("--no-dora", dest="use_dora", action="store_false",
                            help="Disable DoRA (plain LoRA)")

    # LoRA+ (Hayou et al. 2024): B trains faster than A via separate LR group
    lora_plus_group = parser.add_mutually_exclusive_group()
    lora_plus_group.add_argument("--use-lora-plus", dest="use_lora_plus", action="store_true",
                                 default=None,
                                 help="Enable LoRA+ (lr_B = lr_A * ratio) (default: on)")
    lora_plus_group.add_argument("--no-lora-plus", dest="use_lora_plus", action="store_false",
                                 help="Disable LoRA+ (single LR for A and B)")
    parser.add_argument("--lora-plus-ratio", type=float, default=None,
                        help="LoRA+ lr_B / lr_A ratio, must be in [1.0, 64.0] (default: 16.0)")

    # Metacognitive calibration
    parser.add_argument("--enable-calibration-loss", action="store_true",
                        help="Weight training loss by per-step Brier score (rewards calibrated confidences)")
    parser.add_argument("--calibration-loss-weight", type=float, default=None,
                        help="Lambda on the calibration auxiliary term (default: 0.1)")

    # Orchestration mode
    parser.add_argument("--mode", default=None, choices=["classic", "rsi"],
                        help="Execution mode: 'classic' = diagnose→generate→verify→train "
                             "(default); 'rsi' = full RSI tick per spec §4 "
                             "(requires --enable-task-synthesis)")

    # Task synthesis (opt-in)
    parser.add_argument("--enable-task-synthesis", action="store_true",
                        help="Enable synthesis mode: generate novel tasks via task_synthesizer "
                             "between diagnose and generate phases, filter by property consensus")
    parser.add_argument("--synthesis-tasks-per-cycle", type=int, default=None,
                        help="Novel tasks to synthesize per cycle when --enable-task-synthesis "
                             "is active (default: 20)")
    parser.add_argument("--property-consensus-threshold", type=float, default=None,
                        help="Minimum fraction of property checks a synthesized task must pass "
                             "to enter training (default: 0.7)")

    # vLLM
    parser.add_argument("--use-vllm", action="store_true",
                        help="Use vLLM for 5-10x faster inference (pip install vllm)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                        help="vLLM GPU memory fraction (default: 0.85)")

    # TDQ backend — uses an in-house lattice-quantized .tdq file instead of
    # vLLM/HF. Activates TDQModelLoader (src/utils/tdq_backend.py) which
    # decompresses the TDQ file into a real HF AutoModelForCausalLM on load,
    # then exposes the same generate/train interface as the other loaders.
    # When set, --use-vllm is ignored. --model should point at the .tdq file.
    parser.add_argument("--backend", default=None, choices=[None, "tdq"],
                        help="Alternative model backend. 'tdq' loads a .tdq "
                             "compressed file via TDQModelHF and runs HF-only "
                             "(no vLLM). Default: auto (vLLM if --use-vllm).")
    parser.add_argument("--tdq-inference-dir", default=None,
                        help="Path to the directory containing td_inference.py "
                             "and td_decomp.so. Defaults to $TDQ_INFERENCE_DIR "
                             "or /Users/milannarula/Desktop/ai_quatinization/final.")

    # Logging
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # Preflight / dry-run
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip environment preflight checks (NOT recommended)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run preflight, instantiate all components, then exit "
                             "without training. Catches config/env issues before GPU time.")

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
    if args.heldout_repetitions is not None:
        config.orchestrator.heldout_repetitions = args.heldout_repetitions
    if args.write_cycle_metrics:
        config.orchestrator.write_cycle_metrics = True
    if args.write_cycle_samples:
        config.orchestrator.write_cycle_samples = True
    if args.collect_training_loss_trajectory:
        config.orchestrator.collect_training_loss_trajectory = True
    if args.mode is not None:
        config.orchestrator.mode = args.mode

    # Diagnostics config
    if args.questions_per_domain is not None:
        config.diagnostics.questions_per_domain = args.questions_per_domain
    if args.diagnostics_batch_size is not None:
        config.diagnostics.batch_size = args.diagnostics_batch_size
    if args.confidence_threshold is not None:
        config.diagnostics.confidence_threshold = args.confidence_threshold
    if args.domains is not None:
        requested = [d.strip() for d in args.domains.split(",") if d.strip()]
        valid = set(config.diagnostics.domains)
        invalid = [d for d in requested if d not in valid]
        if invalid:
            parser.error(f"unknown domain(s): {invalid}. Valid: {sorted(valid)}")
        if not requested:
            parser.error("--domains must list at least one domain")
        config.diagnostics.domains = requested
        logger.info(f"Domain subset: RSI will only probe/train on {requested}")

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
    if args.use_rslora is not None:
        config.trainer.use_rslora = args.use_rslora
    if args.init_method is not None:
        config.trainer.init_method = args.init_method
    if args.use_dora is not None:
        config.trainer.use_dora = args.use_dora
    if args.use_lora_plus is not None:
        config.trainer.use_lora_plus = args.use_lora_plus
    if args.lora_plus_ratio is not None:
        config.trainer.lora_plus_ratio = args.lora_plus_ratio
    config.trainer.__post_init__()

    # Synthesis config
    if args.enable_task_synthesis:
        synthesis_kwargs = {"enable_task_synthesis": True}
        if args.synthesis_tasks_per_cycle is not None:
            synthesis_kwargs["tasks_per_cycle"] = args.synthesis_tasks_per_cycle
        if args.property_consensus_threshold is not None:
            synthesis_kwargs["property_consensus_threshold"] = args.property_consensus_threshold
        config.synthesis = SynthesisConfig(**synthesis_kwargs)

    # TDQ backend selected: short-circuit vLLM setup, point env at td_inference dir
    if args.backend == "tdq":
        config.backend = "tdq"
        config.use_vllm = False
        if args.tdq_inference_dir:
            import os as _os
            _os.environ["TDQ_INFERENCE_DIR"] = args.tdq_inference_dir
    # vLLM mode
    elif args.use_vllm:
        config.use_vllm = True
        config.vllm = VLLMConfig(
            model_path=args.model,
            dtype=args.dtype,
            max_model_len=config.model.max_seq_length,
            gpu_memory_utilization=args.gpu_memory_utilization,
            quantization_config=quant_config,
            skip_reload_after_training=bool(args.skip_vllm_reload),
        )

    # Preflight: fail-fast validation BEFORE touching the GPU or downloading
    # weights. This is the "before GPU time" safety net — misconfiguration
    # should surface in seconds, not two hours into training.
    if not args.skip_preflight:
        from src.utils.preflight import run_preflight
        report = run_preflight(config, require_cuda=not args.dry_run)
        report.print_summary()
        if not report.ok:
            logger.error("Preflight failed — aborting. Fix the errors above "
                         "and re-run. Use --skip-preflight to bypass (not recommended).")
            import sys
            sys.exit(2)

    # Instantiate the loop. Constructor alone validates many invariants via
    # dataclass __post_init__ validators, so catching exceptions here keeps
    # the error path unified: any problem shows up as a clear failure before
    # the first cycle starts.
    try:
        loop = ImprovementLoop(config)
    except (ValueError, TypeError, ImportError, RuntimeError) as e:
        logger.error(f"Failed to build ImprovementLoop: {type(e).__name__}: {e}")
        import sys
        sys.exit(2)

    if args.dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN COMPLETE — all components instantiated, no training started.")
        logger.info("Remove --dry-run to begin the real training loop.")
        logger.info("=" * 60)
        return

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
