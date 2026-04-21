"""Central configuration for the recursive self-improvement system."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ModelConfig:
    model_path: str = ""  # must be set before use
    quantization_config: Optional[dict] = None
    device_map: str = "auto"
    max_seq_length: int = 4096
    dtype: str = "bfloat16"
    allow_remote_code: bool = True

    def __post_init__(self):
        if self.max_seq_length < 1:
            raise ValueError(f"max_seq_length must be >= 1, got {self.max_seq_length}")
        if self.dtype not in ("bfloat16", "float16", "float32"):
            raise ValueError(f"dtype must be bfloat16/float16/float32, got {self.dtype}")


@dataclass
class DiagnosticsConfig:
    # Tuned for single-A6000 + vLLM: 80 × 8 domains ≈ 640 batched probes,
    # ~3-5 min per diagnostic phase. Bump for deeper probing; clamp range
    # permits 20-400 so --questions-per-domain 50 isn't silently clamped up.
    questions_per_domain: int = 80
    min_questions_per_domain: int = 20
    max_questions_per_domain: int = 400
    domains: list[str] = field(default_factory=lambda: [
        "reasoning", "math", "code", "science", "logic",
        "common_sense", "language_understanding", "abstraction",
    ])
    batch_size: int = 16
    confidence_threshold: float = 0.7
    activation_analysis: bool = True
    weak_layer_percentile: float = 0.2
    code_execution_timeout: int = 10  # seconds for code execution checks in diagnostics

    use_programmatic_generators: bool = True
    difficulty_curriculum: bool = True
    difficulty_mix: dict = field(default_factory=lambda: {
        "easy": 0.30, "medium": 0.35, "hard": 0.25, "expert": 0.10,
    })
    semantic_grading: bool = True
    significance_alpha: float = 0.05
    min_evidence_for_weakness: int = 8
    activation_probes_per_domain: int = 2

    def __post_init__(self):
        if self.batch_size < 1:
            raise ValueError(f"diagnostics.batch_size must be >= 1, got {self.batch_size}")
        if self.questions_per_domain < 1:
            raise ValueError(f"questions_per_domain must be >= 1, got {self.questions_per_domain}")
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(f"confidence_threshold must be in [0, 1], got {self.confidence_threshold}")
        if not (0.0 < self.weak_layer_percentile <= 1.0):
            raise ValueError(f"weak_layer_percentile must be in (0, 1], got {self.weak_layer_percentile}")
        if self.min_questions_per_domain > self.max_questions_per_domain:
            raise ValueError(
                f"min_questions_per_domain ({self.min_questions_per_domain}) > "
                f"max_questions_per_domain ({self.max_questions_per_domain})"
            )
        if self.code_execution_timeout < 1:
            raise ValueError(f"code_execution_timeout must be >= 1, got {self.code_execution_timeout}")


@dataclass
class GeneratorConfig:
    min_reasoning_steps: int = 3
    # 30 samples × ~5 weaknesses = ~150 generation calls per cycle, roughly
    # 3-5 min on vLLM. Bump to 100 for richer training sets on beefier setups.
    samples_per_weakness: int = 30
    temperature: float = 0.7
    top_p: float = 0.9
    # Self-consistency: generate N independent solutions per problem, train
    # only on samples where ≥(consistency_threshold × N) generations reach the
    # same final answer. Higher N = stronger mode-collapse resistance but N×
    # generation cost. N=1 disables the check (legacy behavior).
    consistency_samples: int = 1
    consistency_threshold: float = 0.5
    # STaR (Zelikman et al. 2022): when the diagnostics engine provides real
    # problems with canonical answers, sample K reasoning chains per failed
    # problem at moderate temperature and keep only those whose final answer
    # matches ground truth. Replaces "model makes up problems and grades itself".
    star_k_samples: int = 4  # K chains per failed problem
    star_temperature: float = 0.7  # sampling temp for the K chains
    star_rationalization: bool = True  # rationalize 0/K problems with answer hint
    star_max_rationalizations_per_weakness: int = 16  # cap to bound cost
    # Quality-ranked top-k sample filter (applied post-verify, pre-train).
    # Cycle-6 succeeded on 1 sample; cycle-1 regressed on 1 sample — the
    # difference was sample quality. Rank verified samples by
    #   score = consistency_score * parse_confidence * (1 + 0.5 * star_bonus)
    # (star_bonus=1 for source=='star', 0 for 'star_rationalized'/'synthesized')
    # and keep only the top-k. Set to 0 to disable (use all verified samples).
    sample_quality_top_k: int = 0
    # Floor: when top-k is active, never train on fewer than this many samples
    # (unless verified<floor, in which case we use all). Protects against
    # ranking down to a single marginal sample.
    sample_quality_floor: int = 3

    def __post_init__(self):
        if self.min_reasoning_steps < 1:
            raise ValueError(f"min_reasoning_steps must be >= 1, got {self.min_reasoning_steps}")
        if self.samples_per_weakness < 1:
            raise ValueError(f"samples_per_weakness must be >= 1, got {self.samples_per_weakness}")
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.consistency_samples < 1:
            raise ValueError(f"consistency_samples must be >= 1, got {self.consistency_samples}")
        if not (0.0 < self.consistency_threshold <= 1.0):
            raise ValueError(f"consistency_threshold must be in (0, 1], got {self.consistency_threshold}")
        if self.sample_quality_top_k < 0:
            raise ValueError(f"sample_quality_top_k must be >= 0, got {self.sample_quality_top_k}")
        if self.sample_quality_floor < 1:
            raise ValueError(f"sample_quality_floor must be >= 1, got {self.sample_quality_floor}")


@dataclass
class VerifierConfig:
    check_logical_validity: bool = True
    check_step_completeness: bool = True
    check_assumption_grounding: bool = True
    reject_on_any_gap: bool = False  # True is too strict — rejects nearly all samples early on
    min_confidence_for_accept: float = 0.85
    use_model_verification: bool = False  # escalation: let model assist
    min_chain_steps: int = 2  # minimum steps for chain-level check
    code_exec_timeout: int = 5
    code_exec_memory_mb: int = 256
    enable_sympy_math_check: bool = True
    enable_code_exec_check: bool = True
    # Upper bound must match min_confidence_for_accept — otherwise samples in
    # [escalate_below, min_confidence_for_accept) are silently rejected without
    # model review.
    escalate_to_model_below: float = 0.85
    escalate_to_model_above: float = 0.50
    check_weights: dict = field(default_factory=lambda: {
        "logical_validity": 1.0,
        "step_completeness": 1.0,
        "assumption_grounding": 1.0,
        "domain_exec": 2.0,
        "consistency": 1.5,
    })
    max_prior_steps_to_compare: int = 8
    allow_model_override_reject: bool = True
    # When True, require atomic-step format (step_id/depends_on/rule). Samples
    # missing these fields or with malformed DAGs are rejected outright. Enables
    # external structural verification — any outside checker can audit each step.
    atomic_mode: bool = False

    def __post_init__(self):
        if not (0.0 <= self.min_confidence_for_accept <= 1.0):
            raise ValueError(f"min_confidence_for_accept must be in [0, 1], got {self.min_confidence_for_accept}")
        if self.min_chain_steps < 1:
            raise ValueError(f"min_chain_steps must be >= 1, got {self.min_chain_steps}")
        if self.code_exec_timeout < 1:
            raise ValueError(f"code_exec_timeout must be >= 1, got {self.code_exec_timeout}")
        if self.code_exec_memory_mb < 1:
            raise ValueError(f"code_exec_memory_mb must be >= 1, got {self.code_exec_memory_mb}")
        if self.escalate_to_model_above >= self.escalate_to_model_below:
            raise ValueError(
                f"escalate_to_model_above ({self.escalate_to_model_above}) must be < "
                f"escalate_to_model_below ({self.escalate_to_model_below})"
            )


@dataclass
class TrainerConfig:
    # Defaults lowered after run-10 regression analysis:
    #   run-10 cycle 1: 21 samples × 2 optimizer steps → loss 0.50→0.18 crash,
    #   held-out regressed -0.296. With LoRA+ (16× lr on B) and rsLoRA
    #   (scale=alpha/sqrt(rank)), effective per-step weight delta was too
    #   aggressive — one step on a tiny batch moved the model into the
    #   training distribution at the cost of everything else.
    # Fix: shrink LR 4× (2e-5 → 5e-6), shrink rank 8× (64 → 8), shrink alpha
    # proportionally. rsLoRA scale drops from 128/sqrt(64)=16 to 16/sqrt(8)=5.66.
    # Combined with LoRA+'s 16× on B and 4× lower LR, the effective update
    # per step drops ~10× — landing in the regime where 2 steps don't
    # memorize 20 samples.
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    learning_rate: float = 5e-6
    # Defaults tuned for small-cycle RSI regime (typ. 5-30 verified samples/cycle).
    # Cycle-2 (success) = 1-2 optimizer steps, final loss ~0.4-0.8.
    # Cycle-3 (overfit) = 25+ steps on 9 samples, final loss 0.045.
    # With samples<30: num_epochs=2, grad_accum=4 keeps steps in the 2-4 range
    # instead of the 25+ that caused memorization.
    num_epochs: int = 2
    batch_size: int = 2
    gradient_accumulation_steps: int = 4

    # Regularization knobs (see _train_inner in custom_lora.py).
    # early_stop_loss: if unweighted loss drops below this mid-training, stop
    #   immediately. 0.15 is well below natural SFT floor (~0.4) but above the
    #   0.044 memorization catastrophe observed in cycle 3.
    # max_steps_per_cycle: hard cap on optimizer steps per cycle. If the
    #   naive (num_epochs × len(dataloader)) / grad_accum would exceed this,
    #   grad_accum is scaled up automatically.
    # min_steps_per_cycle: if the computed step budget is below this, the
    #   cycle is skipped with a warning (too little signal to update safely).
    # early_stop_loss raised 0.15 → 0.30. Observed run-8 cycle 1: 18 samples
    # × 3 optimizer steps drove loss from ~0.5 to 0.1112 in 13 batches. By the
    # time we hit loss 0.15, the model had already memorized and held-out
    # dropped from 0.558 → 0.280 (full eval). At 0.30 we stop after ~1 step
    # on tiny batches — less signal per cycle but no catastrophic overfit.
    # Combined with post-Phase-5b revert, this should hold the base model
    # at baseline or improve slowly instead of collapsing.
    # early_stop_loss raised 0.30 → 0.50 after run-10. Observed: loss dropped
    # from ~0.5 → 0.18 in 2 optimizer steps, then early-stop fired at 0.30 AFTER
    # damage. At 0.50 we catch the crash on the first batch whose forward-pass
    # loss dips below natural SFT floor (~0.5-0.7), so we stop before more than
    # one full accumulation group lands.
    # Lowered 0.50 → 0.15 for consistency with skip_if_initial_loss_below.
    # Observed on cycle 20: pre-loss 0.43, first batch loss also ~0.43,
    # triggered early-stop at step 2 after only 2 optimizer updates.
    # That's barely training, and it still managed to regress held-out
    # by 0.051. 0.15 lets training actually land ~5-8 steps of useful
    # gradient before stopping. The regression_revert_threshold=0.03
    # guard (cf1e461) catches any training that turns out to hurt.
    early_stop_loss: float = 0.15
    max_steps_per_cycle: int = 8
    min_steps_per_cycle: int = 1
    # min_train_samples: minimum training-pool size before we actually train.
    # Observed failure: cycle-1 trained on 4 STaR-fallback samples (1 optimizer
    # step), which added enough gradient noise to an 8B model that held-out
    # eval dropped 0.244 and parse-format output collapsed, cascading every
    # subsequent cycle into worse proposals and worse fallbacks. The spec's
    # `min_steps_per_cycle` only gates on STEPS — not samples — so 1 sample ×
    # 1 epoch still qualifies. Require at least this many verified samples
    # before training; below this, skip the cycle's training (the fallback
    # pool will accumulate across cycles if populated-across-cycles is on).
    # Run-9 observed: 21 samples × 2 steps → loss crashed to 0.18, held-out
    # regressed by 0.296. Raised floor to 60 so pool accumulates across
    # 3-4 cycles; lower-variance gradient updates on more diverse problems
    # should stop the instant-memorization failure mode.
    # STACKING-TEST MODE: lowered 30→5 so training fires on cycle 1 even
    # with low per-cycle accept. For the "does RSI stack on top of cycle_7"
    # question we want the fastest possible trained-and-evaled cycle.
    # Regression_revert_threshold=0.03 + early_stop_loss=0.15 still guard
    # against corruption. Bump back to 30 after the stacking answer lands.
    min_train_samples: int = 5
    # regression_revert_threshold: if post-training held-out drops by more
    # than this vs pre-training, revert the checkpoint to pre-training
    # weights. Trainers that blow up the base model shouldn't get to keep
    # their corruption in the next cycle's starting state. Set to a large
    # value (e.g. 1.0) to disable.
    # Tightened 0.10 → 0.03 after observing cycle-20 training drop
    # held-out 0.576 → 0.525 (delta 0.051). The old 0.10 threshold
    # let that regression through — the model ended up running at
    # 0.525, below the 0.558 baseline, when cycle_7's 0.576 would have
    # been kept if the threshold had been tighter. 0.03 still allows
    # ~0.5 std-of-score noise without triggering spurious reverts
    # (observed held-out spread per eval is ~0.05 across 2 reps).
    regression_revert_threshold: float = 0.03
    # skip_if_initial_loss_below: pre-training loss probe. Before the first
    # backward step, we forward a single batch under no_grad and check the
    # loss. If it's already below this threshold, the model has effectively
    # memorized the training distribution and any further step will corrupt
    # it (this is the failure mode that put held-out at 0.000 after cycle 4
    # where step 1 hit loss 0.0547). Training returns zero-step metrics
    # instead of applying the damaging update. Set to 0 to disable the probe.
    # Lowered 0.50 → 0.15 because property-verified RSI samples are
    # reference solutions the model itself wrote — pre-loss is naturally
    # low (observed 0.37 on cycle-5 batch) but "already memorized" is
    # the wrong diagnosis for that distribution. The 0.0547 failure case
    # that set 0.50 was from STaR-era identical-chain samples where the
    # loss truly indicated overfit-to-trivia. Property-quorum samples
    # are diverse enough (admitted by 2+ independence classes) that a
    # <0.50 pre-loss just reflects the fluency of reference code, not
    # memorization. 0.15 is low enough to still catch pathological
    # "loss=0.01 on 3 identical assert strings" failure modes.
    skip_if_initial_loss_below: float = 0.15
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    # max_grad_norm tightened 1.0 → 0.3 after run-10. With the old value,
    # a single noisy batch could produce a >1.0 grad norm whose clipped-to-1.0
    # update still moved weights more than ~1% per step given LoRA+'s
    # 16× B-side LR multiplier. 0.3 caps per-step update magnitude and is
    # standard practice for small-batch preference/RLHF training.
    max_grad_norm: float = 0.3
    target_modules: list[str] = field(default_factory=lambda: [
        # LLaMA/Mistral/Qwen-style
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        # GPT-2/GPT-J-style (Conv1D layers)
        "c_attn", "c_proj", "c_fc",
    ])
    # Custom: weakness-adaptive rank scaling
    min_rank: int = 4
    max_rank: int = 256
    weakness_rank_scale: float = 2.0  # how much extra rank for weak layers

    # DPO / preference-pair training.
    # training_mode:
    #   "sft"    — supervised fine-tuning on positive samples only (default, preserves behavior)
    #   "dpo"    — preference-pair DPO only (requires PreferencePair inputs)
    #   "mixed"  — alternate SFT and DPO batches when both sources are available
    # dpo_beta: KL regularization strength. 0.1 = standard (Rafailov 2023).
    # Higher beta = stay closer to reference; lower beta = diverge more aggressively.
    training_mode: str = "sft"
    dpo_beta: float = 0.1

    # GRPO (Group Relative Policy Optimization, DeepSeek 2024).
    # Only used when training_mode == "grpo".
    #   grpo_group_size: G completions sampled per prompt; rewards normalized within group.
    #   grpo_clip_eps: PPO-style importance-ratio clip; 0.2 = standard.
    #   grpo_rollout_refresh_steps: regenerate cached rollouts every N optimizer steps.
    #   grpo_max_new_tokens: rollout length cap per completion.
    #   grpo_rollout_temperature / top_p: sampling params for rollouts.
    grpo_group_size: int = 8
    grpo_clip_eps: float = 0.2
    grpo_rollout_refresh_steps: int = 64
    grpo_max_new_tokens: int = 512
    grpo_rollout_temperature: float = 1.0
    grpo_rollout_top_p: float = 0.95

    # Process Reward Model (PRM, Lightman et al. 2023). Dense per-step rewards
    # for GRPO. When use_prm=True, rl_engine's reward_fn should score every step
    # via PRM.score_chain() instead of returning an outcome-only scalar.
    # Trained once per RSI cycle on accumulated samples (<4GB extra VRAM on A6000).
    use_prm: bool = False
    prm_lr: float = 1e-4
    prm_epochs: int = 1
    # Aggregation passed to make_prm_reward_fn: "min" (Lightman default — punishes
    # any bad step), "mean", or "last" (~outcome reward).
    prm_aggregate: str = "min"

    # Metacognitive calibration (metacog_calib).
    #   enable_calibration_loss — when True, mix per-sample Brier score into
    #     the SFT objective as an auxiliary penalty (scalar, bounded in [0,1]).
    #   calibration_loss_weight — lambda on the auxiliary term. 0.1 = ~10%
    #     weight vs the base CE loss at worst-case Brier.
    # calibration_ece is always reported for monitoring (no training effect).
    enable_calibration_loss: bool = False
    calibration_loss_weight: float = 0.1

    # LoRA+ (Hayou et al. 2024): B trains faster than A because A starts near
    # identity (Kaiming) while B starts at zero. Split into two optimizer groups
    # with lr_B = lr_A * lora_plus_ratio. Canonical ratio = 16.
    # When DoRA is active, the `magnitude` parameter joins the A (slow) group.
    use_lora_plus: bool = True
    lora_plus_ratio: float = 16.0

    # rsLoRA (Kalajdzievski 2023): scaling = alpha / sqrt(rank) instead of
    # alpha / rank. Strictly dominates classic scaling as rank grows.
    use_rslora: bool = True
    # LoRA initialization: "kaiming" (A kaiming, B zeros) or "pissa"
    # (Meng et al. 2024: A,B from top-r SVD of base weight, with captured
    # components subtracted from the original). PiSSA converges ~2-3x faster
    # at the cost of a one-off SVD per LoRA layer.
    init_method: str = "kaiming"

    # DoRA (Liu et al. 2024): decompose W into magnitude + direction; LoRA
    # adapts direction and a separate trainable magnitude vector scales each
    # input feature independently. Same rank, materially better downstream
    # quality. Cost: ~5-10% extra VRAM (full V materialization per forward for
    # the norm computation, under no_grad), + one scalar per input feature.
    use_dora: bool = False

    # train_max_seq_length: cap on padded sequence length during SFT/DPO
    # training. ModelConfig.max_seq_length governs generation/inference
    # (typically 4096 to fit reasoning tokens) but SFT samples are padded
    # to whichever max_length the dataset is built with. Property-verified
    # code solutions (the v0.2.1 RSI distribution) are typically <500
    # tokens; padding each to 4096 wastes ~8x per-step compute and makes
    # attention O(seq²) dominate the forward pass.
    # 1024 caps attention cost at 1/16 of 4096's while still fitting
    # long samples (longer ones are truncated preserving EOS — existing
    # TrainingDataset behavior). Dataset clamps to min(this, ModelConfig.
    # max_seq_length) so shrinking model seqlen never under-runs this.
    train_max_seq_length: int = 1024

    # use_gradient_checkpointing: when True, recompute activations during
    # backward to save ~3x VRAM at ~2x compute cost. For 32B-4bit (~18GB
    # on A6000 48GB) + rank-8 LoRA (~180MB params + Adam states) +
    # train_max_seq_length=1024 batch=2, activations are ~2-4GB and fit
    # comfortably with ~25GB headroom, so GC is pure slowdown. Kept True
    # by default (safe — matches prior behavior). Set False at next
    # restart to cut train time ~40-50% once held-out VRAM headroom is
    # confirmed by an operator. OOM-retry path still halves batch size.
    use_gradient_checkpointing: bool = True

    def __post_init__(self):
        if self.lora_rank < 1:
            raise ValueError(f"lora_rank must be >= 1, got {self.lora_rank}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.num_epochs < 1:
            raise ValueError(f"num_epochs must be >= 1, got {self.num_epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.gradient_accumulation_steps < 1:
            raise ValueError(f"gradient_accumulation_steps must be >= 1, got {self.gradient_accumulation_steps}")
        if self.early_stop_loss <= 0:
            raise ValueError(f"early_stop_loss must be > 0, got {self.early_stop_loss}")
        if self.skip_if_initial_loss_below < 0:
            raise ValueError(
                f"skip_if_initial_loss_below must be >= 0, got {self.skip_if_initial_loss_below}"
            )
        if self.max_steps_per_cycle < 1:
            raise ValueError(f"max_steps_per_cycle must be >= 1, got {self.max_steps_per_cycle}")
        if self.min_steps_per_cycle < 1:
            raise ValueError(f"min_steps_per_cycle must be >= 1, got {self.min_steps_per_cycle}")
        if self.min_steps_per_cycle > self.max_steps_per_cycle:
            raise ValueError(
                f"min_steps_per_cycle ({self.min_steps_per_cycle}) > "
                f"max_steps_per_cycle ({self.max_steps_per_cycle})"
            )
        if not (0.0 <= self.warmup_ratio <= 1.0):
            raise ValueError(f"warmup_ratio must be in [0, 1], got {self.warmup_ratio}")
        if self.min_rank > self.max_rank:
            raise ValueError(f"min_rank ({self.min_rank}) > max_rank ({self.max_rank})")
        if self.lora_rank < self.min_rank:
            raise ValueError(f"lora_rank ({self.lora_rank}) < min_rank ({self.min_rank})")
        if self.training_mode not in ("sft", "dpo", "mixed", "grpo"):
            raise ValueError(
                f"training_mode must be one of 'sft', 'dpo', 'mixed', 'grpo' — got {self.training_mode!r}"
            )
        if self.dpo_beta <= 0:
            raise ValueError(f"dpo_beta must be > 0, got {self.dpo_beta}")
        if self.grpo_group_size < 2:
            raise ValueError(f"grpo_group_size must be >= 2, got {self.grpo_group_size}")
        if not (0.0 < self.grpo_clip_eps < 1.0):
            raise ValueError(f"grpo_clip_eps must be in (0, 1), got {self.grpo_clip_eps}")
        if self.grpo_rollout_refresh_steps < 1:
            raise ValueError(
                f"grpo_rollout_refresh_steps must be >= 1, got {self.grpo_rollout_refresh_steps}"
            )
        if self.grpo_max_new_tokens < 1:
            raise ValueError(f"grpo_max_new_tokens must be >= 1, got {self.grpo_max_new_tokens}")
        if self.prm_aggregate not in ("min", "mean", "last"):
            raise ValueError(
                f"prm_aggregate must be one of 'min', 'mean', 'last' — got {self.prm_aggregate!r}"
            )
        if self.calibration_loss_weight < 0:
            raise ValueError(
                f"calibration_loss_weight must be >= 0, got {self.calibration_loss_weight}"
            )
        if not (1.0 <= self.lora_plus_ratio <= 64.0):
            raise ValueError(
                f"lora_plus_ratio must be in [1.0, 64.0], got {self.lora_plus_ratio}"
            )
        if self.init_method not in ("kaiming", "pissa"):
            raise ValueError(
                f"init_method must be one of 'kaiming', 'pissa' — got {self.init_method!r}"
            )
        if self.train_max_seq_length < 1:
            raise ValueError(
                f"train_max_seq_length must be >= 1, got {self.train_max_seq_length}"
            )


@dataclass
class EscalationSchedule:
    verification: int = 4
    diagnosis: int = 7
    generation: int = 10


@dataclass
class OrchestratorConfig:
    max_cycles: int = 100
    min_improvement_threshold: float = 0.01
    plateau_patience: int = 3
    escalation_schedule: EscalationSchedule = field(default_factory=EscalationSchedule)
    output_dir: Path = Path("./outputs")
    log_dir: Path = Path("./logs")
    checkpoint_every: int = 1  # save every N cycles
    resume_from: Optional[str] = None  # resume from checkpoint path
    # --- Observability knobs (cycle2-autopsy). All default-off for BC. ---
    # Number of times to run held-out eval per cycle. >1 reveals measurement
    # noise: the spread across repetitions is a lower bound on what a "real"
    # improvement must exceed.
    heldout_repetitions: int = 1
    # Write outputs/cycle_metrics/cycle_N.json with per-sample/per-question
    # records, training loss trajectory, and STaR internals.
    write_cycle_metrics: bool = False
    # Write outputs/cycle_samples/cycle_N.jsonl — all training samples with
    # prompt/chain/response/expected_answer/verified/notes.
    write_cycle_samples: bool = False
    # When True, the trainer populates TrainingMetrics.loss_trajectory with
    # per-step losses. Small overhead; only collect when we plan to dump it.
    collect_training_loss_trajectory: bool = False
    # Execution mode: "classic" = diagnose→generate→verify→train (default);
    # "rsi" = full RSI tick per spec §4 (requires synthesis enabled).
    mode: str = "classic"
    # Speed knob: RSI Step 0 diagnostics refresh period. Reuse the prior tick's
    # DiagnosticResult for N-1 cycles, re-run fully every Nth cycle (and always
    # on cycle 1 / post-training). On 32B-R1 the 240-prompt diagnostic is
    # ~6 min/miss; period=5 cuts miss rate from 1/3 to 1/5, saving ~2.4 min/cycle
    # on average vs period=3. Cache still invalidates after every training step,
    # so staleness is bounded to non-training cycles where mastery is stable.
    # Set to 1 to disable caching.
    rsi_diagnostic_refresh_every: int = 5
    # Quick regression probe size (per domain) — 5 is enough to detect the
    # -0.2+ drops that trigger revert. Default was 8; 5 saves ~5s per training
    # cycle without changing revert semantics.
    regression_probe_questions_per_domain: int = 5

    def __post_init__(self):
        if self.max_cycles < 1:
            raise ValueError(f"max_cycles must be >= 1, got {self.max_cycles}")
        if self.checkpoint_every < 1:
            raise ValueError(f"checkpoint_every must be >= 1, got {self.checkpoint_every}")
        if self.plateau_patience < 1:
            raise ValueError(f"plateau_patience must be >= 1, got {self.plateau_patience}")
        if self.heldout_repetitions < 1:
            raise ValueError(f"heldout_repetitions must be >= 1, got {self.heldout_repetitions}")
        if self.mode not in ("classic", "rsi"):
            raise ValueError(f"orchestrator.mode must be 'classic' or 'rsi', got {self.mode!r}")


@dataclass
class VLLMConfig:
    model_path: str = ""
    dtype: str = "bfloat16"
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.85
    quantization_config: Optional[dict] = None
    # If True, the orchestrator skips reloading vLLM after training and runs
    # post-diagnostic + held-out eval in HF mode. Slower per-prompt but avoids
    # the ~3-5 min vLLM reload + CUDA graph recapture. Net win when the probe
    # count is small (e.g. single-domain RSI with ~40 questions).
    skip_reload_after_training: bool = False

    def __post_init__(self):
        if not (0.0 < self.gpu_memory_utilization <= 1.0):
            raise ValueError(f"gpu_memory_utilization must be in (0, 1], got {self.gpu_memory_utilization}")
        if self.max_model_len < 1:
            raise ValueError(f"max_model_len must be >= 1, got {self.max_model_len}")


@dataclass
class SynthesisConfig:
    """Configuration for the task-synthesis pipeline (opt-in, default-off)."""
    enable_task_synthesis: bool = False
    tasks_per_cycle: int = 20
    property_consensus_threshold: float = 0.7
    # Candidates per proposed problem in rsi_tick Step 3 (SOLVE). Candidate
    # 0 is always the reference the model emitted when proposing (guaranteed
    # quorum pass if the model wrote self-consistent tests). Candidates 1..k-1
    # are freshly sampled attempts for diversity. k=6 balances cost vs
    # chance-of-finding-a-passing-diverse-solution. Run-4 cycle 5 produced
    # 1 passing candidate across ~60 attempts with k=3; at k=6 with reference
    # as candidate 0, every well-formed proposal should yield ≥1 sample.
    candidates_per_problem: int = 6
    # If True, rsi_tick uses the builtin-based code-proposal path
    # (PROBLEM + ENTRY + REFERENCE + TESTS → trusted builtin checks).
    # This is the only path that actually produces training samples with
    # an 8B base model — the legacy §3.1 co-gen format requires the model
    # to emit property source code, which it can't do reliably. Set False
    # to force the legacy path on stronger models.
    use_builtin_code_path: bool = True

    def __post_init__(self):
        if self.tasks_per_cycle < 1:
            raise ValueError(f"tasks_per_cycle must be >= 1, got {self.tasks_per_cycle}")
        if not (0.0 < self.property_consensus_threshold <= 1.0):
            raise ValueError(
                f"property_consensus_threshold must be in (0, 1], "
                f"got {self.property_consensus_threshold}"
            )


@dataclass
class SystemConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)
    use_vllm: bool = False
    vllm: Optional[VLLMConfig] = None
    # Alternative model backend. None = default (HF or vLLM per use_vllm).
    # "tdq" = TDQModelLoader (decompresses a .tdq file into an HF model).
    backend: Optional[str] = None
