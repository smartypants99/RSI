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
    # Task #20 throughput: optional Liger kernels (linkedin/Liger-Kernel).
    # When True AND the `liger_kernel` package is importable AND the model
    # is Qwen2-family, apply fused RMSNorm/RoPE/SwiGLU/CE kernels. Safe
    # with bnb-4bit per upstream README and gemini consult. Graceful
    # no-op when liger_kernel isn't installed OR the model isn't Qwen2.
    # Default True because the no-op branch is free; set False to force-
    # skip on models where upstream has regressions.
    use_liger_kernels: bool = True

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
    # Mirror of GeneratorConfig.use_logprob_continuous_score — consumed
    # by _check_ground_truth_scored() to enable vLLM prompt_logprobs
    # gold-token scoring on non-code items. See GeneratorConfig for the
    # ρ-lift rationale.
    use_logprob_continuous_score: bool = True

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
    # Task #10 speed pass: 30 → 24 (≈20% cut in solve tokens / cycle).
    samples_per_weakness: int = 24
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
    # Sample-quality clean-floor filter (task #14). When the training pool
    # has >= sample_quality_min_clean_floor total samples AND the clean-only
    # subset (no "any_fail" verdict warning) is also >= this floor, drop every
    # sample whose verdict_warnings contains "any_fail" — those are the
    # relaxed-accept-policy (majority/quorum_2of3) admits that bypassed a FAIL
    # verdict. Below the floor, keep all samples so starvation doesn't skip
    # training entirely. Set to 0 to disable.
    sample_quality_min_clean_floor: int = 0  # disabled; down-weight via sample_quality_any_fail_weight instead
    sample_quality_any_fail_weight: float = 0.4  # task #13: down-weight any_fail samples instead of dropping
    # Continuous log-prob-of-gold score on non-code ground-truth items.
    # When True (default), DiagnosticsEngine._check_ground_truth_scored
    # blends the model's mean-per-token gold-prob into per_question['score']
    # for numeric_exact / sympy_equiv / exact_mc / exact_string methods.
    # This lifts the paired-sample correlation ρ from ~0.46 (binary)
    # toward 0.8+ (continuous) — the load-bearing assumption for the
    # ≤1% MDE at N=600 wedge-1 claim (task #25). Disable if vLLM
    # prompt_logprobs adds unacceptable latency or breaks compat.
    use_logprob_continuous_score: bool = True

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
        if self.sample_quality_min_clean_floor < 0:
            raise ValueError(
                f"sample_quality_min_clean_floor must be >= 0, "
                f"got {self.sample_quality_min_clean_floor}"
            )


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

    # External-solver verifier backends (Tasks #3/#4/#5). When enabled, the
    # task synthesizer stamps the corresponding Property kinds onto eligible
    # problems (theorem → lean, SMT → z3, physics/chemistry → sim). Each
    # backend ships graceful-skip behavior if its dependency (lean4/z3/scipy)
    # is unavailable, so flipping on is safe even without the binary installed.
    # Default-off for BC; consolidation flip enables lean+z3+sim together.
    # lean4 binary is not installed on the target GPU; leave off.
    lean_verifier_enabled: bool = False
    # Consolidation flip: z3 + sim default ON. Both ship graceful-skip when
    # their dependency is missing, and gating lives in task_synthesizer
    # (stamps properties onto eligible problems).
    z3_verifier_enabled: bool = True
    sim_verifier_enabled: bool = True

    # Task #11 concern #1: verifier accept policy. "any_fail_veto" is the
    # strict §2.1 behavior (any single FAIL rejects). Live run showed
    # cycle 1 rejected multiple 2-of-3 PASS candidates, starving training.
    # "majority" accepts when ceil(N/2) PASS; "quorum_2of3" accepts
    # 2-of-3 PASS with at most 1 FAIL. Both relaxed policies still
    # enforce distinct-classes + duplicate-author rules and record
    # verdict_warn=any_fail on the accepted record.
    verifier_accept_policy: str = "any_fail_veto"  # strict: live cycle 2 showed 91.7% of majority-accepts carry any_fail warnings → noise dominates training

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
        if self.verifier_accept_policy not in ("any_fail_veto", "majority", "quorum_2of3"):
            raise ValueError(
                "verifier_accept_policy must be one of "
                "'any_fail_veto'|'majority'|'quorum_2of3', "
                f"got {self.verifier_accept_policy!r}"
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
    # Live-run cycle 2 (32 samples, LR=5e-5): held-out AND anchor both
    # regressed -15pp. LR=5e-5 over-drives QLoRA on the 32B base — lowered
    # to 2e-5. Still 4× the original 5e-6 (which was too timid to move the
    # loss) but not the 5e-5 that was causing damage.
    learning_rate: float = 4e-6
    # Defaults tuned for small-cycle RSI regime (typ. 5-30 verified samples/cycle).
    # Cycle-2 (success) = 1-2 optimizer steps, final loss ~0.4-0.8.
    # Cycle-3 (overfit) = 25+ steps on 9 samples, final loss 0.045.
    # With samples<30: num_epochs=2, grad_accum=4 keeps steps in the 2-4 range
    # instead of the 25+ that caused memorization.
    num_epochs: int = 2
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    # Warmup-cycle epoch cap (task #14). Early cycles have the least-
    # calibrated reference (cycle-1 held-out is always thinly sampled), so
    # small per-cycle weight updates reduce the chance that a noisy reference
    # gets locked in as "best". For cycle <= num_epochs_warmup_cycles the
    # effective num_epochs is min(num_epochs, num_epochs_warmup). Set
    # num_epochs_warmup_cycles=0 to disable the warmup cap.
    num_epochs_warmup: int = 1
    num_epochs_warmup_cycles: int = 5

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
    max_grad_norm: float = 1.0
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
    lora_plus_ratio: float = 4.0

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
    # comfortably with ~25GB headroom, so GC is pure slowdown.
    # Task #20 throughput pass: flipped default False. Gemini consult
    # predicts 25-30% speedup on the 32B-4bit / A6000 workload. The OOM-
    # retry path (src/trainer/custom_lora.py) still halves batch size, so
    # a misconfigured restart on a smaller GPU self-recovers. Set True
    # explicitly if running on <48GB VRAM or longer seq lengths.
    use_gradient_checkpointing: bool = True

    def __post_init__(self):
        if self.lora_rank < 1:
            raise ValueError(f"lora_rank must be >= 1, got {self.lora_rank}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.num_epochs < 1:
            raise ValueError(f"num_epochs must be >= 1, got {self.num_epochs}")
        if self.num_epochs_warmup < 1:
            raise ValueError(
                f"num_epochs_warmup must be >= 1, got {self.num_epochs_warmup}"
            )
        if self.num_epochs_warmup_cycles < 0:
            raise ValueError(
                f"num_epochs_warmup_cycles must be >= 0, "
                f"got {self.num_epochs_warmup_cycles}"
            )
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
    # --- Structured observability (src/utils/structured_logs.py) ---
    # Master switch. When True, the five JSONL sinks under output_dir are
    # populated by their respective call sites. Each sub-flag below can
    # silence one sink individually (e.g. training_steps is the highest
    # volume — set its sub-flag False if log write time matters).
    # Default True: the sinks are small, append-only, and critical for the
    # "why did training regress?" post-mortem. Operators who want zero
    # disk churn flip the master flag.
    structured_observability_enabled: bool = True
    structured_log_training_steps: bool = True
    structured_log_heldout_per_prompt: bool = True
    structured_log_verify_decisions: bool = True
    structured_log_propose_attempts: bool = True
    structured_log_cycle_summary: bool = True
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
    # Auto-diagnose: after each cycle finishes, shell out to
    # scripts/analyze_cycle.py and surface its "Bottom line" TL;DR to stderr
    # + append a structured row to outputs/auto_diagnosis.jsonl. Zero effect
    # on loop logic; never crashes the loop. Default on — operator wants to
    # see likely cause inline whenever a cycle regresses/alarms.
    auto_diagnose_enabled: bool = True
    # Speed knob: RSI Step 0 diagnostics refresh period. Reuse the prior tick's
    # DiagnosticResult for N-1 cycles, re-run fully every Nth cycle (and always
    # on cycle 1 / post-training). On 32B-R1 the 240-prompt diagnostic is
    # ~6 min/miss; period=5 cuts miss rate from 1/3 to 1/5, saving ~2.4 min/cycle
    # on average vs period=3. Cache still invalidates after every training step,
    # so staleness is bounded to non-training cycles where mastery is stable.
    # Set to 1 to disable caching.
    # Task #12 (warm-speed): bumped 5 → 7 to further cut diagnostic miss rate
    # from 1/5 to 1/7 without meaningfully increasing staleness on
    # non-training cycles.
    rsi_diagnostic_refresh_every: int = 7
    # Quick regression probe size (per domain) — 5 is enough to detect the
    # -0.2+ drops that trigger revert. Default was 8; 5 saves ~5s per training
    # cycle without changing revert semantics.
    regression_probe_questions_per_domain: int = 5
    # Per-domain oversample target during frozen held-out eval (task #3).
    # After HELD_OUT_ONLY partition filtering (~37% retained) and ×6 active
    # domains, 540 per-domain lands ~1200 total held-out questions, giving
    # SE ≈ 0.014 at p=0.5 — enough to resolve |delta| > 0.02 confidently.
    heldout_questions_per_domain: int = 540
    # Task #10 speed pass: quick-eval subsample. On non-"full" cycles we
    # downscale questions_per_domain so the eval lands ~heldout_quick_subsample_n
    # prompts total after the HELD_OUT_ONLY partition filter (~37% retain).
    # Every heldout_full_every cycles we run the full sweep to refresh the
    # base-model reference.
    #
    # Honest statistical-power calibration (McNemar-style paired binary):
    #   Var(Δ) per item ≈ 2·p·(1-p)·(1-ρ); at p=0.5, ρ=0.9 → σ² ≈ 0.05 / N.
    #   N=128  → SE(Δ) ≈ √(0.05/128)  ≈ 0.020  → z=2 at ≈4pp delta.
    #   N=1200 → SE(Δ) ≈ √(0.05/1200) ≈ 0.0065 → z=2 at ≈1.3pp delta.
    # So the QUICK subsample reliably detects |Δ| ≥ ~3-4%; the FULL sweep
    # every 5 cycles detects |Δ| ≥ ~1%. This aligns with the existing
    # regression_revert_threshold=0.03 — the quick-subsample sensitivity
    # matches the revert gate, and finer <1pp measurement is only needed
    # on the cadence the full sweep already provides.
    # Set heldout_quick_subsample_n=0 to disable (always run full).
    # Task #22 speed-round-2: 128 → 96. With 4 active domains quick eval
    # lands ~96 post-filter (SE(Δ) ≈ √(0.05/96) ≈ 0.023 → z=2 at ≈4.6pp),
    # still matched to regression_revert_threshold=0.03 and the full-sweep
    # every 5 cycles provides the <1pp resolution. Shaves ~25% off quick
    # eval wall-clock on the 8-11 min quick cycle.
    heldout_quick_subsample_n: int = 96
    heldout_full_every: int = 5
    # Task #23 wedge 1: skip the full held-out eval when the quick
    # regression probe already showed a regression beyond this threshold.
    # Full eval takes ~40 min; if the quick probe already says training
    # hurt, there's no point spending 40 more minutes confirming it.
    # Setting skip_full_heldout_on_quick_regression=False restores the
    # prior "always run full" behavior. Threshold=0.10 matches the default
    # regression_revert_threshold.
    skip_full_heldout_on_quick_regression: bool = True
    quick_regression_skip_threshold: float = 0.10
    # Task #23 wedge 2: halve the default full-eval per-domain target.
    # With N=600 + paired-sample VR=2.5 the MDE is ~5.1pp (vs ~3.6pp at
    # N=1200) — the 1pp sensitivity claim from the task spec is NOT
    # reachable by N reduction alone; operator must trade off cost vs.
    # sensitivity. See docs in heldout_full_subsample_n math comment
    # below. Set heldout_full_subsample_n=0 to fall back to the legacy
    # heldout_questions_per_domain behavior.
    # MDE math (paired McNemar, α=0.05, power=0.8, p=0.5):
    #   σ²_paired = 2·p·(1-p) / VR
    #   MDE = 2.802 · √(σ²_paired / N)
    # VR=2.5: N=600 → MDE≈0.0511, N=1200 → MDE≈0.0362
    # VR=4.0: N=600 → MDE≈0.0404, N=1200 → MDE≈0.0286
    # VR=10 : N=600 → MDE≈0.0256, N=1200 → MDE≈0.0181
    # The 1pp sensitivity target in the task spec would require VR≥126
    # at N=600, which is unrealistic (observed VR is 3-5×). Shipping
    # N=600 explicitly for throughput; operator may raise back to 1200
    # via heldout_full_subsample_n=1200 if MDE matters more than wall.
    heldout_full_subsample_n: int = 600
    # Task #23 wedge 3: cache the base-model held-out predictions once at
    # loop startup so cycle-1 paired_delta computation does not require a
    # second full eval run. When True, _cache_base_heldout_predictions()
    # runs during __init__ (or first cycle) and stashes per-question
    # correctness keyed by (prompt, expected). Frozen-eval seed guarantees
    # the question set matches across cycles.
    heldout_cache_base_predictions: bool = True
    # Task #23 wedge 4: bump max_num_seqs during the held-out-only phase.
    # The vLLM engine is configured once at startup (max_num_seqs is not
    # hot-swappable in current vLLM), so this value is read by the vLLM
    # loader when coresident_training is disabled during Phase 5b. Leave
    # at 0 to inherit VLLMConfig.max_num_seqs.
    heldout_max_num_seqs: int = 96
    # Task #23 wedge 5: hard cap on max_new_tokens during held-out eval
    # generation. Most held-out questions have short canonical answers
    # (numbers, function signatures) — 2048-token budgets are wasted KV-
    # cache space. 512 matches the existing hard-coded cap at the two
    # held-out generate_batch sites and is enforced through
    # heldout_eval_max_tokens so it's visible and tunable.
    heldout_eval_max_tokens: int = 512
    # Substrate update: promote the merged checkpoint to a new base every N
    # training cycles. LoRA on a frozen 4-bit base has a fixed ceiling (the
    # only trainable params are the low-rank adapters); periodically snapshot
    # the current merged weights as a new base and restart LoRA fresh on top.
    # Guardrail: only promote if cumulative held-out improvement since the
    # last promotion (or baseline) is >= substrate_merge_min_improvement; a
    # regression or flatline epoch is skipped. Set to 0 to disable entirely.
    merge_into_base_every: int = 10
    substrate_merge_min_improvement: float = 0.005
    # QLoRA adapter persistence. On bnb-4bit bases, merge_lora no-ops (packed
    # weights ≠ dense delta) AND save_checkpoint no-ops (save_pretrained
    # raises). Without adapter persistence every cycle evaluates the untrained
    # base. When True (default) and base is 4bit, each cycle writes a PEFT-
    # format adapter and vLLM loads it at inference via LoRARequest. Auto-
    # gated by base quantization — setting True on a full-precision base is
    # harmless (the gate returns False and this has no effect).
    use_lora_adapter_persistence: bool = True
    # Stable session id for append-only registries under outputs/<kind>/<sid>.jsonl.
    # Previously unset → RSIRegistries.open fell back to uuid.uuid4()[:12] every
    # process restart, siloing each run's artifacts into a fresh file. That
    # meant the property + proposer few-shot banks could NEVER accumulate
    # across restarts — defeating the point of RSI library growth. Default
    # "rsi" gives one stable bank per output_dir; set a unique value when
    # you want isolated runs (e.g. A/B experiments sharing output_dir).
    run_id: str = "rsi"
    # True weight growth (Task #1, weight-growth). Every N trained cycles,
    # distill the current model into a 1.5N-param student via
    # src/trainer/growth.py::grow_and_distill. Full GrowthConfig (growth_factor,
    # distill_epochs, abort_if_worse_by, …) is constructed by the caller in
    # growth.py; this field only gates WHEN the growth step fires. 0 = off.
    grow_every: int = 15
    # Self-editing pipeline (Task #2, self-edit). Every N cycles the model
    # proposes a unified diff against self_edit_candidate_path, which is
    # applied in a worktree sandbox and smoke-evaluated for smoke_cycles
    # before merging only if held-out delta >= min_improvement. 0 = off.
    self_edit_every: int = 8
    self_edit_max_diff_lines: int = 40
    self_edit_min_improvement: float = 0.005
    self_edit_smoke_cycles: int = 2
    self_edit_candidate_path: str = "src/generator/data_generator.py"
    # Fast-start (Task #11, fast-start). Shrinks cold-cycle-1 wall-time so
    # a restart hits its first trained cycle in ≤20 min instead of ~60.
    #  - skip_first_diagnostics: cycle 1 uses a uniform-weakness default
    #    (src/utils/fast_start.default_weakness_diag); real diagnostics
    #    resume cycle 2+.
    #  - prestash_prior_samples: on __init__, glob
    #    outputs/training_pool/*.jsonl and prior outputs_run_*/training_pool
    #    for records with a different session_id, load up to
    #    prestash_max_samples of them into _rsi_pending_pool so cycle 1
    #    can train without waiting for a full synthesis round.
    skip_first_diagnostics: bool = True
    prestash_prior_samples: bool = True
    prestash_max_samples: int = 30

    # Anchor eval (Task #1, ground-truth). External benchmarks run after the
    # internal held-out eval each cycle; if internal score improves while
    # anchor score drops by >= verifier_capture_alarm_threshold, the cycle
    # fires a verifier-capture alarm (self-graded loop drifting from ground
    # truth). Default True; consolidation flip confirms ON.
    anchor_eval_enabled: bool = True
    anchor_eval_size: int = 200
    verifier_capture_alarm_threshold: float = 0.01
    anchor_eval_benchmarks: list[str] = field(default_factory=lambda: [
        "humaneval", "mbpp", "gsm8k", "math",
    ])
    anchor_eval_cache_dir: str = "outputs/external_benchmarks"

    # Consolidation flips (Team RSI-Trust v1). Each gates an opt-in feature
    # module added this round; defaults True so the consolidated pipeline
    # runs with all trust-layer features ON.
    #  - verifier_adequacy_enforce: src/verifier/adequacy.py scores each
    #    verifier property on a curated triple bundle (TPR/TNR) and prunes
    #    library entries that fall below the gate.
    #  - eval_partition_strict: enforce the 4-way HELD_OUT/PROPOSER/TRAIN/
    #    SMOKE partition; proposer and training pools cannot touch held-out.
    #  - meta_meta_enabled: src/orchestrator/meta_meta.py attributes held-out
    #    deltas to per-component contributions and graduates the self-edit
    #    allow-list tier as confidence rises.
    verifier_adequacy_enforce: bool = True
    eval_partition_strict: bool = True
    meta_meta_enabled: bool = True
    # Paired-sample held-out variance reduction (eval-isolation commit
    # abbcb06). Loop pairs current vs previous cycle's per-question records
    # and reports mean delta ± paired SE.
    paired_eval_enabled: bool = True
    # Task #27: held-out eval statistical mode.
    #   "binary"     — legacy McNemar on {0,1} correctness (pre task-#25).
    #   "continuous" — continuous per-question score (defaults to {0,1}
    #                  correctness fallback when records carry no 'score'
    #                  field). SE = sqrt(2σ²(1-ρ)/N). Matches the paired
    #                  schema (mean, se, z, n) so the downstream
    #                  regression_revert_threshold comparison is unchanged.
    # Default "continuous" (task #27 landing). Flip back to "binary" to
    # reproduce pre-task-#25 MDE math.
    heldout_eval_mode: str = "continuous"
    # Task #27: group-sequential early-stop for held-out eval. When True
    # and heldout_repetitions >= 2, sprt_decide() is called after each
    # rep; break the rep loop on stop_reject_null (signal confirmed).
    # Uses OBF K=3 α=0.05 critical values from sequential_eval.
    # Safety gates (promotion eligibility, capture alarm, mode-collapse,
    # regression guard) are untouched — early-stop is a throughput knob
    # only, not a decision-rule change.
    sprt_early_stop_enabled: bool = True
    # Intra-rep chunked SPRT (ships the run_chunked primitive). When True
    # and the held-out branch runs a full (non-quick) eval, the loop
    # iterates diagnostics.run_chunked() and calls sprt_decide after each
    # chunk against the BaseHeldoutCache reference. Breaks on
    # stop_reject_null or stop_accept_null. Safety gates unaffected —
    # promotion eligibility, capture alarm, mode-collapse, regression guard
    # all run against the (partial) accumulated per_question exactly as
    # they would against a full-N result.
    # OBF K=4 α=0.05 critical values are (4.049, 2.863, 2.337, 2.024);
    # see diagnostics/sequential_eval.py for the derivation and JT Table 2.3.
    heldout_chunked_sprt_enabled: bool = True
    heldout_chunk_size: int = 150
    heldout_sprt_max_chunks: int = 4
    # Optional |z| futility threshold checked after each chunk (before the
    # final one). When set, stop_accept_null fires whenever |z| < threshold
    # on a non-final look. Default 0.5: with the continuous log-prob score
    # raising expected paired ρ from 0.46 → 0.8+, per-chunk MDE drops enough
    # that true-null cycles spend the vast majority of their mass inside
    # |z| ≤ 0.5 by chunks 1-2. For K=4 OBF α=0.05 + futility_z=0.5,
    # simulated type-II error remains < 0.2 for effect sizes ≥ 2pp
    # (gemini-cross-checked 2026-04-23). Set to None to disable the
    # futility boundary and preserve pure rejection-only early-stop.
    heldout_sprt_futility_z: float | None = 0.5
    # meta_meta append-only history (src/orchestrator/meta_meta.py). Written
    # each cycle when meta_meta_enabled.
    meta_meta_history_path: str = "outputs/meta_meta_history.jsonl"
    # Per-phase wall-time history sidecar (task #10). Fed each cycle from
    # CycleResult.phase_times; feeds meta_meta.wall_time_trend so end-of-10-
    # cycle windows can log "cycle time trending down by X%".
    meta_meta_wall_time_path: str = "outputs/meta_meta_wall_time.jsonl"
    # Rescore + prune cadence for verifier adequacy library (consumed by the
    # adequacy module when a registry-side owner wires it in).
    verifier_adequacy_rescore_every: int = 10
    # MoE conversion (src/trainer/moe_conversion.py). Post-hoc dense-to-sparse
    # upcycling of FFN layers via MegaBlocks-style block-sparse experts. Fires
    # during growth events as an alternative to layer expansion. Off by default;
    # per gemini consult, SOTA uses clustering-based init + shared-expert
    # architecture, so expect to tune num_experts / top_k / shared_experts
    # together. Flag also gates the MoE path out of test runs.
    moe_conversion_enabled: bool = False
    moe_num_experts: int = 4
    moe_top_k: int = 2
    moe_shared_experts: int = 1
    moe_init_method: str = "clustering"  # "copy_perturb" | "slice" | "clustering"
    moe_router_noise_std: float = 0.02

    # Fast-student distillation (src/utils/fast_student.py). When True, propose/
    # solve generation routes through a periodically-distilled small student
    # (default Qwen2.5-Coder-1.5B). Teacher training is unchanged; verification
    # runs on real ground truth. See FastStudentConfig for knobs.
    use_fast_student: bool = True
    fast_student_model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    # When False, the RSI loop harvests (prompt, completion) pairs into the
    # FastStudentManager buffer but does NOT call on_trained_cycle — the
    # inline distill is suppressed. Default False to avoid co-resident
    # teacher+student OOM risk on the live run (cross-review issue A):
    # the teacher HF weights are still resident at the on_trained_cycle
    # hook point (pre-vLLM-swap-back), and _default_distill_fn loads the
    # 1.5B student in bf16 + AdamW + per-example forward/backward which
    # has no GPU-headroom guarantee alongside a ~16GB QLoRA teacher.
    # Harvest remains active so buffer accumulates and a future operator
    # can flip this True once an offline distill path lands.
    fast_student_distill_inline: bool = False

    # Component-proposer / meta-meta-meta (src/orchestrator/component_proposer.py).
    # Every N cycles the orchestrator proposes entirely new RSI components
    # (not just tuning existing knobs). 0 = off.
    # Default 0 = OFF. Component proposer is safety-sensitive (spawns new
    # pipeline components) so it must be opted-in per run, not on by default.
    component_proposer_every: int = 0
    # Verdict JSONL for component proposer runs (task #13).
    component_proposer_log_path: str = "outputs/component_proposer_verdicts.jsonl"
    # Compute allocator (UCB1 bandit over {k_candidates, token_budget, ...})
    # consulted at start of each cycle to pick a strategy. Default off; when
    # on, the allocator's selection overrides synthesis.candidates_per_problem
    # and generator.max_new_tokens for that cycle.
    compute_allocator_enabled: bool = False
    compute_allocator_history_path: str = "outputs/compute_allocator_history.jsonl"
    compute_allocator_budget_tokens: float = 1_000_000_000.0

    # Architecture search (src/trainer/arch_search.py). Explores small LoRA
    # topology variants during growth events and keeps the winner.
    arch_search_enabled: bool = True
    arch_search_every: int = 30
    arch_search_min_delta: float = 0.005

    # Best-checkpoint promotion gates (task #2). The overnight run saw
    # cycle 1's 1-sample/2-step eval=0.624 become an unassailable reference
    # bank; cycles 2-6 all reverted to it and the loop made zero forward
    # progress for 6 hours. Require (a) minimum sample count per cycle for
    # promotion eligibility, and (b) N≥best_confirm_cycles consecutive eligible
    # cycles at or above the new high-water mark before promoting the new best.
    # Setting best_confirm_cycles=1 restores the old (broken) behavior.
    best_min_samples_verified: int = 8
    best_confirm_cycles: int = 2
    # Task #11 concern #2: when the post-train anchor eval reports
    # distinct/n < this threshold on ANY benchmark (especially offline
    # fixtures), the cycle is marked mode_collapse_detected=True and
    # becomes ineligible for best-promotion. Threshold 0.6 matches
    # overnight-cycle-2 humaneval distinct=7/12=0.58 (would have tripped).
    # Set to 0.0 to disable the gate.
    mode_collapse_distinct_threshold: float = 0.6
    # Verifier-capture response (task #2). When detect_verifier_capture fires
    # (internal-up / anchor-down divergence), revert vLLM to last confirmed-best
    # checkpoint, mark the cycle ineligible for best promotion, bump degradation
    # counter. If the alarm fires N consecutive times, halt self-edit globally.
    verifier_capture_halt_consecutive: int = 2

    def __post_init__(self):
        if self.max_cycles < 1:
            raise ValueError(f"max_cycles must be >= 1, got {self.max_cycles}")
        if self.checkpoint_every < 1:
            raise ValueError(f"checkpoint_every must be >= 1, got {self.checkpoint_every}")
        if self.plateau_patience < 1:
            raise ValueError(f"plateau_patience must be >= 1, got {self.plateau_patience}")
        if self.heldout_repetitions < 1:
            raise ValueError(f"heldout_repetitions must be >= 1, got {self.heldout_repetitions}")
        if self.heldout_eval_mode not in ("binary", "continuous"):
            raise ValueError(
                f"heldout_eval_mode must be 'binary' or 'continuous', "
                f"got {self.heldout_eval_mode!r}"
            )
        if self.quick_regression_skip_threshold < 0:
            raise ValueError(
                "quick_regression_skip_threshold must be >= 0, "
                f"got {self.quick_regression_skip_threshold}"
            )
        if self.heldout_full_subsample_n < 0:
            raise ValueError(
                "heldout_full_subsample_n must be >= 0, "
                f"got {self.heldout_full_subsample_n}"
            )
        if self.heldout_max_num_seqs < 0:
            raise ValueError(
                "heldout_max_num_seqs must be >= 0, "
                f"got {self.heldout_max_num_seqs}"
            )
        if self.heldout_eval_max_tokens < 1:
            raise ValueError(
                "heldout_eval_max_tokens must be >= 1, "
                f"got {self.heldout_eval_max_tokens}"
            )
        if self.heldout_chunk_size < 1:
            raise ValueError(
                f"heldout_chunk_size must be >= 1, got {self.heldout_chunk_size}"
            )
        if self.heldout_sprt_max_chunks not in (3, 4):
            raise ValueError(
                "heldout_sprt_max_chunks must be 3 or 4 (OBF tables), "
                f"got {self.heldout_sprt_max_chunks}"
            )
        if (
            self.heldout_sprt_futility_z is not None
            and self.heldout_sprt_futility_z < 0
        ):
            raise ValueError(
                "heldout_sprt_futility_z must be >= 0 or None, "
                f"got {self.heldout_sprt_futility_z}"
            )
        if self.heldout_quick_subsample_n < 0:
            raise ValueError(
                f"heldout_quick_subsample_n must be >= 0, got {self.heldout_quick_subsample_n}"
            )
        if self.heldout_full_every < 1:
            raise ValueError(
                f"heldout_full_every must be >= 1, got {self.heldout_full_every}"
            )
        if self.mode not in ("classic", "rsi"):
            raise ValueError(f"orchestrator.mode must be 'classic' or 'rsi', got {self.mode!r}")
        if self.merge_into_base_every < 0:
            raise ValueError(
                f"merge_into_base_every must be >= 0 (0=disabled), got {self.merge_into_base_every}"
            )
        if self.substrate_merge_min_improvement < 0:
            raise ValueError(
                f"substrate_merge_min_improvement must be >= 0, got {self.substrate_merge_min_improvement}"
            )
        if self.grow_every < 0:
            raise ValueError(
                f"grow_every must be >= 0 (0=disabled), got {self.grow_every}"
            )
        if self.self_edit_every < 0:
            raise ValueError(
                f"self_edit_every must be >= 0 (0=disabled), got {self.self_edit_every}"
            )
        if self.self_edit_max_diff_lines < 1:
            raise ValueError(
                f"self_edit_max_diff_lines must be >= 1, got {self.self_edit_max_diff_lines}"
            )
        if self.self_edit_smoke_cycles < 1:
            raise ValueError(
                f"self_edit_smoke_cycles must be >= 1, got {self.self_edit_smoke_cycles}"
            )
        if self.prestash_max_samples < 0:
            raise ValueError(
                f"prestash_max_samples must be >= 0, got {self.prestash_max_samples}"
            )
        if self.anchor_eval_size < 1:
            raise ValueError(
                f"anchor_eval_size must be >= 1, got {self.anchor_eval_size}"
            )
        if self.verifier_capture_alarm_threshold < 0:
            raise ValueError(
                f"verifier_capture_alarm_threshold must be >= 0, "
                f"got {self.verifier_capture_alarm_threshold}"
            )
        if not self.anchor_eval_benchmarks:
            raise ValueError("anchor_eval_benchmarks must be non-empty")
        if self.best_min_samples_verified < 0:
            raise ValueError(
                f"best_min_samples_verified must be >= 0, got {self.best_min_samples_verified}"
            )
        if self.best_confirm_cycles < 1:
            raise ValueError(
                f"best_confirm_cycles must be >= 1, got {self.best_confirm_cycles}"
            )
        if self.verifier_capture_halt_consecutive < 1:
            raise ValueError(
                f"verifier_capture_halt_consecutive must be >= 1, "
                f"got {self.verifier_capture_halt_consecutive}"
            )
        if not (0.0 <= self.mode_collapse_distinct_threshold <= 1.0):
            raise ValueError(
                "mode_collapse_distinct_threshold must be in [0.0, 1.0], "
                f"got {self.mode_collapse_distinct_threshold}"
            )


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
    # Task #10 speed pass: vLLM max_num_seqs. Default 32 leverages A6000
    # KV-cache headroom once propose/solve are shrunk — larger concurrent
    # batch amortizes the per-step scheduler/decode overhead. Set 0 to use
    # vLLM's own default (typically 256, overcommits KV on small GPUs).
    # Task #18 speed pass step 1: bumped 32 → 48. Second-opinion (gemini,
    # two-pass drill) corrected KV budget for Qwen2-32B GQA: per-seq at
    # max_model_len=4096 ≈ 128MB (bf16 KV, 8 KV heads × 128 dim × 64
    # layers × 2 × 4096) or ~256MB counting scheduler overhead. On 48GB
    # A6000 minus 18GB weights minus ~1.5GB CUDA graphs, usable KV budget
    # is ~25-28GB, supporting ~40-48 concurrent sequences safely. 256 was
    # first-pass advice but self-corrected to ~40 on the GQA KV math. 48
    # gives ~50% concurrency gain on the 120-prompt solve batch without
    # touching higher-risk levers (fp8_kv storage is memory-only on
    # Ampere with possible -5-10% throughput from cast overhead;
    # speculative decoding is broken with load_format=bitsandbytes per
    # the same consult).
    max_num_seqs: int = 48
    # ──────────────────────────────────────────────────────────────────────
    # Task #19: vLLM resident-during-training (the 4-6 min/cycle sink).
    # ──────────────────────────────────────────────────────────────────────
    # When True, disable vLLM's CUDA-graph capture so vLLM's static VRAM
    # footprint is ~1.5-2GB lower. Required for coresident_training_enabled
    # on 48GB GPUs (gemini consult: CUDA graph static buffers are the
    # hidden 1-2GB that triggers OOM during the HF backward pass). Safe to
    # leave False when training-coresidency is disabled; eager mode costs
    # ~10-15% throughput on inference.
    enforce_eager: bool = False
    # Master flag for the Tier-B co-resident training path. When True:
    #   - _load_vllm uses gpu_memory_utilization=coresident_vllm_mem_frac
    #     (default 0.42 on A6000, gemini consult) instead of the default,
    #   - swap_to_hf_for_training calls vllm.sleep() (vLLM>=0.6) instead
    #     of destroying the engine, freeing KV cache while keeping weights,
    #   - HF training copy is loaded on top; after training, vllm.wake_up()
    #     and set_lora_adapter(new_adapter) replace the full reload.
    # Default False — requires live GPU validation before enabling. Flag
    # is read at VLLMModelLoader construction and in the orchestrator
    # training swap site.
    coresident_training_enabled: bool = False
    # VRAM fraction vLLM is allowed when coresident_training_enabled=True.
    # gemini consult (48GB A6000, 32B-4bit): 0.42 leaves ~28GB for HF +
    # LoRA + AdamW + activations + KV headroom. FP8 KV (task #18, foom-
    # activator) would relax this to ~0.55.
    coresident_vllm_mem_frac: float = 0.42
    # ──────────────────────────────────────────────────────────────────────
    # Task #22 speed-round-2: AWQ + speculative decoding.
    # ──────────────────────────────────────────────────────────────────────
    # `quantization_scheme` is the vLLM-side quantization strategy. Accepted
    # values:
    #   "auto"  — infer from quantization_config (default; back-compat with
    #             the pre-AWQ code path: load_in_4bit → bitsandbytes).
    #   "bnb"   — force bitsandbytes 4-bit (load_format=bitsandbytes).
    #             Blocks vLLM speculative decoding.
    #   "awq"   — pass quantization="awq" to vLLM. Enables speculative
    #             decoding on Ampere. Model path must point at a pre-AWQ
    #             checkpoint (e.g. casperhansen/deepseek-r1-distill-qwen-32b-awq).
    #             WARNING: QLoRA training cannot train on an AWQ checkpoint;
    #             the HF training path requires a separate BF16 base. Gated
    #             opt-in; only flip together with a trainer-auditor sign-off
    #             on the split-checkpoint training plan.
    #   "gptq"  — pass quantization="gptq". Similar constraints to AWQ.
    #   "none"  — no quantization (full-precision); vLLM quantizes on the fly
    #             or uses the model's native dtype.
    # Default "auto" preserves current behavior — this commit is pure
    # plumbing, not a default flip.
    quantization_scheme: str = "auto"
    # Speculative decoding (vLLM). Small draft model predicts N tokens; the
    # target model verifies them in one forward pass. Typical speedup 1.5-2x
    # on solve-phase throughput when draft model ≪ target. Requires
    # quantization_scheme in ("awq", "gptq", "none") — bnb blocks it.
    # Both fields must be set for speculative to activate; either being
    # None/0 leaves speculative off.
    speculative_draft_model: Optional[str] = None
    num_speculative_tokens: int = 0

    # Task #19 secondary win: overlap the verify phase (CPU-bound code
    # execution, property checks) with the NEXT cycle's propose (GPU-bound
    # generate). Thread pool dispatches verify_batch while the orchestrator
    # starts the next propose's prompt build. Default False (same wall-
    # clock as today until flipped); user flips after A/B confirming
    # verify wall-clock > 20s/cycle (the break-even point against thread-
    # dispatch overhead).
    parallel_verify_enabled: bool = False

    # Task #18 speed pass step 2: chunked prefill. vLLM splits long prompt
    # prefills into chunks and interleaves them with decode steps, which
    # prevents the 120-prompt solve batch's long shared <think>...task-
    # template prefix from stalling the decode pipeline for 2-3 seconds
    # per prefill wave. Gemini consult: essential for this workload —
    # without it, prefill spikes dominate wall-time on the solve phase.
    # Default True; safe to disable if a vLLM version rejects the kwarg
    # (the loader's TypeError-retry drops it gracefully).
    enable_chunked_prefill: bool = True

    # Task #18 speed pass step 3: prefix-cache throughput logging. When
    # True, flip vLLM's disable_log_stats=False so the engine emits
    # per-interval prompt_throughput + num_cached_tokens lines. Off by
    # default because the log volume is non-trivial on long runs; flip
    # on for a single diagnostic cycle to VERIFY the long shared prefix
    # is actually caching (gemini consult: a single trailing-space drift
    # in the system prompt invalidates the cache for everything after it,
    # silently, with no other failure mode).
    log_throughput_stats: bool = False

    def __post_init__(self):
        if not (0.0 < self.gpu_memory_utilization <= 1.0):
            raise ValueError(f"gpu_memory_utilization must be in (0, 1], got {self.gpu_memory_utilization}")
        if self.max_num_seqs < 0:
            raise ValueError(f"max_num_seqs must be >= 0, got {self.max_num_seqs}")
        if self.max_model_len < 1:
            raise ValueError(f"max_model_len must be >= 1, got {self.max_model_len}")
        if not (0.0 < self.coresident_vllm_mem_frac <= 1.0):
            raise ValueError(
                f"coresident_vllm_mem_frac must be in (0, 1], got {self.coresident_vllm_mem_frac}"
            )
        if self.quantization_scheme not in ("auto", "bnb", "awq", "gptq", "none"):
            raise ValueError(
                f"quantization_scheme must be one of "
                f"'auto'|'bnb'|'awq'|'gptq'|'none', got {self.quantization_scheme!r}"
            )
        if self.num_speculative_tokens < 0:
            raise ValueError(
                f"num_speculative_tokens must be >= 0, got {self.num_speculative_tokens}"
            )
        if self.num_speculative_tokens > 0 and not self.speculative_draft_model:
            raise ValueError(
                "num_speculative_tokens>0 requires speculative_draft_model to be set"
            )
        if (
            self.num_speculative_tokens > 0
            and self.quantization_scheme == "bnb"
        ):
            raise ValueError(
                "speculative decoding is incompatible with quantization_scheme='bnb' "
                "(vLLM bitsandbytes load_format blocks it). Use 'awq' or 'gptq'."
            )


@dataclass
class SynthesisConfig:
    """Configuration for the task-synthesis pipeline (opt-in, default-off)."""
    enable_task_synthesis: bool = False
    # Task #10 speed pass: 20 → 12. At k=3 candidates_per_problem that's still
    # 36 candidates per propose+solve batch — enough diversity for the quorum
    # verifier while cutting proposer wall-clock ~40%.
    tasks_per_cycle: int = 12
    property_consensus_threshold: float = 0.7
    # Candidates per proposed problem in rsi_tick Step 3 (SOLVE). Candidate
    # 0 is always the reference the model emitted when proposing (guaranteed
    # quorum pass if the model wrote self-consistent tests). Candidates 1..k-1
    # are freshly sampled attempts for diversity. k=6 balances cost vs
    # chance-of-finding-a-passing-diverse-solution. Run-4 cycle 5 produced
    # 1 passing candidate across ~60 attempts with k=3; at k=6 with reference
    # as candidate 0, every well-formed proposal should yield ≥1 sample.
    # Task #10 speed pass: 6 → 3. Reference (candidate 0, guaranteed-pass when
    # the model wrote self-consistent tests) plus 2 fresh samples.
    # Task #22 speed-round-2: 3 → 2. Reference (candidate 0) + 1 fresh sample.
    # Every well-formed proposal still yields ≥1 passing candidate via the
    # reference; the fresh sample supplies diversity for DPO negatives. Cuts
    # solve-phase wall-clock by another ~33% on top of #10's halving.
    candidates_per_problem: int = 2
    # If True, rsi_tick uses the builtin-based code-proposal path
    # (PROBLEM + ENTRY + REFERENCE + TESTS → trusted builtin checks).
    # This is the only path that actually produces training samples with
    # an 8B base model — the legacy §3.1 co-gen format requires the model
    # to emit property source code, which it can't do reliably. Set False
    # to force the legacy path on stronger models.
    use_builtin_code_path: bool = True
    # Curriculum escalation: fraction of per-cycle code proposals whose
    # prompt is augmented with a frontier-skill hint from DifficultyTracker
    # ("Design a problem requiring skill X — the model currently fails
    # here"). Remaining prompts use the canonical / failure-seeded template
    # unchanged. 0.0 disables frontier biasing; 1.0 frontier-samples every slot.
    frontier_fraction: float = 0.5
    # Cross-cycle property + proposer few-shot banks (src/generator/property_library).
    # Gated at call time: takes effect only when the PropertyRegistry carries
    # ≥library_min_admitted distinct admitted property_ids, so cold-start
    # cycle-1 prompt (no library yet) is unchanged. Default True because the
    # whole point of RSI is that the pipeline itself improves across cycles —
    # this is the mechanism.
    use_property_library: bool = True
    library_min_admitted: int = 20
    library_k_properties: int = 5
    library_k_proposer: int = 3
    # Score floor for property inclusion (rejected_adversarial_count *
    # confirmer_pass_rate). 1.0 = at least one corruption caught with a
    # non-zero pass rate; excludes never-tested-adversarially properties.
    library_min_vov_score: float = 1.0

    # Out-of-distribution curriculum (Task #7, curriculum-ood).
    # OODProposer periodically seeds entirely new problem CATEGORIES (not just
    # harder instances of known ones) so the solver keeps discovering skills
    # outside DifficultyTracker's existing subdomain keys. Default-off for BC;
    # consolidation flip turns ood_enabled=True with period=12.
    # Consolidation flip: OOD curriculum default ON with period=12 so the
    # first OOD proposer round fires ~cycle 12 (after the initial SFT ramp).
    ood_enabled: bool = True
    ood_period: int = 12
    ood_domains_per_cycle: int = 3
    ood_seeds_per_domain: int = 8
    ood_state_path: str = "outputs/ood_domains.jsonl"
    ood_mainstream_threshold: float = 0.20

    # Fast-start (Task #11). Cycle 1 uses this smaller propose budget so
    # the first cycle lands fast; cycle>=2 falls back to tasks_per_cycle.
    synthesis_tasks_per_cycle_bootstrap: int = 15

    # Task #10 speed pass: cap proposer + solver generation length. The
    # proposer's output is structured (PROBLEM/ENTRY/REFERENCE/TESTS) and
    # fits well inside 600 tokens for the 12-task batch default. R1-style
    # <think> blocks are stripped post-generation, so capping hard here
    # saves wall-clock on the 32B-R1 path without harming the parser.
    # Solver cap is separately configurable because solver needs more
    # headroom for reasoning-before-code.
    proposer_max_new_tokens: int = 600
    solver_max_new_tokens: int = 1200

    # Reasoning-strategy library (Task #1B). Model-authored reasoning templates
    # stored as system-prompt prefixes, A/B-tested on a held-out slice, winners
    # re-injected as few-shot prefixes for future propose/solve cycles.
    strategy_library_enabled: bool = True
    strategy_ab_holdout_size: int = 4
    strategy_library_path: str = "outputs/reasoning_strategies.jsonl"
    strategy_library_k_few_shot: int = 2

    # Peer-LLM jury verification (Task #1A). Route candidates through multiple
    # independent CLI-accessible LLMs (codex, gemini) and require ≥2/3 consensus.
    peer_jury_enabled: bool = True
    peer_jury_cache_path: str = "outputs/peer_jury_cache.jsonl"
    peer_jury_timeout_s: int = 30
    peer_jury_min_agree: int = 2

    def __post_init__(self):
        if not (0.0 <= self.frontier_fraction <= 1.0):
            raise ValueError(
                f"frontier_fraction must be in [0, 1], got {self.frontier_fraction}"
            )
        if self.library_min_admitted < 0:
            raise ValueError(
                f"library_min_admitted must be >= 0, got {self.library_min_admitted}"
            )
        if self.library_k_properties < 0 or self.library_k_proposer < 0:
            raise ValueError("library_k_* must be >= 0")
        if self.library_min_vov_score < 0:
            raise ValueError(
                f"library_min_vov_score must be >= 0, got {self.library_min_vov_score}"
            )
        if self.tasks_per_cycle < 1:
            raise ValueError(f"tasks_per_cycle must be >= 1, got {self.tasks_per_cycle}")
        if not (0.0 < self.property_consensus_threshold <= 1.0):
            raise ValueError(
                f"property_consensus_threshold must be in (0, 1], "
                f"got {self.property_consensus_threshold}"
            )
        if self.ood_period < 1:
            raise ValueError(f"ood_period must be >= 1, got {self.ood_period}")
        if not (1 <= self.ood_domains_per_cycle <= 10):
            raise ValueError(
                f"ood_domains_per_cycle must be in [1, 10], got {self.ood_domains_per_cycle}"
            )
        if not (1 <= self.ood_seeds_per_domain <= 20):
            raise ValueError(
                f"ood_seeds_per_domain must be in [1, 20], got {self.ood_seeds_per_domain}"
            )
        if not (0.0 < self.ood_mainstream_threshold <= 1.0):
            raise ValueError(
                f"ood_mainstream_threshold must be in (0, 1], "
                f"got {self.ood_mainstream_threshold}"
            )
        if self.synthesis_tasks_per_cycle_bootstrap < 1:
            raise ValueError(
                f"synthesis_tasks_per_cycle_bootstrap must be >= 1, "
                f"got {self.synthesis_tasks_per_cycle_bootstrap}"
            )
        if self.strategy_ab_holdout_size < 0:
            raise ValueError(
                f"strategy_ab_holdout_size must be >= 0, got {self.strategy_ab_holdout_size}"
            )
        if self.strategy_library_k_few_shot < 0:
            raise ValueError(
                f"strategy_library_k_few_shot must be >= 0, "
                f"got {self.strategy_library_k_few_shot}"
            )
        if not (1 <= self.peer_jury_min_agree <= 3):
            raise ValueError(
                f"peer_jury_min_agree must be in [1, 3], got {self.peer_jury_min_agree}"
            )
        if self.peer_jury_timeout_s < 1:
            raise ValueError(
                f"peer_jury_timeout_s must be >= 1, got {self.peer_jury_timeout_s}"
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
