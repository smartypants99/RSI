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
    questions_per_domain: int = 300
    min_questions_per_domain: int = 150
    max_questions_per_domain: int = 600
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
    samples_per_weakness: int = 100
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
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    learning_rate: float = 2e-5
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 16
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    target_modules: list[str] = field(default_factory=lambda: [
        # LLaMA/Mistral/Qwen-style
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        # GPT-2/GPT-J-style (Conv1D layers)
        "c_attn", "c_proj", "c_fc",
    ])
    # Custom: weakness-adaptive rank scaling
    min_rank: int = 8
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
        if not (0.0 <= self.warmup_ratio <= 1.0):
            raise ValueError(f"warmup_ratio must be in [0, 1], got {self.warmup_ratio}")
        if self.min_rank > self.max_rank:
            raise ValueError(f"min_rank ({self.min_rank}) > max_rank ({self.max_rank})")
        if self.lora_rank < self.min_rank:
            raise ValueError(f"lora_rank ({self.lora_rank}) < min_rank ({self.min_rank})")
        if self.training_mode not in ("sft", "dpo", "mixed"):
            raise ValueError(
                f"training_mode must be one of 'sft', 'dpo', 'mixed' — got {self.training_mode!r}"
            )
        if self.dpo_beta <= 0:
            raise ValueError(f"dpo_beta must be > 0, got {self.dpo_beta}")


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

    def __post_init__(self):
        if self.max_cycles < 1:
            raise ValueError(f"max_cycles must be >= 1, got {self.max_cycles}")
        if self.checkpoint_every < 1:
            raise ValueError(f"checkpoint_every must be >= 1, got {self.checkpoint_every}")
        if self.plateau_patience < 1:
            raise ValueError(f"plateau_patience must be >= 1, got {self.plateau_patience}")


@dataclass
class VLLMConfig:
    model_path: str = ""
    dtype: str = "bfloat16"
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.85
    quantization_config: Optional[dict] = None

    def __post_init__(self):
        if not (0.0 < self.gpu_memory_utilization <= 1.0):
            raise ValueError(f"gpu_memory_utilization must be in (0, 1], got {self.gpu_memory_utilization}")
        if self.max_model_len < 1:
            raise ValueError(f"max_model_len must be >= 1, got {self.max_model_len}")


@dataclass
class SystemConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    use_vllm: bool = False
    vllm: Optional[VLLMConfig] = None
