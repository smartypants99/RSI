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
    difficulty_bands: list[str] = field(default_factory=lambda: ["easy", "medium", "hard", "expert"])
    difficulty_mix: dict = field(default_factory=lambda: {
        "easy": 0.30, "medium": 0.35, "hard": 0.25, "expert": 0.10,
    })
    semantic_grading: bool = True
    significance_alpha: float = 0.05
    min_evidence_for_weakness: int = 8
    calibrated_confidence: bool = True
    activation_probes_per_domain: int = 2


@dataclass
class GeneratorConfig:
    min_reasoning_steps: int = 3
    max_reasoning_steps: int = 50
    require_explicit_assumptions: bool = True
    require_step_justification: bool = True
    samples_per_weakness: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    max_retries: int = 3  # retries for insufficient reasoning


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


@dataclass
class VLLMConfig:
    model_path: str = ""
    dtype: str = "bfloat16"
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.85
    quantization_config: Optional[dict] = None


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
