"""Smoke tests: import every module and instantiate every public class with defaults.

No GPU, no real model — just prove the import graph is healthy and ctors don't crash.
"""
from __future__ import annotations

import importlib
import pkgutil
import pytest


def _iter_src_modules():
    import src
    for m in pkgutil.walk_packages(src.__path__, prefix="src."):
        yield m.name


def test_import_all_modules():
    """Every .py under src/ must import cleanly."""
    failures = []
    for modname in _iter_src_modules():
        try:
            importlib.import_module(modname)
        except Exception as e:  # pragma: no cover — diagnostic
            failures.append(f"{modname}: {type(e).__name__}: {e}")
    assert not failures, "Module import failures:\n" + "\n".join(failures)


def test_system_config_defaults():
    from src.utils.config import SystemConfig
    cfg = SystemConfig()
    assert cfg.model is not None
    assert cfg.diagnostics is not None
    assert cfg.generator is not None
    assert cfg.verifier is not None
    assert cfg.trainer is not None
    assert cfg.orchestrator is not None


def test_all_config_dataclasses_instantiate():
    from src.utils import config as cfg
    for name in (
        "ModelConfig", "DiagnosticsConfig", "GeneratorConfig", "VerifierConfig",
        "TrainerConfig", "EscalationSchedule", "OrchestratorConfig", "VLLMConfig",
        "SystemConfig",
    ):
        cls = getattr(cfg, name)
        instance = cls()
        assert instance is not None


def test_verifier_instantiates():
    from src.utils.config import VerifierConfig
    from src.verifier.verifier import Verifier
    v = Verifier(VerifierConfig())
    assert v is not None


def test_dataclasses_from_generator():
    from src.generator.data_generator import TrainingSample, PreferencePair, ReasoningStep
    s = TrainingSample(prompt="p", response="r", domain="math")
    assert s.content_hash  # auto-populated
    step = ReasoningStep(step_number=1, content="c", justification="j")
    assert step.claim == "c"
    pair = PreferencePair(prompt="p", chosen_response="a", rejected_response="b")
    assert pair.content_hash


def test_weakness_and_diagnostic_result():
    from src.diagnostics.engine import WeaknessReport, DiagnosticResult
    w = WeaknessReport(domain="math", subdomain="algebra", severity=0.5)
    assert w.domain == "math"
    d = DiagnosticResult(cycle=1, timestamp=0.0)
    assert d.overall_score == 0.0  # empty domain_scores


@pytest.mark.parametrize("optional_mod", [
    "src.diagnostics.curriculum",
    "src.trainer.prm",
    "src.orchestrator.meta",
    "src.orchestrator.decision_log",
])
def test_optional_new_modules_import_cleanly_if_present(optional_mod):
    """New teammate modules: if present, they must import cleanly. If missing, skip."""
    try:
        importlib.import_module(optional_mod)
    except ModuleNotFoundError:
        pytest.skip(f"{optional_mod} not yet present")
    except Exception as e:
        pytest.fail(f"{optional_mod} failed to import: {type(e).__name__}: {e}")


def test_curriculum_state_roundtrip():
    from src.diagnostics.curriculum import CurriculumState, DEFAULT_CLASSES
    state = CurriculumState.fresh(DEFAULT_CLASSES)
    d = state.to_dict()
    rebuilt = CurriculumState.from_dict(d)
    assert rebuilt is not None


def test_curriculum_pick_frontier_and_record():
    import random
    from src.diagnostics.curriculum import CurriculumState
    s = CurriculumState.fresh()
    qs = s.pick_frontier(20, random.Random(0))
    assert len(qs) == 20
    assert all("_class_id" in q and "_difficulty_int" in q for q in qs)
    s.record_results([(q["_class_id"], q["_difficulty_int"], True) for q in qs])
    s2 = CurriculumState.from_dict(s.to_dict())
    assert s2.active_classes == s.active_classes


def test_meta_controller_instantiates(tmp_path):
    from src.orchestrator.meta import MetaController
    mc = MetaController(log_path=tmp_path / "meta.jsonl", initial_lr=2e-5)
    assert mc is not None


def test_decision_tracker_instantiates(tmp_path):
    from src.orchestrator.decision_log import CausalTracker, DecisionRecord
    t = CausalTracker(log_path=tmp_path / "decisions.jsonl")
    assert t is not None
    rec = DecisionRecord(cycle=1, config_snapshot={"lr": 1e-4}, pre_score=0.3, post_score=0.4)
    assert rec.eval_delta is None  # no eval scores set


def test_training_metrics_has_calibration_fields():
    from src.trainer.custom_lora import TrainingMetrics
    m = TrainingMetrics(
        cycle=1, avg_loss=0.5, final_loss=0.4, steps=10,
        samples_used=5, samples_rejected=0, learning_rate=2e-5,
    )
    assert hasattr(m, "calibration_ece")
    assert hasattr(m, "calibration_brier")
    assert hasattr(m, "calibration_samples")


def test_prm_parse_completion_steps():
    from src.trainer.prm import _parse_completion_steps
    steps = _parse_completion_steps("Step 1: do X\nStep 2: do Y\nConclusion: done")
    assert len(steps) >= 1


def test_trainer_config_has_prm_fields():
    from src.utils.config import TrainerConfig
    t = TrainerConfig()
    assert hasattr(t, "use_prm")
    assert hasattr(t, "prm_lr")
    assert hasattr(t, "enable_calibration_loss")
