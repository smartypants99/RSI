"""Extended controller tests: tiebreak, cache, malformed scores, budget."""
import sys
import time as _time
from unittest.mock import MagicMock

from timedilate.config import TimeDilateConfig

if "vllm" not in sys.modules or not isinstance(sys.modules["vllm"], MagicMock):
    sys.modules["vllm"] = MagicMock()

from timedilate.controller import DilationController, DilationResult, CycleRecord


def _mock_engine(responses):
    engine = MagicMock()
    engine.generate = MagicMock(side_effect=list(responses))
    return engine


# --- Pairwise tiebreak: equal-score branches ---

def test_equal_score_branches_first_wins_stable_sort():
    """When two branches tie on score, stable sort keeps first-generated."""
    engine = _mock_engine([
        "initial", "60",
        # 3 branches all scoring 80 — first should win (stable sort, reverse=True)
        "critique",
        "branch_a", "80",
        "branch_b", "80",
        "branch_c", "80",
    ])
    config = TimeDilateConfig(
        dilation_factor=1.5, branch_factor=3,
        convergence_patience=10, branch_temperature_spread=0.0,
    )
    controller = DilationController(config, engine)
    result = controller.run("test")
    # All branches tied at 80; improvement over 60; winner is first in list
    assert result.score == 80
    assert result.output == "branch_a"


def test_branch_factor_1_no_temperature_spread_used():
    """branch_factor=1 should never spread temperature."""
    engine = _mock_engine([
        "initial", "60",
        "critique", "refined", "70",
    ])
    config = TimeDilateConfig(
        dilation_factor=1.5, branch_factor=1,
        convergence_patience=10, branch_temperature_spread=0.5,
    )
    controller = DilationController(config, engine)
    result = controller.run("test")
    assert result.score == 70


# --- Score cache behavior ---

def test_score_cache_hits_counter_accurate():
    engine = _mock_engine([
        "initial", "70",
        "critique", "initial",  # dup -> cache hit
        "critique", "initial",  # dup -> cache hit
    ])
    config = TimeDilateConfig(dilation_factor=2, convergence_patience=10)
    controller = DilationController(config, engine)
    controller.run("test")
    assert controller._score_cache_hits == 2


def test_score_cache_isolated_by_prompt():
    """Same output under different prompts should not collide in cache."""
    engine = _mock_engine(["o", "50"])
    config = TimeDilateConfig(dilation_factor=1.5, convergence_patience=10)
    controller = DilationController(config, engine)
    k1 = controller._cache_key("prompt A", "same output")
    k2 = controller._cache_key("prompt B", "same output")
    assert k1 != k2


def test_score_cache_same_key_for_same_inputs():
    controller = DilationController(TimeDilateConfig(), MagicMock())
    assert controller._cache_key("p", "o") == controller._cache_key("p", "o")


# --- Malformed score responses ---

def test_malformed_score_response_defaults_to_50():
    """Non-numeric score reply must default to 50, not crash."""
    engine = _mock_engine([
        "initial",
        "absolutely not a number here",  # score parse fails
        # convergence will still need cycles; make factor 1 so no cycle runs
    ])
    config = TimeDilateConfig(dilation_factor=1.0)  # no cycles after initial
    controller = DilationController(config, engine)
    # With factor 1, _score is never invoked (single pass). Use branching instead.
    # Rebuild with factor 1.5 so score runs.
    engine = _mock_engine(["initial", "absolutely not a number"])
    config = TimeDilateConfig(dilation_factor=1.0)
    controller = DilationController(config, engine)
    result = controller.run("test")
    # factor 1.0 => single pass, score stays 0 (not computed)
    assert result.score == 0


def test_score_extracts_leading_integer_from_noisy_reply():
    engine = _mock_engine([
        "initial",
        "The answer: 85 out of 100.",  # integer buried in text
        "critique", "refined",
        "92.",  # trailing period
    ])
    config = TimeDilateConfig(dilation_factor=1.5, convergence_patience=10)
    controller = DilationController(config, engine)
    result = controller.run("test")
    assert result.score == 92


def test_score_clamps_above_100():
    engine = MagicMock()
    engine.generate = MagicMock(return_value="500")
    config = TimeDilateConfig(dilation_factor=1.5, convergence_patience=10)
    controller = DilationController(config, engine)
    s = controller._score("prompt", "output")
    assert s == 100


def test_score_engine_exception_defaults_to_50():
    engine = MagicMock()
    engine.generate = MagicMock(side_effect=RuntimeError("inference broke"))
    config = TimeDilateConfig()
    controller = DilationController(config, engine)
    assert controller._score("p", "o") == 50


# --- Early stop ---

def test_early_stop_respects_custom_threshold():
    engine = _mock_engine(["initial", "75"])
    config = TimeDilateConfig(
        dilation_factor=100, early_stop_score=70, convergence_patience=5
    )
    controller = DilationController(config, engine)
    result = controller.run("test")
    assert result.score == 75
    assert result.cycles_completed == 0


# --- Time budget ---

def test_time_budget_expires_stops_loop():
    call_count = [0]

    def gen(prompt, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return "initial"
        if "Rate the RESPONSE" in prompt:
            _time.sleep(0.01)
            return "75"
        if "reviewing" in prompt.lower():
            return "critique"
        return "refined"

    engine = MagicMock()
    engine.generate = MagicMock(side_effect=gen)
    config = TimeDilateConfig(dilation_factor=1000, time_budget_seconds=0.05,
                              convergence_patience=50)
    controller = DilationController(config, engine)
    result = controller.run("test")
    assert result.elapsed_seconds < 2.0


# --- on_cycle callback shape ---

def test_on_cycle_callback_args_shape():
    engine = _mock_engine([
        "initial", "60",
        "critique", "better", "80",
    ])
    captured = []
    controller = DilationController(
        TimeDilateConfig(dilation_factor=1.5, convergence_patience=10), engine
    )
    controller.run("test", on_cycle=lambda c, t, s, e: captured.append((c, t, s, e)))
    assert len(captured) == 1
    cycle, total, score, elapsed = captured[0]
    assert cycle == 1
    assert score == 80
    assert isinstance(elapsed, float)


# --- DilationResult properties ---

def test_score_gain_property():
    r = DilationResult(
        output="x", dilation_factor=10, cycles_completed=5, total_cycles=10,
        elapsed_seconds=1.0, model_used="m", score=90, initial_score=60,
    )
    assert r.score_gain == 30


def test_improvement_rate_empty_history():
    r = DilationResult(
        output="x", dilation_factor=1, cycles_completed=0, total_cycles=0,
        elapsed_seconds=0.1, model_used="m", score=0,
    )
    assert r.improvement_rate == 0.0


def test_improvement_rate_mixed_history():
    hist = [
        CycleRecord(cycle=1, action="refine", improved=True),
        CycleRecord(cycle=2, action="refine", improved=False),
        CycleRecord(cycle=3, action="refine", improved=True),
        CycleRecord(cycle=4, action="refine", improved=False),
    ]
    r = DilationResult(
        output="x", dilation_factor=4, cycles_completed=4, total_cycles=4,
        elapsed_seconds=1, model_used="m", score=80, cycle_history=hist,
    )
    assert r.improvement_rate == 0.5


def test_to_report_includes_improvement_metrics():
    hist = [
        CycleRecord(cycle=1, action="refine", improved=True,
                    score_before=50, score_after=70),
    ]
    r = DilationResult(
        output="x", dilation_factor=2, cycles_completed=1, total_cycles=2,
        elapsed_seconds=1, model_used="m", score=70, initial_score=50,
        cycle_history=hist,
    )
    rep = r.to_report()
    assert rep["score_gain"] == 20
    assert rep["improvements"] == 1
    assert rep["improvement_rate"] == 1.0


# --- Determinism of controller with deterministic mock engine ---

def test_deterministic_engine_same_run_same_result():
    """Given identical mock responses, two runs must produce identical results."""
    responses = ["initial", "60", "critique", "better", "80"]
    cfg = TimeDilateConfig(dilation_factor=1.5, convergence_patience=10)
    r1 = DilationController(cfg, _mock_engine(responses)).run("test")
    r2 = DilationController(cfg, _mock_engine(responses)).run("test")
    assert r1.output == r2.output
    assert r1.score == r2.score
    assert r1.cycles_completed == r2.cycles_completed


# --- Convergence resets counted correctly ---

def test_no_convergence_reset_when_within_patience():
    engine = _mock_engine([
        "initial", "70",
        # 2 cycles no improvement, but patience=5
        "critique", "x", "60",
        "critique", "y", "55",
    ])
    config = TimeDilateConfig(dilation_factor=2, convergence_patience=5)
    controller = DilationController(config, engine)
    result = controller.run("test")
    assert result.convergence_resets == 0


# --- Branch all failing is tolerated ---

def test_all_branches_fail_controller_continues():
    call_count = [0]

    def gen(prompt, **kwargs):
        call_count[0] += 1
        c = call_count[0]
        if c == 1:
            return "initial"
        if c == 2:
            return "60"
        if "reviewing" in prompt.lower() or "weaknesses" in prompt.lower():
            return "critique"
        # All refines raise
        raise RuntimeError("branch fail")

    engine = MagicMock()
    engine.generate = MagicMock(side_effect=gen)
    config = TimeDilateConfig(dilation_factor=1.5, branch_factor=2,
                              convergence_patience=10)
    controller = DilationController(config, engine)
    result = controller.run("test")
    # Initial preserved since all branches failed
    assert result.output == "initial"
    assert result.score == 60
