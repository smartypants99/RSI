"""Tests for the cross-cycle property + proposer few-shot library (task #2).

Coverage:
  - top_k_admitted_properties ranks by rejected_adversarial_count × confirmer_pass_rate
  - min_vov_score gate filters out weak / untested properties
  - top_k_proposer_exemplars ranks by training-pool accept count
  - build_library_prefix gates on library_min_admitted (cold-start safety)
  - rendered blocks carry the `### EXAMPLES FROM PRIOR CYCLES` delimiter so
    DeepSeek-R1-style parsers don't get confused by undelimited few-shot text
  - OrchestratorConfig.run_id gives stable sid across restarts (persistence audit)
"""
from __future__ import annotations

from pathlib import Path

from src.generator.property_library import (
    PropertyExemplar,
    ProposerExemplar,
    top_k_admitted_properties,
    top_k_proposer_exemplars,
    count_admitted_properties,
    render_property_exemplars_block,
    render_proposer_exemplars_block,
    build_library_prefix,
)
from src.orchestrator.registries import RSIRegistries


# ─── helpers ─────────────────────────────────────────────────────────────

def _open_regs(tmp_path: Path, sid: str = "test") -> RSIRegistries:
    return RSIRegistries.open(tmp_path / "outputs", sid=sid)


def _write_property(regs: RSIRegistries, pid: str, **kw) -> None:
    rec = {
        "property_id": pid,
        "problem_id": kw.get("problem_id", f"prob_{pid}"),
        "author": kw.get("author", "model:test"),
        "independence_class": kw.get("independence_class", "exec.behavioral"),
        "kind": kw.get("kind", "UNIT_TEST"),
        "name": kw.get("name", f"prop_{pid}"),
        "description": kw.get("description", f"desc for {pid}"),
    }
    regs.property_registry.append_property(rec, bundle_passed_vov=True)


def _write_verification(
    regs: RSIRegistries,
    *,
    property_id: str,
    passed: bool,
    adversarial: bool = False,
    quorum_accepted: bool = True,
    verdict: str = "",
) -> None:
    verdict = verdict or ("PASS" if passed else "FAIL")
    rec = {
        "record_id": f"ver_{property_id}_{adversarial}_{passed}",
        "problem_id": "prob_x",
        "candidate_id": "cand_x",
        "property_ids": [property_id],
        "per_property_verdicts": [{
            "property_id": property_id, "passed": passed,
            "verdict": verdict,
            "reason": "", "class": "exec.behavioral",
        }],
        "quorum_accepted": quorum_accepted,
        "quorum_reason": "",
        "adversarial": adversarial,
    }
    regs.verification_log.append_verification(rec)


# ─── property bank ───────────────────────────────────────────────────────


def test_top_k_admitted_properties_ranks_by_kill_times_pass_rate(tmp_path):
    regs = _open_regs(tmp_path)
    _write_property(regs, "p_killer")      # 3 kills × 1.0 = 3.0
    _write_property(regs, "p_balanced")    # 2 kills × 0.5 = 1.0
    _write_property(regs, "p_toothless")   # 0 kills × 1.0 = 0.0 (dropped)

    for _ in range(3):
        _write_verification(regs, property_id="p_killer", passed=False, adversarial=True)
    _write_verification(regs, property_id="p_killer", passed=True, adversarial=False)

    for _ in range(2):
        _write_verification(regs, property_id="p_balanced", passed=False, adversarial=True)
    _write_verification(regs, property_id="p_balanced", passed=True, adversarial=False)
    _write_verification(regs, property_id="p_balanced", passed=False, adversarial=False)

    _write_verification(regs, property_id="p_toothless", passed=True, adversarial=False)

    out = top_k_admitted_properties(
        regs.property_registry, regs.verification_log, k=5, min_vov_score=1.0,
    )
    ids = [e.property_id for e in out]
    assert ids == ["p_killer", "p_balanced"], f"got {ids!r}"
    assert out[0].rejected_adversarial_count == 3
    assert out[0].confirmer_pass_count == 1 and out[0].confirmer_total == 1
    assert out[0].score == 3.0
    assert out[1].score == 1.0  # 2 * 0.5


def test_min_vov_score_filters_weak_properties(tmp_path):
    regs = _open_regs(tmp_path)
    _write_property(regs, "p_weak")
    _write_verification(regs, property_id="p_weak", passed=False, adversarial=True)
    _write_verification(regs, property_id="p_weak", passed=False, adversarial=False)

    strong = top_k_admitted_properties(
        regs.property_registry, regs.verification_log, k=5, min_vov_score=1.0,
    )
    assert strong == [], "p_weak has 0 confirmer pass rate so score=0"

    # Lower the floor and it should appear.
    any_ = top_k_admitted_properties(
        regs.property_registry, regs.verification_log, k=5, min_vov_score=0.0,
    )
    assert [e.property_id for e in any_] == ["p_weak"]


def test_errors_do_not_count_as_adversarial_kills(tmp_path):
    regs = _open_regs(tmp_path)
    _write_property(regs, "p_err")
    _write_verification(
        regs, property_id="p_err", passed=False, adversarial=True, verdict="ERROR",
    )
    _write_verification(regs, property_id="p_err", passed=True, adversarial=False)
    out = top_k_admitted_properties(
        regs.property_registry, regs.verification_log, k=5, min_vov_score=0.0,
    )
    assert out and out[0].rejected_adversarial_count == 0


def test_count_admitted_properties_deduplicates(tmp_path):
    regs = _open_regs(tmp_path)
    _write_property(regs, "pa")
    _write_property(regs, "pa")  # re-appended
    _write_property(regs, "pb")
    assert count_admitted_properties(regs.property_registry) == 2


def test_render_property_block_has_delimiter(tmp_path):
    ex = [PropertyExemplar(
        property_id="p1", name="checks_return", description="must return int",
        independence_class="exec.behavioral", kind="UNIT_TEST",
        rejected_adversarial_count=3, confirmer_pass_count=4, confirmer_total=5,
        score=3.0 * 0.8,
    )]
    block = render_property_exemplars_block(ex)
    assert "### EXAMPLES FROM PRIOR CYCLES" in block
    assert "### END EXAMPLES" in block
    assert "checks_return" in block
    assert "exec.behavioral" in block


def test_render_property_block_empty_is_empty():
    assert render_property_exemplars_block([]) == ""


# ─── proposer bank ───────────────────────────────────────────────────────


def test_top_k_proposer_exemplars_ranks_by_accept_count(tmp_path):
    regs = _open_regs(tmp_path)
    regs.problem_registry.append({
        "problem_id": "q_hot",
        "problem_text": "invert a balanced BST",
        "domain": "code",
        "problem_ctx": {"entry_point": "solve", "reference": "def solve(x): ...", "tests": ["assert solve(1)==1"]},
    })
    regs.problem_registry.append({
        "problem_id": "q_cold",
        "problem_text": "reverse a list",
        "domain": "code",
    })
    regs.problem_registry.append({
        "problem_id": "q_unused",
        "problem_text": "unused",
        "domain": "code",
    })
    # Two accepts for q_hot, one for q_cold, zero for q_unused.
    for _ in range(2):
        regs.training_pool.append_sample({
            "pool_record_id": "pr1", "problem_id": "q_hot",
            "candidate_id": "c", "verification_record_id": "v",
            "domain": "code", "prompt": "", "response": "",
        })
    regs.training_pool.append_sample({
        "pool_record_id": "pr2", "problem_id": "q_cold",
        "candidate_id": "c", "verification_record_id": "v",
        "domain": "code", "prompt": "", "response": "",
    })
    out = top_k_proposer_exemplars(
        regs.problem_registry, regs.training_pool, k=5, min_accept_count=1,
    )
    ids = [e.problem_id for e in out]
    assert ids == ["q_hot", "q_cold"]
    assert out[0].quorum_accept_count == 2
    assert out[0].entry_point == "solve"
    assert out[0].tests == ("assert solve(1)==1",)


def test_render_proposer_block_has_delimiter_and_shape(tmp_path):
    ex = [ProposerExemplar(
        problem_id="q1", problem_text="reverse a list",
        domain="code", quorum_accept_count=4,
        entry_point="solve", reference="def solve(x):\n    return x[::-1]",
        tests=("assert solve([1,2,3])==[3,2,1]",),
    )]
    block = render_proposer_exemplars_block(ex)
    assert "### EXAMPLES FROM PRIOR CYCLES" in block
    assert "### END EXAMPLES" in block
    # Proposer block must mirror CODE_PROPOSAL_TEMPLATE label shape.
    assert "PROBLEM:" in block and "ENTRY:" in block
    assert "REFERENCE:" in block and "TESTS:" in block


# ─── combined prefix gate ────────────────────────────────────────────────


def test_build_library_prefix_empty_below_min_admitted(tmp_path):
    regs = _open_regs(tmp_path)
    _write_property(regs, "only_one")
    _write_verification(regs, property_id="only_one", passed=False, adversarial=True)
    _write_verification(regs, property_id="only_one", passed=True, adversarial=False)
    # Only 1 admitted property, but gate is 20 → empty.
    prefix = build_library_prefix(
        property_registry=regs.property_registry,
        verification_log=regs.verification_log,
        problem_registry=regs.problem_registry,
        training_pool=regs.training_pool,
        min_admitted_for_gate=20,
    )
    assert prefix == ""


def test_build_library_prefix_non_empty_above_gate(tmp_path):
    regs = _open_regs(tmp_path)
    for i in range(25):
        _write_property(regs, f"p{i}")
    # Give one property some kills + passes so it scores.
    _write_verification(regs, property_id="p0", passed=False, adversarial=True)
    _write_verification(regs, property_id="p0", passed=True, adversarial=False)
    prefix = build_library_prefix(
        property_registry=regs.property_registry,
        verification_log=regs.verification_log,
        problem_registry=regs.problem_registry,
        training_pool=regs.training_pool,
        min_admitted_for_gate=20,
        k_properties=3, k_proposer=2,
    )
    assert prefix != ""
    assert "### EXAMPLES FROM PRIOR CYCLES" in prefix


# ─── persistence audit: run_id gives stable sid across "restarts" ────────


def test_run_id_stabilizes_registry_sid(tmp_path):
    """Without a stable run_id, RSIRegistries.open falls back to uuid() →
    each process restart opens a fresh file and the few-shot banks NEVER
    accumulate. OrchestratorConfig.run_id should give one deterministic sid.
    """
    from src.utils.config import OrchestratorConfig
    cfg = OrchestratorConfig()
    # run_id must exist and be a non-empty string so the registry file path
    # is stable across restarts.
    assert hasattr(cfg, "run_id")
    assert isinstance(cfg.run_id, str) and cfg.run_id

    # Opening twice with the same sid must produce the same on-disk file —
    # writes in the first "process" must be visible to the second.
    regs1 = RSIRegistries.open(tmp_path / "outputs", sid=cfg.run_id)
    _write_property(regs1, "persistent_pid")
    regs2 = RSIRegistries.open(tmp_path / "outputs", sid=cfg.run_id)
    ids = [r.get("property_id") for r in regs2.property_registry.iter_records()]
    assert "persistent_pid" in ids, "registry writes didn't persist across opens"


# ─── integration with task_synthesizer ───────────────────────────────────


def test_task_synthesizer_compute_library_prefix_respects_config(tmp_path):
    from src.generator.task_synthesizer import TaskSynthesizer
    from src.utils.config import SynthesisConfig

    regs = _open_regs(tmp_path)
    for i in range(25):
        _write_property(regs, f"p{i}")
    _write_verification(regs, property_id="p0", passed=False, adversarial=True)
    _write_verification(regs, property_id="p0", passed=True, adversarial=False)

    # use_property_library=False → prefix empty even with a full bank.
    ts_off = TaskSynthesizer(
        SynthesisConfig(use_property_library=False), model_loader=None,
        generate_fn=lambda _p: "",
    )
    ts_off.set_registries(regs)
    assert ts_off._compute_library_prefix() == ""

    # use_property_library=True → prefix non-empty once gate passes.
    ts_on = TaskSynthesizer(
        SynthesisConfig(
            use_property_library=True, library_min_admitted=20,
            library_k_properties=2, library_k_proposer=1, library_min_vov_score=0.0,
        ),
        model_loader=None, generate_fn=lambda _p: "",
    )
    ts_on.set_registries(regs)
    prefix = ts_on._compute_library_prefix()
    assert "### EXAMPLES FROM PRIOR CYCLES" in prefix


def test_task_synthesizer_prefix_absent_when_registries_unset(tmp_path):
    from src.generator.task_synthesizer import TaskSynthesizer
    from src.utils.config import SynthesisConfig

    ts = TaskSynthesizer(SynthesisConfig(), model_loader=None, generate_fn=lambda _p: "")
    # No set_registries call → prefix empty.
    assert ts._compute_library_prefix() == ""
