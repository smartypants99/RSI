# Team RSI-Foom Collaboration Protocol

10 agents pushing RSI from ~20% → 50%+ toward foom (first-in-world goal).

## Hard rules — all agents

1. **pytest green BEFORE every commit.** Run `python3 -m pytest tests/ -q`.
2. **NO force-push, no destructive git.**
3. **Do NOT touch the GPU run.** GPU at ssh -p 41024 root@82.79.85.125 has a LIVE vLLM run — do not kill, restart, or write files there. Your commits apply after team finishes when integration-lead coordinates a restart.
4. **External LLM consultants allowed.** `/opt/homebrew/bin/codex` and `/opt/homebrew/bin/gemini -p "..."` are available. Use them for design review on risky changes (self-editing, weight-growth, RL).
5. **Coordinate via SendMessage, not silent edits.** If you touch a file another agent owns, message them first.
6. **Peer review: EVERY feature commit needs a second pair of eyes.** Before merging, SendMessage the diff summary to your designated reviewer (see pairings below). They respond `lgtm` or `blocked: <reason>`.
7. **Integration lead owns shared files**: `src/utils/config.py`, `src/orchestrator/loop.py`. Coordinate changes to these via integration-lead.

## File ownership

| Agent | Primary files |
|---|---|
| integration-lead | src/utils/config.py, src/orchestrator/loop.py, test harness |
| weight-growth | src/trainer/growth.py (new), src/trainer/custom_lora.py |
| self-edit | src/orchestrator/self_edit.py (new), safety sandbox |
| verifier-lean | src/verifier/lean_backend.py (new), sandbox integration |
| verifier-z3 | src/verifier/z3_backend.py (new) |
| verifier-sim | src/verifier/simulator_backend.py (new) |
| cycle-accel | src/utils/fast_student.py (new), generation path |
| curriculum-ood | src/generator/ood_proposer.py (new), domain exploration |
| rl-transition | src/trainer/grpo.py (new), reward shaping |
| safety-gate | src/safety/ (new dir), reviews self-edit + weight-growth changes |

## Peer review pairings

- weight-growth ↔ safety-gate (mandatory review — this is dangerous)
- self-edit ↔ safety-gate (mandatory review — also dangerous)
- verifier-lean ↔ verifier-z3 (peer, same domain)
- verifier-sim ↔ cycle-accel (peer)
- curriculum-ood ↔ rl-transition (peer, reward/exploration interplay)
- integration-lead reviews the cross-cutting work of all

## Escalation

- If blocked >20 min, SendMessage integration-lead.
- If a risky change is vetoed by safety-gate twice, integration-lead arbitrates with codex/gemini consulted.

## Shutdown criteria

When your task is done + pytest green + peer review approved, TaskUpdate status=completed and SendMessage team-lead. Then go idle.

## Shared context

Current state: DeepSeek-R1-Distill-Qwen-32B-bnb-4bit via vLLM, Team RSI commits a7c5e58+ landed. Read `update-log.txt` and recent commits for the full story. Target: components 1-4 of the 6-component foom bar (see `memory/user_rsi_target.md` via the main process — or just read this doc).
