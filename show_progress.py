#!/usr/bin/env python3
"""Quick progress viewer — run locally after pulling outputs from the GPU.

Usage:
    python3 show_progress.py                  # default outputs/ path
    python3 show_progress.py outputs_from_gpu # or pass a snapshot dir

Prints a compact trajectory of key signals across all completed cycles
so it's obvious at a glance whether RSI is improving or thrashing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _fmt(x, w=6):
    if x is None:
        return " " * w
    if isinstance(x, float):
        return f"{x:{w}.3f}"
    return str(x).rjust(w)


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs")
    logs = sorted((root / "logs").glob("cycle_*.json"),
                  key=lambda p: int(p.stem.split("_")[1]))
    if not logs:
        print(f"No cycle logs found under {root / 'logs'}")
        return 1

    print(f"Reading {len(logs)} cycles from {root}")
    print()
    # Header
    cols = [
        ("cycle", 5), ("pre", 6), ("post", 6), ("Δ", 7),
        ("eval", 6), ("Δeval", 7), ("spread", 7),
        ("steps", 6), ("loss", 7), ("verified", 9), ("errors", 6),
    ]
    print(" ".join(h.ljust(w) for h, w in cols))
    print(" ".join("-" * w for _, w in cols))

    prev_eval = None
    eval_history = []
    for p in logs:
        d = json.loads(p.read_text())
        t = d.get("training") or {}
        eval_score = d.get("eval_score")
        # Eval spread if multiple reps were recorded
        eval_all = d.get("eval_scores_all") or []
        spread = (max(eval_all) - min(eval_all)) if len(eval_all) > 1 else 0.0
        delta_eval = (eval_score - prev_eval) if (
            eval_score is not None and prev_eval is not None
        ) else None

        row = [
            _fmt(d.get("cycle"), 5),
            _fmt(d.get("pre_score"), 6),
            _fmt(d.get("post_score"), 6),
            _fmt(d.get("improvement"), 7),
            _fmt(eval_score, 6),
            _fmt(delta_eval, 7),
            _fmt(spread, 7),
            _fmt(t.get("steps"), 6),
            _fmt(t.get("final_loss"), 7),
            f"{d.get('samples_verified', 0)}/{d.get('samples_generated', 0)}".rjust(9),
            _fmt(len(d.get("errors") or []), 6),
        ]
        print(" ".join(row))

        if eval_score is not None:
            eval_history.append(eval_score)
            prev_eval = eval_score

    # Trajectory summary
    print()
    if len(eval_history) >= 2:
        total = eval_history[-1] - eval_history[0]
        print(f"Held-out trajectory: {eval_history[0]:.3f} → {eval_history[-1]:.3f} "
              f"({total:+.3f} over {len(eval_history)} cycles)")
        improving = sum(
            1 for a, b in zip(eval_history[:-1], eval_history[1:]) if b > a
        )
        print(f"Cycles where held-out went up: {improving}/{len(eval_history) - 1}")
        # Variance of held-out — if reps had spread=0, noise comes from
        # model change only. With n=16 questions, binomial σ at p=0.25 is
        # ~0.108 so any delta < 0.10 is within noise.
        best = max(eval_history)
        best_cycle = eval_history.index(best) + 1
        print(f"Best held-out: {best:.3f} at cycle {best_cycle}")
    else:
        print("Not enough cycles to compute trajectory yet.")

    # Meta decisions summary
    mpath = root / "meta_decisions.jsonl"
    if mpath.exists():
        lines = [line for line in mpath.read_text().splitlines() if line.strip()]
        print(f"\nMeta decisions logged: {len(lines)}")
        for line in lines[-5:]:
            try:
                d = json.loads(line)
                print(f"  cycle {d.get('cycle', '?')}: {d.get('decision', '?')} "
                      f"(reason: {d.get('reason', '')[:80]})")
            except json.JSONDecodeError:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
