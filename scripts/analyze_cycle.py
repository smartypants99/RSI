#!/usr/bin/env python3
"""
analyze_cycle.py — post-mortem analyzer for instrumented RSI cycles.

Consumes 5 jsonl logs produced by instrument-agent:
  - training_steps.jsonl     (per optimizer step)
  - heldout_per_prompt.jsonl (per held-out prompt, pre/post score)
  - verify_decisions.jsonl   (per verifier decision / warnings)
  - propose_attempts.jsonl   (per proposer attempt, success/failure reason)
  - cycle_summary.jsonl      (one-line cycle-level summary)

Usage:
  python scripts/analyze_cycle.py <cycle_number|latest> [--logs-dir outputs/] [--rsync host:path]

Outputs:
  - human-readable report to stdout
  - outputs/cycle_N_analysis.md (markdown persistence)

Design: defensive field access. Schema may evolve; we surface "FIELD MISSING"
rather than crash. If a whole log is missing, we note it explicitly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOGS_DIR = REPO_ROOT / "outputs"

LOG_FILES = [
    "training_steps.jsonl",
    "heldout_per_prompt.jsonl",
    "verify_decisions.jsonl",
    "propose_attempts.jsonl",
    "cycle_summary.jsonl",
]

B_MOVE_THRESHOLD = 1e-5


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARN: {path.name}:{line_no} bad JSON ({e})", file=sys.stderr)
    return rows


def _find_cycle_dir(logs_dir: Path, cycle: str) -> tuple[int | None, Path]:
    """Resolve 'latest' or a cycle number to a concrete directory.

    Search layout (accept whichever instrument-agent picked):
      logs_dir/cycle_N/*.jsonl
      logs_dir/cycles/N/*.jsonl
      logs_dir/*.jsonl                 (single-cycle flat layout, cycle='latest')
    """
    candidates: list[tuple[int, Path]] = []
    for pat in ("cycle_*", "cycles/*"):
        for p in logs_dir.glob(pat):
            if not p.is_dir():
                continue
            m = re.search(r"(\d+)$", p.name)
            if m:
                candidates.append((int(m.group(1)), p))
    if cycle == "latest":
        if candidates:
            n, d = max(candidates, key=lambda x: x[0])
            return n, d
        # Fallback: flat layout in logs_dir itself
        if any((logs_dir / lf).exists() for lf in LOG_FILES):
            return None, logs_dir
        return None, logs_dir  # no data; still point at logs_dir
    # Specific cycle number
    want = int(cycle)
    for n, d in candidates:
        if n == want:
            return n, d
    # Not found; return placeholder
    return want, logs_dir / f"cycle_{want}"


def _maybe_rsync(remote: str, local: Path) -> None:
    local.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-av", "--include=*.jsonl", "--include=*/",
           "--exclude=*", remote.rstrip("/") + "/", str(local) + "/"]
    print(f"[rsync] {' '.join(cmd)}")
    subprocess.run(cmd, check=False)


# ---------------------------------------------------------------------------
# Defensive field access
# ---------------------------------------------------------------------------

def get(d: dict, *keys: str, default: Any = None) -> Any:
    """Try several keys, return first non-None match."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


# ---------------------------------------------------------------------------
# Section analyzers — each returns (stdout_lines, md_lines, signals_dict)
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    lines: list[str] = field(default_factory=list)
    md: list[str] = field(default_factory=list)
    signals: dict = field(default_factory=dict)


def analyze_training_health(rows: list[dict]) -> AnalysisResult:
    r = AnalysisResult()
    r.lines.append("## Training health")
    r.md.append("## Training health\n")
    if not rows:
        r.lines.append("  [MISSING] training_steps.jsonl empty or absent")
        r.md.append("- **MISSING** `training_steps.jsonl` empty or absent\n")
        r.signals["training_health"] = "missing"
        return r

    losses = [get(x, "loss", "train_loss") for x in rows]
    losses = [l for l in losses if isinstance(l, (int, float))]
    initial = losses[0] if losses else None
    final = losses[-1] if losses else None

    grad_b = [get(x, "grad_norm_B", "grad_norm_b") for x in rows]
    grad_b = [g for g in grad_b if isinstance(g, (int, float))]
    max_grad_b = max(grad_b) if grad_b else None

    b_moves = [get(x, "post_step_B_max_abs", "post_step_b_max_abs", default=0.0) for x in rows]
    b_moves_num = [m for m in b_moves if isinstance(m, (int, float))]
    frac_moved = (
        sum(1 for m in b_moves_num if m > B_MOVE_THRESHOLD) / len(b_moves_num)
        if b_moves_num else None
    )

    lrs_b = [get(x, "lr_B", "lr_b") for x in rows]
    lrs_b = [l for l in lrs_b if isinstance(l, (int, float))]
    lr_b_applied = statistics.mean(lrs_b) if lrs_b else None

    def fmt(x, p="{:.4g}"):
        return p.format(x) if isinstance(x, (int, float)) else "N/A"

    line = (
        f"  steps={len(rows)}  loss: {fmt(initial)} → {fmt(final)}  "
        f"max|grad_B|={fmt(max_grad_b)}  B-moved frac={fmt(frac_moved,'{:.2%}') if frac_moved is not None else 'N/A'}  "
        f"LR_B mean={fmt(lr_b_applied)}"
    )
    r.lines.append(line)
    r.md.append(f"- Steps: **{len(rows)}**\n")
    r.md.append(f"- Loss: `{fmt(initial)}` → `{fmt(final)}`\n")
    r.md.append(f"- max(grad_norm_B): `{fmt(max_grad_b)}`\n")
    r.md.append(f"- Fraction of steps where B moved (>{B_MOVE_THRESHOLD}): "
                f"`{fmt(frac_moved,'{:.2%}') if frac_moved is not None else 'N/A'}`\n")
    r.md.append(f"- Mean applied LR_B: `{fmt(lr_b_applied)}`\n")

    r.signals.update(
        loss_initial=initial, loss_final=final,
        max_grad_B=max_grad_b, frac_B_moved=frac_moved, lr_B_mean=lr_b_applied,
        n_steps=len(rows),
    )
    return r


def analyze_training_damage_probe(
    training_rows: list[dict], heldout_rows: list[dict]
) -> AnalysisResult:
    r = AnalysisResult()
    r.lines.append("## Training damage probe (per-domain pre→post score delta)")
    r.md.append("## Training damage probe (per-domain pre→post score delta)\n")

    if not heldout_rows:
        r.lines.append("  [MISSING] heldout_per_prompt.jsonl")
        r.md.append("- **MISSING** `heldout_per_prompt.jsonl`\n")
        r.signals["damage_probe"] = "missing"
        return r

    # Build per-domain delta
    by_domain_pre: dict[str, list[float]] = defaultdict(list)
    by_domain_post: dict[str, list[float]] = defaultdict(list)
    by_domain_delta: dict[str, list[float]] = defaultdict(list)

    for row in heldout_rows:
        dom = get(row, "domain", "task", "category", default="unknown")
        pre = get(row, "score_pre", "pre_score", "baseline_score")
        post = get(row, "score_post", "post_score", "score")
        if isinstance(pre, (int, float)):
            by_domain_pre[dom].append(pre)
        if isinstance(post, (int, float)):
            by_domain_post[dom].append(post)
        if isinstance(pre, (int, float)) and isinstance(post, (int, float)):
            by_domain_delta[dom].append(post - pre)

    # Domains present in training samples
    trained_domains: Counter = Counter()
    for t in training_rows:
        d = get(t, "domain", "task", "category")
        if d:
            trained_domains[d] += 1

    r.md.append("| domain | n_heldout | pre_mean | post_mean | Δ | trained_in_cycle |\n")
    r.md.append("|---|---:|---:|---:|---:|---:|\n")
    worst_dom = None
    worst_delta = math.inf
    for dom in sorted(set(list(by_domain_pre) + list(by_domain_post))):
        pre_m = statistics.mean(by_domain_pre[dom]) if by_domain_pre[dom] else None
        post_m = statistics.mean(by_domain_post[dom]) if by_domain_post[dom] else None
        deltas = by_domain_delta[dom]
        dmean = statistics.mean(deltas) if deltas else None
        trained_n = trained_domains.get(dom, 0)
        pre_s = f"{pre_m:.4f}" if pre_m is not None else "N/A"
        post_s = f"{post_m:.4f}" if post_m is not None else "N/A"
        d_s = f"{dmean:+.4f}" if dmean is not None else "N/A"
        r.lines.append(
            f"  {dom:<20s} n={len(deltas):<4d} pre={pre_s} post={post_s} Δ={d_s} trained={trained_n}"
        )
        r.md.append(f"| {dom} | {len(deltas)} | {pre_s} | {post_s} | {d_s} | {trained_n} |\n")
        if dmean is not None and dmean < worst_delta:
            worst_delta = dmean
            worst_dom = dom

    r.signals["worst_domain"] = worst_dom
    r.signals["worst_delta"] = None if worst_delta is math.inf else worst_delta
    return r


def analyze_verifier_noise(
    verify_rows: list[dict], heldout_rows: list[dict], training_rows: list[dict]
) -> AnalysisResult:
    r = AnalysisResult()
    r.lines.append("## Verifier noise")
    r.md.append("## Verifier noise\n")

    if not verify_rows:
        r.lines.append("  [MISSING] verify_decisions.jsonl")
        r.md.append("- **MISSING** `verify_decisions.jsonl`\n")
        r.signals["verifier_noise"] = "missing"
        return r

    accepted = [
        v for v in verify_rows
        if get(v, "accepted", "accept", "decision") in (True, "accept", "accepted")
    ]
    def has_any_fail(v: dict) -> bool:
        w = get(v, "verdict_warnings", "warnings", default=[]) or []
        if isinstance(w, str):
            return "any_fail" in w
        return any("any_fail" in str(x) for x in w)

    n_acc = len(accepted)
    n_acc_warned = sum(1 for v in accepted if has_any_fail(v))
    frac = (n_acc_warned / n_acc) if n_acc else None

    r.lines.append(
        f"  accepted={n_acc}  with any_fail warning={n_acc_warned}  "
        f"({'N/A' if frac is None else f'{frac:.2%}'})"
    )
    r.md.append(f"- Accepted samples: **{n_acc}**\n")
    r.md.append(f"- ...with `verdict_warnings` containing `any_fail`: **{n_acc_warned}** "
                f"({'N/A' if frac is None else f'{frac:.2%}'})\n")

    # Task #5: under `quorum_2of3`, any single failing property trips `any_fail`
    # without signalling real verifier disagreement. The true "noise" signal is
    # accepts that saw ≥2 non-PASS verdicts — those are candidates the policy
    # admitted despite majority-failure-ish evidence. Compute the multi-fail
    # rate separately so the TL;DR can distinguish structural-quorum noise
    # from actual verifier flakiness.
    def _fail_count(v: dict) -> int:
        fc = get(v, "fail_count", default=None)
        if isinstance(fc, int):
            return fc
        pb = get(v, "per_backend", default=[]) or []
        return sum(1 for b in pb if str(b.get("verdict", "")).upper() != "PASS")
    n_acc_multifail = sum(1 for v in accepted if _fail_count(v) >= 2)
    frac_multifail = (n_acc_multifail / n_acc) if n_acc else None
    r.lines.append(
        f"  accepted multi-fail (fail_count≥2)={n_acc_multifail}  "
        f"({'N/A' if frac_multifail is None else f'{frac_multifail:.2%}'})"
    )
    r.md.append(f"- ...with ≥2 non-PASS verdicts (real disagreement): "
                f"**{n_acc_multifail}** "
                f"({'N/A' if frac_multifail is None else f'{frac_multifail:.2%}'})\n")

    # Mean score_delta for prompts whose training samples had warnings vs not.
    # Keyed by training sample_id → warning flag; match to held-out score_delta via domain/prompt tag.
    warned_ids = {
        get(v, "sample_id", "training_sample_id", "id")
        for v in accepted if has_any_fail(v)
    }
    warned_ids.discard(None)
    warned_domains = set()
    clean_domains = set()
    for t in training_rows:
        tid = get(t, "sample_id", "id")
        dom = get(t, "domain", "task")
        if tid in warned_ids:
            warned_domains.add(dom)
        else:
            clean_domains.add(dom)

    def domain_mean_delta(domset: set) -> float | None:
        vals: list[float] = []
        for h in heldout_rows:
            if get(h, "domain", "task") not in domset:
                continue
            pre = get(h, "score_pre", "pre_score")
            post = get(h, "score_post", "post_score", "score")
            if isinstance(pre, (int, float)) and isinstance(post, (int, float)):
                vals.append(post - pre)
        return statistics.mean(vals) if vals else None

    md_w = domain_mean_delta(warned_domains)
    md_c = domain_mean_delta(clean_domains - warned_domains)
    r.lines.append(
        f"  mean Δ on domains w/ warned training samples: "
        f"{'N/A' if md_w is None else f'{md_w:+.4f}'}  "
        f"vs clean-only: {'N/A' if md_c is None else f'{md_c:+.4f}'}"
    )
    r.md.append(f"- Mean heldout Δ on domains touched by *warned* training samples: "
                f"`{'N/A' if md_w is None else f'{md_w:+.4f}'}`\n")
    r.md.append(f"- Mean heldout Δ on *clean-only* domains: "
                f"`{'N/A' if md_c is None else f'{md_c:+.4f}'}`\n")
    r.signals.update(
        frac_accepted_warned=frac,
        frac_accepted_multifail=frac_multifail,
        delta_warned=md_w,
        delta_clean=md_c,
    )
    return r


def _paired_rho(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def analyze_rho_decomposition(heldout_rows: list[dict]) -> AnalysisResult:
    r = AnalysisResult()
    r.lines.append("## ρ decomposition (per-domain paired-sample correlation pre vs post)")
    r.md.append("## ρ decomposition\n")

    if not heldout_rows:
        r.lines.append("  [MISSING] heldout_per_prompt.jsonl")
        r.md.append("- **MISSING** `heldout_per_prompt.jsonl`\n")
        r.signals["rho"] = "missing"
        return r

    by_dom: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for h in heldout_rows:
        dom = get(h, "domain", "task", default="unknown")
        pre = get(h, "score_pre", "pre_score")
        post = get(h, "score_post", "post_score", "score")
        if isinstance(pre, (int, float)) and isinstance(post, (int, float)):
            by_dom[dom].append((pre, post))

    r.md.append("| domain | n | ρ(pre,post) |\n|---|---:|---:|\n")
    rhos: dict[str, float] = {}
    for dom in sorted(by_dom):
        pairs = by_dom[dom]
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        rho = _paired_rho(xs, ys)
        rhos[dom] = rho if rho is not None else float("nan")
        rho_s = "N/A" if rho is None else f"{rho:+.3f}"
        r.lines.append(f"  {dom:<20s} n={len(pairs):<4d} ρ={rho_s}")
        r.md.append(f"| {dom} | {len(pairs)} | {rho_s} |\n")

    # Bottleneck: domain with lowest ρ (where variance reduction fails)
    finite = {d: v for d, v in rhos.items() if not math.isnan(v)}
    if finite:
        worst = min(finite, key=lambda k: finite[k])
        r.lines.append(f"  ρ bottleneck domain: {worst} (ρ={finite[worst]:+.3f})")
        r.md.append(f"\n**ρ bottleneck domain**: `{worst}` (ρ={finite[worst]:+.3f})\n")
        r.signals["rho_bottleneck"] = worst
        r.signals["rho_bottleneck_val"] = finite[worst]
    return r


def analyze_proposer_bottleneck(propose_rows: list[dict]) -> AnalysisResult:
    r = AnalysisResult()
    r.lines.append("## Proposer bottleneck")
    r.md.append("## Proposer bottleneck\n")

    if not propose_rows:
        r.lines.append("  [MISSING] propose_attempts.jsonl")
        r.md.append("- **MISSING** `propose_attempts.jsonl`\n")
        r.signals["proposer"] = "missing"
        return r

    fail_counts: Counter = Counter()
    fail_time: dict[str, float] = defaultdict(float)
    total_time = 0.0
    n_success = 0
    for row in propose_rows:
        ok = get(row, "success", "accepted", "passed", default=None)
        reason = get(row, "failure_reason", "fail_reason", "reason", default="ok" if ok else "unknown")
        dt = get(row, "time_s", "duration_s", "elapsed_s", default=0.0)
        if isinstance(dt, (int, float)):
            total_time += dt
        if ok is True or reason == "ok":
            n_success += 1
            continue
        fail_counts[reason] += 1
        if isinstance(dt, (int, float)):
            fail_time[reason] += dt

    r.md.append(f"- Total attempts: **{len(propose_rows)}**, succeeded: **{n_success}**, "
                f"total time: `{total_time:.1f}s`\n")
    r.md.append("| failure_reason | count | total_time_s |\n|---|---:|---:|\n")
    for reason, cnt in fail_counts.most_common():
        r.lines.append(f"  {reason:<30s} count={cnt:<5d} time={fail_time[reason]:.1f}s")
        r.md.append(f"| {reason} | {cnt} | {fail_time[reason]:.1f} |\n")

    if fail_counts:
        top = fail_counts.most_common(1)[0][0]
        r.signals["top_failure_reason"] = top
        r.signals["fail_counts"] = dict(fail_counts)
    r.signals["propose_success"] = n_success
    r.signals["propose_total"] = len(propose_rows)
    return r


def compose_tldr(summary_rows: list[dict], sig: dict) -> AnalysisResult:
    r = AnalysisResult()
    r.lines.append("## Bottom line — 3-bullet TL;DR")
    r.md.append("## Bottom line — 3-bullet TL;DR\n")

    bullets: list[str] = []

    # (1) Training actually moved the model?
    fm = sig.get("frac_B_moved")
    lr_b = sig.get("lr_B_mean")
    li, lf = sig.get("loss_initial"), sig.get("loss_final")
    if fm is not None and fm < 0.5:
        bullets.append(
            f"Training barely moved B: only {fm:.1%} of steps had "
            f"|ΔB|>{B_MOVE_THRESHOLD}. LR_B mean={lr_b}. Suspect LR too cold "
            f"or gradient gating is over-aggressive."
        )
    elif isinstance(li, (int, float)) and isinstance(lf, (int, float)) and lf > li:
        bullets.append(
            f"Loss went UP ({li:.4f} → {lf:.4f}). Training regressed. "
            f"Suspect LR too hot (LR_B={lr_b}) or bad sample mix."
        )
    elif isinstance(li, (int, float)) and isinstance(lf, (int, float)):
        bullets.append(f"Training loss {li:.4f} → {lf:.4f}; LR_B={lr_b}; B-moved frac={fm}.")
    else:
        bullets.append("Training-health signals missing — cannot attribute.")

    # (2) Damage probe
    wd = sig.get("worst_domain")
    wdelta = sig.get("worst_delta")
    if wd is not None and isinstance(wdelta, (int, float)):
        verdict = "REGRESSED" if wdelta < -1e-4 else ("neutral" if abs(wdelta) <= 1e-4 else "improved")
        bullets.append(
            f"Domain mix: worst domain on held-out is `{wd}` with Δ={wdelta:+.4f} ({verdict}). "
            f"If this domain was heavily trained, the objective is misaligned with eval."
        )
    else:
        bullets.append("Damage-probe signals missing.")

    # (3) ρ bottleneck + verifier
    rb = sig.get("rho_bottleneck")
    rbv = sig.get("rho_bottleneck_val")
    fw = sig.get("frac_accepted_warned")
    parts = []
    if rb is not None and isinstance(rbv, (int, float)):
        parts.append(f"ρ bottleneck domain `{rb}` ρ={rbv:+.3f} → that domain will never hit MDE at sane N.")
    # Task #5 recalibration: under `quorum_2of3` with 3 properties, `any_fail`
    # fires structurally whenever 1 of 3 properties fails (policy admits 2/3).
    # Observed empirically: 14/15 accepts = pass=2,fail=1 deterministically —
    # a property-generator artifact (one builtin systematically mismatches on
    # clean candidates), not verifier flake. The real "noisy" signal is when
    # ≥2 of 3 properties fail on an accept (quorum actually disagreed with
    # itself). Fire only when frac_multifail > 0.1 OR frac_any_fail >> the
    # structural floor (~0.66 for 3 props under quorum_2of3).
    fwm = sig.get("frac_accepted_multifail")
    any_fail_floor = 0.66  # loose upper bound for 3-property quorum_2of3
    fired = False
    if isinstance(fwm, (int, float)) and fwm > 0.1:
        parts.append(
            f"Verifier noisy: {fwm:.1%} of accepted samples have ≥2 property "
            f"FAILs (real quorum disagreement)."
        )
        fired = True
    if not fired and isinstance(fw, (int, float)) and fw > any_fail_floor:
        parts.append(
            f"Verifier noisy: {fw:.1%} of accepted samples carry `any_fail` "
            f"(above structural quorum floor ~{any_fail_floor:.0%})."
        )
    bullets.append(" ".join(parts) if parts else "ρ/verifier within acceptable ranges (or data missing).")

    for i, b in enumerate(bullets, 1):
        r.lines.append(f"  {i}. {b}")
        r.md.append(f"{i}. {b}\n")
    return r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze(cycle: str, logs_dir: Path, rsync: str | None) -> int:
    if rsync:
        _maybe_rsync(rsync, logs_dir)

    n, cycle_dir = _find_cycle_dir(logs_dir, cycle)

    # Try per-cycle subdir first, then flat fallback
    def _load(name: str) -> list[dict]:
        p = cycle_dir / name
        if not p.exists():
            p = logs_dir / name
        return _read_jsonl(p)

    training = _load("training_steps.jsonl")
    heldout = _load("heldout_per_prompt.jsonl")
    verify = _load("verify_decisions.jsonl")
    propose = _load("propose_attempts.jsonl")
    summary = _load("cycle_summary.jsonl")

    missing = [
        name for name, rows in (
            ("training_steps", training), ("heldout_per_prompt", heldout),
            ("verify_decisions", verify), ("propose_attempts", propose),
            ("cycle_summary", summary),
        ) if not rows
    ]

    out: list[str] = []
    md: list[str] = []
    label = f"cycle={n if n is not None else cycle}"
    header = f"# Cycle analysis — {label}"
    out.append(header)
    out.append(f"cycle_dir={cycle_dir}")
    md.append(f"# Cycle analysis — {label}\n\n")
    md.append(f"- cycle_dir: `{cycle_dir}`\n")
    if missing:
        out.append(f"MISSING LOGS: {', '.join(missing)}")
        md.append(f"- **MISSING LOGS**: {', '.join(missing)}\n")
    out.append("")
    md.append("\n")

    results: list[AnalysisResult] = []
    results.append(analyze_training_health(training))
    results.append(analyze_training_damage_probe(training, heldout))
    results.append(analyze_verifier_noise(verify, heldout, training))
    results.append(analyze_rho_decomposition(heldout))
    results.append(analyze_proposer_bottleneck(propose))

    combined_sig: dict = {}
    for res in results:
        combined_sig.update(res.signals)
        out.extend(res.lines)
        out.append("")
        md.extend(res.md)
        md.append("\n")

    tldr = compose_tldr(summary, combined_sig)
    out.extend(tldr.lines)
    md.extend(tldr.md)

    text = "\n".join(out)
    print(text)

    # Persist
    out_md = logs_dir / (f"cycle_{n}_analysis.md" if n is not None else f"cycle_{cycle}_analysis.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("".join(md))
    print(f"\n[wrote] {out_md}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cycle", help="cycle number (int) or 'latest'")
    ap.add_argument("--logs-dir", default=str(DEFAULT_LOGS_DIR),
                    help="local outputs dir containing jsonl logs")
    ap.add_argument("--rsync", default=None,
                    help="optional remote source e.g. host:/path/to/outputs to rsync first")
    args = ap.parse_args(argv)
    return analyze(args.cycle, Path(args.logs_dir), args.rsync)


if __name__ == "__main__":
    sys.exit(main())
