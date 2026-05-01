"""External ground-truth benchmarks — anti verifier-capture canary.

The RSI pipeline generates its own problems, its own solutions, and its own
adversarial bank. That's a closed loop: the model could drift in a
direction the verifier happens to accept while genuinely regressing on
real tasks. Detecting that failure mode requires a signal the proposer
and the verifier cannot influence.

This module provides:

1. Loaders for widely-used external benchmarks (HumanEval, MBPP, GSM8K,
   MATH) with deterministic partitioning and local caching.
2. ``run_anchor_eval(model, config)`` — a secondary held-out eval that
   runs a fixed sampled subset every cycle. Its score is logged as
   ``cycle_metrics.anchor_score``.
3. ``detect_verifier_capture(prev, curr)`` — the canary: if the internal
   held-out score improved but the anchor score dropped, the verifier is
   probably capturing. Callers append an alarm to update-log.txt.
4. ``seed_adversarial_bank_from_external(...)`` — loads known-wrong
   solutions (HumanEval+ style mutants) into the VoV bank so the bank is
   not 100% self-generated.

All network/dataset access is optional. If the `datasets` package or
network is unavailable, loaders fall back to a small bundled offline
fixture so tests remain deterministic and green.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)


SUPPORTED_BENCHMARKS = ("humaneval", "mbpp", "gsm8k", "math")


@dataclass
class BenchmarkItem:
    """One external question with a trusted ground-truth answer."""
    benchmark: str
    item_id: str
    prompt: str
    answer: str              # canonical expected answer (string)
    domain: str              # "code" | "math"
    meta: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "benchmark": self.benchmark,
            "item_id": self.item_id,
            "prompt": self.prompt,
            "answer": self.answer,
            "domain": self.domain,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_json(cls, obj: dict) -> "BenchmarkItem":
        return cls(
            benchmark=str(obj["benchmark"]),
            item_id=str(obj["item_id"]),
            prompt=str(obj["prompt"]),
            answer=str(obj["answer"]),
            domain=str(obj.get("domain", "code")),
            meta=dict(obj.get("meta") or {}),
        )


# ---------------------------------------------------------------------------
# Offline fallback fixtures. Tiny but real-shaped so the pipeline stays
# exercised even when `datasets` or network is unavailable. Canonical
# answers chosen so a grader can check them exactly.
# ---------------------------------------------------------------------------

# NOTE: Every offline fixture row carries meta["source"]="offline_fixture".
# run_anchor_eval checks this marker and emits an alarm when an anchor
# benchmark is populated entirely from the offline fallback — without it,
# a gsm8k=1.0 result from the 12-item toy fixture is indistinguishable
# in telemetry from gsm8k=1.0 on the 1319-item HF test set. Closes task #9.
_OFFLINE_FIXTURES: dict[str, list[dict]] = {
    "humaneval": [
        {"item_id": f"HE/{i}",
         "prompt": f"def add_{i}(a, b):\n    '''Return a + b.'''\n",
         "answer": f"def add_{i}(a, b):\n    return a + b\n",
         "domain": "code",
         "meta": {
             "source": "offline_fixture",
             "entry_point": f"add_{i}",
             "test": (
                 "def check(candidate):\n"
                 "    assert candidate(1, 2) == 3\n"
                 "    assert candidate(-1, 1) == 0\n"
                 "    assert candidate(0, 0) == 0\n"
             ),
         }}
        for i in range(12)
    ],
    "mbpp": [
        {"item_id": f"MBPP/{i}",
         "prompt": f"Write a function mul_{i}(a, b) that returns a * b.",
         "answer": f"def mul_{i}(a, b):\n    return a * b\n",
         "domain": "code",
         "meta": {
             "source": "offline_fixture",
             "entry_point": f"mul_{i}",
             "test_list": [
                 f"assert mul_{i}(2, 3) == 6",
                 f"assert mul_{i}(0, 5) == 0",
                 f"assert mul_{i}(-2, 3) == -6",
             ],
         }}
        for i in range(12)
    ],
    "gsm8k": [
        {"item_id": f"GSM/{i}",
         "prompt": f"If I have {i} apples and buy {i} more, how many apples do I have?",
         "answer": str(i + i),
         "domain": "math",
         "meta": {"source": "offline_fixture"}}
        for i in range(12)
    ],
    "math": [
        {"item_id": f"MATH/{i}",
         "prompt": f"Compute {i} * {i}.",
         "answer": str(i * i),
         "domain": "math",
         "meta": {"source": "offline_fixture"}}
        for i in range(12)
    ],
}


def _cache_path(cache_dir: Path, benchmark: str) -> Path:
    return Path(cache_dir) / f"{benchmark}.jsonl"


def _write_cache(path: Path, items: list[BenchmarkItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it.to_json()) + "\n")
    tmp.replace(path)


def _read_cache(path: Path) -> list[BenchmarkItem]:
    items: list[BenchmarkItem] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            items.append(BenchmarkItem.from_json(json.loads(line)))
    return items


def _try_load_from_datasets(benchmark: str) -> Optional[list[BenchmarkItem]]:
    """Attempt to load via HF `datasets`. Returns None on any failure."""
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        logger.debug("datasets package unavailable (%s); using offline fixture for %s", e, benchmark)
        return None
    try:
        if benchmark == "humaneval":
            ds = load_dataset("openai_humaneval", split="test")
            return [
                BenchmarkItem(
                    benchmark="humaneval",
                    item_id=str(r.get("task_id", i)),
                    prompt=str(r.get("prompt", "")),
                    answer=str(r.get("canonical_solution", "")),
                    domain="code",
                    meta={
                        "test": r.get("test", ""),
                        # Without entry_point the grader short-circuits to
                        # False — observed 0/50 pre-fix even with correct
                        # completions. The HF row carries it; propagate it.
                        "entry_point": r.get("entry_point", ""),
                    },
                )
                for i, r in enumerate(ds)
            ]
        if benchmark == "mbpp":
            ds = load_dataset("mbpp", split="test")
            out_items = []
            for i, r in enumerate(ds):
                text = str(r.get("text", ""))
                test_list = list(r.get("test_list", []) or [])
                test_setup = str(r.get("test_setup_code", "") or "")
                # Standard MBPP eval prompt format includes the first test so
                # the model can infer the canonical function name. Without it
                # the model improvises a name → assert fails. This is the
                # documented HF/lm-eval-harness format.
                prompt_parts = [text, "Your code should pass these tests:"]
                if test_list:
                    prompt_parts.append("\n".join(test_list))
                full_prompt = "\n".join(prompt_parts).strip()
                out_items.append(BenchmarkItem(
                    benchmark="mbpp",
                    item_id=str(r.get("task_id", i)),
                    prompt=full_prompt,
                    answer=str(r.get("code", "")),
                    domain="code",
                    meta={
                        "test_list": test_list,
                        "test_setup_code": test_setup,
                    },
                ))
            return out_items
        if benchmark == "gsm8k":
            ds = load_dataset("gsm8k", "main", split="test")
            return [
                BenchmarkItem(
                    benchmark="gsm8k",
                    item_id=f"gsm8k/{i}",
                    prompt=str(r.get("question", "")),
                    answer=_extract_gsm8k_final_answer(str(r.get("answer", ""))),
                    domain="math",
                )
                for i, r in enumerate(ds)
            ]
        if benchmark == "math":
            # `hendrycks/competition_math` was removed from the Hub. Try
            # known mirrors in priority order; first one that loads wins.
            _math_repos = (
                ("HuggingFaceH4/MATH-500", "test", "problem", "solution",
                 "answer"),
                ("lighteval/MATH", "test", "problem", "solution", None),
                ("EleutherAI/hendrycks_math", None, "problem", "solution",
                 None),
            )
            ds = None
            chosen_keys: tuple[str, str, str | None] = ("problem", "solution", None)
            for repo_id, split, prob_k, sol_k, ans_k in _math_repos:
                try:
                    if split:
                        ds = load_dataset(repo_id, split=split)
                    else:
                        ds = load_dataset(repo_id)
                        ds = ds.get("test") or ds.get("train") or next(iter(ds.values()))
                    chosen_keys = (prob_k, sol_k, ans_k)
                    logger.info("MATH benchmark loaded from %s", repo_id)
                    break
                except Exception as e:
                    logger.debug("MATH load %s failed: %s", repo_id, e)
                    ds = None
            if ds is None:
                return None
            prob_key, sol_key, ans_key = chosen_keys
            out = []
            for i, r in enumerate(ds):
                prob = str(r.get(prob_key, "") or "")
                sol = str(r.get(sol_key, "") or "")
                # Some mirrors (MATH-500) ship a separate clean `answer` field;
                # prefer it when present so the grader gets a parseable target
                # instead of a multi-paragraph LaTeX solution.
                if ans_key and r.get(ans_key):
                    ans = str(r.get(ans_key))
                else:
                    ans = sol
                if not prob:
                    continue
                out.append(BenchmarkItem(
                    benchmark="math",
                    item_id=f"math/{i}",
                    prompt=prob,
                    answer=ans,
                    domain="math",
                    meta={
                        "level": r.get("level", ""),
                        "type": r.get("type", r.get("subject", "")),
                    },
                ))
            return out
    except Exception as e:
        logger.warning("datasets load failed for %s (%s); falling back to offline fixture", benchmark, e)
        return None
    return None


def _extract_gsm8k_final_answer(raw_answer: str) -> str:
    # GSM8K answers end with "#### <number>"
    if "####" in raw_answer:
        return raw_answer.split("####")[-1].strip()
    return raw_answer.strip()


def _load_offline_fixture(benchmark: str) -> list[BenchmarkItem]:
    rows = _OFFLINE_FIXTURES[benchmark]
    return [
        BenchmarkItem(
            benchmark=benchmark,
            item_id=r["item_id"],
            prompt=r["prompt"],
            answer=r["answer"],
            domain=r["domain"],
            meta=dict(r.get("meta") or {}),
        )
        for r in rows
    ]


def load_benchmark(
    benchmark: str,
    cache_dir: Path | str = Path("outputs/external_benchmarks"),
    *,
    force_offline: bool = False,
) -> list[BenchmarkItem]:
    """Load an external benchmark. Reads from cache if present.

    Falls back to an offline fixture if `datasets` / network is unavailable.
    """
    if benchmark not in SUPPORTED_BENCHMARKS:
        raise ValueError(f"unknown benchmark {benchmark!r}; supported: {SUPPORTED_BENCHMARKS}")
    cache_dir = Path(cache_dir)
    cpath = _cache_path(cache_dir, benchmark)
    if cpath.exists():
        try:
            return _read_cache(cpath)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("cache read failed for %s (%s); re-fetching", benchmark, e)

    items: Optional[list[BenchmarkItem]] = None
    if not force_offline:
        items = _try_load_from_datasets(benchmark)
    if items is None:
        items = _load_offline_fixture(benchmark)
    try:
        _write_cache(cpath, items)
    except OSError as e:
        logger.warning("cache write failed for %s (%s)", benchmark, e)
    return items


# ---------------------------------------------------------------------------
# Deterministic partitioning.
# ---------------------------------------------------------------------------

ANCHOR_SEED = 0xA11CE  # stable across cycles — anchor set must be fixed


def _stable_hash(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16)


def sample_anchor_set(
    items_by_benchmark: dict[str, list[BenchmarkItem]],
    per_benchmark: int,
    seed: int = ANCHOR_SEED,
    include_surprise: bool = False,
) -> list[BenchmarkItem]:
    """Deterministic subset: same (items, per_benchmark, seed) → same list.

    Sorts candidates by stable hash(item_id) seeded by `seed`, takes the
    first `per_benchmark` per benchmark. This is reproducible across
    processes and Python runs (unlike `random.sample`, which depends on
    hash randomization).

    When ``include_surprise=True`` the surprise-eval benchmarks (codex
    surprise, ARC-AGI, LiveCodeBench post-cutoff) are merged into the
    sampling pool via ``surprise_eval.load_surprise_anchor_items``. Any
    failure in the surprise loader is logged and skipped — the base
    external benchmarks still sample normally.
    """
    merged: dict[str, list[BenchmarkItem]] = dict(items_by_benchmark)
    if include_surprise:
        try:
            from .surprise_eval import load_surprise_anchor_items
            surprise = load_surprise_anchor_items()
            for k, v in surprise.items():
                merged.setdefault(k, []).extend(v)
        except Exception as e:
            logger.warning("sample_anchor_set: surprise merge failed (%s)", e)

    out: list[BenchmarkItem] = []
    for bench in sorted(merged.keys()):
        pool = list(merged[bench])
        # Sort by (seeded hash, item_id) for a stable, seed-varying order.
        pool.sort(key=lambda it: (_stable_hash(f"{seed}:{it.item_id}"), it.item_id))
        out.extend(pool[:per_benchmark])
    return out


# ---------------------------------------------------------------------------
# Grading. Intentionally simple string-match; a runtime grader can replace
# this by passing its own grade_fn into run_anchor_eval. The crucial property
# is that grading uses the CANONICAL answer, not a self-generated verifier.
# ---------------------------------------------------------------------------

def _default_grade(item: BenchmarkItem, prediction: str) -> bool:
    pred = prediction or ""
    if not pred.strip():
        return False
    if item.domain == "math":
        ans = (item.answer or "").strip()
        return _normalize_number(pred) == _normalize_number(ans)
    if item.benchmark == "humaneval":
        return _grade_humaneval(item, pred)
    if item.benchmark == "mbpp":
        return _grade_mbpp(item, pred)
    ans = (item.answer or "").strip()
    pred_s = pred.strip()
    return ans in pred_s or pred_s in ans


def _extract_code(text: str) -> str:
    """Extract python code from a possibly-markdown-wrapped completion.

    Handles ```python fences, ``` fences, and stripping <think>...</think>.
    """
    import re as _re
    s = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL | _re.IGNORECASE)
    m = _re.search(r"```(?:python|py)?\s*\n(.*?)```", s, _re.DOTALL)
    if m:
        return m.group(1)
    return s


def _grade_humaneval(item: BenchmarkItem, prediction: str) -> bool:
    """HumanEval grader: build full program = prompt + completion, then run
    the benchmark's check() function against the entry_point."""
    test = item.meta.get("test") or ""
    entry = item.meta.get("entry_point") or ""
    if not test or not entry:
        return False
    pred_code = _extract_code(prediction)
    # Always prepend the prompt's import/setup lines (everything before the
    # first `def entry(`). Chat-tuned instruct models output a fresh function
    # signature like `def foo(x: str) -> List[str]:` but usually drop the
    # `from typing import List` that was in the prompt. Concatenating the
    # prompt's preamble lines preserves those imports.
    import re as _re
    prompt_preamble = item.prompt
    m = _re.search(rf"(?m)^\s*def\s+{_re.escape(entry)}\b", item.prompt)
    if m:
        prompt_preamble = item.prompt[: m.start()]
    if f"def {entry}" in pred_code:
        program = prompt_preamble + "\n" + pred_code
    else:
        program = item.prompt + pred_code
    source = program + "\n\n" + test + f"\n\ncheck({entry})\n"
    try:
        from .sandbox import run_python_sandboxed
        ok, _tail = run_python_sandboxed(source, timeout_s=5, memory_mb=256)
        return bool(ok)
    except Exception as e:
        logger.debug("humaneval sandbox failure on %s: %s", item.item_id, e)
        return False


def _grade_mbpp(item: BenchmarkItem, prediction: str) -> bool:
    """MBPP grader: execute prediction's code, then run each assert in test_list.

    Includes test_setup_code (imports the assert lines depend on) so that e.g.
    `assert math.isclose(foo(...), ...)` doesn't NameError on `math`.
    """
    tests = item.meta.get("test_list") or []
    if not tests:
        return False
    pred_code = _extract_code(prediction)
    if "def " not in pred_code:
        return False
    setup = item.meta.get("test_setup_code") or ""
    asserts = "\n".join(tests)
    parts = [pred_code]
    if setup.strip():
        parts.append(setup)
    parts.append(asserts)
    source = "\n\n".join(parts) + "\n"
    try:
        from .sandbox import run_python_sandboxed
        ok, _tail = run_python_sandboxed(source, timeout_s=5, memory_mb=256)
        return bool(ok)
    except Exception as e:
        logger.debug("mbpp sandbox failure on %s: %s", item.item_id, e)
        return False


def _normalize_number(s: str) -> str:
    # Extract trailing number-looking substring.
    import re
    m = re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    return m[-1] if m else s.strip()


ModelFn = Callable[[str], str]  # prompt -> prediction
BatchModelFn = Callable[[list], list]  # prompts -> predictions


def run_anchor_eval(
    model_fn: ModelFn,
    *,
    benchmarks: Iterable[str] = SUPPORTED_BENCHMARKS,
    per_benchmark: int = 50,
    cache_dir: Path | str = Path("outputs/external_benchmarks"),
    seed: int = ANCHOR_SEED,
    grade_fn: Optional[Callable[[BenchmarkItem, str], bool]] = None,
    batch_model_fn: Optional[BatchModelFn] = None,
) -> dict:
    """Run the anchor eval and return a summary dict.

    ``model_fn`` is a prompt→prediction callable. Kept abstract so the
    orchestrator can pass in either an HF-model wrapper or a vLLM wrapper.
    Returns::

        {"anchor_score": float in [0,1],
         "per_benchmark": {name: float, ...},
         "n": int,
         "timestamp": float}
    """
    grade = grade_fn or _default_grade
    items_by_bench: dict[str, list[BenchmarkItem]] = {}
    for b in benchmarks:
        try:
            items_by_bench[b] = load_benchmark(b, cache_dir=cache_dir)
        except Exception as e:
            logger.warning("anchor_eval: skipping benchmark %s (%s)", b, e)
    sample = sample_anchor_set(items_by_bench, per_benchmark, seed=seed)

    correct = 0
    per_bench_correct: dict[str, int] = {b: 0 for b in items_by_bench}
    per_bench_total: dict[str, int] = {b: 0 for b in items_by_bench}

    # Fast path: single batched call through vLLM. Per-prompt serial calls on a
    # 100-item anchor set were ~15 min/cycle — one batch collapses to ~1 min.
    preds: list[str]
    if batch_model_fn is not None and sample:
        try:
            preds = list(batch_model_fn([it.prompt for it in sample]))
            if len(preds) < len(sample):
                preds = preds + [""] * (len(sample) - len(preds))
        except Exception as e:
            logger.warning("anchor_eval batch_model_fn failed, falling back to serial: %s", e)
            preds = []
            for it in sample:
                try:
                    preds.append(model_fn(it.prompt))
                except Exception as ee:
                    logger.debug("model_fn raised on item %s: %s", it.item_id, ee)
                    preds.append("")
    else:
        preds = []
        for it in sample:
            try:
                preds.append(model_fn(it.prompt))
            except Exception as e:
                logger.debug("model_fn raised on item %s: %s", it.item_id, e)
                preds.append("")

    per_item_results: list[dict] = []
    for it, pred in zip(sample, preds):
        ok = bool(grade(it, pred))
        per_bench_total[it.benchmark] = per_bench_total.get(it.benchmark, 0) + 1
        if ok:
            correct += 1
            per_bench_correct[it.benchmark] = per_bench_correct.get(it.benchmark, 0) + 1
        per_item_results.append({
            "benchmark": it.benchmark,
            "item_id": it.item_id,
            "passed": ok,
        })
    n = len(sample)
    anchor_score = (correct / n) if n else 0.0
    per_bench = {
        b: (per_bench_correct[b] / per_bench_total[b]) if per_bench_total[b] else 0.0
        for b in per_bench_total
    }
    # Distinct-prediction count per benchmark — flags "sampling isn't constant"
    # suspiciously-clean results (e.g. gsm8k=1.0 from a model always emitting "0").
    per_bench_distinct: dict[str, int] = {b: 0 for b in per_bench_total}
    seen_by_bench: dict[str, set] = {b: set() for b in per_bench_total}
    for it, pred in zip(sample, preds):
        seen_by_bench[it.benchmark].add((pred or "").strip()[:256])
    for b in per_bench_distinct:
        per_bench_distinct[b] = len(seen_by_bench[b])

    # Offline-fixture detection per benchmark (task #9): gsm8k=1.0 in the
    # overnight run turned out to be the 12-item offline fixture — trivial
    # arithmetic the model always gets right — not real HF gsm8k. Mark any
    # benchmark whose sampled items are entirely stamped with
    # meta["source"] == "offline_fixture" so telemetry callers can
    # distinguish "true held-out result" from "toy-fixture placeholder".
    per_bench_offline: dict[str, bool] = {b: False for b in per_bench_total}
    offline_counts: dict[str, int] = {b: 0 for b in per_bench_total}
    for it in sample:
        if (it.meta or {}).get("source") == "offline_fixture":
            offline_counts[it.benchmark] = offline_counts.get(it.benchmark, 0) + 1
    for b in per_bench_offline:
        total = per_bench_total.get(b, 0)
        per_bench_offline[b] = total > 0 and offline_counts.get(b, 0) == total

    # Suspiciously-clean alarm (task #9): per-benchmark score >= 0.95 while
    # either (a) fewer than 15% of predictions were distinct (degenerate
    # output — model is emitting ~the same string every time and the grader
    # is accepting it), or (b) the benchmark is running on the offline
    # fixture (non-signal: toy arithmetic the model cannot fail). Returned
    # as a list so cycle telemetry can log + alarm without recomputing.
    per_bench_suspect: list[dict] = []
    CLEAN_SCORE = 0.95
    LOW_DIVERSITY_RATIO = 0.15
    for b, score in per_bench.items():
        total = per_bench_total.get(b, 0)
        distinct = per_bench_distinct.get(b, 0)
        diversity_ratio = (distinct / total) if total else 1.0
        reasons: list[str] = []
        if per_bench_offline.get(b):
            reasons.append("offline_fixture")
        if score >= CLEAN_SCORE and diversity_ratio <= LOW_DIVERSITY_RATIO:
            reasons.append(
                f"degenerate_predictions(distinct={distinct}/{total})"
            )
        if reasons and score >= CLEAN_SCORE:
            # Only alarm when the score itself is clean — a failing
            # benchmark on offline fixtures is still informative.
            per_bench_suspect.append({
                "benchmark": b,
                "score": score,
                "n": total,
                "distinct": distinct,
                "reasons": reasons,
            })
            logger.warning(
                "anchor_eval suspicious-clean: benchmark=%s score=%.3f n=%d "
                "distinct=%d reasons=%s — score is NOT a valid held-out signal",
                b, score, total, distinct, ",".join(reasons),
            )

    return {
        "anchor_score": anchor_score,
        "per_benchmark": per_bench,
        "per_benchmark_n": dict(per_bench_total),
        "per_benchmark_distinct": per_bench_distinct,
        "per_benchmark_offline": per_bench_offline,
        "per_benchmark_suspect": per_bench_suspect,
        "per_item_results": per_item_results,
        "n": n,
        "timestamp": time.time(),
    }


# ---------------------------------------------------------------------------
# Verifier-capture canary.
# ---------------------------------------------------------------------------

def detect_verifier_capture(
    *,
    internal_delta: float,
    anchor_delta: float,
    threshold: float = 0.01,
    anchor_drop_tolerance: float = 0.005,
) -> bool:
    """Canary: internal eval improves by > threshold while anchor drops
    by > anchor_drop_tolerance. That divergence is the silent-failure
    signature of verifier capture.
    """
    return (internal_delta > threshold) and (anchor_delta < -anchor_drop_tolerance)


def fire_capture_alarm(
    *,
    cycle: int,
    internal_delta: float,
    anchor_delta: float,
    log_path: Path | str = Path("update-log.txt"),
) -> str:
    """Append a verifier-capture alarm to update-log.txt. Returns the line written."""
    line = (
        f"[VERIFIER-CAPTURE-ALARM] cycle={cycle} "
        f"internal_delta={internal_delta:+.4f} anchor_delta={anchor_delta:+.4f} "
        f"ts={time.time():.0f}\n"
    )
    try:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
    except OSError as e:
        logger.warning("capture-alarm write failed (%s)", e)
    logger.error(line.strip())
    return line


# ---------------------------------------------------------------------------
# Adversarial bank seeding. Produces known-wrong-but-plausible mutations of
# canonical solutions: off-by-one, swapped operator, returned input unchanged.
# ---------------------------------------------------------------------------

def _mutate_code_solution(src: str, rng: random.Random) -> Optional[str]:
    if not src or "def " not in src:
        return None
    # Pick one of a few textual mutations. All produce valid-but-wrong code
    # for the kinds of tasks we seed from HumanEval / MBPP.
    choices = []
    if "+ b" in src:
        choices.append(src.replace("+ b", "- b", 1))
    if "* b" in src:
        choices.append(src.replace("* b", "+ b", 1))
    if "return " in src:
        # Off-by-one: replace "return X" with "return X + 1" on a simple expr line.
        lines = src.splitlines()
        for i, ln in enumerate(lines):
            stripped = ln.strip()
            if stripped.startswith("return ") and "#" not in stripped:
                lines[i] = ln.rstrip() + " + 1"
                choices.append("\n".join(lines) + ("\n" if src.endswith("\n") else ""))
                break
    if not choices:
        return None
    return rng.choice(choices)


def seed_adversarial_bank_from_external(
    bank: Any,  # AdversarialBank, kept loose to avoid circular import
    *,
    per_benchmark: int = 10,
    cache_dir: Path | str = Path("outputs/external_benchmarks"),
    seed: int = 0xBAD_BAD,
    cycle: int = -1,
) -> int:
    """Append wrong-but-plausible external solutions into the VoV bank.

    This guarantees the adversarial bank is not 100% self-generated — a
    property that is accepted by every bank entry is truly too weak,
    because at least some bank entries come from an external source the
    model cannot have over-fit to.

    Returns the number of entries added.
    """
    rng = random.Random(seed)
    added = 0
    for bench in ("humaneval", "mbpp"):
        try:
            items = load_benchmark(bench, cache_dir=cache_dir)
        except Exception as e:
            logger.warning("seed: skip %s (%s)", bench, e)
            continue
        # Deterministic sub-selection before mutation.
        items = sorted(items, key=lambda it: _stable_hash(f"{seed}:{it.item_id}"))
        for it in items[: per_benchmark * 3]:
            mutated = _mutate_code_solution(it.answer, rng)
            if not mutated:
                continue
            try:
                bank.append(
                    problem_id=f"external:{it.benchmark}:{it.item_id}",
                    candidate=mutated,
                    domain="code",
                    problem_ctx={"source": "external_benchmark", "benchmark": bench},
                    cycle=cycle,
                    reason="external_seed: known-wrong mutation of canonical solution",
                )
                added += 1
            except Exception as e:
                logger.debug("bank.append failed for %s: %s", it.item_id, e)
            if added >= per_benchmark * 2:
                break
    return added


__all__ = [
    "SUPPORTED_BENCHMARKS",
    "BenchmarkItem",
    "load_benchmark",
    "sample_anchor_set",
    "run_anchor_eval",
    "detect_verifier_capture",
    "fire_capture_alarm",
    "seed_adversarial_bank_from_external",
    "ANCHOR_SEED",
]
