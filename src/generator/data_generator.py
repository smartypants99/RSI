"""Training data generator — the model creates its own training data with full COT.

Rebuilt parser: structured line classification, schema validation, near-duplicate
detection, and retry-with-prompt-adjustment on parse failure. Backwards-compatible
public surface (TrainingSample, ReasoningStep, DataGenerator) for verifier/trainer.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import re
from dataclasses import dataclass, field
from enum import Enum

import torch

from typing import Callable, Optional

from ..diagnostics.engine import DiagnosticResult, WeaknessReport
from ..utils.config import GeneratorConfig
from ..utils.model_loader import ModelLoader

logger = logging.getLogger(__name__)


# =============================================================================
# Public dataclasses — preserved for verifier/trainer compatibility
# =============================================================================


@dataclass
class ReasoningStep:
    """A single step in a chain-of-thought.

    Atomic-step fields (step_id, depends_on, rule) enable external structural
    verification. Legacy samples leave step_id=0, depends_on=[], rule="".
    """
    step_number: int
    content: str
    justification: str
    assumptions: list[str] = field(default_factory=list)
    step_id: int = 0
    depends_on: list[int] = field(default_factory=list)
    rule: str = ""
    # Metacognitive confidence — value in [0,1] parsed from a trailing
    # [C:0.xx] token emitted by the model after a step. -1.0 means unset
    # (no marker was emitted); downstream code should skip unset values.
    confidence: float = -1.0

    @property
    def claim(self) -> str:
        return self.content

    @claim.setter
    def claim(self, value: str) -> None:
        self.content = value

    @property
    def is_atomic(self) -> bool:
        return self.step_id > 0 and bool(self.rule)


@dataclass
class TrainingSample:
    """A single training sample with full chain-of-thought."""
    prompt: str
    response: str
    reasoning_chain: list[ReasoningStep] = field(default_factory=list)
    target_weakness: str = ""
    domain: str = ""
    verified: bool = False
    verification_notes: str = ""
    confidence: float = 0.0
    content_hash: str = ""
    parse_confidence: float = 1.0
    parse_issues: list[str] = field(default_factory=list)
    severity_at_generation: float = 0.0
    expected_answer: str = ""
    # Metacognitive per-step confidence in [0,1]. Parallel to reasoning_chain —
    # unset steps use -1.0. Populated by the parser from [C:0.xx] markers.
    per_step_confidence: list[float] = field(default_factory=list)
    # Fraction of independent generations that reached the same final answer
    # on this problem (self-consistency, Wang et al. 2022). 1.0 = perfect
    # agreement; 0.0 = no check was run. Training weight multiplies this in,
    # so low-agreement samples contribute less.
    consistency_score: float = 1.0
    consistency_votes: int = 1  # total generations used for the check
    # STaR (Zelikman et al. 2022) ground-truth fields. Populated when this
    # sample survived because its final answer matched a CANONICAL answer from
    # a real diagnostic problem — not self-agreement, not heuristic acceptance.
    ground_truth_check_type: str = ""  # "contains"/"math_equiv"/"code_executes"/...
    ground_truth_verified: bool = False
    source: str = "synthesized"  # "star" | "star_rationalized" | "synthesized"
    sample_index: int = 0  # which of the K sampled chains this is (debug)

    def __post_init__(self):
        # Sync per-step confidences from the chain if not explicitly provided.
        # A step's confidence == -1.0 means "no marker emitted by the model" —
        # preserved here so downstream calibration code can filter them out.
        if not self.per_step_confidence and self.reasoning_chain:
            self.per_step_confidence = [
                float(getattr(s, "confidence", -1.0)) for s in self.reasoning_chain
            ]
        if not self.content_hash and self.prompt:
            chain_text = "|".join(
                f"{s.content}::{getattr(s, 'justification', '')}::"
                f"{';'.join(getattr(s, 'assumptions', []) or [])}"
                for s in self.reasoning_chain
            ) if self.reasoning_chain else ""
            self.content_hash = hashlib.md5(
                f"{self.domain}:{self.target_weakness}:{self.prompt}{chain_text}{self.response}".encode()
            ).hexdigest()

    def to_training_format(self) -> dict:
        full_response = self._build_full_response()
        return {
            "prompt": self.prompt,
            "completion": full_response,
            "metadata": {
                "domain": self.domain,
                "target_weakness": self.target_weakness,
                "num_reasoning_steps": len(self.reasoning_chain),
                "verified": self.verified,
                "parse_confidence": self.parse_confidence,
            },
        }

    def _build_full_response(self) -> str:
        parts = []
        for i, step in enumerate(self.reasoning_chain, 1):
            if self.domain == "code":
                content = re.sub(r'(?m)^\d+[.)]\s+', '', step.content).strip()
            else:
                content = " ".join(step.content.split())
            content = re.sub(r'^(?:step\s*\d+\s*[.):–—-]\s*)', '', content, flags=re.IGNORECASE)
            header_id = step.step_id if step.step_id > 0 else i
            parts.append(f"Step {header_id}: {content}")
            if step.depends_on:
                parts.append(f"  Depends: {', '.join(str(d) for d in step.depends_on)}")
            elif step.is_atomic:
                parts.append("  Depends: premise")
            if step.rule:
                parts.append(f"  Rule: {step.rule}")
            if step.justification and step.justification != "implicit":
                parts.append(f"  Justification: {step.justification}")
            if step.assumptions:
                parts.append(f"  Assumptions: {'; '.join(step.assumptions)}")
        parts.append(f"\nConclusion: {self.response}")
        return "\n".join(parts)


@dataclass
class PreferencePair:
    """A DPO preference pair: same prompt, chosen (correct) vs rejected (incorrect) response.

    Produced naturally by K-sample rollouts: for each problem, chains that verified
    correctly become `chosen`, chains that verified incorrectly become `rejected`.
    Pairing is done star_loop-side; the trainer treats each pair as one unit.
    """
    prompt: str
    chosen_response: str
    rejected_response: str
    domain: str = ""
    target_weakness: str = ""
    weight: float = 1.0
    chosen_confidence: float = 1.0
    rejected_confidence: float = 0.0
    severity_at_generation: float = 0.0
    content_hash: str = ""

    def __post_init__(self):
        if not self.content_hash and self.prompt:
            self.content_hash = hashlib.md5(
                f"{self.domain}:{self.target_weakness}:{self.prompt}|"
                f"{self.chosen_response}|{self.rejected_response}".encode()
            ).hexdigest()


# =============================================================================
# Parser — structured, line-classifying, reports diagnostics
# =============================================================================


class LineKind(Enum):
    BLANK = 0
    STEP_HEADER = 1
    JUSTIFICATION = 2
    ASSUMPTION = 3
    CONCLUSION = 4
    CONTENT = 5


@dataclass
class ParseResult:
    chain: list[ReasoningStep]
    conclusion: str
    issues: list[str] = field(default_factory=list)
    confidence: float = 1.0

    @property
    def ok(self) -> bool:
        return bool(self.chain) and bool(self.conclusion)


_STEP_PATTERNS = [
    re.compile(r'^(?:\*{1,2}|#{1,4}\s*)?step\s*(\d{1,3})\s*\*{0,2}\s*[.):\-–—]?\s*(.*)$', re.IGNORECASE),
    re.compile(r'^\(?(\d{1,2})\)?\s*[.):\-–—]\s+(.*)$'),
    re.compile(r'^\*{1,2}(\d{1,2})[.):\-–—]?\*{1,2}\s*(.*)$'),
]

_ORDINAL_MAP = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}
_ORDINAL_RE = re.compile(
    r'^(' + '|'.join(_ORDINAL_MAP) + r')(?:ly)?\s*[,.:;)\-]\s*(.*)$',
    re.IGNORECASE,
)

_CONCLUSION_RE = re.compile(
    r'^(?:#{1,4}\s*|>\s*|\*{1,2})?\s*'
    r'(?:conclusion|final\s+answer|in\s+conclusion|final\s+result|the\s+answer\s+is)'
    r'\s*\*{0,2}\s*[:\-–—]?\s*(.*)$',
    re.IGNORECASE,
)

_JUSTIFICATION_RE = re.compile(
    r'^(?:\*{1,2})?\s*'
    r'(?:justification|because|reason|rationale|this\s+(?:is\s+because|follows\s+because|works\s+because)|the\s+reason\s+is)'
    r'\s*\*{0,2}\s*[:\-–—]\s*(.*)$',
    re.IGNORECASE,
)

_ASSUMPTION_RE = re.compile(
    r'^(?:\*{1,2})?\s*'
    r'(?:assumptions?|assuming|given\s+that|premise)'
    r'\s*\*{0,2}\s*[:\-–—]\s*(.*)$',
    re.IGNORECASE,
)

_BULLET_RE = re.compile(r'^[-*•·]\s*')
_MD_WRAP_RE = re.compile(r'^\*{1,2}(.+?)\*{1,2}$')
_MD_HEADER_RE = re.compile(r'^(?:#{1,6}|>)+\s*')
_MD_INLINE_RE = re.compile(r'(?:\*{1,3}|`+|_{1,2})')


def _strip_inline_md(s: str) -> str:
    """Remove leftover emphasis/code marks that survived line-level stripping."""
    return _MD_INLINE_RE.sub('', s).strip()

_CONTAM_PATTERNS = re.compile(
    r'(?:\n|(?<=[.?!;] ))(?:solution|answer|step\s*1[.:)]|first,\s*(?:we|let|I)|'
    r"let's solve|to solve this|hint:|the answer|we can solve|let me solve)",
    re.IGNORECASE,
)

# Anchored at string start — catches responses that open with a solution/answer
# header, which _CONTAM_PATTERNS misses because it requires a preceding newline
# or sentence boundary. If this matches, the entire "problem" is unusable.
_LEADING_CONTAM_RE = re.compile(
    r'^\s*(?:solution|answer|hint|step\s*1|first,\s*(?:we|let|I)|'
    r"let's solve|to solve this|let me solve|we can solve|the answer is)\b",
    re.IGNORECASE,
)

_TOKEN_RE = re.compile(r'[A-Za-z0-9]+')

# Metacognitive confidence marker: [C:0.85] anywhere in a line.
# Accepts optional whitespace and 0-1 float. Captured value is stripped from
# content; last match in a step wins (rarely more than one).
_CONFIDENCE_RE = re.compile(r'\[\s*[Cc]\s*:\s*([01](?:\.\d+)?|0?\.\d+)\s*\]')

# Refusal patterns — model declined to answer
_REFUSAL_RE = re.compile(
    r"(?:i\s+can'?t|i\s+cannot|i\s+(?:am|'m)\s+(?:not\s+able|unable)|"
    r"i\s+(?:don'?t|do\s+not)\s+(?:have|know)|"
    r"as\s+an?\s+(?:ai|language\s+model)|"
    r"i\s+(?:apologize|sorry)|"
    r"not\s+(?:possible|able)\s+to\s+(?:solve|answer|help)|"
    r"this\s+(?:is\s+)?(?:beyond|outside)\s+(?:my|the\s+scope))",
    re.IGNORECASE,
)


def _is_gibberish(text: str) -> bool:
    """Detect gibberish: low alpha ratio, excessive repetition, or too short."""
    if not text or len(text.strip()) < 10:
        return True
    stripped = text.strip()
    alpha_chars = sum(1 for c in stripped if c.isalpha())
    if len(stripped) > 20 and alpha_chars / len(stripped) < 0.3:
        return True
    tokens = _TOKEN_RE.findall(stripped.lower())
    if len(tokens) >= 5:
        from collections import Counter
        most_common_count = Counter(tokens).most_common(1)[0][1]
        if most_common_count / len(tokens) > 0.6:
            return True
    return False


def _is_refusal(text: str) -> bool:
    """Check if model output is a refusal to answer."""
    if not text:
        return False
    return bool(_REFUSAL_RE.search(text[:300]))


def _strip_markdown_line(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    s = _BULLET_RE.sub('', s).strip()
    s = _MD_HEADER_RE.sub('', s).strip()
    m = _MD_WRAP_RE.match(s)
    if m:
        s = m.group(1).strip()
    return s


def _classify_line(line: str) -> tuple[LineKind, tuple]:
    if not line:
        return LineKind.BLANK, ()

    m = _CONCLUSION_RE.match(line)
    if m:
        return LineKind.CONCLUSION, (_strip_inline_md(m.group(1)),)

    for pat in _STEP_PATTERNS:
        m = pat.match(line)
        if m:
            try:
                n = int(m.group(1))
            except (TypeError, ValueError):
                continue
            if 0 <= n <= 99:
                return LineKind.STEP_HEADER, (n, _strip_inline_md(m.group(2) or ""))

    m = _ORDINAL_RE.match(line)
    if m:
        n = _ORDINAL_MAP[m.group(1).lower()]
        return LineKind.STEP_HEADER, (n, _strip_inline_md(m.group(2) or ""))

    m = _JUSTIFICATION_RE.match(line)
    if m:
        return LineKind.JUSTIFICATION, (_strip_inline_md(m.group(1)),)

    m = _ASSUMPTION_RE.match(line)
    if m:
        return LineKind.ASSUMPTION, (_strip_inline_md(m.group(1)),)

    return LineKind.CONTENT, (_strip_inline_md(line),)


def _split_assumptions(text: str) -> list[str]:
    if not text or text.lower() in ("none", "n/a", "-", "implicit"):
        return []
    return [a.strip() for a in re.split(r'[;,]|\s+and\s+', text) if a.strip()]


class ResponseParser:
    """Parses model output into a structured chain + conclusion."""

    def parse(self, text: str) -> ParseResult:
        issues: list[str] = []
        if not text or not text.strip():
            return ParseResult(chain=[], conclusion="", issues=["empty response"], confidence=0.0)

        if _is_refusal(text):
            return ParseResult(chain=[], conclusion="", issues=["model refusal"], confidence=0.0)

        if _is_gibberish(text):
            return ParseResult(chain=[], conclusion="", issues=["gibberish response"], confidence=0.0)

        classified: list[tuple[LineKind, tuple]] = []
        for raw in text.splitlines():
            cleaned = _strip_markdown_line(raw)
            classified.append(_classify_line(cleaned))

        chain: list[ReasoningStep] = []
        conclusion_parts: list[str] = []
        in_conclusion = False

        cur_num: int | None = None
        cur_content: list[str] = []
        cur_just_parts: list[str] = []
        cur_assumptions: list[str] = []
        last_field: str | None = None

        def flush() -> None:
            nonlocal cur_num, cur_content, cur_just_parts, cur_assumptions, last_field
            if cur_num is not None:
                content = "\n".join(x for x in cur_content if x).strip()
                content = re.sub(r'^(?:step\s*\d+\s*[.):\-–—]\s*)', '', content, flags=re.IGNORECASE)
                # Extract metacognitive confidence marker [C:0.xx] from content
                # and/or justification. Use the last match (rightmost), strip
                # all occurrences from the rendered fields.
                just_raw = " ".join(p for p in cur_just_parts if p).strip()
                conf_val = -1.0
                for field_text in (content, just_raw):
                    m_iter = list(_CONFIDENCE_RE.finditer(field_text))
                    if m_iter:
                        try:
                            v = float(m_iter[-1].group(1))
                            if 0.0 <= v <= 1.0:
                                conf_val = v
                        except ValueError:
                            pass
                content = _CONFIDENCE_RE.sub('', content).strip()
                just_raw = _CONFIDENCE_RE.sub('', just_raw).strip()
                if content or cur_just_parts or cur_assumptions:
                    just = just_raw or "implicit"
                    chain.append(ReasoningStep(
                        step_number=cur_num,
                        content=content,
                        justification=just,
                        assumptions=cur_assumptions[:],
                        confidence=conf_val,
                    ))
            cur_num = None
            cur_content = []
            cur_just_parts = []
            cur_assumptions = []
            last_field = None

        for kind, payload in classified:
            if in_conclusion:
                if kind == LineKind.STEP_HEADER:
                    break
                if kind == LineKind.CONTENT:
                    conclusion_parts.append(payload[0])
                elif kind == LineKind.BLANK and conclusion_parts:
                    break
                continue

            if kind == LineKind.BLANK:
                last_field = None
                continue

            if kind == LineKind.CONCLUSION:
                flush()
                in_conclusion = True
                if payload[0]:
                    conclusion_parts.append(payload[0])
                continue

            if kind == LineKind.STEP_HEADER:
                flush()
                cur_num, trailing = payload
                if trailing:
                    cur_content.append(trailing)
                last_field = "content"
                continue

            if kind == LineKind.JUSTIFICATION:
                if cur_num is None:
                    cur_num = len(chain) + 1
                cur_just_parts.append(payload[0])
                last_field = "justification"
                continue

            if kind == LineKind.ASSUMPTION:
                if cur_num is None:
                    cur_num = len(chain) + 1
                cur_assumptions.extend(_split_assumptions(payload[0]))
                last_field = "assumption"
                continue

            text_line = payload[0]
            if cur_num is None:
                cur_num = 1
                cur_content.append(text_line)
                last_field = "content"
                continue

            if last_field == "justification":
                cur_just_parts.append(text_line)
            elif last_field == "assumption":
                cur_assumptions.extend(_split_assumptions(text_line))
            else:
                cur_content.append(text_line)

        flush()

        conclusion = " ".join(conclusion_parts).strip()
        if not conclusion:
            conclusion = self._fallback_conclusion(text)
            if conclusion:
                issues.append("conclusion extracted via fallback")
            else:
                issues.append("no conclusion found")

        score = 1.0
        if not chain:
            score = 0.0
            issues.append("no steps parsed")
        else:
            if not any(s.justification and s.justification != "implicit" for s in chain):
                score -= 0.2
                issues.append("no justifications parsed")
            if all(len(s.content) < 10 for s in chain):
                score -= 0.3
                issues.append("all step contents very short")
        if "no conclusion found" in issues:
            score -= 0.3
        score = max(0.0, min(1.0, score))

        return ParseResult(chain=chain, conclusion=conclusion, issues=issues, confidence=score)

    @staticmethod
    def _fallback_conclusion(text: str) -> str:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not lines:
            return ""
        for line in reversed(lines):
            cleaned = _strip_markdown_line(line)
            kind, payload = _classify_line(cleaned)
            if kind in (LineKind.JUSTIFICATION, LineKind.ASSUMPTION, LineKind.BLANK):
                continue
            if kind == LineKind.STEP_HEADER:
                trailing = payload[1]
                if trailing:
                    return trailing
                continue
            if kind == LineKind.CONTENT and payload[0]:
                return payload[0]
        return ""


# =============================================================================
# Schema validation & near-duplicate detection
# =============================================================================


def _normalize_for_dedup(text: str) -> frozenset[str]:
    return frozenset(t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 2)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


@dataclass
class SchemaValidation:
    ok: bool
    issues: list[str] = field(default_factory=list)


def validate_sample(sample: TrainingSample, min_steps: int) -> SchemaValidation:
    """Structural invariants, not logical validity — runs before verifier."""
    issues: list[str] = []
    if not sample.prompt or not sample.prompt.strip():
        issues.append("missing prompt")
    if not sample.response or not sample.response.strip():
        issues.append("missing conclusion")
    if len(sample.reasoning_chain) < min_steps:
        issues.append(f"only {len(sample.reasoning_chain)} steps (need {min_steps})")
    for i, step in enumerate(sample.reasoning_chain):
        if not isinstance(step, ReasoningStep):
            issues.append(f"step {i} wrong type")
            continue
        if not step.content or not step.content.strip():
            issues.append(f"step {i} empty content")
        if step.justification is None:
            issues.append(f"step {i} justification is None")
    if sample.response and _is_gibberish(sample.response):
        issues.append("conclusion is gibberish")
    if sample.prompt and _is_gibberish(sample.prompt):
        issues.append("prompt is gibberish")
    return SchemaValidation(ok=not issues, issues=issues)


# =============================================================================
# DataGenerator
# =============================================================================


class DataGenerator:
    """Generates training data with robust parsing, schema validation, and dedup."""

    DEDUP_JACCARD_THRESHOLD = 0.85
    MAX_DOMAIN_FRACTION = 0.40

    def __init__(self, config: GeneratorConfig, model_loader: ModelLoader):
        self.config = config
        self.model = model_loader
        self._seen_hashes: set[str] = set()
        self._seen_signatures: list[frozenset[str]] = []
        self._custom_solution_template: str | None = None
        self._parser = ResponseParser()
        # Training data cache: weakness key -> list of samples from last cycle.
        # Avoids regenerating samples for unchanged weaknesses.
        self._sample_cache: dict[str, list[TrainingSample]] = {}
        self._last_weakness_keys: dict[str, float] = {}  # weakness key -> severity
        # Diversity stats for the most recent generation run
        self._diversity_stats: dict = {}
        # STaR ground-truth grader: (response, expected, check_type) -> bool.
        # Injected by the orchestrator (typically DiagnosticsEngine._check_answer).
        # Without it, the STaR path is disabled and we fall back to the legacy
        # self-synthesis path.
        self._grader: Optional[Callable[[str, str, str], bool]] = None
        # DPO preference pairs emitted during the most recent STaR run. Each
        # problem that produced ≥1 correct and ≥1 incorrect chain yields pairs
        # (chosen × rejected) consumed by the trainer in dpo/mixed mode.
        self._last_preference_pairs: list[PreferencePair] = []
        # Cap on DPO pairs per problem to avoid combinatorial explosion
        # (K correct × K incorrect = K² pairs if unbounded).
        self.MAX_DPO_PAIRS_PER_PROBLEM = 4

    def set_grader(self, grader: Callable[[str, str, str], bool]) -> None:
        """Register a ground-truth grading function for the STaR path.

        Expected signature matches DiagnosticsEngine._check_answer:
            grader(response: str, expected: str, check_type: str) -> bool
        """
        self._grader = grader

    def _generate_batch_with_oom_retry(
        self, prompts: list[str], max_new_tokens: int = 2048,
        temperature: float = 0.7, top_p: float = 0.9, max_retries: int = 3,
    ) -> list[str]:
        """Call generate_batch with OOM retry — halves chunk size on each failure."""
        chunk_size = len(prompts)
        for attempt in range(max_retries + 1):
            try:
                all_responses: list[str] = []
                for start in range(0, len(prompts), chunk_size):
                    batch = prompts[start:start + chunk_size]
                    all_responses.extend(
                        self.model.generate_batch(batch, max_new_tokens=max_new_tokens,
                                                  temperature=temperature, top_p=top_p)
                    )
                return all_responses
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                chunk_size = max(1, chunk_size // 2)
                if attempt < max_retries:
                    logger.warning(
                        f"OOM in generator generate_batch (attempt {attempt+1}/{max_retries}), "
                        f"retrying with chunk_size={chunk_size}"
                    )
                else:
                    logger.error(f"OOM in generator after {max_retries} retries — returning empty")
                    return [""] * len(prompts)
        return [""] * len(prompts)

    def set_custom_solution_template(self, template: str) -> None:
        """Override the default solution prompt template."""
        self._custom_solution_template = template

    def get_diversity_stats(self) -> dict:
        """Return diversity metrics from the most recent generation run."""
        return dict(self._diversity_stats)

    def _compute_num_samples(self, weakness: WeaknessReport) -> int:
        """Sample count from severity, tempered by significance and difficulty.

        Uses additive WeaknessReport fields (`significance`, `difficulty`) when
        populated; falls back gracefully to severity-only scaling when they are
        at defaults (significance=1.0, difficulty='mixed').
        """
        severity = max(0.0, min(1.0, weakness.severity))
        scale = max(0.2, min(1.0, severity * 2))
        # Down-weight non-significant weaknesses (p > 0.1) — noisy signal.
        # Skip when significance is at its default (1.0), signalling "not populated"
        # by the diagnostician; penalizing that would break back-compat.
        sig = getattr(weakness, "significance", 1.0)
        if 0.1 < sig < 1.0:
            scale *= max(0.3, 1.0 - (sig - 0.1))
        # Harder weaknesses deserve more samples; easier ones fewer.
        difficulty = getattr(weakness, "difficulty", "mixed") or "mixed"
        difficulty_mult = {
            "easy": 0.7, "medium": 1.0, "mixed": 1.0, "hard": 1.2, "expert": 1.4,
        }.get(difficulty.lower(), 1.0)
        scale *= difficulty_mult
        return max(10, int(self.config.samples_per_weakness * scale))

    def generate_for_weaknesses(self, weaknesses: list[WeaknessReport]) -> list[TrainingSample]:
        self._seen_hashes.clear()
        self._seen_signatures.clear()
        all_samples: list[TrainingSample] = []
        new_cache: dict[str, list[TrainingSample]] = {}
        cache_hits = 0

        for weakness in weaknesses:
            cache_key = f"{weakness.domain}/{weakness.subdomain}"
            # Cache hit: reuse samples if weakness severity hasn't changed significantly
            prev_severity = self._last_weakness_keys.get(cache_key)
            if (prev_severity is not None
                    and abs(prev_severity - weakness.severity) < 0.05
                    and cache_key in self._sample_cache
                    and self._sample_cache[cache_key]):
                samples = self._sample_cache[cache_key]
                cache_hits += 1
                logger.info(f"  Cache hit for {cache_key} ({len(samples)} samples)")
            else:
                samples = self._generate_for_weakness(weakness)
            new_cache[cache_key] = samples
            for s in samples:
                if self._accept_unique(s):
                    all_samples.append(s)

        # Update cache for next cycle
        self._sample_cache = new_cache
        self._last_weakness_keys = {
            f"{w.domain}/{w.subdomain}": w.severity for w in weaknesses
        }

        # Diversity rebalancing: cap any single domain at MAX_DOMAIN_FRACTION
        all_samples = self._rebalance_domains(all_samples)

        # Compute diversity stats
        self._diversity_stats = self._compute_diversity(all_samples)
        if cache_hits:
            logger.info(f"  Cache: {cache_hits}/{len(weaknesses)} weaknesses reused from prior cycle")
        logger.info(
            f"Generator produced {len(all_samples)} unique samples across "
            f"{len(weaknesses)} weaknesses "
            f"(diversity: {self._diversity_stats.get('topic_coverage', 0):.0%} topic coverage, "
            f"{self._diversity_stats.get('unique_domains', 0)} domains)"
        )
        return all_samples

    # =========================================================================
    # STaR path: train on rationales that reach KNOWN-CORRECT canonical answers
    # on REAL diagnostic problems. (Zelikman et al., 2022 — Self-Taught Reasoner.)
    # =========================================================================

    def generate_from_diagnostic_result(
        self, result: DiagnosticResult,
    ) -> list[TrainingSample]:
        """STaR-style generation: real problems, K chains, ground-truth filter.

        For every diagnostic question the model got WRONG, sample K chains at
        moderate temperature, parse each, and keep only those whose final
        answer matches the canonical `expected` under the diagnostics
        `check_type` grader. If 0/K chains succeed, optionally run the
        rationalization fallback (generate with the answer as a hint, strip
        the hint from the training example).

        `expected_answer` is set to the CANONICAL answer from diagnostics —
        not a model-extracted guess. `consistency_score` is (correct_k / K) —
        agreement with ground truth, not self-agreement.

        Requires set_grader() to have been called. If no grader is installed,
        logs a warning and falls back to the legacy self-synthesis path.
        """
        if self._grader is None:
            logger.warning(
                "STaR path invoked but no grader installed (call set_grader). "
                "Falling back to legacy generate_for_weaknesses."
            )
            return self.generate_for_weaknesses(result.weaknesses)

        failed_items = self._collect_failed_items(result)
        if not failed_items:
            logger.warning(
                "STaR: no failed diagnostic items found in result "
                f"(weaknesses={len(result.weaknesses)}). Falling back to legacy path."
            )
            return self.generate_for_weaknesses(result.weaknesses)

        self._seen_hashes.clear()
        self._seen_signatures.clear()

        all_samples: list[TrainingSample] = []
        all_pairs: list[PreferencePair] = []
        star_kept = 0
        star_rejected = 0
        rationalized = 0
        for_rationalization: list[dict] = []

        by_weakness: dict[tuple[str, str], list[dict]] = {}
        for item in failed_items:
            key = (item["domain"], item["subdomain"])
            by_weakness.setdefault(key, []).append(item)

        for (domain, subdomain), items in by_weakness.items():
            weakness = self._weakness_for(result, domain, subdomain)
            kept_here, rej_here, zero_hits, pairs_here = self._star_sample_batch(items, weakness)
            star_kept += len(kept_here)
            star_rejected += rej_here
            for s in kept_here:
                if self._accept_unique(s):
                    all_samples.append(s)
            all_pairs.extend(pairs_here)
            for_rationalization.extend(zero_hits)

        if self.config.star_rationalization and for_rationalization:
            cap = max(1, int(self.config.star_max_rationalizations_per_weakness))
            per_w: dict[tuple[str, str], list[dict]] = {}
            for item in for_rationalization:
                k = (item["domain"], item["subdomain"])
                per_w.setdefault(k, []).append(item)
            bounded: list[dict] = []
            for items in per_w.values():
                bounded.extend(items[:cap])
            rationalized_samples = self._rationalize_batch(bounded, result)
            for s in rationalized_samples:
                if self._accept_unique(s):
                    all_samples.append(s)
                    rationalized += 1

        all_samples = self._rebalance_domains(all_samples)
        self._diversity_stats = self._compute_diversity(all_samples)
        # Stash pairs so the orchestrator can retrieve them for DPO/mixed training
        # without changing the return type (list[TrainingSample]).
        self._last_preference_pairs = all_pairs
        logger.info(
            f"STaR: kept={star_kept}, rejected={star_rejected}, "
            f"rationalized={rationalized}, final={len(all_samples)} samples, "
            f"{len(all_pairs)} DPO pairs "
            f"across {len(by_weakness)} weakness buckets "
            f"({len(failed_items)} failed diagnostic items processed)"
        )
        return all_samples

    def get_last_preference_pairs(self) -> list[PreferencePair]:
        """DPO preference pairs produced by the most recent STaR run.

        Empty when the last run used the legacy path or produced no problems
        with both correct and incorrect chains.
        """
        return list(self._last_preference_pairs)

    @staticmethod
    def _render_chain_for_training(
        item: dict,
        chain: list[ReasoningStep],
        conclusion: str,
    ) -> str:
        """Render a chain the same way TrainingSample._build_full_response does.

        DPO requires chosen/rejected strings that share tokenizer conventions.
        We construct a throwaway TrainingSample and reuse its rendering so the
        two sides of every pair are byte-identical in format to SFT samples.
        """
        if not chain or not conclusion:
            return ""
        tmp = TrainingSample(
            prompt=item["question"],
            response=DataGenerator._postprocess_conclusion(
                conclusion, item["domain"], chain,
            ),
            reasoning_chain=chain,
            domain=item["domain"],
        )
        return tmp._build_full_response()

    @staticmethod
    def _collect_failed_items(result: DiagnosticResult) -> list[dict]:
        """Pull every `correct=False` evidence entry from every weakness.

        Items are deduplicated on (question, expected, check_type) to avoid
        double-counting when the same failure shows up in multiple weakness reports.
        """
        seen: set[tuple[str, str, str]] = set()
        failed: list[dict] = []
        for w in result.weaknesses:
            for ev in w.evidence or []:
                if not isinstance(ev, dict):
                    continue
                if ev.get("correct"):
                    continue
                q = ev.get("question") or ""
                exp = ev.get("expected") or ""
                ct = ev.get("check_type") or "contains"
                if not q or not exp:
                    continue
                key = (q, exp, ct)
                if key in seen:
                    continue
                seen.add(key)
                failed.append({
                    "question": q,
                    "expected": exp,
                    "check_type": ct,
                    "domain": ev.get("domain") or w.domain,
                    "subdomain": ev.get("subdomain") or w.subdomain,
                    "difficulty": ev.get("difficulty") or "medium",
                    "severity": w.severity,
                })
        return failed

    @staticmethod
    def _weakness_for(
        result: DiagnosticResult, domain: str, subdomain: str,
    ) -> WeaknessReport:
        for w in result.weaknesses:
            if w.domain == domain and w.subdomain == subdomain:
                return w
        return WeaknessReport(domain=domain, subdomain=subdomain, severity=0.5)

    def _star_sample_batch(
        self, items: list[dict], weakness: WeaknessReport,
    ) -> tuple[list[TrainingSample], int, list[dict], list[PreferencePair]]:
        """Sample K chains per item, grade each, keep correct ones + emit DPO pairs.

        Returns (kept_samples, rejected_count, items_with_zero_correct, preference_pairs).
        Preference pairs: for each problem with ≥1 correct AND ≥1 incorrect chain,
        emit up to MAX_DPO_PAIRS_PER_PROBLEM pairs (correct × incorrect). Chains are
        rendered via TrainingSample._build_full_response so tokenizer conventions
        match between chosen/rejected sides.
        """
        if not items:
            return [], 0, [], []
        k = max(1, int(self.config.star_k_samples))
        prompts: list[str] = []
        for item in items:
            for _ in range(k):
                prompts.append(self._build_solution_prompt(item["question"], weakness))

        solutions = self._generate_batch_with_oom_retry(
            prompts,
            max_new_tokens=2048,
            temperature=self.config.star_temperature,
            top_p=self.config.top_p,
        )
        if len(solutions) < len(prompts):
            solutions = list(solutions) + [""] * (len(prompts) - len(solutions))

        kept: list[TrainingSample] = []
        rejected = 0
        zero_correct: list[dict] = []
        pairs: list[PreferencePair] = []
        for i, item in enumerate(items):
            start = i * k
            chain_solutions = solutions[start:start + k]
            correct_indices: list[int] = []
            incorrect_indices: list[int] = []
            parsed_chains: list[tuple[list[ReasoningStep], str]] = []
            for j, sol in enumerate(chain_solutions):
                parsed = self._parser.parse(sol)
                parsed_chains.append((parsed.chain, parsed.conclusion))
                if not parsed.conclusion or not parsed.chain:
                    continue
                if len(parsed.chain) < self.config.min_reasoning_steps:
                    continue
                try:
                    ok = bool(self._grader(parsed.conclusion, item["expected"], item["check_type"]))
                except Exception as e:
                    logger.debug(f"grader raised on candidate: {e}")
                    ok = False
                if ok:
                    correct_indices.append(j)
                else:
                    incorrect_indices.append(j)

            if not correct_indices:
                rejected += k
                zero_correct.append(item)
                continue

            agreement = len(correct_indices) / k

            # Build DPO pairs from this problem's K rollouts: each correct chain
            # paired with each incorrect chain (same prompt, chosen=correct,
            # rejected=incorrect). Capped to avoid combinatorial blowup.
            if correct_indices and incorrect_indices:
                pair_budget = self.MAX_DPO_PAIRS_PER_PROBLEM
                pair_count = 0
                for ci in correct_indices:
                    if pair_count >= pair_budget:
                        break
                    for ri in incorrect_indices:
                        if pair_count >= pair_budget:
                            break
                        chosen_str = self._render_chain_for_training(
                            item, parsed_chains[ci][0], parsed_chains[ci][1],
                        )
                        rejected_str = self._render_chain_for_training(
                            item, parsed_chains[ri][0], parsed_chains[ri][1],
                        )
                        if not chosen_str or not rejected_str:
                            continue
                        if chosen_str == rejected_str:
                            continue  # Guard against near-duplicate rollouts
                        pairs.append(PreferencePair(
                            prompt=item["question"],
                            chosen_response=chosen_str,
                            rejected_response=rejected_str,
                            domain=item["domain"],
                            target_weakness=f"{item['domain']}/{item['subdomain']}",
                            weight=agreement,  # more correct chains → higher pair weight
                            chosen_confidence=1.0,
                            rejected_confidence=0.0,
                            severity_at_generation=max(0.0, min(1.0, item["severity"])),
                        ))
                        pair_count += 1

            for j in correct_indices:
                chain, conclusion = parsed_chains[j]
                conclusion = self._postprocess_conclusion(conclusion, item["domain"], chain)
                sample = TrainingSample(
                    prompt=item["question"],
                    response=conclusion,
                    reasoning_chain=chain,
                    target_weakness=f"{item['domain']}/{item['subdomain']}",
                    domain=item["domain"],
                    severity_at_generation=max(0.0, min(1.0, item["severity"])),
                    expected_answer=item["expected"],
                    consistency_score=agreement,
                    consistency_votes=k,
                    ground_truth_check_type=item["check_type"],
                    ground_truth_verified=True,
                    source="star",
                    sample_index=j,
                )
                kept.append(sample)
            rejected += (k - len(correct_indices))
        return kept, rejected, zero_correct, pairs

    def _rationalize_batch(
        self, items: list[dict], result: DiagnosticResult,
    ) -> list[TrainingSample]:
        """STaR rationalization: generate with the canonical answer as a hint.

        We do NOT include the hint in the training example — the whole point
        is to teach the model to produce the same reasoning WITHOUT being told
        the answer. The generated chain is graded defensively.
        """
        if not items or self._grader is None:
            return []
        prompts: list[str] = []
        for item in items:
            weakness = self._weakness_for(result, item["domain"], item["subdomain"])
            prompts.append(self._build_rationalization_prompt(item, weakness))

        solutions = self._generate_batch_with_oom_retry(
            prompts,
            max_new_tokens=2048,
            temperature=self.config.star_temperature,
            top_p=self.config.top_p,
        )
        if len(solutions) < len(prompts):
            solutions = list(solutions) + [""] * (len(prompts) - len(solutions))

        kept: list[TrainingSample] = []
        for item, sol in zip(items, solutions):
            parsed = self._parser.parse(sol)
            if not parsed.chain or not parsed.conclusion:
                continue
            if len(parsed.chain) < self.config.min_reasoning_steps:
                continue
            try:
                ok = bool(self._grader(parsed.conclusion, item["expected"], item["check_type"]))
            except Exception:
                ok = False
            if not ok:
                continue
            stripped_chain = self._strip_hint_from_chain(parsed.chain, item["expected"])
            if not stripped_chain:
                continue
            conclusion = self._postprocess_conclusion(
                parsed.conclusion, item["domain"], stripped_chain,
            )
            sample = TrainingSample(
                prompt=item["question"],
                response=conclusion,
                reasoning_chain=stripped_chain,
                target_weakness=f"{item['domain']}/{item['subdomain']}",
                domain=item["domain"],
                severity_at_generation=max(0.0, min(1.0, item["severity"])),
                expected_answer=item["expected"],
                consistency_score=0.5,
                consistency_votes=1,
                ground_truth_check_type=item["check_type"],
                ground_truth_verified=True,
                source="star_rationalized",
                sample_index=0,
            )
            kept.append(sample)
        return kept

    def _build_rationalization_prompt(
        self, item: dict, weakness: WeaknessReport,
    ) -> str:
        """Prompt that reveals the canonical answer and asks for reasoning TO it.

        The hint is in the PROMPT, not in the training example. Downstream we
        only keep the chain + conclusion, so the model is trained to produce
        these rationales from the problem alone.
        """
        conclusion_guidance = {
            "code": (
                "Conclusion: <ONLY the complete function source code — no prose, "
                "no markdown fences, no explanation>"
            ),
            "math": (
                "Conclusion: <ONLY the symbolic or numeric answer, e.g. 'x = 5', "
                "'pi/2', '42' — NO prose>"
            ),
        }.get(item["domain"], "Conclusion: <your final answer>")

        return (
            f"<|im_start|>system\n"
            f"You are a precise {item['domain']} solver. You show all work in "
            f"structured numbered steps. You never skip reasoning.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Solve the following {item['domain']}/{item['subdomain']} problem.\n\n"
            f"PROBLEM:\n{item['question']}\n\n"
            f"HINT (for your reasoning only — do NOT mention the hint or that you were given one): "
            f"the correct final answer is: {item['expected']}\n\n"
            f"Produce the reasoning chain that NATURALLY arrives at this answer "
            f"from the problem alone. Do not reference the hint. Do not write "
            f"'given the answer is X' or 'since we know the answer'. Reason as if "
            f"you derived the answer yourself.\n\n"
            f"Use EXACTLY this format — one item per line:\n\n"
            f"Step 1: <action>\n"
            f"Justification: <why>\n"
            f"Assumptions: <assumptions or 'none'>\n\n"
            f"Step 2: <action>\n"
            f"Justification: <why>\n"
            f"Assumptions: <assumptions or 'none'>\n\n"
            f"... (minimum {self.config.min_reasoning_steps} steps)\n\n"
            f"{conclusion_guidance}\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"Step 1:"
        )

    @staticmethod
    def _strip_hint_from_chain(
        chain: list[ReasoningStep], expected: str,
    ) -> list[ReasoningStep]:
        """Remove steps that leak the hint.

        Drop any step whose content or justification contains giveaway phrases
        ("given the answer", "as hinted", etc.). If more than half the chain
        would be dropped, return [] so the caller rejects the sample.
        """
        giveaways = re.compile(
            r"(?:given\s+(?:the\s+)?(?:answer|hint)|since\s+we\s+know\s+the\s+answer|"
            r"we\s+are\s+told(?:\s+the\s+answer)?|as\s+hinted|from\s+the\s+hint|"
            r"the\s+hint\s+(?:says|tells\s+us)|the\s+problem\s+gives\s+us\s+the\s+answer)",
            re.IGNORECASE,
        )
        cleaned: list[ReasoningStep] = []
        for step in chain:
            if giveaways.search(step.content or "") or giveaways.search(step.justification or ""):
                continue
            cleaned.append(step)
        if len(cleaned) < max(1, len(chain) // 2):
            return []
        return cleaned

    def _compute_diversity(self, samples: list[TrainingSample]) -> dict:
        """Measure and report diversity of generated samples.

        Computes topic coverage (fraction of configured domains represented),
        reasoning style variation (spread of chain lengths), and subdomain coverage.
        """
        if not samples:
            return {"topic_coverage": 0.0, "unique_domains": 0,
                    "unique_subdomains": 0, "chain_length_spread": 0.0,
                    "avg_chain_length": 0.0}

        domains = {s.domain for s in samples}
        subdomains = {s.target_weakness for s in samples}
        chain_lengths = [len(s.reasoning_chain) for s in samples]
        avg_len = sum(chain_lengths) / len(chain_lengths)
        spread = (max(chain_lengths) - min(chain_lengths)) / max(avg_len, 1)

        # Topic coverage: what fraction of weakness-targeted domains are represented
        target_domains = {s.domain for s in samples if s.domain}
        all_domains = set()
        try:
            from ..utils.config import DiagnosticsConfig
            all_domains = set(DiagnosticsConfig().domains)
        except Exception:
            pass
        coverage = len(target_domains) / max(len(all_domains), len(target_domains), 1)

        return {
            "topic_coverage": coverage,
            "unique_domains": len(domains),
            "unique_subdomains": len(subdomains),
            "chain_length_spread": round(spread, 2),
            "avg_chain_length": round(avg_len, 1),
            "samples_per_domain": {d: sum(1 for s in samples if s.domain == d) for d in domains},
        }

    def _rebalance_domains(self, samples: list[TrainingSample]) -> list[TrainingSample]:
        """Cap over-represented domains so no single domain dominates training."""
        if not samples:
            return samples
        max_per_domain = max(10, int(len(samples) * self.MAX_DOMAIN_FRACTION))
        domain_buckets: dict[str, list[TrainingSample]] = {}
        for s in samples:
            domain_buckets.setdefault(s.domain, []).append(s)

        over_represented = {d for d, ss in domain_buckets.items() if len(ss) > max_per_domain}
        if not over_represented:
            return samples

        result: list[TrainingSample] = []
        for domain, domain_samples in domain_buckets.items():
            if domain in over_represented:
                domain_samples.sort(key=lambda s: s.severity_at_generation, reverse=True)
                kept = domain_samples[:max_per_domain]
                logger.info(
                    f"  Rebalance: capped {domain} from {len(domain_samples)} to {len(kept)} samples"
                )
                result.extend(kept)
            else:
                result.extend(domain_samples)
        return result

    def _accept_unique(self, sample: TrainingSample) -> bool:
        if sample.content_hash in self._seen_hashes:
            return False
        sig = _normalize_for_dedup(f"{sample.prompt} {sample.response}")
        for prior in self._seen_signatures:
            if _jaccard(sig, prior) >= self.DEDUP_JACCARD_THRESHOLD:
                return False
        self._seen_hashes.add(sample.content_hash)
        self._seen_signatures.append(sig)
        return True

    def _generate_for_weakness(self, weakness: WeaknessReport) -> list[TrainingSample]:
        num_samples = self._compute_num_samples(weakness)

        problem_prompts = [
            self._build_problem_generation_prompt(weakness, i)
            for i in range(num_samples)
        ]
        # Send all prompts in one generate_batch call for maximum throughput.
        # vLLM handles continuous batching internally; splitting into smaller
        # Python-side batches just adds round-trip overhead.
        problems: list[str] = self._generate_batch_with_oom_retry(
            problem_prompts, max_new_tokens=512,
            temperature=self.config.temperature, top_p=self.config.top_p,
        )
        if len(problems) < len(problem_prompts):
            problems = list(problems) + [""] * (len(problem_prompts) - len(problems))

        valid_problems = [p for p in (self._clean_problem(p) for p in problems) if p]
        if not valid_problems:
            logger.warning(f"{weakness.domain}/{weakness.subdomain}: no valid problems after cleaning")
            return []

        first_prompts = [self._build_solution_prompt(p, weakness) for p in valid_problems]
        first_solutions: list[str] = self._generate_batch_with_oom_retry(
            first_prompts, max_new_tokens=2048, temperature=0.3,
        )
        if len(first_solutions) < len(first_prompts):
            first_solutions = list(first_solutions) + [""] * (len(first_prompts) - len(first_solutions))

        samples: list[TrainingSample] = []
        parse_failures = 0
        schema_failures = 0
        # First pass: build samples, collect failures for batched retry
        retry_items: list[tuple[int, str, str]] = []  # (index, problem, solution)
        first_pass_samples: list[TrainingSample | None] = []
        for i, (problem, solution) in enumerate(zip(valid_problems, first_solutions)):
            sample = self._build_sample(problem, weakness, [solution])
            first_pass_samples.append(sample)
            if sample is None:
                retry_items.append((i, problem, solution))

        # Batched retry: build all retry prompts at once and send as one batch
        if retry_items:
            retry_prompts = []
            for _, problem, prev_solution in retry_items:
                prev_result = self._parser.parse(prev_solution)
                prompt = self._build_adjusted_retry_prompt(problem, prev_solution, weakness, prev_result)
                retry_prompts.append(prompt)
            # Use a single batch call for all retries
            if retry_prompts:
                retry_solutions = self._generate_batch_with_oom_retry(
                    retry_prompts, max_new_tokens=2048, temperature=0.45,
                )
                if len(retry_solutions) < len(retry_prompts):
                    retry_solutions = list(retry_solutions) + [""] * (len(retry_prompts) - len(retry_solutions))
                for (idx, problem, prev_solution), retry_sol in zip(retry_items, retry_solutions):
                    sample = self._build_sample(problem, weakness, [prev_solution, retry_sol])
                    first_pass_samples[idx] = sample

        valid_for_answer: list[TrainingSample] = []
        for sample in first_pass_samples:
            if sample is None:
                parse_failures += 1
                continue
            validation = validate_sample(sample, self.config.min_reasoning_steps)
            if not validation.ok:
                schema_failures += 1
                logger.debug(f"schema failure: {validation.issues}")
                continue
            valid_for_answer.append(sample)

        # Batch-generate expected answers for math/code domains
        if valid_for_answer and weakness.domain in ("math", "code"):
            self._fill_expected_answers(valid_for_answer, weakness)

        # Self-consistency: only runs when consistency_samples > 1.
        # Generates (N-1) additional independent solutions per sample and
        # rejects anything below the agreement threshold. The sample's
        # consistency_score is set to the agreement fraction (used downstream
        # as a training weight multiplier).
        consistency_rejected = 0
        if valid_for_answer and getattr(self.config, "consistency_samples", 1) > 1:
            valid_for_answer, consistency_rejected = self._apply_self_consistency(
                valid_for_answer, weakness, valid_problems,
            )

        samples.extend(valid_for_answer)

        logger.info(
            f"{weakness.domain}/{weakness.subdomain}: "
            f"{len(samples)}/{len(valid_problems)} samples "
            f"(parse_fail={parse_failures}, schema_fail={schema_failures}"
            + (f", consistency_fail={consistency_rejected}" if consistency_rejected else "")
            + ")"
        )
        return samples

    def _apply_self_consistency(
        self,
        samples: list[TrainingSample],
        weakness: WeaknessReport,
        problems: list[str],
    ) -> tuple[list[TrainingSample], int]:
        """Verify each sample by sampling (N-1) more solutions and checking agreement.

        For each sample we generate (N-1) additional independent solutions to
        the same problem at moderate temperature. We extract each solution's
        final answer, normalize it, and count agreement with the sample's
        existing answer. Samples below consistency_threshold are rejected.

        Returns (kept_samples, rejected_count).
        """
        n = int(self.config.consistency_samples)
        threshold = float(self.config.consistency_threshold)
        extra_per = max(0, n - 1)
        if extra_per == 0:
            return samples, 0

        # Build batched retry prompts: for each sample, extra_per independent
        # solutions. We interleave so order is [s0_try1, s0_try2, s1_try1, ...]
        prompts: list[str] = []
        for s in samples:
            for _ in range(extra_per):
                prompts.append(self._build_solution_prompt(s.prompt, weakness))

        # Use a temperature above the first-pass 0.3 so the extra samples are
        # genuinely independent — low-temp resampling produces near-duplicates,
        # which inflates agreement scores without catching mode collapse.
        extra_solutions = self._generate_batch_with_oom_retry(
            prompts,
            max_new_tokens=2048,
            temperature=max(0.6, self.config.temperature),
            top_p=self.config.top_p,
        )
        if len(extra_solutions) < len(prompts):
            extra_solutions = list(extra_solutions) + [""] * (len(prompts) - len(extra_solutions))

        kept: list[TrainingSample] = []
        rejected = 0
        for i, sample in enumerate(samples):
            primary_answer = self._normalize_answer_for_vote(
                sample.expected_answer or sample.response, weakness.domain,
            )
            votes = [primary_answer] if primary_answer else []
            parseable = 1 if primary_answer else 0

            start = i * extra_per
            for extra in extra_solutions[start:start + extra_per]:
                if not extra:
                    continue
                parsed = self._parser.parse(extra)
                candidate = parsed.conclusion if parsed.conclusion else ""
                if not candidate:
                    continue
                normalized = self._normalize_answer_for_vote(candidate, weakness.domain)
                if not normalized:
                    continue
                votes.append(normalized)
                parseable += 1

            # Agreement = fraction of parseable votes matching the plurality answer.
            if len(votes) == 0:
                sample.consistency_score = 0.0
                sample.consistency_votes = n
                rejected += 1
                continue

            counts: dict[str, int] = {}
            for v in votes:
                counts[v] = counts.get(v, 0) + 1
            top_answer, top_count = max(counts.items(), key=lambda kv: kv[1])
            agreement = top_count / n  # against the INTENDED vote count, not just parseable
            sample.consistency_score = agreement
            sample.consistency_votes = n

            if agreement < threshold:
                rejected += 1
                continue
            # If the plurality disagrees with the primary, prefer the plurality
            # answer — more voters agreed on it.
            if top_answer != primary_answer and weakness.domain in ("math", "code"):
                sample.expected_answer = top_answer
            kept.append(sample)

        return kept, rejected

    @staticmethod
    def _normalize_answer_for_vote(text: str, domain: str) -> str:
        """Reduce a conclusion to a canonical form for agreement counting.

        Cheap normalization that catches most equivalent-answer cases without
        requiring sympy-level symbolic equality. The goal is to flag clear
        agreement; marginal cases should fail open (get rejected) rather than
        be falsely counted as agreement.
        """
        if not text:
            return ""
        t = text.strip().lower()
        # Strip common preambles ("the answer is ...", "final: ...", etc.)
        t = re.sub(r"^(the\s+)?(final\s+)?(answer|result|output|value|conclusion)\s*[:=]\s*", "", t)
        t = re.sub(r"^\\boxed\s*\{\s*", "", t)
        t = re.sub(r"\s*\}\s*$", "", t)
        t = t.strip(" .`\"'")
        # For math: extract the first number if one exists and the rest is short.
        if domain == "math":
            m = re.search(r"-?\d+(?:\.\d+)?(?:/\d+)?", t)
            if m and len(t) <= len(m.group(0)) + 10:
                return m.group(0)
        # For code: keep first line (most code answers are a single expression/return).
        if domain == "code":
            first_line = t.splitlines()[0] if t else ""
            # Collapse whitespace inside the line to fold formatting-only diffs.
            return re.sub(r"\s+", " ", first_line).strip()
        # Generic: collapse whitespace, trim to first ~80 chars for robust comparison.
        return re.sub(r"\s+", " ", t)[:80]

    def _build_sample(
        self, problem: str, weakness: WeaknessReport, attempts: list[str],
    ) -> TrainingSample | None:
        latest = attempts[-1]
        result = self._parser.parse(latest)
        if len(result.chain) < self.config.min_reasoning_steps:
            return None
        conclusion = result.conclusion
        if not conclusion:
            for prev in reversed(attempts[:-1]):
                prev_result = self._parser.parse(prev)
                if prev_result.conclusion:
                    conclusion = prev_result.conclusion
                    break
        if not conclusion:
            return None
        conclusion = self._postprocess_conclusion(conclusion, weakness.domain, result.chain)
        return TrainingSample(
            prompt=problem,
            response=conclusion,
            reasoning_chain=result.chain,
            target_weakness=f"{weakness.domain}/{weakness.subdomain}",
            domain=weakness.domain,
            parse_confidence=result.confidence,
            parse_issues=result.issues,
            severity_at_generation=max(0.0, min(1.0, weakness.severity)),
        )

    def _fill_expected_answers(
        self, samples: list[TrainingSample], weakness: WeaknessReport,
    ) -> None:
        """Batch-generate short expected answers so the verifier can cross-check."""
        if weakness.domain == "math":
            prompts = [
                f"<|im_start|>system\nYou are a math answer extractor. "
                f"Output ONLY the final numeric or symbolic answer, nothing else.<|im_end|>\n"
                f"<|im_start|>user\nProblem: {s.prompt}\n\n"
                f"Solution conclusion: {s.response}\n\n"
                f"What is the final answer? Output ONLY the value (e.g. '42', 'x=5', 'pi/2')."
                f"<|im_end|>\n<|im_start|>assistant\n"
                for s in samples
            ]
        elif weakness.domain == "code":
            prompts = [
                f"<|im_start|>system\nYou extract expected test outputs from code problems. "
                f"Output ONLY the expected function behavior summary.<|im_end|>\n"
                f"<|im_start|>user\nProblem: {s.prompt}\n\n"
                f"Output ONLY the expected return values from the test cases, one per line."
                f"<|im_end|>\n<|im_start|>assistant\n"
                for s in samples
            ]
        else:
            return

        answers = self._generate_batch_with_oom_retry(
            prompts, max_new_tokens=128, temperature=0.1,
        )
        if len(answers) < len(samples):
            answers = list(answers) + [""] * (len(samples) - len(answers))

        for sample, answer in zip(samples, answers):
            cleaned = answer.strip()
            if cleaned and not _is_gibberish(cleaned) and not _is_refusal(cleaned):
                sample.expected_answer = cleaned

    @staticmethod
    def _postprocess_conclusion(conclusion: str, domain: str, chain: list[ReasoningStep]) -> str:
        """Domain-specific cleanup so the verifier can extract+execute `response`."""
        if domain == "code":
            # Strip markdown code fences; keep the code body verbatim.
            m = re.search(r'```(?:python|py)?\s*\n?(.*?)```', conclusion, re.DOTALL)
            if m:
                return m.group(1).strip()
            # If the conclusion is just a prose pointer but the final step has
            # a code-shaped body, prefer the final step's content.
            if chain and not re.search(r'\bdef\s+\w+\s*\(', conclusion):
                last = chain[-1].content
                m2 = re.search(r'```(?:python|py)?\s*\n?(.*?)```', last, re.DOTALL)
                if m2:
                    return m2.group(1).strip()
                if re.search(r'\bdef\s+\w+\s*\(', last):
                    return last.strip()
            return conclusion.strip()
        if domain == "math":
            c = conclusion.strip()
            # Strip leading prose like "The answer is" / "So we get" / "Therefore,"
            c = re.sub(
                r'^(?:the\s+answer\s+is|the\s+result\s+is|so\s+we\s+get|'
                r'therefore[,:]?|thus[,:]?|hence[,:]?|we\s+get|we\s+find)\s*',
                '', c, flags=re.IGNORECASE,
            ).strip()
            # Strip trailing period.
            c = re.sub(r'\.$', '', c).strip()
            # Strip inline code/math backticks or $...$
            c = c.strip('`').strip()
            c = re.sub(r'^\$+|\$+$', '', c).strip()
            return c
        return conclusion.strip()

    def _clean_problem(self, raw: str) -> str:
        """Strip solution contamination; drop the problem if contamination starts
        at the top (nothing usable precedes it).
        """
        p = (raw or "").strip()
        if not p:
            return ""
        if _is_refusal(p):
            return ""
        if _is_gibberish(p):
            return ""
        if _LEADING_CONTAM_RE.match(p):
            return ""
        m = _CONTAM_PATTERNS.search(p)
        if not m:
            return p
        if m.start() > 20:
            return p[:m.start()].strip()
        return ""

    _STYLE_ANGLES = [
        "numeric edge case",
        "adversarial distractor",
        "multi-step composition",
        "symbolic generalization",
        "inverse problem",
    ]

    def _build_problem_generation_prompt(self, weakness: WeaknessReport, index: int) -> str:
        import hashlib
        difficulty = getattr(weakness, "difficulty", "mixed") or "mixed"
        target_difficulty = difficulty if difficulty != "mixed" else "challenging"
        overconfident = getattr(weakness, "calibrated_confidence", 0.0) > 0.6
        style = self._STYLE_ANGLES[index % len(self._STYLE_ANGLES)]
        variation_key = hashlib.md5(
            f"{weakness.domain}:{weakness.subdomain}:{weakness.severity}:{index}".encode()
        ).hexdigest()[:8]

        parts = [
            f"<|im_start|>system\n"
            f"You are an expert problem author for {weakness.domain}/{weakness.subdomain}. "
            f"You write self-contained problems with unambiguous correct answers.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Generate ONE {target_difficulty} {weakness.subdomain} problem (#{index + 1}).\n\n"
            f"Requirements:\n"
            f"- Emphasize: {style}\n"
            f"- Must require multi-step reasoning to solve\n"
            f"- Must be self-contained with a definite correct answer\n"
            f"- variation_key: {variation_key}\n"
        ]

        if overconfident:
            parts.append(
                "\nIMPORTANT: The solver has been OVERCONFIDENT in wrong answers here. "
                "Design a problem where a plausible-looking shortcut leads to a WRONG answer, "
                "so the correct approach requires careful, non-obvious reasoning.\n"
            )

        usable_evidence = [e for e in weakness.evidence if isinstance(e, dict) and e.get('question')]
        if usable_evidence:
            failed = usable_evidence[(index * 7 + 3) % len(usable_evidence)]
            parts.append(
                f"\nPreviously failed question (for reference — generate a DIFFERENT problem "
                f"testing the same skill):\n"
                f"Q: {failed.get('question', 'N/A')}\n"
                f"Expected: {failed.get('expected', 'N/A')}\n"
            )

        # Domain-specific shaping
        if weakness.domain == "code":
            parts.append(
                "\nDomain constraints:\n"
                "- MUST be a function-implementation task\n"
                "- Specify the function signature explicitly (e.g. `def func_name(args) -> type:`)\n"
                "- Include 2-4 test cases in EXACTLY this form:\n"
                "  Test: func_name(input) == expected_output\n"
                "- Do NOT include the solution\n"
            )
        elif weakness.domain == "math":
            parts.append(
                "\nDomain constraints:\n"
                "- The answer MUST be a single clean numeric or symbolic value "
                "(e.g. 'x = 5', 'pi/2', '42')\n"
                "- NOT a prose explanation\n"
                "- Do NOT include the solution\n"
            )
        else:
            parts.append(
                "\nDomain constraints:\n"
                "- The answer must be concise and verifiable\n"
                "- Do NOT include the solution\n"
            )

        parts.append(
            "\nOutput ONLY the problem statement, nothing else. "
            "Do not add preamble like 'Here is a problem:' or 'Sure!'."
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        return "".join(parts)

    def _build_solution_prompt(self, problem: str, weakness: WeaknessReport) -> str:
        if self._custom_solution_template:
            try:
                return self._custom_solution_template.format(
                    problem=problem,
                    domain=weakness.domain,
                    subdomain=weakness.subdomain,
                )
            except (KeyError, ValueError) as e:
                logger.warning(f"Custom solution template failed ({e}) — using default")
                self._custom_solution_template = None

        conclusion_guidance = {
            "code": (
                "Conclusion: <ONLY the complete function source code — no prose, "
                "no markdown fences, no explanation>"
            ),
            "math": (
                "Conclusion: <ONLY the symbolic or numeric answer, e.g. 'x = 5', "
                "'pi/2', '42' — NO prose>"
            ),
        }.get(weakness.domain, "Conclusion: <your final answer>")

        return (
            f"<|im_start|>system\n"
            f"You are a precise {weakness.domain} solver. You show all work in "
            f"structured numbered steps. You never skip reasoning.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Solve the following {weakness.domain}/{weakness.subdomain} problem.\n\n"
            f"PROBLEM:\n{problem}\n\n"
            f"You MUST use EXACTLY this format — one item per line:\n\n"
            f"Step 1: <what you are doing in this step>\n"
            f"Justification: <why this step follows logically from prior steps or premises>\n"
            f"Assumptions: <any assumptions made, or 'none'>\n\n"
            f"Step 2: <next action>\n"
            f"Justification: <reasoning>\n"
            f"Assumptions: <assumptions or 'none'>\n\n"
            f"... (continue for all steps)\n\n"
            f"{conclusion_guidance}\n\n"
            f"Rules:\n"
            f"- Minimum {self.config.min_reasoning_steps} steps. More is better.\n"
            f"- NO skipping steps. Every logical leap must be its own step.\n"
            f"- If uncertain about any step, state your uncertainty in the Justification.\n"
            f"- Populate Assumptions explicitly when grounding is non-trivial "
            f"(do NOT write 'none' if you relied on a precondition).\n"
            f"- Use the EXACT prefixes 'Step N:', 'Justification:', 'Assumptions:', 'Conclusion:'.\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"Step 1:"
        )

    def _build_adjusted_retry_prompt(
        self, problem: str, prev_solution: str, weakness: WeaknessReport,
        prev_result: ParseResult,
    ) -> str:
        diagnostic_lines: list[str] = []
        if not prev_result.chain:
            diagnostic_lines.append(
                "Your previous attempt had NO parseable steps. Use the exact prefix 'Step N:' "
                "on its own line for each step."
            )
        elif len(prev_result.chain) < self.config.min_reasoning_steps:
            diagnostic_lines.append(
                f"Your previous attempt had only {len(prev_result.chain)} step(s); "
                f"minimum is {self.config.min_reasoning_steps}. Break every inference into a separate step."
            )
        if not prev_result.conclusion:
            diagnostic_lines.append(
                "Your previous attempt had no 'Conclusion:' line. You MUST end with 'Conclusion: <your final answer>'."
            )
        if prev_result.chain and not any(
            s.justification and s.justification != "implicit" for s in prev_result.chain
        ):
            diagnostic_lines.append(
                "Your previous attempt had no 'Justification:' lines. Every step needs one."
            )
        if any("refusal" in issue for issue in prev_result.issues):
            diagnostic_lines.append(
                "Your previous attempt was a refusal. You MUST attempt to solve the problem. "
                "If you are uncertain, show your best-effort reasoning with stated uncertainty."
            )
        if not diagnostic_lines:
            diagnostic_lines.append("Your previous attempt could not be parsed; follow the format exactly.")

        truncated_prev = self._truncate_at_boundary(prev_solution, 800) if prev_solution else ""

        parts = [
            f"<|im_start|>system\n"
            f"You are a precise {weakness.domain} solver. Your previous answer "
            f"could not be parsed. Follow the format EXACTLY this time.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"PROBLEM:\n{problem}\n\n"
            f"PARSE ERRORS from your previous attempt:\n- " + "\n- ".join(diagnostic_lines) + "\n\n",
        ]
        if truncated_prev:
            parts.append(f"YOUR PREVIOUS ATTEMPT (for reference):\n{truncated_prev}\n\n")
        parts.append(
            f"REQUIRED FORMAT — use these EXACT prefixes, one item per line:\n\n"
            f"Step 1: <action>\n"
            f"Justification: <why>\n"
            f"Assumptions: <what you assume, or 'none'>\n\n"
            f"Step 2: <action>\n"
            f"Justification: <why>\n"
            f"Assumptions: <what you assume, or 'none'>\n\n"
            f"... (minimum {self.config.min_reasoning_steps} steps)\n\n"
            f"Conclusion: <your final answer>\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"Step 1:"
        )
        return "".join(parts)

    @staticmethod
    def _truncate_at_boundary(text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text
        truncated = text[:max_len]
        last_nl = truncated.rfind("\n")
        if last_nl > max_len // 2:
            return truncated[:last_nl] + "..."
        for sep in (". ", "! ", "? "):
            last_sep = truncated.rfind(sep)
            if last_sep > max_len // 2:
                return truncated[:last_sep + 1] + "..."
        last_space = truncated.rfind(" ")
        if last_space > max_len // 2:
            return truncated[:last_space] + "..."
        return truncated + "..."

    # Backward-compat shims for any existing callers/tests.
    def _parse_reasoning_chain(self, solution: str) -> list[ReasoningStep]:
        return self._parser.parse(solution).chain

    def _extract_conclusion(self, solution: str) -> str:
        return self._parser.parse(solution).conclusion
