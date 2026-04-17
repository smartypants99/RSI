"""Process Reward Model (PRM) — dense per-step rewards for reasoning chains.

Based on Lightman et al. 2023 ("Let's Verify Step by Step"). Outcome-only
rewards are sparse: one bit of signal per rollout. A PRM scores EVERY step,
giving the RL engine dense signal that dramatically accelerates GRPO.

Architecture:
  * PRMHead: 2-layer MLP classifier over the base model's hidden state at each
    "Step N:" marker position. Outputs P(step is correct) in [0, 1].
  * Self-labeling: derives step-level labels from outcome + Monte Carlo rollouts.
  * Training: binary cross-entropy on labeled (step, context) pairs.

VRAM budget (A6000, 7B base model):
  * Base hidden size ~4096 → head params ≈ 4096*512 + 512*1 ≈ 2.1M → ~8MB fp32
  * Activations for training: batch_size×seq_len×hidden ≈ tens of MB per batch
  * Total extra VRAM over base model: <4GB in practice
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..generator.data_generator import TrainingSample
from ..utils.config import TrainerConfig
from ..utils.model_loader import ModelLoader

logger = logging.getLogger(__name__)

STEP_MARKER_RE = re.compile(r"(?mi)^\s*step\s*(\d+)\s*[:.)\-]")


# =============================================================================
# Head
# =============================================================================


class PRMHead(nn.Module):
    """2-layer MLP that maps a hidden state to P(step correct).

    Kept lightweight deliberately: the expressive power comes from the base
    model's frozen representation — the head only needs to project it to a
    scalar correctness score. A bigger head would risk overfitting the small
    self-labeled dataset produced each cycle.
    """

    def __init__(self, hidden_size: int, mlp_size: int = 512, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_size)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mlp_size, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: (..., hidden_size) → logits (...,)."""
        x = self.fc1(hidden.float())
        x = self.act(x)
        x = self.drop(x)
        return self.fc2(x).squeeze(-1)


# =============================================================================
# Label generation
# =============================================================================


@dataclass
class PRMStepLabel:
    """One (context, step, label) training tuple for the PRM."""
    prompt: str
    chain_so_far: list[str]   # all prior step texts
    step_text: str
    label: float              # in [0, 1]; 0.5 means ambiguous (dropped by default)
    source: str = "outcome"   # "outcome" | "mc" | "manual"


@dataclass
class PRMTrainingMetrics:
    cycle: int
    avg_loss: float
    final_loss: float
    steps: int
    labels_used: int
    labels_skipped: int
    learning_rate: float
    positive_fraction: float = 0.0


def _extract_steps(sample: TrainingSample) -> list[str]:
    """Pull ordered step texts from a TrainingSample.

    Prefer the structured reasoning_chain. Fall back to regex-split on the raw
    response so samples that bypassed the parser (e.g., manual curation) still
    yield labels.
    """
    if sample.reasoning_chain:
        return [s.content for s in sample.reasoning_chain]

    text = sample.response or ""
    if not text:
        return []
    marks = list(STEP_MARKER_RE.finditer(text))
    if not marks:
        return [text.strip()] if text.strip() else []
    steps: list[str] = []
    for i, m in enumerate(marks):
        start = m.end()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            steps.append(chunk)
    return steps


def label_from_outcome(sample: TrainingSample, *, drop_ambiguous: bool = True
                       ) -> list[PRMStepLabel]:
    """Self-label a sample's steps using only the outcome (verified flag).

    Rules (weakest-possible supervision, per Lightman §3.1):
      * verified=True → all steps labeled 1.0 ("good").
      * verified=False → the LAST step is labeled 0.0 ("bad"); earlier steps are
        ambiguous. If drop_ambiguous (default), we skip the earlier steps rather
        than label them 0.5; a 0.5 sigmoid target teaches the head to hedge.
    """
    steps = _extract_steps(sample)
    if not steps:
        return []
    labels: list[PRMStepLabel] = []
    if sample.verified:
        for i, st in enumerate(steps):
            labels.append(PRMStepLabel(
                prompt=sample.prompt,
                chain_so_far=steps[:i],
                step_text=st,
                label=1.0,
                source="outcome",
            ))
    else:
        if not drop_ambiguous:
            for i, st in enumerate(steps[:-1]):
                labels.append(PRMStepLabel(
                    prompt=sample.prompt,
                    chain_so_far=steps[:i],
                    step_text=st,
                    label=0.5,
                    source="outcome",
                ))
        last = steps[-1]
        labels.append(PRMStepLabel(
            prompt=sample.prompt,
            chain_so_far=steps[:-1],
            step_text=last,
            label=0.0,
            source="outcome",
        ))
    return labels


RolloutFn = Callable[[str, list[str], str, int], list[bool]]
"""Rollout signature: (prompt, chain_so_far, candidate_step, n) -> list of booleans
indicating whether each of n sampled continuations reached the correct answer."""


def label_from_monte_carlo(
    prompt: str,
    chain: list[str],
    *,
    rollout_fn: RolloutFn,
    n_rollouts: int = 8,
    threshold: float = 0.5,
) -> list[PRMStepLabel]:
    """Label each step in a chain by Monte Carlo continuation (Lightman §3.2).

    For each prefix ending at step k, sample ``n_rollouts`` continuations from
    that prefix. The step is "good" if the fraction that reach a correct final
    answer is ≥ threshold. This is the gold-standard PRM label when an oracle
    verifier is available (we have one — the diagnostics verifier).

    rollout_fn returns an iterable of booleans (hit/miss); its implementation
    lives in rl_engine / generator to avoid a circular dep here.
    """
    out: list[PRMStepLabel] = []
    for i, step in enumerate(chain):
        hits = rollout_fn(prompt, chain[:i], step, n_rollouts)
        if not hits:
            continue
        good_frac = sum(1 for h in hits if h) / len(hits)
        out.append(PRMStepLabel(
            prompt=prompt,
            chain_so_far=chain[:i],
            step_text=step,
            label=1.0 if good_frac >= threshold else 0.0,
            source="mc",
        ))
    return out


# =============================================================================
# Dataset
# =============================================================================


class PRMDataset(Dataset):
    """Encodes (prompt + chain_so_far + step) sequences and marks the last-token
    position for the head to read.

    We use the hidden state of the final token of ``step_text`` as the feature
    for classification. This matches Lightman's setup and avoids needing to find
    an explicit "Step N:" token that may be split by BPE.
    """

    def __init__(self, labels: list[PRMStepLabel], tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._encoded: list[dict] = []
        self._encode(labels)

    def _encode(self, labels: list[PRMStepLabel]) -> None:
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        for lab in labels:
            prefix_parts = [lab.prompt, ""]
            for i, st in enumerate(lab.chain_so_far, 1):
                prefix_parts.append(f"Step {i}: {st}")
            prefix_text = "\n".join(prefix_parts).rstrip() + "\n"
            step_header = f"Step {len(lab.chain_so_far) + 1}: "
            step_full = step_header + lab.step_text.strip()

            prefix_ids = self.tokenizer(
                prefix_text, add_special_tokens=True, return_tensors="pt",
            )["input_ids"][0]
            step_ids = self.tokenizer(
                step_full, add_special_tokens=False, return_tensors="pt",
            )["input_ids"][0]

            if eos_id is not None and len(prefix_ids) > 1 and prefix_ids[-1].item() == eos_id:
                prefix_ids = prefix_ids[:-1]

            combined = torch.cat([prefix_ids, step_ids])
            if len(combined) > self.max_length:
                # Truncate from the FRONT to preserve the step (the thing we're
                # classifying). Keep at least the last 8 prompt tokens so the
                # model has some context.
                keep = self.max_length
                combined = combined[-keep:]
                score_pos = len(combined) - 1
            else:
                score_pos = len(combined) - 1

            if score_pos < 0:
                continue

            attention_mask = torch.ones(len(combined), dtype=torch.long)
            if len(combined) < self.max_length:
                pad_len = self.max_length - len(combined)
                combined = torch.cat([
                    combined,
                    torch.full((pad_len,), pad_id, dtype=combined.dtype),
                ])
                attention_mask = torch.cat([
                    attention_mask,
                    torch.zeros(pad_len, dtype=torch.long),
                ])

            self._encoded.append({
                "input_ids": combined,
                "attention_mask": attention_mask,
                "score_pos": torch.tensor(score_pos, dtype=torch.long),
                "label": torch.tensor(lab.label, dtype=torch.float32),
            })

    def __len__(self) -> int:
        return len(self._encoded)

    def __getitem__(self, idx):
        return self._encoded[idx]


# =============================================================================
# PRM
# =============================================================================


class PRM:
    """Process Reward Model: wraps a PRMHead on top of a frozen base model.

    The base model is provided by ModelLoader (shared with the main trainer).
    The head lives in fp32, moved to the base model's device. Base weights stay
    frozen — we only train the head. This keeps extra VRAM <4GB on an A6000.

    Integration with rl_engine:
      rl_engine's reward_fn should call score_chain(prompt, chain) to get
      per-step floats, then use those as dense advantages for GRPO. Call
      train_from_samples() once per RSI cycle on accumulated TrainingSamples.
    """

    def __init__(
        self,
        model_loader: ModelLoader,
        trainer_config: TrainerConfig,
        *,
        mlp_size: int = 512,
        dropout: float = 0.1,
    ):
        self.model_loader = model_loader
        self.trainer_config = trainer_config
        base = model_loader.model
        hidden_size = self._infer_hidden_size(base)
        self.head = PRMHead(hidden_size, mlp_size=mlp_size, dropout=dropout)
        device = model_loader.device
        self.head.to(device=device, dtype=torch.float32)
        self._device = device
        self._hidden_size = hidden_size
        self._max_length = model_loader.config.max_seq_length

    @staticmethod
    def _infer_hidden_size(model) -> int:
        cfg = getattr(model, "config", None)
        for attr in ("hidden_size", "n_embd", "d_model"):
            val = getattr(cfg, attr, None)
            if isinstance(val, int) and val > 0:
                return val
        raise RuntimeError("Could not infer hidden size from base model config")

    # -------------------------- inference --------------------------

    @torch.no_grad()
    def _hidden_at(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                   score_pos: torch.Tensor) -> torch.Tensor:
        """Forward the base model and gather hidden states at score_pos.

        input_ids: (batch, seq), score_pos: (batch,) → (batch, hidden).
        """
        base = self.model_loader.model
        base.eval()
        outputs = base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        # last layer hidden state
        hs = outputs.hidden_states[-1]  # (batch, seq, hidden)
        idx = score_pos.view(-1, 1, 1).expand(-1, 1, hs.size(-1))
        gathered = hs.gather(1, idx).squeeze(1)  # (batch, hidden)
        return gathered

    @torch.no_grad()
    def score_step(self, prompt: str, chain_so_far: list[str], step_text: str) -> float:
        """Return P(step is correct) in [0, 1] for a single step."""
        self.head.eval()
        label = PRMStepLabel(prompt=prompt, chain_so_far=list(chain_so_far),
                             step_text=step_text, label=0.0, source="infer")
        ds = PRMDataset([label], self.model_loader.tokenizer, self._max_length)
        if len(ds) == 0:
            return 0.5
        item = ds[0]
        input_ids = item["input_ids"].unsqueeze(0).to(self._device)
        attn = item["attention_mask"].unsqueeze(0).to(self._device)
        pos = item["score_pos"].unsqueeze(0).to(self._device)
        hidden = self._hidden_at(input_ids, attn, pos)
        logit = self.head(hidden)
        return float(torch.sigmoid(logit).item())

    @torch.no_grad()
    def score_chain(self, prompt: str, chain: list[str]) -> list[float]:
        """Return per-step P(correct) for every step in the chain."""
        if not chain:
            return []
        labels = [
            PRMStepLabel(prompt=prompt, chain_so_far=chain[:i],
                         step_text=step, label=0.0, source="infer")
            for i, step in enumerate(chain)
        ]
        ds = PRMDataset(labels, self.model_loader.tokenizer, self._max_length)
        if len(ds) == 0:
            return [0.5] * len(chain)
        loader = DataLoader(ds, batch_size=max(1, self.trainer_config.batch_size),
                            shuffle=False)
        self.head.eval()
        scores: list[float] = []
        for batch in loader:
            hidden = self._hidden_at(
                batch["input_ids"].to(self._device),
                batch["attention_mask"].to(self._device),
                batch["score_pos"].to(self._device),
            )
            logits = self.head(hidden)
            scores.extend(torch.sigmoid(logits).detach().cpu().float().tolist())
        # ds may have dropped items with score_pos<0; pad output to len(chain).
        if len(scores) < len(chain):
            scores.extend([0.5] * (len(chain) - len(scores)))
        return scores[:len(chain)]

    # -------------------------- training --------------------------

    def train_from_samples(
        self,
        samples: list[TrainingSample],
        *,
        cycle: int = 0,
        rollout_fn: Optional[RolloutFn] = None,
        n_rollouts: int = 8,
    ) -> PRMTrainingMetrics:
        """Collect labels, then run binary-cross-entropy training on the head.

        Label sources:
          * Always: outcome-based labels from sample.verified (drops ambiguous).
          * Optional: if ``rollout_fn`` is provided, MC labels for every step of
            every sample (richer, denser signal). Use this for GRPO rollouts
            where the rl_engine already has a resample-continuation function.
        """
        labels: list[PRMStepLabel] = []
        for s in samples:
            labels.extend(label_from_outcome(s, drop_ambiguous=True))
            if rollout_fn is not None:
                chain = _extract_steps(s)
                if chain:
                    labels.extend(label_from_monte_carlo(
                        s.prompt, chain, rollout_fn=rollout_fn,
                        n_rollouts=n_rollouts,
                    ))
        return self.train_from_labels(labels, cycle=cycle)

    def train_from_labels(
        self, labels: list[PRMStepLabel], *, cycle: int = 0,
    ) -> PRMTrainingMetrics:
        cfg = self.trainer_config
        prm_lr = getattr(cfg, "prm_lr", 1e-4)
        prm_epochs = getattr(cfg, "prm_epochs", 1)

        if not labels:
            return PRMTrainingMetrics(
                cycle=cycle, avg_loss=0.0, final_loss=0.0, steps=0,
                labels_used=0, labels_skipped=0, learning_rate=prm_lr,
            )

        ds = PRMDataset(labels, self.model_loader.tokenizer, self._max_length)
        skipped = len(labels) - len(ds)
        if len(ds) == 0:
            return PRMTrainingMetrics(
                cycle=cycle, avg_loss=0.0, final_loss=0.0, steps=0,
                labels_used=0, labels_skipped=skipped, learning_rate=prm_lr,
            )

        loader = DataLoader(ds, batch_size=max(1, cfg.batch_size), shuffle=True,
                            drop_last=False)

        # Class imbalance: downstream labeling tends to produce many more
        # positives than negatives on early cycles (most steps in a verified
        # chain are labeled 1). Use pos_weight in BCE to rebalance.
        n_pos = sum(1 for lab in labels if lab.label >= 0.5)
        n_neg = len(labels) - n_pos
        pos_weight = torch.tensor(
            max(n_neg, 1) / max(n_pos, 1), device=self._device, dtype=torch.float32,
        )

        optimizer = torch.optim.AdamW(
            self.head.parameters(), lr=prm_lr, weight_decay=cfg.weight_decay,
        )
        self.head.train()
        self.model_loader.model.eval()  # base stays frozen

        total_loss = 0.0
        batch_count = 0
        last_loss = 0.0
        step_count = 0
        try:
            for _ in range(prm_epochs):
                for batch in loader:
                    input_ids = batch["input_ids"].to(self._device)
                    attn = batch["attention_mask"].to(self._device)
                    pos = batch["score_pos"].to(self._device)
                    targets = batch["label"].to(self._device)

                    hidden = self._hidden_at(input_ids, attn, pos)
                    logits = self.head(hidden.detach())
                    loss = F.binary_cross_entropy_with_logits(
                        logits, targets, pos_weight=pos_weight,
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.head.parameters(), cfg.max_grad_norm,
                    )
                    optimizer.step()

                    last_loss = float(loss.detach().item())
                    total_loss += last_loss
                    batch_count += 1
                    step_count += 1
        finally:
            self.head.eval()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return PRMTrainingMetrics(
            cycle=cycle,
            avg_loss=total_loss / max(batch_count, 1),
            final_loss=last_loss,
            steps=step_count,
            labels_used=len(ds),
            labels_skipped=skipped,
            learning_rate=prm_lr,
            positive_fraction=n_pos / max(len(labels), 1),
        )

    # -------------------------- persistence --------------------------

    def save(self, path: Path, cycle: int) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "head_state_dict": self.head.state_dict(),
                "hidden_size": self._hidden_size,
                "cycle": cycle,
            },
            path / f"prm_cycle_{cycle}.pt",
        )

    def load(self, path: Path | str) -> None:
        path = Path(path)
        if path.is_dir():
            # pick latest by mtime
            candidates = sorted(path.glob("prm_cycle_*.pt"), key=lambda p: p.stat().st_mtime)
            if not candidates:
                raise FileNotFoundError(f"No PRM checkpoint in {path}")
            path = candidates[-1]
        state = torch.load(path, map_location=self._device, weights_only=True)
        if state.get("hidden_size") != self._hidden_size:
            raise RuntimeError(
                f"PRM checkpoint hidden_size={state.get('hidden_size')} "
                f"!= current base {self._hidden_size}"
            )
        self.head.load_state_dict(state["head_state_dict"])
        self.head.to(self._device)


# =============================================================================
# rl_engine reward-function adapter
# =============================================================================


def _parse_completion_steps(completion: str) -> list[str]:
    """Split a raw GRPO completion into step texts using the Step-marker regex.

    Mirrors the extraction logic used for TrainingSample so training and
    inference see the same boundaries. If no markers are found, treats the
    whole completion as one step (so the PRM still produces a signal).
    """
    if not completion:
        return []
    marks = list(STEP_MARKER_RE.finditer(completion))
    if not marks:
        stripped = completion.strip()
        return [stripped] if stripped else []
    steps: list[str] = []
    for i, m in enumerate(marks):
        start = m.end()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(completion)
        chunk = completion[start:end].strip()
        if chunk:
            steps.append(chunk)
    return steps


def make_prm_reward_fn(
    prm: PRM, *, aggregate: Optional[str] = None,
) -> Callable[[str, str, "TrainingSample"], float]:
    """Build a reward_fn matching rl_engine's GRPO signature.

    Signature: (prompt, completion, sample) -> float. The third arg is accepted
    but unused — GRPO normalizes within group, so only relative scale matters.

    The completion is parsed into steps via the Step-N marker regex, scored
    per-step by the PRM, then aggregated to a scalar:
      * "min"  — weakest-step score. Punishes any bad step (Lightman-style).
      * "mean" — average step quality.
      * "last" — final-step score. Closest to outcome reward.

    Min is the default: it most strongly discourages faulty intermediate
    reasoning, which is the whole point of a process reward. If ``aggregate``
    is None, falls back to ``trainer_config.prm_aggregate`` when present, then
    to "min".
    """
    if aggregate is None:
        aggregate = getattr(prm.trainer_config, "prm_aggregate", "min") or "min"
    if aggregate not in ("min", "mean", "last"):
        raise ValueError(f"aggregate must be min/mean/last, got {aggregate!r}")

    def reward_fn(prompt: str, completion: str, sample) -> float:
        steps = _parse_completion_steps(completion)
        if not steps:
            return 0.0
        scores = prm.score_chain(prompt, steps)
        if not scores:
            return 0.0
        if aggregate == "min":
            return float(min(scores))
        if aggregate == "mean":
            return float(sum(scores) / len(scores))
        return float(scores[-1])  # "last"

    return reward_fn
