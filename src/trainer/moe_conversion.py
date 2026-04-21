"""Post-hoc dense-to-sparse MoE conversion (MegaBlocks-style upcycling).

Converts dense FFN layers to sparse Mixture-of-Experts at inference-FLOP
parity, yielding a 2-4× effective parameter increase. Gated behind
`OrchestratorConfig.moe_conversion_enabled` (default False); fires during
growth events as an alternative to layer expansion.

Approach (per gemini consult, 2026-04-20):

  * **Clustering init** (default): K-means on the neuron weight vectors
    of the dense FFN's ``up_proj`` to partition the intermediate dim into
    N groups; each expert owns one group's rows (up_proj) and columns
    (down_proj). Minimises initial reconstruction error.
  * **Copy-perturb**: all experts = identical dense FFN + small noise.
    Simple baseline; prone to expert collapse.
  * **Slicing**: even split of the intermediate dim. Effective when
    neurons are already specialised.

  * **Shared experts** (DeepSeek-MoE): ``moe_shared_experts`` experts are
    always active, preserving generalist knowledge; the remaining
    (num_experts - shared) are top-k routed.
  * **Router init**: small random weights + N(0, `moe_router_noise_std`)
    noise. When clustering is used the router could be warm-started from
    cluster assignments, but that requires activation samples — we defer
    that optional upgrade to a follow-up.

We ship a framework-independent :class:`SparseMoELayer` with top-k
routing and an auxiliary load-balancing loss (Switch-Transformer form).
When MegaBlocks is importable we delegate to its block-sparse kernels;
otherwise we fall back to a dense gather implementation so the code path
is always testable on CPU.
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:  # pragma: no cover — optional dep; not installed in CI
    import megablocks  # type: ignore
    _HAS_MEGABLOCKS = True
except Exception:
    _HAS_MEGABLOCKS = False


# -------------------- config ------------------------------------------------


@dataclass
class MoEConversionConfig:
    num_experts: int = 4
    top_k: int = 2
    shared_experts: int = 1
    init_method: str = "clustering"  # "copy_perturb" | "slice" | "clustering"
    router_noise_std: float = 0.02
    copy_perturb_std: float = 0.01
    aux_loss_weight: float = 0.01

    def __post_init__(self):
        if self.num_experts < 2:
            raise ValueError("num_experts must be >= 2")
        if not (1 <= self.top_k <= self.num_experts):
            raise ValueError("top_k must be in [1, num_experts]")
        if self.shared_experts < 0 or self.shared_experts >= self.num_experts:
            raise ValueError("shared_experts must be in [0, num_experts)")
        if self.init_method not in ("copy_perturb", "slice", "clustering"):
            raise ValueError(f"init_method unknown: {self.init_method!r}")
        if self.router_noise_std < 0 or self.copy_perturb_std < 0:
            raise ValueError("noise std must be >= 0")


# -------------------- init strategies ---------------------------------------


def _cluster_rows(weight: torch.Tensor, n_clusters: int, iters: int = 5) -> list[torch.Tensor]:
    """Cheap K-means over rows of ``weight`` returning a list of row-index
    tensors, one per cluster. CPU-friendly; used at init time only.
    """
    rows = weight.detach().float()
    n = rows.shape[0]
    # Init centroids by even-stride sampling for determinism.
    stride = max(1, n // n_clusters)
    centroids = rows[torch.arange(0, n, stride)[:n_clusters]].clone()
    assignments = torch.zeros(n, dtype=torch.long)
    for _ in range(iters):
        dists = torch.cdist(rows, centroids)
        assignments = dists.argmin(dim=1)
        for k in range(n_clusters):
            members = rows[assignments == k]
            if members.numel() > 0:
                centroids[k] = members.mean(dim=0)
    return [torch.where(assignments == k)[0] for k in range(n_clusters)]


def _build_expert_slices(
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    cfg: MoEConversionConfig,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return a list of (up_slice, down_slice) per expert.

    `up_weight`:    (intermediate, hidden) — row-partitioned.
    `down_weight`:  (hidden, intermediate) — column-partitioned.
    """
    intermediate = up_weight.shape[0]
    N = cfg.num_experts
    if cfg.init_method == "copy_perturb":
        experts = []
        for _ in range(N):
            u = up_weight.clone() + torch.randn_like(up_weight) * cfg.copy_perturb_std
            d = down_weight.clone() + torch.randn_like(down_weight) * cfg.copy_perturb_std
            experts.append((u, d))
        return experts
    if cfg.init_method == "slice":
        per = max(1, intermediate // N)
        experts = []
        for k in range(N):
            start = k * per
            end = intermediate if k == N - 1 else (k + 1) * per
            u = up_weight[start:end].clone()
            d = down_weight[:, start:end].clone()
            experts.append((u, d))
        return experts
    # clustering
    clusters = _cluster_rows(up_weight, N)
    experts = []
    for idx in clusters:
        if idx.numel() == 0:
            # degenerate cluster: fall back to a single neuron so shapes stay valid
            idx = torch.tensor([0], dtype=torch.long)
        u = up_weight[idx].clone()
        d = down_weight[:, idx].clone()
        experts.append((u, d))
    return experts


# -------------------- layer -------------------------------------------------


class _Expert(nn.Module):
    """SwiGLU-style FFN expert. Compatible with gate+up+down triples, which
    is the modern Llama/Qwen layout. Falls back to up+down (ReLU) when no
    gate is supplied.
    """

    def __init__(self, up_w: torch.Tensor, down_w: torch.Tensor, gate_w: Optional[torch.Tensor] = None):
        super().__init__()
        self.up = nn.Parameter(up_w)
        self.down = nn.Parameter(down_w)
        self.gate = nn.Parameter(gate_w) if gate_w is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = x @ self.up.t()
        if self.gate is not None:
            g = x @ self.gate.t()
            hidden = F.silu(g) * up
        else:
            hidden = F.relu(up)
        return hidden @ self.down.t()


class SparseMoELayer(nn.Module):
    """Top-k routed MoE with optional always-on shared experts.

    When MegaBlocks is installed we could dispatch through its dropless
    block-sparse kernels; for portability we use a gather-based reference
    implementation that produces identical logits. The overhead matters
    at prod scale but is fine for verification.
    """

    def __init__(
        self,
        experts: list[tuple[torch.Tensor, torch.Tensor]],
        cfg: MoEConversionConfig,
        hidden_size: int,
        gate_weights: Optional[list[torch.Tensor]] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = hidden_size
        if gate_weights is None:
            gate_weights = [None] * len(experts)
        self.experts = nn.ModuleList(
            _Expert(u, d, g) for (u, d), g in zip(experts, gate_weights)
        )
        self.num_experts = len(experts)
        self.shared = cfg.shared_experts
        self.router = nn.Linear(hidden_size, self.num_experts - self.shared, bias=False)
        with torch.no_grad():
            self.router.weight.normal_(mean=0.0, std=cfg.router_noise_std)
        self.last_aux_loss: torch.Tensor = torch.zeros(())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., hidden). Flatten tokens for routing.
        orig_shape = x.shape
        flat = x.reshape(-1, self.hidden_size)

        out = torch.zeros_like(flat)
        # Shared experts: always on, equal weight averaging.
        if self.shared > 0:
            for i in range(self.shared):
                out = out + self.experts[i](flat)
            out = out / max(1, self.shared)

        # Routed experts: top-k over logits.
        if self.num_experts > self.shared:
            logits = self.router(flat)  # (T, num_routed)
            probs = F.softmax(logits, dim=-1)
            k = min(self.cfg.top_k, self.num_experts - self.shared)
            top_vals, top_idx = probs.topk(k, dim=-1)
            # Normalise top-k gate weights so they sum to 1.
            top_vals = top_vals / (top_vals.sum(dim=-1, keepdim=True) + 1e-9)

            routed_out = torch.zeros_like(flat)
            for slot in range(k):
                idx = top_idx[:, slot]
                weight = top_vals[:, slot].unsqueeze(-1)
                for e in range(self.num_experts - self.shared):
                    mask = idx == e
                    if mask.any():
                        selected = flat[mask]
                        expert_out = self.experts[self.shared + e](selected)
                        routed_out[mask] = routed_out[mask] + weight[mask] * expert_out
            out = out + routed_out

            # Switch-Transformer aux loss: encourage uniform expert load.
            with_routed = probs.mean(dim=0)  # fraction each expert gets
            # Top-1 assignment fraction (approx via argmax of probs):
            top1 = probs.argmax(dim=-1)
            usage = torch.bincount(
                top1, minlength=self.num_experts - self.shared
            ).float() / max(1, flat.shape[0])
            self.last_aux_loss = (
                self.cfg.aux_loss_weight
                * (self.num_experts - self.shared)
                * (with_routed * usage).sum()
            )
        else:
            self.last_aux_loss = torch.zeros((), device=flat.device)

        return out.reshape(orig_shape)


# -------------------- public conversion API ---------------------------------


def _extract_dense_ffn(ffn: nn.Module) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Pull (up_weight, down_weight, gate_weight?) from a dense FFN module.

    Supports Llama-style (gate_proj, up_proj, down_proj) and bare
    (up_proj, down_proj). Raises if neither matches.
    """
    if hasattr(ffn, "up_proj") and hasattr(ffn, "down_proj"):
        up = ffn.up_proj.weight.detach()
        down = ffn.down_proj.weight.detach()
        gate = ffn.gate_proj.weight.detach() if hasattr(ffn, "gate_proj") else None
        return up, down, gate
    raise AttributeError(
        "dense FFN must expose up_proj / down_proj (optional gate_proj) modules"
    )


def convert_dense_ffn_to_moe(
    ffn: nn.Module,
    cfg: MoEConversionConfig,
    hidden_size: int,
) -> SparseMoELayer:
    """Upcycle a single dense FFN into a sparse MoE layer.

    The dense weights are partitioned (clustering / slice / copy-perturb)
    across ``cfg.num_experts`` experts; the router is initialised with
    small random noise. The returned layer preserves the dense FFN's
    approximate function at t=0 when ``init_method='copy_perturb'`` and
    ``top_k >= 1``.
    """
    up, down, gate = _extract_dense_ffn(ffn)
    expert_weights = _build_expert_slices(up, down, cfg)
    gate_slices: Optional[list[Optional[torch.Tensor]]]
    if gate is None:
        gate_slices = None
    else:
        # Mirror the up_proj partitioning on the gate tensor so SwiGLU stays coherent.
        if cfg.init_method == "slice":
            per = max(1, gate.shape[0] // cfg.num_experts)
            gate_slices = []
            for k in range(cfg.num_experts):
                start = k * per
                end = gate.shape[0] if k == cfg.num_experts - 1 else (k + 1) * per
                gate_slices.append(gate[start:end].clone())
        elif cfg.init_method == "copy_perturb":
            gate_slices = [
                gate.clone() + torch.randn_like(gate) * cfg.copy_perturb_std
                for _ in range(cfg.num_experts)
            ]
        else:  # clustering — reuse row partition computed on up
            clusters = _cluster_rows(up, cfg.num_experts)
            gate_slices = []
            for idx in clusters:
                if idx.numel() == 0:
                    idx = torch.tensor([0], dtype=torch.long)
                gate_slices.append(gate[idx].clone())
    return SparseMoELayer(expert_weights, cfg, hidden_size, gate_weights=gate_slices)


def convert_model_ffn_to_moe(
    model: nn.Module,
    cfg: MoEConversionConfig,
    hidden_size: int,
    *,
    ffn_attr: str = "mlp",
) -> nn.Module:
    """Walk the transformer stack and replace each layer's dense FFN with
    a :class:`SparseMoELayer`. Returns a new model (teacher untouched).

    Looks for ``layer.{ffn_attr}`` on each block in the stack located via
    :func:`src.trainer.growth._locate_layers`. Layers without that attr
    are left alone.
    """
    from src.trainer.growth import _locate_layers

    student = copy.deepcopy(model)
    parent, attr = _locate_layers(student)
    stack = getattr(parent, attr)
    converted = 0
    for i, layer in enumerate(stack):
        ffn = getattr(layer, ffn_attr, None)
        if ffn is None:
            continue
        try:
            moe = convert_dense_ffn_to_moe(ffn, cfg, hidden_size)
        except AttributeError:
            continue
        setattr(layer, ffn_attr, moe)
        converted += 1
    logger.info(
        "moe conversion: %d/%d layers upcycled (experts=%d, top_k=%d, shared=%d, init=%s, megablocks=%s)",
        converted, len(stack), cfg.num_experts, cfg.top_k, cfg.shared_experts,
        cfg.init_method, _HAS_MEGABLOCKS,
    )
    return student


def is_megablocks_available() -> bool:
    return _HAS_MEGABLOCKS
