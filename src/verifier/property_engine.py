"""Property-based consensus verifier for synthesized tasks.

This module provides verify_by_consensus(), which checks a list of TrainingSample
objects against named properties and returns only those that pass at least
`threshold` fraction of the registered property checks.

The full property set is defined by the property_verifier teammate. This stub
provides the public interface so the orchestrator loop can import it and the
test suite stays green before the full implementation lands.
"""

from __future__ import annotations

import logging
from typing import Callable

from ..generator.data_generator import TrainingSample

logger = logging.getLogger(__name__)

# Type alias: a property check is a callable that returns True/False.
PropertyFn = Callable[[TrainingSample], bool]

# Registry of named property functions. teammate property_verifier populates
# this at import time by calling register_property().
_REGISTRY: dict[str, PropertyFn] = {}


def register_property(name: str, fn: PropertyFn) -> None:
    """Register a named property check (idempotent — later wins)."""
    _REGISTRY[name] = fn


def verify_by_consensus(
    samples: list[TrainingSample],
    threshold: float = 0.7,
    *,
    properties: dict[str, PropertyFn] | None = None,
) -> list[TrainingSample]:
    """Return samples that pass >= threshold fraction of registered properties.

    Args:
        samples: Candidate TrainingSample objects to filter.
        threshold: Minimum pass-rate in (0, 1] to accept a sample.
        properties: Override the global registry for this call (for testing).

    Returns:
        Subset of samples that cleared the consensus bar.
    """
    registry = properties if properties is not None else _REGISTRY

    if not registry:
        # No properties registered — accept all (preserves classic behavior
        # when this module is present but teammate hasn't landed checks yet).
        logger.debug("property_engine: no properties registered, accepting all %d samples", len(samples))
        return list(samples)

    accepted: list[TrainingSample] = []
    for sample in samples:
        passes = sum(1 for fn in registry.values() if _safe_check(fn, sample))
        rate = passes / len(registry)
        if rate >= threshold:
            accepted.append(sample)

    logger.info(
        "property_engine: %d/%d samples passed consensus (threshold=%.2f, properties=%d)",
        len(accepted), len(samples), threshold, len(registry),
    )
    return accepted


def _safe_check(fn: PropertyFn, sample: TrainingSample) -> bool:
    try:
        return bool(fn(sample))
    except Exception as exc:
        logger.debug("property check raised %s: %s", type(exc).__name__, exc)
        return False
