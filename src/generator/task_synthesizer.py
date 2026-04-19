"""Task synthesizer — generates novel training tasks and associated properties.

Called by the orchestrator loop between the diagnose and generate phases when
`synthesis_config.enable_task_synthesis` is True. The full synthesis logic is
implemented by the task_synthesizer teammate. This stub provides a clean public
interface so the orchestrator can import and call it before that implementation
lands, with graceful fallback (returns an empty list).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..generator.data_generator import TrainingSample
from ..diagnostics.engine import DiagnosticResult
from ..utils.config import SynthesisConfig

logger = logging.getLogger(__name__)


@dataclass
class SynthesisResult:
    """Output of one synthesis run."""
    tasks: list[TrainingSample] = field(default_factory=list)
    properties: dict[str, object] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


class TaskSynthesizer:
    """Generates novel tasks targeting diagnosed weaknesses.

    The teammate's implementation overrides _synthesize_tasks(). This base
    class ensures the interface is stable and the orchestrator loop can call
    synthesize() safely even with the stub.
    """

    def __init__(self, config: SynthesisConfig, model_loader=None):
        self.config = config
        self.model_loader = model_loader

    def synthesize(self, diag: DiagnosticResult) -> SynthesisResult:
        """Generate up to config.tasks_per_cycle novel tasks from diagnostic result.

        Returns an empty SynthesisResult if the full implementation hasn't
        been provided yet (stub behavior).
        """
        try:
            return self._synthesize_tasks(diag)
        except NotImplementedError:
            logger.debug("task_synthesizer: stub active, returning empty result")
            return SynthesisResult()
        except Exception as exc:
            logger.warning(
                "task_synthesizer: synthesis failed (%s: %s) — returning empty result",
                type(exc).__name__, exc,
            )
            return SynthesisResult()

    def _synthesize_tasks(self, diag: DiagnosticResult) -> SynthesisResult:
        """Override in the full implementation."""
        raise NotImplementedError
