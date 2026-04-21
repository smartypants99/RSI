"""Bridge between growth and the external TDQ compressor.

When growth produces a model whose fp16 weights would exceed the VRAM
budget, the trainer must avoid ever instantiating the grown stack on-GPU.
Instead it saves the grown module to a HF-format directory (safetensors),
then calls `compress_model_dir_to_tdq`, which imports td_quant in-process
from TDQ_INFERENCE_DIR and invokes compress_and_save directly — no
subprocess, no shell. The resulting .tdq file is then loaded via
:class:`src.utils.tdq_backend.TDQModelLoader` for subsequent cycles.

External tooling contract (READ-only from our side):
  - ai_quatinization/final/td_quant.py :: compress_and_save(model_id, output_path, config)
    where `model_id` may be a local directory path (HF save_pretrained layout).

We do NOT import td_quant at module load — compression is rare and
import pulls heavy deps (transformers, AutoModelForCausalLM). Deferred
until the bridge is actually invoked.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


DEFAULT_TDQ_DIR = "/Users/milannarula/Desktop/ai_quatinization/final"


def _import_td_quant():
    tdq_dir = os.environ.get("TDQ_INFERENCE_DIR", DEFAULT_TDQ_DIR)
    # Append, don't prepend: we don't want the TDQ dir to shadow stdlib or
    # site-packages on name collisions. td_quant's own relative imports
    # still resolve fine because they're package-relative / absolute to
    # modules that live alongside td_quant in the same dir.
    if tdq_dir and tdq_dir not in sys.path:
        sys.path.append(tdq_dir)
    import td_quant  # noqa: F401
    return sys.modules["td_quant"]


def save_model_to_hf_dir(model, tokenizer, output_dir: Path) -> Path:
    """Persist an in-memory model + tokenizer in HF layout (safetensors).

    This is the handoff format `compress_and_save` accepts as `model_id`.
    Kept thin — any error from save_pretrained propagates.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    if tokenizer is not None:
        tokenizer.save_pretrained(str(output_dir))
    return output_dir


def compress_model_dir_to_tdq(
    hf_dir: Path,
    output_tdq_path: Path,
    config: str = "A",
    compressor=None,
) -> Path:
    """Compress a HF-format model directory into a .tdq file.

    `compressor` is injectable for tests; defaults to
    td_quant.compress_and_save. Returns the path to the written .tdq.
    """
    hf_dir = Path(hf_dir)
    output_tdq_path = Path(output_tdq_path)
    output_tdq_path.parent.mkdir(parents=True, exist_ok=True)
    if compressor is None:
        td_quant = _import_td_quant()
        compressor = td_quant.compress_and_save
    logger.info("TDQ-bridge: compressing %s -> %s (config=%s)",
                hf_dir, output_tdq_path, config)
    compressor(str(hf_dir), str(output_tdq_path), config=config)
    if not output_tdq_path.exists():
        raise RuntimeError(
            f"TDQ compression reported success but output file missing: "
            f"{output_tdq_path}"
        )
    return output_tdq_path


def load_tdq_as_loader(tdq_path: Path, **kwargs):
    """Convenience: load a .tdq file via TDQModelLoader (the existing backend)."""
    from .tdq_backend import TDQModelLoader
    loader = TDQModelLoader(str(tdq_path), **kwargs)
    loader.load()
    return loader
