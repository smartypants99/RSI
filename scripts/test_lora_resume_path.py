"""Self-test for ImprovementLoop._lora_resume_path. Doesn't need GPU/torch
beyond the import — exercises the dir-walk logic with mock files.

Validates:
1. Returns None when no lora_weights/ dir exists
2. Returns the best_checkpoint_cycle's dir when set + adapter exists
3. Falls back to highest-numbered cycle dir when no best yet
4. Skips dirs without lora_weights.pt sentinel
5. Returns None when use_lora_adapter_persistence=False
"""
from __future__ import annotations
import sys, pathlib, tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.orchestrator.loop import ImprovementLoop  # noqa: E402


class _Stub:
    def __init__(self, output_dir, use_persistence=True):
        self.orchestrator = type("O", (), {
            "output_dir": output_dir,
            "use_lora_adapter_persistence": use_persistence,
        })()


def make_loop(tmpdir: pathlib.Path, *, persistence=True, best_cycle=None):
    loop = ImprovementLoop.__new__(ImprovementLoop)
    loop.config = _Stub(tmpdir, use_persistence=persistence)
    loop._best_checkpoint_cycle = best_cycle
    return loop


def make_adapter(root: pathlib.Path, cycle: int, with_pt: bool = True):
    d = root / "lora_weights" / f"lora_cycle_{cycle}"
    d.mkdir(parents=True, exist_ok=True)
    if with_pt:
        (d / "lora_weights.pt").write_text("stub")


def main() -> None:
    with tempfile.TemporaryDirectory() as t:
        tmp = pathlib.Path(t)

        # 1. No dir → None
        loop = make_loop(tmp)
        assert loop._lora_resume_path() is None, "expected None on empty dir"

        # 2. Best cycle present
        make_adapter(tmp, 1)
        make_adapter(tmp, 2)
        make_adapter(tmp, 3)
        loop = make_loop(tmp, best_cycle=2)
        got = loop._lora_resume_path()
        assert got is not None and got.name == "lora_cycle_2", f"expected cycle_2, got {got}"

        # 3. No best → highest-numbered fallback
        loop = make_loop(tmp, best_cycle=None)
        got = loop._lora_resume_path()
        assert got is not None and got.name == "lora_cycle_3", f"expected cycle_3, got {got}"

        # 4. Skip dirs without lora_weights.pt
        make_adapter(tmp, 4, with_pt=False)
        loop = make_loop(tmp, best_cycle=4)
        got = loop._lora_resume_path()
        # cycle 4 has no .pt → falls back to highest valid (cycle_3)
        assert got is not None and got.name == "lora_cycle_3", \
            f"expected cycle_3 fallback, got {got}"

        # 5. Persistence disabled → None
        loop = make_loop(tmp, persistence=False, best_cycle=2)
        assert loop._lora_resume_path() is None, \
            "expected None when persistence disabled"

        # 6. .reverted marker on a cycle skips it (capture-alarm rejection)
        (tmp / "lora_weights" / "lora_cycle_3" / ".reverted").write_text("alarmed")
        loop = make_loop(tmp, best_cycle=None)
        got = loop._lora_resume_path()
        # cycle 3 has .reverted, cycle 4 has no .pt (case 4) → fallback cycle_2
        assert got is not None and got.name == "lora_cycle_2", \
            f"expected cycle_2 with cycle_3 reverted, got {got}"

        # 7. .reverted on the BEST cycle is also skipped
        (tmp / "lora_weights" / "lora_cycle_2" / ".reverted").write_text("alarmed")
        loop = make_loop(tmp, best_cycle=2)
        got = loop._lora_resume_path()
        # cycle 2 (best) is .reverted; cycle 3 also reverted; cycle 4 has no
        # .pt → cycle 1
        assert got is not None and got.name == "lora_cycle_1", \
            f"expected cycle_1 fallback when best is reverted, got {got}"

    print("PASS — _lora_resume_path: 7/7 cases")


if __name__ == "__main__":
    main()
