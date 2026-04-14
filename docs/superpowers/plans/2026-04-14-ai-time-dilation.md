# AI Time Dilation Runtime — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI tool that wraps a local Qwen model and applies iterative self-refinement loops to produce arbitrarily better output via a user-specified dilation factor.

**Architecture:** Inference engine (vLLM + speculative decoding) -> Dilation controller (cycle scheduler) -> Improvement engine (branch-score-select loop) -> CLI. Each component is a focused Python module with a clean interface.

**Tech Stack:** Python 3.11+, vLLM, Qwen2.5 models, Click, Rich

**Spec:** `docs/superpowers/specs/2026-04-14-ai-time-dilation-design.md`

---

## File Structure

```
src/
  timedilate/
    __init__.py          — package init, version
    engine.py            — InferenceEngine: wraps vLLM, exposes generate()
    controller.py        — DilationController: manages cycles, timing, checkpointing
    improver.py          — ImprovementEngine: branch-score-select loop
    scorer.py            — Scorer: structured rubric-based evaluation
    directives.py        — Directive generation (task-aware + self-generated)
    checkpoint.py        — Save/load intermediate results
    cli.py               — Click CLI entry point
    config.py            — Configuration dataclass
tests/
  test_directives.py     — Directive generation tests
  test_scorer.py         — Scoring logic tests
  test_improver.py       — Improvement loop tests
  test_controller.py     — Dilation controller tests
  test_engine.py         — Inference engine tests (with mock)
  test_checkpoint.py     — Checkpoint save/load tests
  test_cli.py            — CLI integration tests
  conftest.py            — Shared fixtures
pyproject.toml           — Project config, dependencies, entry point
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/timedilate/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "timedilate"
version = "0.1.0"
description = "AI Time Dilation Runtime"
requires-python = ">=3.11"
dependencies = [
    "vllm>=0.6.0",
    "click>=8.0",
    "rich>=13.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov",
]

[project.scripts]
timedilate = "timedilate.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/timedilate"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 2: Create package init**

```python
# src/timedilate/__init__.py
__version__ = "0.1.0"
```

- [ ] **Step 3: Create conftest.py**

```python
# tests/conftest.py
import pytest

@pytest.fixture
def sample_prompt():
    return "Write a Python function that reverses a string."

@pytest.fixture
def sample_output():
    return "def reverse_string(s):\n    return s[::-1]"
```

- [ ] **Step 4: Install in dev mode and verify**

Run: `pip install -e ".[dev]"`
Expected: Installs successfully

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/timedilate/__init__.py tests/conftest.py
git commit -m "feat: project scaffolding with pyproject.toml"
```

---

### Task 2: Configuration

**Files:**
- Create: `src/timedilate/config.py`

- [ ] **Step 1: Write config dataclass**

```python
# src/timedilate/config.py
from dataclasses import dataclass, field

@dataclass
class TimeDilateConfig:
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    draft_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dilation_factor: int = 2
    budget_seconds: float = 30.0
    branch_factor: int = 3
    max_tokens: int = 4096
    temperature: float = 0.7
    scoring_temperature: float = 0.0
    checkpoint_dir: str = ".timedilate_checkpoints"
    checkpoint_interval: int = 10
    convergence_threshold: int = 5
```

- [ ] **Step 2: Commit**

```bash
git add src/timedilate/config.py
git commit -m "feat: add configuration dataclass"
```

---

### Task 3: Directive Generation

**Files:**
- Create: `src/timedilate/directives.py`
- Create: `tests/test_directives.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_directives.py
from timedilate.directives import DirectiveGenerator

def test_classify_code_task():
    gen = DirectiveGenerator()
    task_type = gen.classify_task("Write a Python function to sort a list")
    assert task_type == "code"

def test_classify_prose_task():
    gen = DirectiveGenerator()
    task_type = gen.classify_task("Write an essay about climate change")
    assert task_type == "prose"

def test_get_directives_code():
    gen = DirectiveGenerator()
    directives = gen.get_directives("code")
    assert len(directives) > 0
    assert all(isinstance(d, str) for d in directives)

def test_get_directives_prose():
    gen = DirectiveGenerator()
    directives = gen.get_directives("prose")
    assert len(directives) > 0

def test_cycle_through_directives():
    gen = DirectiveGenerator()
    directives = gen.get_directives("code")
    # Should cycle and never raise IndexError
    for i in range(len(directives) + 5):
        d = gen.next_directive("code", i)
        assert isinstance(d, str)

def test_custom_directive_generation_prompt():
    gen = DirectiveGenerator()
    prompt = gen.generate_custom_directive_prompt("code", "Write a sorting function", "def sort(lst): return sorted(lst)")
    assert "sorting" in prompt.lower() or "improve" in prompt.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_directives.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement directives module**

```python
# src/timedilate/directives.py

CODE_DIRECTIVES = [
    "Fix any bugs or logical errors in this code.",
    "Add handling for edge cases (empty input, invalid types, boundary values).",
    "Optimize the performance of this code.",
    "Add comprehensive error handling and input validation.",
    "Refactor for readability and clean code principles.",
    "Explore an alternative algorithmic approach.",
    "Add inline comments explaining complex logic.",
    "Stress-test: what happens with very large inputs?",
    "Add useful features that serve the original goal.",
]

PROSE_DIRECTIVES = [
    "Improve clarity and conciseness of the writing.",
    "Strengthen the arguments with better evidence or reasoning.",
    "Add concrete examples to illustrate key points.",
    "Consider and address counterarguments.",
    "Improve the overall structure and flow.",
    "Refine the tone to better match the intended audience.",
    "Deepen the analysis where it's currently surface-level.",
    "Remove redundancy and filler.",
    "Add a stronger opening and conclusion.",
]

GENERAL_DIRECTIVES = [
    "Improve the overall quality and completeness.",
    "Fix any errors or inaccuracies.",
    "Add more detail where it's needed.",
    "Consider alternative approaches.",
    "Make the output more useful for the end user.",
    "Improve structure and organization.",
    "Add examples or illustrations.",
    "Polish and refine the output.",
]

TASK_KEYWORDS = {
    "code": ["function", "code", "program", "script", "implement", "build", "create a", "write a", "class", "api", "app", "game", "website", "algorithm"],
    "prose": ["essay", "write about", "explain", "describe", "article", "blog", "letter", "email", "story", "poem", "report"],
}


class DirectiveGenerator:
    def classify_task(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        scores = {"code": 0, "prose": 0}
        for task_type, keywords in TASK_KEYWORDS.items():
            for kw in keywords:
                if kw in prompt_lower:
                    scores[task_type] += 1
        if scores["code"] > scores["prose"]:
            return "code"
        if scores["prose"] > scores["code"]:
            return "prose"
        return "general"

    def get_directives(self, task_type: str) -> list[str]:
        if task_type == "code":
            return CODE_DIRECTIVES
        if task_type == "prose":
            return PROSE_DIRECTIVES
        return GENERAL_DIRECTIVES

    def next_directive(self, task_type: str, cycle_index: int) -> str:
        directives = self.get_directives(task_type)
        return directives[cycle_index % len(directives)]

    def generate_custom_directive_prompt(self, task_type: str, original_prompt: str, current_output: str) -> str:
        return (
            f"Given the task: {original_prompt}\n\n"
            f"And the current output:\n{current_output}\n\n"
            f"Suggest one specific improvement that would make this output better. "
            f"Respond with just the improvement directive, nothing else."
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_directives.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/timedilate/directives.py tests/test_directives.py
git commit -m "feat: add task-aware directive generation"
```

---

### Task 4: Scorer

**Files:**
- Create: `src/timedilate/scorer.py`
- Create: `tests/test_scorer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_scorer.py
from timedilate.scorer import Scorer

def test_build_scoring_prompt():
    scorer = Scorer()
    prompt = scorer.build_scoring_prompt(
        original_prompt="Write a sort function",
        output="def sort(lst): return sorted(lst)"
    )
    assert "sort" in prompt.lower()
    assert "correctness" in prompt.lower() or "score" in prompt.lower()

def test_parse_score_valid():
    scorer = Scorer()
    assert scorer.parse_score("85") == 85
    assert scorer.parse_score("Score: 72") == 72
    assert scorer.parse_score("I'd give this a 90 out of 100") == 90

def test_parse_score_invalid():
    scorer = Scorer()
    assert scorer.parse_score("no number here") == 0
    assert scorer.parse_score("") == 0

def test_parse_score_clamps():
    scorer = Scorer()
    assert scorer.parse_score("150") == 100
    assert scorer.parse_score("-5") == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_scorer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement scorer**

```python
# src/timedilate/scorer.py
import re


class Scorer:
    RUBRIC = (
        "Rate the following output on a scale of 0-100 based on these criteria:\n"
        "- Correctness: Does it accurately address the task? (0-25)\n"
        "- Completeness: Does it fully address all aspects? (0-25)\n"
        "- Quality: Is it well-structured and polished? (0-25)\n"
        "- Usefulness: Would the end user be satisfied? (0-25)\n\n"
        "Respond with ONLY a single integer 0-100. Nothing else."
    )

    def build_scoring_prompt(self, original_prompt: str, output: str) -> str:
        return (
            f"Original task: {original_prompt}\n\n"
            f"Output to score:\n{output}\n\n"
            f"{self.RUBRIC}"
        )

    def parse_score(self, raw: str) -> int:
        numbers = re.findall(r'\d+', raw)
        if not numbers:
            return 0
        score = int(numbers[0])
        return max(0, min(100, score))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_scorer.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/timedilate/scorer.py tests/test_scorer.py
git commit -m "feat: add structured rubric-based scorer"
```

---

### Task 5: Checkpoint System

**Files:**
- Create: `src/timedilate/checkpoint.py`
- Create: `tests/test_checkpoint.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_checkpoint.py
import json
import tempfile
from pathlib import Path
from timedilate.checkpoint import CheckpointManager

def test_save_and_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)
        mgr.save(cycle=5, output="hello world", score=85)
        result = mgr.load_latest()
        assert result["cycle"] == 5
        assert result["output"] == "hello world"
        assert result["score"] == 85

def test_load_latest_picks_highest_cycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)
        mgr.save(cycle=5, output="v1", score=70)
        mgr.save(cycle=10, output="v2", score=85)
        result = mgr.load_latest()
        assert result["cycle"] == 10
        assert result["output"] == "v2"

def test_load_empty_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)
        assert mgr.load_latest() is None

def test_cleanup():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)
        mgr.save(cycle=5, output="v1", score=70)
        mgr.cleanup()
        assert mgr.load_latest() is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_checkpoint.py -v`
Expected: FAIL

- [ ] **Step 3: Implement checkpoint manager**

```python
# src/timedilate/checkpoint.py
import json
from pathlib import Path


class CheckpointManager:
    def __init__(self, checkpoint_dir: str):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, cycle: int, output: str, score: int) -> None:
        path = self.dir / f"cycle_{cycle:06d}.json"
        path.write_text(json.dumps({
            "cycle": cycle,
            "output": output,
            "score": score,
        }))

    def load_latest(self) -> dict | None:
        files = sorted(self.dir.glob("cycle_*.json"))
        if not files:
            return None
        return json.loads(files[-1].read_text())

    def cleanup(self) -> None:
        for f in self.dir.glob("cycle_*.json"):
            f.unlink()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_checkpoint.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/timedilate/checkpoint.py tests/test_checkpoint.py
git commit -m "feat: add checkpoint save/load system"
```

---

### Task 6: Inference Engine

**Files:**
- Create: `src/timedilate/engine.py`
- Create: `tests/test_engine.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_engine.py
import pytest
from unittest.mock import MagicMock, patch
from timedilate.engine import InferenceEngine
from timedilate.config import TimeDilateConfig

def test_engine_init():
    """Test engine can be instantiated with config."""
    config = TimeDilateConfig()
    # Don't actually load the model in tests
    with patch("timedilate.engine.LLM") as mock_llm:
        engine = InferenceEngine(config)
        assert engine.config == config

def test_generate_calls_model():
    config = TimeDilateConfig()
    with patch("timedilate.engine.LLM") as MockLLM:
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="def hello(): pass")]
        MockLLM.return_value.generate.return_value = [mock_output]

        engine = InferenceEngine(config)
        result = engine.generate("Write a function")
        assert result == "def hello(): pass"

def test_generate_with_max_tokens():
    config = TimeDilateConfig(max_tokens=512)
    with patch("timedilate.engine.LLM") as MockLLM:
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="output")]
        MockLLM.return_value.generate.return_value = [mock_output]

        engine = InferenceEngine(config)
        engine.generate("test", max_tokens=512)
        call_args = MockLLM.return_value.generate.call_args
        assert call_args is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_engine.py -v`
Expected: FAIL

- [ ] **Step 3: Implement inference engine**

```python
# src/timedilate/engine.py
from vllm import LLM, SamplingParams
from timedilate.config import TimeDilateConfig


class InferenceEngine:
    def __init__(self, config: TimeDilateConfig):
        self.config = config
        self.llm = LLM(
            model=config.model,
            speculative_model=config.draft_model,
            num_speculative_tokens=5,
            trust_remote_code=True,
        )

    def generate(self, prompt: str, max_tokens: int | None = None, temperature: float | None = None) -> str:
        params = SamplingParams(
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature if temperature is not None else self.config.temperature,
        )
        outputs = self.llm.generate([prompt], params)
        return outputs[0].outputs[0].text
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_engine.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/timedilate/engine.py tests/test_engine.py
git commit -m "feat: add vLLM inference engine with speculative decoding"
```

---

### Task 7: Improvement Engine

**Files:**
- Create: `src/timedilate/improver.py`
- Create: `tests/test_improver.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_improver.py
from unittest.mock import MagicMock
from timedilate.improver import ImprovementEngine
from timedilate.config import TimeDilateConfig

def make_mock_engine(responses: list[str]):
    engine = MagicMock()
    engine.generate = MagicMock(side_effect=responses)
    return engine

def test_branch_generates_n_variants():
    # 3 branches + 3 scores
    engine = make_mock_engine([
        "variant_1", "variant_2", "variant_3",
        "90", "85", "70",
    ])
    config = TimeDilateConfig(branch_factor=3)
    improver = ImprovementEngine(engine, config)
    best, score = improver.run_cycle(
        original_prompt="Write hello world",
        current_best="print('hi')",
        current_score=50,
        directive="Improve this code.",
    )
    assert best == "variant_1"
    assert score == 90

def test_keeps_current_if_no_improvement():
    engine = make_mock_engine([
        "worse_v1", "worse_v2", "worse_v3",
        "30", "20", "10",
    ])
    config = TimeDilateConfig(branch_factor=3)
    improver = ImprovementEngine(engine, config)
    best, score = improver.run_cycle(
        original_prompt="Write hello world",
        current_best="print('hello world')",
        current_score=95,
        directive="Improve this code.",
    )
    assert best == "print('hello world')"
    assert score == 95

def test_single_branch_mode():
    engine = make_mock_engine(["improved", "88"])
    config = TimeDilateConfig(branch_factor=1)
    improver = ImprovementEngine(engine, config)
    best, score = improver.run_cycle(
        original_prompt="test",
        current_best="original",
        current_score=50,
        directive="Improve.",
    )
    assert best == "improved"
    assert score == 88
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_improver.py -v`
Expected: FAIL

- [ ] **Step 3: Implement improvement engine**

```python
# src/timedilate/improver.py
from timedilate.config import TimeDilateConfig
from timedilate.scorer import Scorer


class ImprovementEngine:
    def __init__(self, engine, config: TimeDilateConfig):
        self.engine = engine
        self.config = config
        self.scorer = Scorer()

    def _build_improvement_prompt(self, original_prompt: str, current_best: str, directive: str) -> str:
        return (
            f"Original task: {original_prompt}\n\n"
            f"Current solution:\n{current_best}\n\n"
            f"Improvement directive: {directive}\n\n"
            f"Produce an improved version of the solution. "
            f"Output ONLY the improved solution, nothing else."
        )

    def run_cycle(
        self,
        original_prompt: str,
        current_best: str,
        current_score: int,
        directive: str,
    ) -> tuple[str, int]:
        variants = []
        for _ in range(self.config.branch_factor):
            prompt = self._build_improvement_prompt(original_prompt, current_best, directive)
            variant = self.engine.generate(prompt)
            variants.append(variant)

        best_variant = current_best
        best_score = current_score

        for variant in variants:
            score_prompt = self.scorer.build_scoring_prompt(original_prompt, variant)
            raw_score = self.engine.generate(score_prompt, temperature=self.config.scoring_temperature)
            score = self.scorer.parse_score(raw_score)
            if score > best_score:
                best_variant = variant
                best_score = score

        return best_variant, best_score
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_improver.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/timedilate/improver.py tests/test_improver.py
git commit -m "feat: add branch-score-select improvement engine"
```

---

### Task 8: Dilation Controller

**Files:**
- Create: `src/timedilate/controller.py`
- Create: `tests/test_controller.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_controller.py
from unittest.mock import MagicMock, patch
from timedilate.controller import DilationController
from timedilate.config import TimeDilateConfig

def test_controller_runs_correct_number_of_cycles():
    config = TimeDilateConfig(dilation_factor=3, branch_factor=1)
    mock_engine = MagicMock()
    # Initial generation + (2 refinement cycles * (1 improve + 1 score)) = 5 calls
    mock_engine.generate = MagicMock(side_effect=[
        "initial output",   # first generation
        "improved v1", "90",  # cycle 1
        "improved v2", "95",  # cycle 2
    ])

    controller = DilationController(config, mock_engine)
    result = controller.run("Write hello world")
    # dilation_factor=3 means 1 initial + 2 refinement cycles
    assert result.output == "improved v2"
    assert result.cycles_completed == 2

def test_controller_dilation_1_means_no_refinement():
    config = TimeDilateConfig(dilation_factor=1, branch_factor=1)
    mock_engine = MagicMock()
    mock_engine.generate = MagicMock(return_value="initial output")

    controller = DilationController(config, mock_engine)
    result = controller.run("test")
    assert result.output == "initial output"
    assert result.cycles_completed == 0

def test_controller_tracks_convergence():
    config = TimeDilateConfig(dilation_factor=10, branch_factor=1, convergence_threshold=3)
    mock_engine = MagicMock()
    responses = ["initial output", "80"]  # initial + score
    # 9 cycles where nothing improves (score stays at 80)
    for _ in range(9):
        responses.extend(["same output", "70"])  # always worse
    mock_engine.generate = MagicMock(side_effect=responses)

    controller = DilationController(config, mock_engine)
    result = controller.run("test")
    assert result.cycles_completed == 9
    assert result.convergence_detected
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_controller.py -v`
Expected: FAIL

- [ ] **Step 3: Implement dilation controller**

```python
# src/timedilate/controller.py
import time
from dataclasses import dataclass
from timedilate.config import TimeDilateConfig
from timedilate.improver import ImprovementEngine
from timedilate.scorer import Scorer
from timedilate.directives import DirectiveGenerator
from timedilate.checkpoint import CheckpointManager


@dataclass
class DilationResult:
    output: str
    score: int
    cycles_completed: int
    elapsed_seconds: float
    convergence_detected: bool


class DilationController:
    def __init__(self, config: TimeDilateConfig, engine):
        self.config = config
        self.engine = engine
        self.improver = ImprovementEngine(engine, config)
        self.scorer = Scorer()
        self.directives = DirectiveGenerator()
        self.checkpoint = CheckpointManager(config.checkpoint_dir)

    def run(self, prompt: str, on_cycle=None) -> DilationResult:
        start = time.time()
        task_type = self.directives.classify_task(prompt)
        refinement_cycles = self.config.dilation_factor - 1

        # Initial generation
        current_best = self.engine.generate(prompt)

        if refinement_cycles <= 0:
            return DilationResult(
                output=current_best,
                score=0,
                cycles_completed=0,
                elapsed_seconds=time.time() - start,
                convergence_detected=False,
            )

        # Score the initial output
        score_prompt = self.scorer.build_scoring_prompt(prompt, current_best)
        raw_score = self.engine.generate(score_prompt, temperature=self.config.scoring_temperature)
        current_score = self.scorer.parse_score(raw_score)

        no_improvement_count = 0
        convergence_detected = False
        directive_offset = 0

        for cycle in range(refinement_cycles):
            directive = self.directives.next_directive(task_type, cycle + directive_offset)

            new_best, new_score = self.improver.run_cycle(
                original_prompt=prompt,
                current_best=current_best,
                current_score=current_score,
                directive=directive,
            )

            if new_score > current_score:
                current_best = new_best
                current_score = new_score
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= self.config.convergence_threshold:
                convergence_detected = True
                directive_offset += len(self.directives.get_directives(task_type))
                no_improvement_count = 0

            if (cycle + 1) % self.config.checkpoint_interval == 0:
                self.checkpoint.save(cycle + 1, current_best, current_score)

            if on_cycle:
                on_cycle(cycle + 1, refinement_cycles, current_score)

        self.checkpoint.cleanup()

        return DilationResult(
            output=current_best,
            score=current_score,
            cycles_completed=refinement_cycles,
            elapsed_seconds=time.time() - start,
            convergence_detected=convergence_detected,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_controller.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/timedilate/controller.py tests/test_controller.py
git commit -m "feat: add dilation controller with cycle management"
```

---

### Task 9: CLI

**Files:**
- Create: `src/timedilate/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_cli.py
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
from timedilate.cli import main

def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "budget" in result.output
    assert "factor" in result.output

def test_cli_parses_args():
    runner = CliRunner()
    with patch("timedilate.cli.run_dilation") as mock_run:
        mock_run.return_value = MagicMock(
            output="result",
            score=85,
            cycles_completed=5,
            elapsed_seconds=2.5,
            convergence_detected=False,
        )
        result = runner.invoke(main, ["Write hello", "--factor", "5", "--budget", "10"])
        assert result.exit_code == 0
        mock_run.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL

- [ ] **Step 3: Implement CLI**

```python
# src/timedilate/cli.py
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from timedilate.config import TimeDilateConfig
from timedilate.engine import InferenceEngine
from timedilate.controller import DilationController, DilationResult

console = Console()


def run_dilation(prompt: str, config: TimeDilateConfig) -> DilationResult:
    engine = InferenceEngine(config)
    controller = DilationController(config, engine)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[bold]{task.completed}/{task.total} cycles"),
        TextColumn("score: {task.fields[score]}"),
        console=console,
    ) as progress:
        task = progress.add_task("Dilating time...", total=config.dilation_factor - 1, score=0)

        def on_cycle(cycle, total, score):
            progress.update(task, completed=cycle, score=score)

        result = controller.run(prompt, on_cycle=on_cycle)

    return result


@click.command()
@click.argument("prompt")
@click.option("--factor", default=2, help="Dilation factor (e.g., 2, 100, 1000000)")
@click.option("--budget", default=30.0, help="Advisory time budget in seconds")
@click.option("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model name or path")
@click.option("--draft-model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Draft model for speculative decoding")
@click.option("--branches", default=3, help="Branch factor per cycle")
@click.option("--output", "output_file", default=None, help="Save output to file")
@click.option("--verbose", is_flag=True, help="Show detailed progress")
def main(prompt, factor, budget, model, draft_model, branches, output_file, verbose):
    """AI Time Dilation Runtime — make AI think longer in less time."""
    config = TimeDilateConfig(
        model=model,
        draft_model=draft_model,
        dilation_factor=factor,
        budget_seconds=budget,
        branch_factor=branches,
    )

    console.print(f"[bold green]Time Dilation Runtime[/]")
    console.print(f"  Model: {config.model}")
    console.print(f"  Factor: {factor}x ({factor - 1} refinement cycles)")
    console.print(f"  Branches: {branches} per cycle")
    console.print()

    result = run_dilation(prompt, config)

    console.print()
    console.print(f"[bold green]Complete![/] {result.cycles_completed} cycles in {result.elapsed_seconds:.1f}s | Score: {result.score}/100")
    if result.convergence_detected:
        console.print("[yellow]Note: convergence detected — output may have plateaued[/]")
    console.print()
    console.print(result.output)

    if output_file:
        with open(output_file, "w") as f:
            f.write(result.output)
        console.print(f"\n[dim]Saved to {output_file}[/]")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/timedilate/cli.py tests/test_cli.py
git commit -m "feat: add CLI with progress display"
```

---

### Task 10: Integration Test & Final Verification

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test with mock engine**

```python
# tests/test_integration.py
from unittest.mock import MagicMock
from timedilate.controller import DilationController
from timedilate.config import TimeDilateConfig

def test_full_pipeline_dilation_2x():
    """End-to-end: dilation=2 means 1 refinement cycle."""
    config = TimeDilateConfig(dilation_factor=2, branch_factor=1)
    mock_engine = MagicMock()
    mock_engine.generate = MagicMock(side_effect=[
        "def sort(lst): return sorted(lst)",  # initial
        "75",                                   # score initial
        "def sort(lst):\n    if not lst:\n        return []\n    return sorted(lst)",  # improved
        "90",                                   # score improved
    ])

    controller = DilationController(config, mock_engine)
    result = controller.run("Write a sort function")

    assert result.cycles_completed == 1
    assert result.score == 90
    assert "if not lst" in result.output

def test_full_pipeline_dilation_5x():
    """End-to-end: dilation=5 means 4 refinement cycles."""
    config = TimeDilateConfig(dilation_factor=5, branch_factor=1)
    mock_engine = MagicMock()
    responses = [
        "v0", "50",           # initial + score
        "v1", "60",           # cycle 1
        "v2", "70",           # cycle 2
        "v3", "80",           # cycle 3
        "v4", "90",           # cycle 4
    ]
    mock_engine.generate = MagicMock(side_effect=responses)

    controller = DilationController(config, mock_engine)
    result = controller.run("test prompt")

    assert result.cycles_completed == 4
    assert result.score == 90
    assert result.output == "v4"
```

- [ ] **Step 2: Run all tests**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 3: Run linting**

Run: `python -m py_compile src/timedilate/cli.py && python -m py_compile src/timedilate/controller.py && python -m py_compile src/timedilate/improver.py`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "feat: add integration tests for full dilation pipeline"
```

- [ ] **Step 5: Final commit — tag v0.1.0**

```bash
git tag v0.1.0
```
