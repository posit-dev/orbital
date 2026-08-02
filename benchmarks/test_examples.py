"""Benchmark Orbital's end-to-end model examples."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

PROJECT_ROOT = Path(__file__).parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples"
DATA_HOME = Path(tempfile.gettempdir()) / "orbital-benchmark-data"
EXAMPLES = sorted(EXAMPLES_DIR.glob("pipeline_*.py")) + sorted(
    EXAMPLES_DIR.glob("pytorch_*.py")
)


def run_example(example: Path) -> None:
    """Run one example with the same validation enabled by test_examples.sh."""
    env = os.environ.copy()
    env.update(
        BACKEND="duckdb",
        PRINT_SQL="1",
        ASSERT="1",
        SCIKIT_LEARN_DATA=str(DATA_HOME),
    )
    result = subprocess.run(
        [sys.executable, str(example)],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode:
        raise AssertionError(f"{example.name} failed:\n{result.stdout}")


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda example: example.stem)
def test_example(benchmark: BenchmarkFixture, example: Path) -> None:
    """Measure one complete model conversion and execution scenario."""
    # Each example is an end-to-end scenario that can take several seconds.
    # One round keeps CI practical while Perfall provides the historical sample set.
    benchmark.pedantic(run_example, args=(example,), rounds=1, iterations=1)
