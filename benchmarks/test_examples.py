"""Benchmark the Orbital portion of each model example."""

import importlib
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

PROJECT_ROOT = Path(__file__).parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples"
DATA_HOME = Path(tempfile.gettempdir()) / "orbital-benchmark-data"
EXAMPLE_MODULES = [
    f"examples.{path.stem}"
    for path in sorted(
        [*EXAMPLES_DIR.glob("pipeline_*.py"), *EXAMPLES_DIR.glob("pytorch_*.py")]
    )
]

sys.path.insert(0, str(PROJECT_ROOT))
os.environ.update(
    BACKEND="duckdb",
    PRINT_SQL="1",
    ASSERT="1",
    SCIKIT_LEARN_DATA=str(DATA_HOME),
)


def load_example(module_name: str) -> ModuleType:
    """Import an example, preparing its data and fitted model once."""
    return importlib.import_module(module_name)


@pytest.mark.parametrize(
    "module_name", EXAMPLE_MODULES, ids=lambda name: name.rsplit(".", 1)[-1]
)
def test_example(benchmark: BenchmarkFixture, module_name: str) -> None:
    """Measure model conversion and execution without setup or fitting."""
    example = load_example(module_name)
    # One round keeps CI practical while Perfall provides the historical sample set.
    benchmark.pedantic(example.main, rounds=1, iterations=1)
