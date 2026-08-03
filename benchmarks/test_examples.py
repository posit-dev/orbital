"""Benchmark the Orbital portion of each model example."""

import importlib.util
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
EXAMPLE_FILES = sorted(
    [*EXAMPLES_DIR.glob("pipeline_*.py"), *EXAMPLES_DIR.glob("pytorch_*.py")]
)

os.environ.update(
    BACKEND="duckdb",
    PRINT_SQL="1",
    PREDICT_WITH_LIBRARY="0",
    SCIKIT_LEARN_DATA=str(DATA_HOME),
)


def load_example(path: Path) -> ModuleType:
    """Import an example, preparing its data and fitted model once."""
    module_name = f"orbital_benchmark_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import example from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=lambda path: path.stem)
def test_example(benchmark: BenchmarkFixture, path: Path) -> None:
    """Measure Orbital without training or reference-library prediction."""
    example = load_example(path)
    # One round keeps CI practical while Perfall provides the historical sample set.
    benchmark.pedantic(example.main, rounds=1, iterations=1)
