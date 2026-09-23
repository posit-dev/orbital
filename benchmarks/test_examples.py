"""Benchmark the Orbital portion of each model example."""

import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType

import ibis
import pytest
from ibis.expr import operations as ops
from pytest_benchmark.fixture import BenchmarkFixture

import orbital

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

_LOADED_EXAMPLES: dict[Path, ModuleType] = {}


def load_example(path: Path) -> ModuleType:
    """Import an example, preparing its data and fitted model once."""
    if cached := _LOADED_EXAMPLES.get(path):
        return cached

    module_name = f"orbital_benchmark_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import example from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _LOADED_EXAMPLES[path] = module
    return module


def example_table(example: ModuleType) -> tuple[str, ibis.Table]:
    """Return the name and expression of the table an example reads from.

    Every example aliases its input table to the name its exported SQL reads
    from, so the alias in the translated expression is that input table.
    """
    (source,) = example.ibis_expression.op().find(
        ops.AliasedRelation, filter=ops.Relation
    )
    return source.name, source.to_expr()


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=lambda path: path.stem)
def test_example(benchmark: BenchmarkFixture, path: Path) -> None:
    """Measure Orbital without training or reference-library prediction."""
    example = load_example(path)
    # One round keeps CI practical while Perfall provides the historical sample set.
    benchmark.pedantic(example.main, rounds=1, iterations=1)


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=lambda path: path.stem)
def test_translate(benchmark: BenchmarkFixture, path: Path) -> None:
    """Measure building the Ibis expression an example is translated into."""
    example = load_example(path)
    _, table = example_table(example)
    benchmark.pedantic(
        lambda: orbital.translate(table, example.orbital_pipeline),
        rounds=1,
        iterations=1,
    )


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=lambda path: path.stem)
def test_execute(benchmark: BenchmarkFixture, path: Path) -> None:
    """Measure running the SQL generated for an example on DuckDB.

    The SQL is exported before the benchmark starts, so this number covers the
    execution phase alone and not translation or compilation.
    """
    example = load_example(path)
    table_name, _ = example_table(example)
    sql = orbital.export_sql(table_name, example.orbital_pipeline, dialect="duckdb")
    benchmark.pedantic(
        lambda: example.con.raw_sql(sql).fetchall(), rounds=1, iterations=1
    )
