"""Export SQL for a pipeline

This module provides a function to export SQL queries for a given pipeline.
It allows to use the prediction pipeline on any supported SQL dialect
without the need for a python runtime environment.
"""

import math
import typing

import ibis
import ibis.backends.sql.compilers as sc
import sqlglot.expressions
import sqlglot.optimizer
import sqlglot.optimizer.optimizer
import sqlglot.schema
from ibis.expr.sql import Catalog

from .ast import ParsedPipeline
from .translate import ResultsProjection, translate

OPTIMIZER_RULES = (
    sqlglot.optimizer.optimizer.qualify,
    sqlglot.optimizer.optimizer.pushdown_projections,
    sqlglot.optimizer.optimizer.normalize,
    sqlglot.optimizer.optimizer.unnest_subqueries,
    sqlglot.optimizer.optimizer.pushdown_predicates,
    sqlglot.optimizer.optimizer.optimize_joins,
    sqlglot.optimizer.optimizer.eliminate_subqueries,
    # sqlglot.optimizer.optimizer.merge_subqueries,  # This makes the SQLGlot optimizer choke with OOMs
    sqlglot.optimizer.optimizer.eliminate_joins,
    sqlglot.optimizer.optimizer.eliminate_ctes,
    sqlglot.optimizer.optimizer.quote_identifiers,
    sqlglot.optimizer.optimizer.canonicalize,
    # sqlglot.optimizer.optimizer.annotate_types,  # This makes the SQLGlot optimizer choke with maximum recursion
    sqlglot.optimizer.optimizer.simplify,
)


def export_sql(
    table_name: str,
    pipeline: ParsedPipeline,
    dialect: str = "duckdb",
    projection: ResultsProjection = ResultsProjection(),
    optimize: bool = True,
    allow_text_tensors: bool = False,
) -> str:
    """Export SQL for a given pipeline.

    Generates a SQL statement equivalent to the provided pipeline for the
    requested dialect. The statement can be executed directly on a database
    that exposes a table matching ``table_name``. Dialect names correspond to
    those listed in ``sqlglot.dialects.DIALECTS``.

    If ``optimize`` is ``True`` the statement is post-processed by sqlglot's
    optimizer, which often produces more compact SQL but may fail on complex
    expressions.

    :param table_name: Name of the source table used in generated SQL.
    :param pipeline: Parsed pipeline to export.
    :param dialect: Target SQL dialect (any supported by sqlglot).
    :param projection: Optional result projection helper.
    :param optimize: Whether to run the sqlglot optimizer (default ``True``).
    :param allow_text_tensors: Forwarded to [orbital.translate.translate][]; controls whether
        numeric/bool tensors coerced to text in ONNX should remain text in the
        resulting SQL. Defaults to ``False`` to keep encoded columns numeric.
    """
    unbound_table = ibis.table(
        schema={
            fname: ftype._to_ibistype() for fname, ftype in pipeline.features.items()
        },
        name=table_name,
    )

    if projection._omit:
        raise ValueError(
            "Projection is empty. Please provide a projection to export SQL."
        )

    ibis_expr = translate(
        unbound_table,
        pipeline,
        projection=projection,
        allow_text_tensors=allow_text_tensors,
    )
    if dialect == "duckdb":
        sqlglot_expr = _OrbitalDuckDBCompiler().to_sqlglot(ibis_expr)
    else:
        sqlglot_expr = getattr(sc, dialect).compiler.to_sqlglot(ibis_expr)

    if optimize:
        c = Catalog()
        catalog = sqlglot.schema.MappingSchema(
            {unbound_table.get_name(): c.to_sqlglot_schema(unbound_table.schema())},
            normalize=False,
        )
        sqlglot_expr = sqlglot.optimizer.optimize(
            sqlglot_expr, schema=catalog, rules=OPTIMIZER_RULES
        )

    return sqlglot_expr.sql(dialect=dialect)


class _OrbitalDuckDBCompiler(sc.duckdb.DuckDBCompiler):
    """DuckDB compiler that emits floating-point literals in scientific notation.

    DuckDB infers bare decimal literals (``0.94``) as ``DECIMAL(n,m)`` rather
    than ``DOUBLE``.  When many such values are summed — as happens in tree
    ensemble models — the internal ``DECIMAL(18)`` storage overflows.
    Appending ``E0`` (e.g. ``0.94E0``) makes DuckDB treat the literal as
    ``DOUBLE`` without the verbosity of an explicit ``CAST``.
    """

    def visit_NonNullLiteral(
        self, op: typing.Any, *, value: typing.Any, dtype: typing.Any
    ) -> typing.Any:
        """Emit floating-point literals with an ``E0`` suffix for DOUBLE inference."""
        if (
            dtype.is_floating()
            and isinstance(value, (int, float))
            and math.isfinite(value)
        ):
            # Values that Python already renders in scientific notation (e.g. 1e-19)
            # are already inferred as DOUBLE by DuckDB — only bare decimals need E0.
            text = str(value)
            if "e" not in text and "E" not in text:
                abs_text = str(-value) if value < 0 else text
                lit = sqlglot.expressions.Literal(this=abs_text + "E0", is_string=False)
                return sqlglot.expressions.Neg(this=lit) if value < 0 else lit
        return super().visit_NonNullLiteral(op, value=value, dtype=dtype)
