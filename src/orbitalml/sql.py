"""Export SQL for a pipeline

This module provides a function to export SQL queries for a given pipeline.
It allows to use the prediction pipeline on any supported SQL dialect
without the need for a python runtime environment.
"""

import ibis
import ibis.backends.sql.compilers as sc
import sqlglot.optimizer
import sqlglot.optimizer.optimizer
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
) -> str:
    """Export SQL for a given pipeline.

    Given a orbitalml pipeline, this function generates a SQL query that can be
    used to execute the pipeline on a database. The generated SQL is compatible
    with the specified SQL dialect.

    `dialect` can be any of the SQL dialects supported by sqlglot,
    see :class:`sqlglot.dialects.DIALECTS` for a complete list of supported dialects.

    If `optimize` is set to True, the SQL query will be optimized using
    sqlglot's optimizer. This can improve performance, but may fail if
    the query is complex.
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

    ibis_expr = translate(unbound_table, pipeline, projection=projection)
    sqlglot_expr = getattr(sc, dialect).compiler.to_sqlglot(ibis_expr)

    if optimize:
        c = Catalog()
        catalog = {
            unbound_table.get_name(): c.to_sqlglot_schema(unbound_table.schema())
        }
        sqlglot_expr = sqlglot.optimizer.optimize(
            sqlglot_expr, schema=catalog, rules=OPTIMIZER_RULES
        )

    return sqlglot_expr.sql(dialect=dialect)
