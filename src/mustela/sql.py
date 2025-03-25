"""Export SQL for a pipeline

This module provides a function to export SQL queries for a given pipeline.
It allows to use the prediction pipeline on any supported SQL dialect
without the need for a python runtime environment.
"""

import ibis
import ibis.backends.sql.compilers as sc
import sqlglot.optimizer
from ibis.expr.sql import Catalog

from .ast import ParsedPipeline
from .translate import translate


def export_sql(
    table_name: str,
    pipeline: ParsedPipeline,
    dialect: str = "duckdb",
    optimize: bool = True,
) -> str:
    """Export SQL for a given pipeline.

    Given a mustela pipeline, this function generates a SQL query that can be
    used to execute the pipeline on a database. The generated SQL is compatible
    with the specified SQL dialect.

    `dialect` can be any of the SQL dialects supported by sqlglot,
    see :class:`sqlglot.dialects.DIALECTS`` for a complete list of supported dialects.

    If `optimize` is set to True, the SQL query will be optimized using
    sqlglot's optimizer.
    """
    unbound_table = ibis.table(
        schema={
            fname: ftype._to_ibistype() for fname, ftype in pipeline.features.items()
        },
        name=table_name,
    )

    ibis_expr = translate(unbound_table, pipeline)
    sqlglot_expr = sc.duckdb.compiler.to_sqlglot(ibis_expr)

    if optimize:
        catalog = Catalog({unbound_table.get_name(): unbound_table}).to_sqlglot()
        sqlglot_expr = sqlglot.optimizer.optimize(sqlglot_expr, schema=catalog)

    return sqlglot_expr.sql(dialect=dialect)
