import pandas as pd
import sqlglot


def tree_columns(sql, dialect="duckdb"):
    """Split the per-tree columns of ``sql`` into constant and computed.

    Only meaningful for SQL exported with ``separate_trees=True``, which is what
    names a column per tree.
    """
    constant, computed = [], []
    for alias in sqlglot.parse_one(sql, read=dialect).find_all(sqlglot.exp.Alias):
        if not alias.alias.startswith("tre_"):
            continue
        if isinstance(alias.this, sqlglot.exp.Literal):
            constant.append(alias.alias)
        else:
            computed.append(alias.alias)
    return constant, computed


def execute_sql(sql, conn, dialect, data):
    """Execute SQL query on the appropriate database connection."""
    if dialect == "duckdb":
        conn.execute("CREATE TABLE data AS SELECT * FROM data")
        result = conn.execute(sql).fetchdf()
    elif dialect in ("sqlite", "postgres"):
        data.to_sql("data", conn, index=False, if_exists="replace")
        result = pd.read_sql(sql, conn)
    return result
