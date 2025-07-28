import pandas as pd


def execute_sql(sql, conn, dialect, data):
    """Execute SQL query on the appropriate database connection."""
    if dialect == "duckdb":
        conn.execute("CREATE TABLE data AS SELECT * FROM data")
        result = conn.execute(sql).fetchdf()
    elif dialect in ("sqlite", "postgres"):
        data.to_sql("data", conn, index=False, if_exists="replace")
        result = pd.read_sql(sql, conn)
    return result
