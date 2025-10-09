import os
import logging
import ibis
import numpy as np
import pyarrow as pa
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import orbital
import orbital.types

PRINT_SQL = int(os.environ.get("PRINTSQL", "0"))
ASSERT = int(os.environ.get("ASSERT", "0"))


BACKEND = os.environ.get("BACKEND", "duckdb").lower()

if BACKEND not in {"duckdb", "sqlite"}:
    raise ValueError(f"Unsupported backend {BACKEND!r}")


logging.basicConfig(level=logging.INFO)
logging.getLogger("orbital").setLevel(logging.INFO)  # Set DEBUG to see translation process.

# Load the Iris dataset
iris = load_iris(as_frame=True)
iris_x = iris.data

# SQL and orbital don't like dots in column names, replace them with underscores
iris_x.columns = [cname.replace(".", "_") for cname in iris_x.columns]

numeric_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
iris_x = iris_x.set_axis(numeric_cols, axis=1)

# Create a pipeline with ElasticNet instead of LinearRegression
pipeline = Pipeline(
    [
        (
            "preprocess",
            ColumnTransformer(
                [("scaler", StandardScaler(with_std=False), numeric_cols)],
                remainder="passthrough",
            ),
        ),
        ("elastic_net", ElasticNet(alpha=0.1, l1_ratio=0.5)),  # ElasticNet with L1/L2 regularization
    ]
)

pipeline.fit(iris_x, iris.target)

# Convenience for this example to avoid repeating the schema,
# in real cases, the user would know the schema of its database.
features = orbital.types.guess_datatypes(iris_x)

orbital_pipeline = orbital.parse_pipeline(pipeline, features=features)
print(orbital_pipeline)

# Example data for testing, including at least one training value
example_data = pa.table(
    {
        "sepal_length": [5.0, 6.1, 7.2, 5.843333],
        "sepal_width": [3.2, 2.8, 3.0, 3.057333],
        "petal_length": [1.2, 4.7, 6.1, 3.758000],
        "petal_width": [0.2, 1.2, 2.3, 1.199333],
    }
)

# Generate a query expression using orbital
ibis_table = ibis.memtable(example_data, name="DATA_TABLE")
ibis_expression = orbital.translate(ibis_table, orbital_pipeline)

con = {
    "sqlite": lambda: ibis.sqlite.connect(":memory:"),
    "duckdb": ibis.duckdb.connect,
}[BACKEND]()
if PRINT_SQL:
    sql = orbital.export_sql("DATA_TABLE", orbital_pipeline, dialect=BACKEND)
    print(f"\nGenerated Query for {BACKEND.upper()}:")
    print(sql)
    print("\nPrediction with SQL")
    # We need to create the table for the SQL to query it.
    con.create_table(ibis_table.get_name(), obj=example_data)
    print(con.execute(con.sql(sql)))

print("\nPrediction with Ibis")
ibis_target = con.execute(ibis_expression)["variable"].to_numpy()
print(ibis_target)

print("\nPrediction with SKLearn")
target = pipeline.predict(example_data.to_pandas())
print(target)

if ASSERT:
    assert np.allclose(target, ibis_target), "Predictions do not match!"
    print("\nPredictions match!")
