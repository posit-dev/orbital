import os
import logging

import ibis
import numpy as np
import pyarrow as pa
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import orbital
import orbital.types

PRINT_SQL = int(os.environ.get("PRINT_SQL", "0"))
ASSERT = int(os.environ.get("ASSERT", "0"))
BACKEND = os.environ.get("BACKEND", "duckdb").lower()

if BACKEND not in {"duckdb", "sqlite"}:
    raise ValueError(f"Unsupported backend {BACKEND!r}")

logging.basicConfig(level=logging.INFO)
logging.getLogger("orbital").setLevel(logging.INFO)  # Set DEBUG to see translation process.

iris = load_iris(as_frame=True)
iris_x = iris.data

# SQL and orbital don't like dots in column names, replace them with underscores
iris_x.columns = [cname.replace(".", "_") for cname in iris_x.columns]

numeric_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
iris_x = iris_x.set_axis(numeric_cols, axis=1)

pipeline = Pipeline(
    [
        (
            "preprocess",
            ColumnTransformer(
                [("scaler", StandardScaler(with_std=False), numeric_cols)],
                remainder="passthrough",
            ),
        ),
        ("Lasso", Lasso()),
    ]
)
pipeline.fit(iris_x, iris.target)

features = orbital.types.guess_datatypes(iris_x)

orbital_pipeline = orbital.parse_pipeline(pipeline, features=features)
print(orbital_pipeline)

# Include at least 1 value from training set to confirm the right computation happened
example_data = pa.table(
    {
        "sepal_length": [5.0, 6.1, 7.2, 5.843333],
        "sepal_width": [3.2, 2.8, 3.0, 3.057333],
        "petal_length": [1.2, 4.7, 6.1, 3.758000],
        "petal_width": [0.2, 1.2, 2.3, 1.199333],
    }
)

ibis_table = ibis.memtable(example_data, name="DATA_TABLE")
ibis_expression = orbital.translate(ibis_table, orbital_pipeline)

con = {
    "sqlite": lambda: ibis.sqlite.connect(":memory:"),
    "duckdb": lambda: ibis.duckdb.connect(),
}[BACKEND]()
if PRINT_SQL:
    sql = orbital.export_sql("DATA_TABLE", orbital_pipeline, dialect=BACKEND)
    print(f"\nGenerated Query for {BACKEND.upper()}:")
    print(sql)
    print("\nPrediction with SQL")
    # We need to create the table for the SQL to query it.
    con.create_table(ibis_table.get_name(), obj=ibis_table)
    print(con.raw_sql(sql).fetchall())

print("\nPrediction with Ibis")
ibis_target = con.execute(ibis_expression)["variable"].to_numpy()
print(ibis_target)

print("\nPrediction with SKLearn")
target = pipeline.predict(example_data.to_pandas())
print(target)

if ASSERT:
    assert np.allclose(target, ibis_target), "Predictions do not match!"
    print("\nPredictions match!")
