import os
import logging

import ibis
import pyarrow as pa
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import mustela
import mustela.types

PRINT_SQL = int(os.environ.get("PRINTSQL", "0"))
ASSERT = int(os.environ.get("ASSERT", "0"))

logging.basicConfig(level=logging.INFO)
logging.getLogger("mustela").setLevel(logging.INFO)  # Set DEBUG to see translation process.

iris = load_iris(as_frame=True)
iris_x = iris.data

# SQL and Mustela don't like dots in column names, replace them with underscores
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
        ("linear_regression", LinearRegression()),
    ]
)
pipeline.fit(iris_x, iris.target)

# Convenience for this example to avoid repeating the schema,
# in real cases, the user would know the schema of its database.
features = mustela.types.guess_datatypes(iris_x)

mustela_pipeline = mustela.parse_pipeline(pipeline, features=features)
print(mustela_pipeline)

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

ibis_expression = mustela.translate(ibis_table, mustela_pipeline)
con = ibis.duckdb.connect()

if PRINT_SQL:
    sql = mustela.export_sql("DATA_TABLE", mustela_pipeline, dialect="duckdb")
    print("\nGenerated Query for DuckDB:")
    print(sql)
    print("\nPrediction with SQL")
    # We need to create the table for the SQL to query it.
    con.create_table(ibis_table.get_name(), obj=ibis_table)
    print(con.raw_sql(sql).df())

print("\nPrediction with Ibis")
ibis_predictions = ibis_expression.execute()
print(ibis_predictions)

print("\nPrediction with SKLearn")
predictions = pipeline.predict(example_data.to_pandas())
print(predictions)

if ASSERT:
    assert np.allclose(ibis_predictions["variable"], predictions), "Predictions do not match!"
    print("\nPredictions match!")