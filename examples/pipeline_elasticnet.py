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

import orbitalml
import orbitalml.types

PRINT_SQL = int(os.environ.get("PRINTSQL", "0"))
ASSERT = int(os.environ.get("ASSERT", "0"))

logging.basicConfig(level=logging.INFO)
logging.getLogger("orbitalml").setLevel(logging.INFO)  # Set DEBUG to see translation process.

# Load the Iris dataset
iris = load_iris(as_frame=True)
iris_x = iris.data

# SQL and Mustela don't like dots in column names, replace them with underscores
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
features = orbitalml.types.guess_datatypes(iris_x)

orbitalml_pipeline = orbitalml.parse_pipeline(pipeline, features=features)
print(orbitalml_pipeline)

# Example data for testing, including at least one training value
example_data = pa.table(
    {
        "sepal_length": [5.0, 6.1, 7.2, 5.843333],
        "sepal_width": [3.2, 2.8, 3.0, 3.057333],
        "petal_length": [1.2, 4.7, 6.1, 3.758000],
        "petal_width": [0.2, 1.2, 2.3, 1.199333],
    }
)

# Generate a query expression using Mustela
ibis_table = ibis.memtable(example_data, name="DATA_TABLE")
ibis_expression = orbitalml.translate(ibis_table, orbitalml_pipeline)

con = ibis.duckdb.connect()
if PRINT_SQL:
    sql = orbitalml.export_sql("DATA_TABLE", orbitalml_pipeline, dialect="duckdb")
    print("\nGenerated Query for DuckDB:")
    print(sql)
    print("\nPrediction with SQL")
    # We need to create the table for the SQL to query it.
    con.create_table(ibis_table.get_name(), obj=ibis_table)
    print(con.raw_sql(sql).df())

print("\nPrediction with Ibis")
ibis_target = ibis_expression.execute()["variable"].to_numpy()
print(ibis_target)

print("\nPrediction with SKLearn")
target = pipeline.predict(example_data.to_pandas())
print(target)

if ASSERT:
    assert np.allclose(target, ibis_target), "Predictions do not match!"
    print("\nPredictions match!")