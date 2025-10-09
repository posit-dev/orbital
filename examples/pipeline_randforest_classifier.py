import os
import logging
import ibis
import numpy as np
import pandas as pd
import pyarrow as pa
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

import orbital
import orbital.types

PRINT_SQL = int(os.environ.get("PRINT_SQL", "0"))
ASSERT = int(os.environ.get("ASSERT", "0"))
BACKEND = os.environ.get("BACKEND", "duckdb").lower()

if BACKEND not in {"duckdb", "sqlite"}:
    raise ValueError(f"Unsupported backend {BACKEND!r}")


logging.basicConfig(level=logging.INFO)
logging.getLogger("orbital").setLevel(logging.INFO)  # Set DEBUG to see translation process.

iris = load_iris()
df = pd.DataFrame(iris.data, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])

# Create a categorical column based on petal_width
df["petal_width_cat"] = np.where(df["petal_width"] < 1.0, "narrow", "wide")

# Introduce missing values in sepal_width to need imputation
df.loc[[0, 10, 20], "sepal_width"] = np.nan

X = df[["sepal_length", "sepal_width", "petal_length", "petal_width", "petal_width_cat"]]
y = iris.target

pipeline = Pipeline([
    (
        "preprocessor",
        ColumnTransformer([
            ("num", SimpleImputer(strategy="mean"), ["sepal_length", "sepal_width", "petal_length", "petal_width"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["petal_width_cat"]),
        ])
    ),
    ("rf", RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X, y)


features = orbital.types.guess_datatypes(X)
print("orbital Features:", features)

orbital_pipeline = orbital.parse_pipeline(pipeline, features=features)
print(orbital_pipeline)

# Prepare test data for predictions
example_data = pa.table(
    {
        "sepal_length": [5.0, 6.1, 7.2, 5.843333],
        "sepal_width": [3.2, 2.8, 3.0, 3.057333],
        "petal_length": [1.2, 4.7, 6.1, 3.758000],
        "petal_width": [0.2, 1.2, 2.3, 1.199333],
        "petal_width_cat": ["narrow", "wide", "wide", "wide"],
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
ibis_target = con.execute(ibis_expression)
print(ibis_target)

print("\nPrediction with SKLearn")
test_df = example_data.to_pandas()
target = pipeline.predict(test_df)
print(target)

if ASSERT:
    assert np.array_equal(target, ibis_target["output_label"]), "Predictions do not match!"
    print("\nPredictions match!")