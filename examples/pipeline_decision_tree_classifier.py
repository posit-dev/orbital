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
from sklearn.tree import DecisionTreeClassifier

import orbitalml
import orbitalml.types

PRINT_SQL = int(os.environ.get("PRINTSQL", "0"))
ASSERT = int(os.environ.get("ASSERT", "0"))

logging.basicConfig(level=logging.INFO)
logging.getLogger("orbitalml").setLevel(logging.INFO)  # Change to DEBUG to see each translation step.

iris = load_iris()
df = pd.DataFrame(
    iris.data, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"]
)

# Let's create a new categorical column "petal_width_cat" based on the petal_width values.
# So we have a reason to use OneHotEncoder in the pipeline.
df["petal_width_cat"] = np.where(df["petal_width"] < 1.0, "narrow", "wide")

# Introduce some missing values in the sepal_width column.
# So we have a reason to use SimpleImputer in the pipeline.
df.loc[[0, 10, 20], "sepal_width"] = np.nan

# Create some kind of categorical that we can predict.
# Divide the sepal area into 3 categories: small, medium, large.
df["area"] = df["sepal_length"] * df["sepal_width"]
df = df.dropna(subset=["area"]).copy()

def categorize_area(a: float) -> str:
    """Given the area, divide the sepal into 3 categories."""
    if a < 15:
        return "small"
    elif a < 18:
        return "medium"
    else:
        return "large"
df["sepal_size_cat"] = df["area"].apply(categorize_area)
df.drop(columns=["area"], inplace=True)

# Split between features and prediction
X = df[
    ["sepal_length", "sepal_width", "petal_length", "petal_width", "petal_width_cat"]
]
y = df["sepal_size_cat"]

# Predict the sepal_size_cat based on the other columns.
pipeline = Pipeline(
    [
        (
            "preprocessor",
            ColumnTransformer(
                transformers=[
                    (
                        "num",
                        SimpleImputer(strategy="mean"),
                        ["sepal_length", "sepal_width", "petal_length", "petal_width"],
                    ),
                    (
                        "cat",
                        OneHotEncoder(handle_unknown="ignore"),
                        ["petal_width_cat"],
                    ),
                ]
            ),
        ),
        ("tree", DecisionTreeClassifier(random_state=42)),
    ]
)

pipeline.fit(X, y)

features = orbitalml.types.guess_datatypes(X)
print("Mustela Features:", features)

orbitalml_pipeline = orbitalml.parse_pipeline(pipeline, features=features)
print(orbitalml_pipeline)

# Test data
example_data = pa.table(
    {
        "sepal_length": [5.0, 6.1, 7.2, 5.843333],
        "sepal_width": [3.2, 2.8, 3.0, 3.057333],
        "petal_length": [1.2, 4.7, 6.1, 3.758000],
        "petal_width": [0.2, 1.2, 2.3, 1.199333],
        "petal_width_cat": ["narrow", "wide", "wide", "wide"],
    }
)

con = ibis.duckdb.connect()
ibis_table = ibis.memtable(example_data, name="DATA_TABLE")
ibis_expression = orbitalml.translate(ibis_table, orbitalml_pipeline)

if PRINT_SQL:
    sql = orbitalml.export_sql("DATA_TABLE", orbitalml_pipeline, dialect="duckdb")
    print("\nGenerated Query for DuckDB:")
    print(sql)
    print("\nPrediction with SQL")
    # We need to create the table for the SQL to query it.
    con.create_table(ibis_table.get_name(), obj=ibis_table)
    print(con.raw_sql(sql).df())

print("\nPrediction with Ibis")
ibis_target = ibis_expression.execute()
print(ibis_target)

print("\nPrediction with SKLearn")
test_df = example_data.to_pandas()
target = pipeline.predict(test_df)
print(target)

if ASSERT:
    assert np.array_equal(target, ibis_target["output_label"]), "Predictions do not match!"
    print("\nPredictions match!")
