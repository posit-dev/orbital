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
from sklearn.tree import DecisionTreeRegressor

import orbital
import orbital.types

PRINT_SQL = int(os.environ.get("PRINT_SQL", "0"))
ASSERT = int(os.environ.get("ASSERT", "0"))
PREDICT_WITH_LIBRARY = int(os.environ.get("PREDICT_WITH_LIBRARY", "1"))
BACKEND = os.environ.get("BACKEND", "duckdb").lower()

if BACKEND not in {"duckdb", "sqlite"}:
    raise ValueError(f"Unsupported backend {BACKEND!r}")


logging.basicConfig(level=logging.INFO)
logging.getLogger("orbital").setLevel(logging.INFO)  # Set DEBUG to see translation process.

# Load the dataset
iris = load_iris()
df = pd.DataFrame(
    iris.data, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"]
)

# Add a categorical column to exercise OneHotEncoder
df["petal_width_cat"] = np.where(df["petal_width"] < 1.0, "narrow", "wide")

# Introduce some missing values
df.loc[[0, 10, 20], "sepal_width"] = np.nan

# Numeric target: predict petal length
y = df["petal_length"]
X = df.drop(columns=["petal_length"])  # The other columns are features

# Create the preprocessing and regression pipeline
pipeline = Pipeline(
    [
        (
            "preprocessor",
            ColumnTransformer(
                transformers=[
                    (
                        "num",
                        SimpleImputer(strategy="mean"),
                        ["sepal_length", "sepal_width", "petal_width"],
                    ),
                    (
                        "cat",
                        OneHotEncoder(handle_unknown="ignore"),
                        ["petal_width_cat"],
                    ),
                ]
            ),
        ),
        ("tree", DecisionTreeRegressor(random_state=42)),
    ]
)

pipeline.fit(X, y)

# Prepare the inputs outside the benchmarked function.
features = orbital.types.guess_datatypes(X)
example_data = pa.table(
    {
        "sepal_length": [5.0, 6.1, 7.2, 5.843333],
        "sepal_width": [3.2, 2.8, 3.0, 3.057333],
        "petal_width": [0.2, 1.2, 2.3, 1.199333],
        "petal_width_cat": ["narrow", "wide", "wide", "wide"],
    }
)
con = {
    "sqlite": lambda: ibis.sqlite.connect(":memory:"),
    "duckdb": lambda: ibis.duckdb.connect(),
}[BACKEND]()
if PRINT_SQL:
    con.create_table("DATA_TABLE", obj=example_data)


def main():
    print("orbital Features:", features)

    # Convert the pipeline to SQL with Orbital
    orbital_pipeline = orbital.parse_pipeline(pipeline, features=features)
    print(orbital_pipeline)

    if PRINT_SQL:
        sql = orbital.export_sql("DATA_TABLE", orbital_pipeline, dialect=BACKEND)
        print(f"\nGenerated Query for {BACKEND.upper()}:")
        print(sql)
        print("\nPrediction with SQL")
        print(con.raw_sql(sql).fetchall())

    if PREDICT_WITH_LIBRARY:
        print("\nPrediction with SKLearn")
        test_df = example_data.to_pandas()
        target = pipeline.predict(test_df)
        print(target)

    print("\nPrediction with Ibis")
    ibis_table = ibis.memtable(example_data).alias("DATA_TABLE")
    ibis_expression = orbital.translate(ibis_table, orbital_pipeline)
    ibis_target = con.execute(ibis_expression)["variable"].to_numpy()
    print(ibis_target)

    if ASSERT and PREDICT_WITH_LIBRARY:
        assert np.allclose(target, ibis_target), "Predictions do not match!"
        print("\nPredictions match!")


if __name__ == "__main__":
    main()
