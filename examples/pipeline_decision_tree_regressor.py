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
BACKEND = os.environ.get("BACKEND", "duckdb").lower()

if BACKEND not in {"duckdb", "sqlite"}:
    raise ValueError(f"Unsupported backend {BACKEND!r}")


logging.basicConfig(level=logging.INFO)
logging.getLogger("orbital").setLevel(logging.INFO)  # Set DEBUG to see translation process.

# Carica il dataset
iris = load_iris()
df = pd.DataFrame(
    iris.data, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"]
)

# Aggiungi una colonna categorica per testare OneHotEncoder
df["petal_width_cat"] = np.where(df["petal_width"] < 1.0, "narrow", "wide")

# Introduci alcuni valori mancanti
df.loc[[0, 10, 20], "sepal_width"] = np.nan

# Variabile target numerica: prevediamo la lunghezza dei petali
y = df["petal_length"]
X = df.drop(columns=["petal_length"])  # Le altre colonne sono feature

# Creiamo la pipeline di preprocessing + regressione
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

# Converti le feature per orbital
features = orbital.types.guess_datatypes(X)
print("orbital Features:", features)

# Converti la pipeline in SQL con orbital
orbital_pipeline = orbital.parse_pipeline(pipeline, features=features)
print(orbital_pipeline)

# Test data
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
    sql = orbital.export_sql("DATA_TABLE", orbital_pipeline, dialect=BACKEND)
    print(f"\nGenerated Query for {BACKEND.upper()}:")
    print(sql)
    print("\nPrediction with SQL")
    # We need to create the table for the SQL to query it.
    con.create_table("DATA_TABLE", obj=example_data)
    print(con.raw_sql(sql).fetchall())


print("\nPrediction with SKLearn")
test_df = example_data.to_pandas()
target = pipeline.predict(test_df)
print(target)

print("\nPrediction with Ibis")
ibis_table = ibis.memtable(example_data, name="DATA_TABLE")
ibis_expression = orbital.translate(ibis_table, orbital_pipeline)
ibis_target = con.execute(ibis_expression)["variable"].to_numpy()
print(ibis_target)

if ASSERT:
    assert np.allclose(target, ibis_target), "Predictions do not match!"
    print("\nPredictions match!")
