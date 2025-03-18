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

import mustela
import mustela.types

PRINT_SQL = False
logging.basicConfig(level=logging.INFO)
logging.getLogger("mustela").setLevel(logging.DEBUG)

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

# Converti le feature per Mustela
features = mustela.types.guess_datatypes(X)
print("Mustela Features:", features)

# Converti la pipeline in SQL con Mustela
mustela_pipeline = mustela.parse_pipeline(pipeline, features=features)
print(mustela_pipeline)

# Test data
example_data = pa.table(
    {
        "sepal_length": [5.0, 6.1, 7.2, 5.843333],
        "sepal_width": [3.2, 2.8, 3.0, 3.057333],
        "petal_width": [0.2, 1.2, 2.3, 1.199333],
        "petal_width_cat": ["narrow", "wide", "wide", "wide"],
    }
)

# Genera la query SQL con Mustela
ibis_expression = mustela.translate(ibis.memtable(example_data), mustela_pipeline)

if PRINT_SQL:
    print("\nGenerated Query:")
    con = ibis.duckdb.connect()
    print(con.compile(ibis_expression))

print("\nPrediction with Ibis")
print(ibis_expression.execute())

print("\nPrediction with SKLearn")
test_df = example_data.to_pandas()
pred = pipeline.predict(test_df)
print(pred)