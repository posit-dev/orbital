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

import mustela
import mustela.types

PRINT_SQL = False

logging.basicConfig(level=logging.INFO)
logging.getLogger("mustela").setLevel(logging.INFO)  # Set DEBUG to see translation process.

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


features = mustela.types.guess_datatypes(X)
print("Mustela Features:", features)

mustela_pipeline = mustela.parse_pipeline(pipeline, features=features)
print(mustela_pipeline)

# Prepare test data for predictions
example_data = pa.table({
    "sepal_length": [5.0, 6.1, 7.2, 5.843333],
    "sepal_width": [3.2, 2.8, 3.0, 3.057333],
    "petal_length": [1.2, 4.7, 6.1, 3.758000],
    "petal_width": [0.2, 1.2, 2.3, 1.199333],
    "petal_width_cat": ["narrow", "wide", "wide", "wide"],
})


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