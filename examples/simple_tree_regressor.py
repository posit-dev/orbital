import logging

import ibis
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor

import orbital
import orbital.types

PRINT_SQL = False

logging.basicConfig(level=logging.INFO)
logging.getLogger("orbital").setLevel(logging.DEBUG)

X_np, y = make_regression(n_samples=20, n_features=1, noise=0.1, random_state=42)
df = pd.DataFrame(X_np, columns=["feature1"])
df["target"] = y

X = df.drop(columns="target")
y = df["target"]

# Train an estimator with the minimum complexity
model = GradientBoostingRegressor(n_estimators=10, max_depth=2, random_state=42)
model.fit(X, y)

# Sample of data on which we will run the regression
data_sample = X.head(5)

features = orbital.types.guess_datatypes(X)
print("orbital Features:", features)

orbital_pipeline = orbital.parse_pipeline(model, features=features)
print(orbital_pipeline)

ibis_expression = orbital.translate(ibis.memtable(data_sample), orbital_pipeline)

# This currently doesn't work with DuckDB, need to investigate
# duckdb.duckdb.ConversionException: Conversion Error: Casting value "19.7795275449752808" to type DECIMAL(18,17) failed: value is out of range!
#   LINE 1: ...1841) ELSE -0.3673345446586609 END) ELSE CASE  WHEN ((t0.feature1 <= 1.1452322006225586)) THEN (1.5154987573623657...
con = ibis.sqlite.connect()

if PRINT_SQL:
    print("\nGenerated Query for SQLite:")
    print(con.compile(ibis_expression))

print("\nPrediction with SKLearn")
target = model.predict(data_sample)
print(target)

print("\nPrediction with Ibis")
print(con.execute(ibis_expression))
