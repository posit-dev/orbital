import os
import logging

import ibis
import numpy as np
import pandas as pd
import pyarrow as pa
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import orbital
import orbital.types

PRINT_SQL = int(os.environ.get("PRINT_SQL", "0"))
ASSERT = int(os.environ.get("ASSERT", "0"))
PREDICT_WITH_LIBRARY = int(os.environ.get("PREDICT_WITH_LIBRARY", "1")) or ASSERT
BACKEND = os.environ.get("BACKEND", "duckdb").lower()

if BACKEND not in {"duckdb", "sqlite"}:
    raise ValueError(f"Unsupported backend {BACKEND!r}")

logging.basicConfig(level=logging.INFO)
logging.getLogger("orbital").setLevel(
    logging.INFO
)  # Set DEBUG to see translation process.

# House price prediction: sqft, bedrooms and age drive the price, with
# some noise to keep it realistic. activation="tanh" exercises the Tanh
# translator on the sklearn MLP path (PyTorch already covers ReLU/Sigmoid).
rng = np.random.default_rng(7)
num_samples = 800
sqft = rng.uniform(500, 4000, num_samples)
bedrooms = rng.integers(1, 6, num_samples).astype(np.float64)
age_years = rng.uniform(0, 80, num_samples)
noise = rng.normal(0, 1.5, num_samples)
# Price in $10,000s (48.2 == $482,000): MLPRegressor's default Adam learning
# rate saturates tanh units when trained directly on raw dollar-scale
# targets, keeping the target in the tens/hundreds avoids that without
# changing any solver hyperparameters.
price = sqft * 0.021 + bedrooms * 0.9 - age_years * 0.04 + noise
price = np.clip(price, 5, None)

X = pd.DataFrame({"sqft": sqft, "bedrooms": bedrooms, "age_years": age_years})

pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "mlp",
            MLPRegressor(
                hidden_layer_sizes=(32, 16),
                activation="tanh",
                max_iter=2000,
                random_state=7,
            ),
        ),
    ]
)
pipeline.fit(X, price)

# Prepare the inputs outside the benchmarked function.
features = orbital.types.guess_datatypes(X)
example_data = pa.Table.from_pandas(X.head(4))
con = {
    "sqlite": lambda: ibis.sqlite.connect(":memory:"),
    "duckdb": lambda: ibis.duckdb.connect(),
}[BACKEND]()
if PRINT_SQL:
    con.create_table("DATA_TABLE", obj=example_data)


def translate_to_orbital():
    orbital_pipeline = orbital.parse_pipeline(pipeline, features=features)
    print(orbital_pipeline)
    ibis_table = ibis.memtable(example_data).alias("DATA_TABLE")
    ibis_expression = orbital.translate(ibis_table, orbital_pipeline)
    return orbital_pipeline, ibis_expression


orbital_pipeline, ibis_expression = translate_to_orbital()


def main():
    if PRINT_SQL:
        sql = orbital.export_sql("DATA_TABLE", orbital_pipeline, dialect=BACKEND)
        print(f"\nGenerated Query for {BACKEND.upper()}:")
        print(sql)
        print("\nPrediction with SQL")
        print(con.raw_sql(sql).fetchall())

    print("\nPrediction with Ibis")
    ibis_predictions = con.execute(ibis_expression)
    print(ibis_predictions)

    if PREDICT_WITH_LIBRARY:
        print("\nPrediction with SKLearn")
        predictions = pipeline.predict(example_data.to_pandas())
        print(predictions)

    if ASSERT:
        assert np.allclose(ibis_predictions["variable"], predictions, atol=1e-3), (
            "Predictions do not match!"
        )
        print("\nPredictions match!")


if __name__ == "__main__":
    main()
