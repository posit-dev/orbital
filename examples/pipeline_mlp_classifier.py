import os
import logging

import ibis
import numpy as np
import pyarrow as pa
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import orbital
import orbital.types

PRINT_SQL = int(os.environ.get("PRINT_SQL", "0"))
ASSERT = int(os.environ.get("ASSERT", "0"))
PREDICT_WITH_LIBRARY = int(os.environ.get("PREDICT_WITH_LIBRARY", "1"))
BACKEND = os.environ.get("BACKEND", "duckdb").lower()

if BACKEND not in {"duckdb", "sqlite"}:
    raise ValueError(f"Unsupported backend {BACKEND!r}")

logging.basicConfig(level=logging.INFO)
logging.getLogger("orbital").setLevel(
    logging.INFO
)  # Set DEBUG to see translation process.

# Breast cancer diagnosis: binary classification (malignant vs benign) is
# the most common real-world MLPClassifier use case (churn, fraud, credit risk).
cancer = load_breast_cancer(as_frame=True)
X = cancer.data
# SQL and orbital don't like spaces in column names, replace them with underscores
X.columns = [cname.replace(" ", "_") for cname in X.columns]
y = cancer.target

pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "mlp",
            MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
        ),
    ]
)
pipeline.fit(X, y)

# Prepare the inputs outside the benchmarked function.
features = orbital.types.guess_datatypes(X)
# Rows 0/1 are malignant and 19/20 are benign, mixing both classes in the demo output.
# reset_index avoids pyarrow adding the non-contiguous original index as an extra column.
example_data = pa.Table.from_pandas(X.iloc[[0, 1, 19, 20]].reset_index(drop=True))
con = {
    "sqlite": lambda: ibis.sqlite.connect(":memory:"),
    "duckdb": lambda: ibis.duckdb.connect(),
}[BACKEND]()
if PRINT_SQL:
    con.create_table("DATA_TABLE", obj=example_data)


def main():
    orbital_pipeline = orbital.parse_pipeline(pipeline, features=features)
    print(orbital_pipeline)

    if PRINT_SQL:
        sql = orbital.export_sql("DATA_TABLE", orbital_pipeline, dialect=BACKEND)
        print(f"\nGenerated Query for {BACKEND.upper()}:")
        print(sql)
        print("\nPrediction with SQL")
        print(con.raw_sql(sql).fetchall())

    print("\nPrediction with Ibis")
    ibis_table = ibis.memtable(example_data).alias("DATA_TABLE")
    ibis_expression = orbital.translate(ibis_table, orbital_pipeline)
    ibis_result = con.execute(ibis_expression)
    print(ibis_result)

    if PREDICT_WITH_LIBRARY:
        print("\nPrediction with SKLearn")
        test_df = example_data.to_pandas()
        sklearn_labels = pipeline.predict(test_df)
        sklearn_probabilities = pipeline.predict_proba(test_df)
        print(f"Labels: {sklearn_labels}")
        print(f"Probabilities: {sklearn_probabilities}")

    if ASSERT and PREDICT_WITH_LIBRARY:
        assert np.array_equal(sklearn_labels, ibis_result["output_label"]), (
            "Labels do not match!"
        )

        # Binary classification should produce exactly 2 probability columns
        prob_cols = [
            col for col in ibis_result.columns if col.startswith("output_probability.")
        ]
        assert len(prob_cols) == 2, (
            f"Expected exactly 2 probability columns, got {len(prob_cols)}"
        )
        for i, col in enumerate(prob_cols):
            np.testing.assert_allclose(
                sklearn_probabilities[:, i],
                ibis_result[col].to_numpy(),
                atol=1e-4,
                err_msg=f"Probabilities for {col} don't match sklearn",
            )
        print("\nLabels and probabilities match!")


if __name__ == "__main__":
    main()
