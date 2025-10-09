import os
import logging

import ibis
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import orbital
import orbital.types

PRINT_SQL = int(os.environ.get("PRINT_SQL", "0"))
ASSERT = int(os.environ.get("ASSERT", "0"))
BACKEND = os.environ.get("BACKEND", "duckdb").lower()

if BACKEND not in {"duckdb", "sqlite"}:
    raise ValueError(f"Unsupported backend {BACKEND!r}")


logging.basicConfig(level=logging.INFO)
logging.getLogger("orbital").setLevel(logging.INFO)  # Set DEBUG to see translation process.

# Binary classification example using Ames Housing dataset
ames = fetch_openml(name="house_prices", as_frame=True)
ames = ames.frame

# SQL column names cannot start with digits
ames.columns = ["_" + col if col[0].isdigit() else col for col in ames.columns]

numeric_features = [
    col
    for col in ames.select_dtypes(include=["int64", "float64"]).columns
    if col != "SalePrice"
]
ames[numeric_features] = ames[numeric_features].astype(np.float64)

# Orbital requires consistent string types for categorical data
categorical_features = ames.select_dtypes(include=["object", "category"]).columns
ames[categorical_features] = ames[categorical_features].fillna("missing")

# Binary classification: expensive vs affordable houses
def categorize_price_binary(price: float) -> str:
    return "expensive" if price >= 200000 else "affordable"

ames["price_category"] = ames["SalePrice"].apply(categorize_price_binary)

X = ames.drop(columns=["SalePrice", "price_category"])
y = ames["price_category"]

numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="constant", missing_values="missing", fill_value="missing"
            ),
        ),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("var_threshold", VarianceThreshold()),
        ("pca", PCA(n_components=5)),
        ("classifier", GradientBoostingClassifier()),
    ]
)

model.fit(X, y)

features = orbital.types.guess_datatypes(X)

# Small sample for easier verification of results
data_sample = X.head(5)

# Convert the model to an execution pipeline
orbital_pipeline = orbital.parse_pipeline(model, features=features)
print(orbital_pipeline)

# Translate the pipeline to a query
ibis_table = ibis.memtable(data_sample, name="DATA_TABLE")
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
    con.create_table(ibis_table.get_name(), obj=data_sample)
    print(con.raw_sql(sql).fetchall())

print("\nPrediction with SKLearn")
sklearn_predictions = model.predict(data_sample)
sklearn_probabilities = model.predict_proba(data_sample)
print(f"Predictions: {sklearn_predictions}")
print(f"Probabilities: {sklearn_probabilities}")

print("\nPrediction with Ibis")
ibis_result = con.execute(ibis_expression)
print(ibis_result)

if ASSERT:
    assert np.array_equal(sklearn_predictions, ibis_result["output_label"]), "Predictions do not match!"
    
    # Binary classification should produce exactly 2 probability columns
    prob_cols = [col for col in ibis_result.columns if col.startswith("output_probability.")]
    assert len(prob_cols) == 2, f"Expected exactly 2 probability columns for binary classification, got {len(prob_cols)}"
    
    prob_matrix = np.column_stack([ibis_result[col].values for col in prob_cols])
    prob_sums = prob_matrix.sum(axis=1)
    assert np.allclose(prob_sums, 1.0, atol=1e-6), f"Probabilities don't sum to 1.0: {prob_sums}"
    
    # Verify probabilities match sklearn exactly
    for i, col in enumerate(prob_cols):
        sklearn_prob = sklearn_probabilities[:, i]
        ibis_prob = ibis_result[col].values
        np.testing.assert_allclose(
            sklearn_prob, ibis_prob, 
            rtol=1e-4, atol=1e-4, 
            err_msg=f"Probabilities for {col} don't match sklearn"
        )
        
    print("\nPredictions and probabilities match!")
