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

# Load Ames Housing for classification
ames = fetch_openml(name="house_prices", as_frame=True)
ames = ames.frame

# SQL does not allow columns to start with a number.
ames.columns = ["_" + col if col[0].isdigit() else col for col in ames.columns]

# Pic numeric features to uniform them types to float
numeric_features = [
    col
    for col in ames.select_dtypes(include=["int64", "float64"]).columns
    if col != "SalePrice"
]
ames[numeric_features] = ames[numeric_features].astype(np.float64)

# Fill the categorical types with a value for missing values
# (NaN is not a string and we can't mix types in the same column)
categorical_features = ames.select_dtypes(include=["object", "category"]).columns
ames[categorical_features] = ames[categorical_features].fillna("missing")


# Let's create classes of prices, we will divide prices in 3 categories
# We will use this as the target of the prediction
def categorize_price(price: float) -> str:
    if price < 130000:
        return "low"
    elif price < 250000:
        return "medium"
    else:
        return "high"

ames["price_category"] = ames["SalePrice"].apply(categorize_price)

# Split target of prediction (sales cateogiry) from features used for prediction
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

# Convert types from numpy to orbital types
features = orbital.types.guess_datatypes(X)

# Target only 5 rows, so that it's easier for a human to understand
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
    # We need to create the table for the SQL to query it.
    con.create_table(ibis_table.get_name(), obj=ibis_table)
    print(con.raw_sql(sql).fetchall())

print("\nPrediction with SKLearn")
target = model.predict(data_sample)
print(target)

print("\nPrediction with Ibis")
ibis_target = con.execute(ibis_expression)
print(ibis_target)

if ASSERT:
    assert np.array_equal(target, ibis_target["output_label"]), "Predictions do not match!"
    print("\nPredictions match!")
