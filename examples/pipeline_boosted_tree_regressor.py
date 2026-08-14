import os
import logging

import ibis
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import orbital
import orbital.types

PRINT_SQL = int(os.environ.get("PRINT_SQL", "0"))
ASSERT = int(os.environ.get("ASSERT", "0"))
PREDICT_WITH_LIBRARY = int(os.environ.get("PREDICT_WITH_LIBRARY", "1")) or ASSERT
BACKEND = os.environ.get("BACKEND", "duckdb").lower()

if BACKEND not in {"duckdb", "sqlite"}:
    raise ValueError(f"Unsupported backend {BACKEND!r}")


logging.basicConfig(level=logging.INFO)
logging.getLogger("orbital").setLevel(logging.INFO)  # Set DEBUG to see translation process.

ames = fetch_openml(name="house_prices", as_frame=True)
ames = ames.frame

# SQL does not allow columns to start with a number.
# See http://www.postgresql.org/docs/current/static/sql-syntax-lexical.html#SQL-SYNTAX-IDENTIFIERS
# So we need to rename them
ames.columns = ["_" + col if col[0].isdigit() else col for col in ames.columns]

numeric_features = [
    col
    for col in ames.select_dtypes(include=["int64", "float64"]).columns
    if col != "SalePrice"
]
categorical_features = ames.select_dtypes(include=["object", "category"]).columns

# Orbital requires the inputs and outputs of an imputer to
# be of the same type, as SimpleImputer has to compute the mean
# the result is always a float. Which makes sense.
# Let's convert all numeric features to doubles so
# that the inputs and outputs are always double.
ames[numeric_features] = ames[numeric_features].astype(np.float64)

# Categorical features are all strings, so they shouldn't have any NaN values.
# Let's add a missing value to ensure type consistency
ames[categorical_features] = ames[categorical_features].fillna("missing")

X = ames.drop("SalePrice", axis=1)
y = ames["SalePrice"]

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
        ("regressor", GradientBoostingRegressor()),
    ]
)
model.fit(X, y)

# Prepare the inputs outside the benchmarked function.
data_sample = X.head(5)
features = orbital.types.guess_datatypes(X)
con = {
    "sqlite": lambda: ibis.sqlite.connect(":memory:"),
    "duckdb": lambda: ibis.duckdb.connect(),
}[BACKEND]()
if PRINT_SQL:
    con.create_table("DATA_TABLE", obj=data_sample)


def translate_to_orbital():
    orbital_pipeline = orbital.parse_pipeline(model, features=features)
    print(orbital_pipeline)
    ibis_table = ibis.memtable(data_sample).alias("DATA_TABLE")
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

    if PREDICT_WITH_LIBRARY:
        print("\nPrediction with SKLearn")
        target = model.predict(data_sample)
        print(target)

    print("\nPrediction with Ibis")
    ibis_target = con.execute(ibis_expression)["variable"].to_numpy()
    print(ibis_target)

    if ASSERT:
        assert np.allclose(target, ibis_target), "Predictions do not match!"
        print("\nPredictions match!")


if __name__ == "__main__":
    main()
