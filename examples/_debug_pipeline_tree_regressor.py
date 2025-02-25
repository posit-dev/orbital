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

import mustela
import mustela.types

PRINT_SQL = False

logging.basicConfig(level=logging.INFO)
logging.getLogger("mustela").setLevel(logging.DEBUG)

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

# Mustela requires the input and outputs of an imputer to
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

if False:
    import sys
    import pandas as pd
    ibis_con = ibis.sqlite.connect()
    features = mustela.types.guess_datatypes(X)
    numeric_transformer.fit(X[numeric_features])
    mustela_pipeline = mustela.parse_pipeline(numeric_transformer, features={i[0]: i[1] for i in features.items() if i[0] in numeric_features})
    ibis_numeric_transformer = mustela.translate(ibis.memtable(X[numeric_features].head(5)), mustela_pipeline)
    print("Data")
    print(X[numeric_features].head(5))
    print("SKLearn Transform")
    values = numeric_transformer.transform(X[numeric_features].head(5))
    pdf = pd.DataFrame(values, columns=numeric_features)
    print(pdf)
    print("Ibis Transform")
    ibisdf = ibis_con.execute(ibis_numeric_transformer)
    ibisdf.columns = pdf.columns
    print(ibisdf)
    print("Comparison")
    print(pdf.compare(ibisdf))
    sys.exit(0)

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

if False:
    import sys
    import pandas as pd
    from mustela._utils.debug import map_sklearn2onnx_names
    ibis_con = ibis.sqlite.connect()
    features = mustela.types.guess_datatypes(X)
    categorical_cols = X[categorical_features]
    categorical_transformer.fit_transform(categorical_cols)
    mustela_pipeline = mustela.parse_pipeline(categorical_transformer, features={i[0]: i[1] for i in features.items() if i[0] in categorical_features})
    ibis_categorical_transformer = mustela.translate(ibis.memtable(X[categorical_features].head(5)), mustela_pipeline)
    print("SKLearn Transform")
    values = categorical_transformer.transform(categorical_cols.head(5)).toarray()
    # print("Values", values.shape, values[1])
    encoder = categorical_transformer.named_steps['onehot']
    feature_names = encoder.get_feature_names_out(categorical_features)
    pdf = pd.DataFrame(values, columns=feature_names)
    print(pdf)
    print("Ibis prediction")
    ibisdf = ibis_con.execute(ibis_categorical_transformer)
    # The name of the onnx results are different from those of SKLearn
    # But they retain the same category names.
    # Try to guess what are the expected colum names.
    # This should probably be handled by onnx2ibis who shouldn't have to guess.
    name_maps = map_sklearn2onnx_names(feature_names, ibisdf.columns)
    ibisdf.columns = list(name_maps.values())
    print(ibisdf)
    print("Comparison")
    comparison = pdf.compare(ibisdf)
    print(comparison)
    sys.exit(0)

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

features = mustela.types.guess_datatypes(X)
print("Mustela Features:", features)

# Create a small set of data for the prediction
# It's easier to understand if it's small
data_sample = X.head(5)

mustela_pipeline = mustela.parse_pipeline(model, features=features)
print(mustela_pipeline)

ibis_expression = mustela.translate(ibis.memtable(data_sample), mustela_pipeline)
con = ibis.sqlite.connect()

if PRINT_SQL:
    print("\nGenerated Query for SQLite:")
    print(con.compile(ibis_expression))

print("\nPrediction with SKLearn")
target = model.predict(data_sample)
print(target)

# NOTE: When the Mustela optimizer is enabled this is significantly faster
#       But it's currently disabled due to a bug.
# NOTE: Interestingly the DuckDB optimizer has a bug on this query too
#       and unless disabled the query never completes.
#       That's why we run using SQLite.
#       The Mustela optimizer when enabled is able to preoptimize the query
#       which seems to allow DuckDB to complete the query as probably the DuckDB
#       optimizer has less work to do in that case.
print("\nPrediction with Ibis")
print(con.execute(ibis_expression))
