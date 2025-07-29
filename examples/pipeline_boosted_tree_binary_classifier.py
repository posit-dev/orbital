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

PRINT_SQL = int(os.environ.get("PRINTSQL", "0"))
ASSERT = int(os.environ.get("ASSERT", "0"))

logging.basicConfig(level=logging.INFO)
logging.getLogger("orbital").setLevel(logging.INFO)  # Set DEBUG to see translation process.

# Load Ames Housing for binary classification
ames = fetch_openml(name="house_prices", as_frame=True)
ames = ames.frame

# SQL does not allow columns to start with a number.
ames.columns = ["_" + col if col[0].isdigit() else col for col in ames.columns]

# Pick numeric features to uniform them types to float
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

# Let's create BINARY classes of prices (high vs low/medium)
# This will test binary classification with LOGISTIC post-transform
def categorize_price_binary(price: float) -> str:
    if price < 200000:
        return "affordable"  # low/medium prices
    else:
        return "expensive"   # high prices

ames["price_category"] = ames["SalePrice"].apply(categorize_price_binary)

# Split target of prediction (binary sales category) from features used for prediction
X = ames.drop(columns=["SalePrice", "price_category"])
y = ames["price_category"]

print(f"Binary classification target distribution:")
print(y.value_counts())

# Use a smaller subset of features for this binary classification example
# to make it simpler and faster
selected_numeric_features = numeric_features[:3]  # Just first 3 numeric features  
selected_categorical_features = []  # Skip categorical features to avoid OneHotEncoder issues

print(f"Using {len(selected_numeric_features)} numeric and {len(selected_categorical_features)} categorical features")

# Create simpler transformers for the binary classification example
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, selected_numeric_features),
    ]
)

# Binary classification model with GradientBoostingClassifier
# This will use LOGISTIC post-transform for binary classification
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("var_threshold", VarianceThreshold()),
        ("pca", PCA(n_components=3)),  # Fewer components for binary case
        ("classifier", GradientBoostingClassifier(n_estimators=10, max_depth=3, random_state=42)),
    ]
)

# Use subset of features for training
X_subset = X[selected_numeric_features]
model.fit(X_subset, y)

# Convert types from numpy to orbital types
features = orbital.types.guess_datatypes(X_subset)

# Target only 5 rows, so that it's easier for a human to understand
data_sample = X_subset.head(5)

print(f"\nSample data shape: {data_sample.shape}")
print(f"Features: {list(features.keys())}")

# Convert the model to an execution pipeline
orbital_pipeline = orbital.parse_pipeline(model, features=features)
print(f"\nOrbital pipeline: {orbital_pipeline}")

# Translate the pipeline to a query
ibis_table = ibis.memtable(data_sample, name="DATA_TABLE")
ibis_expression = orbital.translate(ibis_table, orbital_pipeline)

con = ibis.duckdb.connect()
if PRINT_SQL:
    sql = orbital.export_sql("DATA_TABLE", orbital_pipeline, dialect="duckdb")
    print("\nGenerated Query for DuckDB:")
    print(sql)
    print("\nPrediction with SQL")
    # We need to create the table for the SQL to query it.
    con.create_table(ibis_table.get_name(), obj=ibis_table)
    sql_result = con.raw_sql(sql).df()
    print(sql_result)

print("\nPrediction with SKLearn")
sklearn_predictions = model.predict(data_sample)
sklearn_probabilities = model.predict_proba(data_sample)
print(f"Predictions: {sklearn_predictions}")
print(f"Probabilities shape: {sklearn_probabilities.shape}")
print(f"Class probabilities (first 3 rows):")
for i in range(min(3, len(sklearn_probabilities))):
    print(f"  Row {i}: {sklearn_probabilities[i]} (sum={sklearn_probabilities[i].sum():.6f})")

print("\nPrediction with Ibis")
ibis_result = con.execute(ibis_expression)
print(f"Ibis result columns: {list(ibis_result.columns)}")
print(ibis_result)

# Check probability sums for binary classification
if "output_probability.affordable" in ibis_result.columns and "output_probability.expensive" in ibis_result.columns:
    prob_affordable = ibis_result["output_probability.affordable"].values
    prob_expensive = ibis_result["output_probability.expensive"].values
    prob_sums = prob_affordable + prob_expensive
    print(f"\nBinary probability sums: {prob_sums}")
    print(f"All probabilities sum to ~1.0: {np.allclose(prob_sums, 1.0, atol=1e-6)}")

if ASSERT:
    assert np.array_equal(sklearn_predictions, ibis_result["output_label"]), "Predictions do not match!"
    
    # Also check that probabilities are close (binary classification should match sklearn exactly)
    if "output_probability.affordable" in ibis_result.columns:
        sklearn_prob_affordable = sklearn_probabilities[:, 0]  # First class
        ibis_prob_affordable = ibis_result["output_probability.affordable"].values
        np.testing.assert_allclose(
            sklearn_prob_affordable, ibis_prob_affordable, 
            rtol=1e-4, atol=1e-4, 
            err_msg="Affordable class probabilities don't match sklearn"
        )
        
        sklearn_prob_expensive = sklearn_probabilities[:, 1]  # Second class  
        ibis_prob_expensive = ibis_result["output_probability.expensive"].values
        np.testing.assert_allclose(
            sklearn_prob_expensive, ibis_prob_expensive, 
            rtol=1e-4, atol=1e-4, 
            err_msg="Expensive class probabilities don't match sklearn"
        )
        
    print("\nAll predictions and probabilities match!")
