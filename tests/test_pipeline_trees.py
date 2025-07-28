import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import orbital
from orbital import types
from orbital_testing_helpers import execute_sql


class TestTreeBasedPipelines:
    """Test suite for tree-based machine learning pipelines and their SQL exports."""

    # Decision Tree Tests
    def test_decision_tree_classifier(self, iris_data, db_connection):
        """Test a decision tree classifier pipeline with preprocessing."""
        df, _ = iris_data
        conn, dialect = db_connection

        # Use binary classification for simplicity
        binary_df = df[df["target"].isin([0, 1])].copy()
        binary_df = pd.concat([binary_df.iloc[:10], binary_df.iloc[-10:]])
        binary_df = binary_df.reset_index(drop=True)

        # Add StandardScaler as preprocessing step
        sklearn_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),  # Normalize features
                ("classifier", DecisionTreeClassifier(max_depth=3, random_state=42)),
            ]
        )

        X = binary_df["petal_length"].to_frame()
        y = binary_df["target"]
        sklearn_pipeline.fit(X, y)
        sklearn_proba = sklearn_pipeline.predict_proba(X)
        sklearn_class = sklearn_pipeline.predict(X)

        features = {fname: types.FloatColumnType() for fname in ["petal_length"]}
        parsed_pipeline = orbital.parse_pipeline(sklearn_pipeline, features=features)

        sql = orbital.export_sql("data", parsed_pipeline, dialect=dialect)
        sql_results = execute_sql(sql, conn, dialect, binary_df)

        sklearn_proba_df = pd.DataFrame(
            sklearn_proba, columns=sklearn_pipeline.classes_, index=binary_df.index
        )

        np.testing.assert_allclose(
            sql_results["output_label"].to_numpy(), sklearn_class
        )
        for class_label in sklearn_pipeline.classes_:
            np.testing.assert_allclose(
                sql_results[f"output_probability.{class_label}"].values.flatten(),
                sklearn_proba_df[class_label].values.flatten(),
            )

    def test_decision_tree_regressor(self, iris_data, db_connection):
        """Test a decision tree regressor pipeline with feature selection."""
        df, feature_names = iris_data
        conn, dialect = db_connection

        # Add feature selection as preprocessing step
        sklearn_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),  # Standardize features
                ("normalizer", MinMaxScaler()),  # Then scale to
                ("regressor", DecisionTreeRegressor(max_depth=3, random_state=42)),
            ]
        )

        X = df[feature_names]
        y = df["target"]
        sklearn_pipeline.fit(X, y)
        sklearn_preds = sklearn_pipeline.predict(X)

        features = {fname: types.FloatColumnType() for fname in feature_names}
        parsed_pipeline = orbital.parse_pipeline(sklearn_pipeline, features=features)

        sql = orbital.export_sql("data", parsed_pipeline, dialect=dialect)

        sql_results = execute_sql(sql, conn, dialect, df)
        np.testing.assert_allclose(
            sql_results.values.flatten(),
            sklearn_preds.flatten(),
            rtol=1e-4,
            atol=1e-4,
        )

    # Gradient Boosting Tests
    def test_gradient_boosting_classifier(self, iris_data, db_connection):
        """Test a gradient boosting classifier with categorical preprocessing."""
        df, feature_names = iris_data
        conn, dialect = db_connection

        # Create a deterministic categorical feature based on petal_length
        # This creates predictable categories that can be debugged
        def assign_quality(length):
            if length < 3:
                return "low"
            elif length < 5:
                return "medium"
            else:
                return "high"

        df["quality"] = df["petal_length"].apply(assign_quality)

        # Use ColumnTransformer for mixed preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), feature_names),
                ("cat", OneHotEncoder(), ["quality"]),
            ]
        )

        sklearn_pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    GradientBoostingClassifier(
                        n_estimators=10, max_depth=3, random_state=42
                    ),
                ),
            ]
        )

        # Use all classes, not just binary
        X = df[feature_names + ["quality"]]
        y = df["target"]
        sklearn_pipeline.fit(X, y)
        sklearn_class = sklearn_pipeline.predict(X)

        features = {fname: types.FloatColumnType() for fname in feature_names}
        features["quality"] = types.StringColumnType()
        parsed_pipeline = orbital.parse_pipeline(sklearn_pipeline, features=features)

        sql = orbital.export_sql("data", parsed_pipeline, dialect=dialect)
        sql_results = execute_sql(sql, conn, dialect, df)

        np.testing.assert_allclose(
            sql_results["output_label"].to_numpy(), sklearn_class
        )

        sklearn_proba = sklearn_pipeline.predict_proba(X)
        sklearn_proba_df = pd.DataFrame(
            sklearn_proba, columns=sklearn_pipeline.classes_, index=df.index
        )
        for class_label in sklearn_pipeline.classes_:
            np.testing.assert_allclose(
                sql_results[f"output_probability.{class_label}"].to_numpy(),
                sklearn_proba_df[class_label].values.flatten(),
                rtol=1e-4,
                atol=1e-4,
            )

    def test_gradient_boosting_regressor(self, iris_data, db_connection):
        """Test a gradient boosting regressor with standardization."""
        df, feature_names = iris_data
        conn, dialect = db_connection

        # Add StandardScaler as preprocessing
        sklearn_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "regressor",
                    GradientBoostingRegressor(
                        n_estimators=10, max_depth=3, random_state=42
                    ),
                ),
            ]
        )

        X = df[feature_names]
        y = df["target"]
        sklearn_pipeline.fit(X, y)
        sklearn_preds = sklearn_pipeline.predict(X)

        features = {fname: types.FloatColumnType() for fname in feature_names}
        parsed_pipeline = orbital.parse_pipeline(sklearn_pipeline, features=features)

        sql = orbital.export_sql("data", parsed_pipeline, dialect=dialect)
        sql_results = execute_sql(sql, conn, dialect, df)

        np.testing.assert_allclose(
            sql_results["variable"].to_numpy(),
            sklearn_preds.flatten(),
            rtol=1e-4,
            atol=1e-4,
        )

    # Random Forest Tests
    def test_random_forest_classifier(self, iris_data, db_connection):
        """Test a random forest classifier with mixed preprocessing."""
        df, feature_names = iris_data
        conn, dialect = db_connection

        # Create a deterministic categorical feature based on sepal_width
        # This creates predictable regions that can be debugged
        def assign_region(width):
            if width < 3.0:
                return "north"
            elif width < 3.4:
                return "east"
            elif width < 3.8:
                return "south"
            else:
                return "west"

        # Apply to the full dataset (all classes)
        df["region"] = df["sepal_width"].apply(assign_region)

        # Use ColumnTransformer to handle mixed data types
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), feature_names),
                ("cat", OneHotEncoder(), ["region"]),
            ]
        )

        sklearn_pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=10, max_depth=3, random_state=42
                    ),
                ),
            ]
        )

        # Use all classes, not just binary
        X = df[feature_names + ["region"]]
        y = df["target"]
        sklearn_pipeline.fit(X, y)
        sklearn_class = sklearn_pipeline.predict(X)

        features = {fname: types.FloatColumnType() for fname in feature_names}
        features["region"] = types.StringColumnType()
        parsed_pipeline = orbital.parse_pipeline(sklearn_pipeline, features=features)

        sql = orbital.export_sql("data", parsed_pipeline, dialect=dialect)
        sql_results = execute_sql(sql, conn, dialect, df)

        np.testing.assert_allclose(
            sql_results["output_label"].to_numpy(), sklearn_class
        )

        sklearn_proba = sklearn_pipeline.predict_proba(X)
        sklearn_proba_df = pd.DataFrame(
            sklearn_proba, columns=sklearn_pipeline.classes_, index=df.index
        )
        for class_label in sklearn_pipeline.classes_:
            np.testing.assert_allclose(
                sql_results[f"output_probability.{class_label}"].to_numpy(),
                sklearn_proba_df[class_label].values.flatten(),
                rtol=1e-3,
                atol=1e-6,
            )

    def test_binary_random_forest_classifier(self, iris_data, db_connection):
        """Test a binary random forest classifier with mixed preprocessing."""
        df, feature_names = iris_data
        conn, dialect = db_connection

        # Add categorical feature for more realistic preprocessing
        binary_df = df[df["target"].isin([0, 1])].copy()
        binary_df["region"] = np.random.choice(
            ["north", "south", "east", "west"], size=binary_df.shape[0]
        )

        # Use ColumnTransformer to handle mixed data types
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), feature_names),
                ("cat", OneHotEncoder(), ["region"]),
            ]
        )

        sklearn_pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=10, max_depth=3, random_state=42
                    ),
                ),
            ]
        )

        X = binary_df[feature_names + ["region"]]
        y = binary_df["target"]
        sklearn_pipeline.fit(X, y)
        sklearn_class = sklearn_pipeline.predict(X)

        features = dict(
            {fname: types.FloatColumnType() for fname in feature_names},
            region=types.StringColumnType(),
        )
        parsed_pipeline = orbital.parse_pipeline(sklearn_pipeline, features=features)

        # Test prediction
        sql = orbital.export_sql("data", parsed_pipeline, dialect=dialect)
        sql_results = execute_sql(sql, conn, dialect, binary_df)
        np.testing.assert_allclose(
            sql_results["output_label"].to_numpy(), sklearn_class
        )

        # Test probabilities
        sklearn_proba = sklearn_pipeline.predict_proba(X)
        sklearn_proba_df = pd.DataFrame(
            sklearn_proba, columns=sklearn_pipeline.classes_
        )
        for class_label in sklearn_pipeline.classes_:
            np.testing.assert_allclose(
                sql_results[f"output_probability.{class_label}"].to_numpy(),
                sklearn_proba_df[class_label].values.flatten(),
                rtol=0.05,
                atol=0.05,
            )
