import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.model_selection import train_test_split
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

        # Test probabilities with more tolerance for rounding differences
        sklearn_proba = sklearn_pipeline.predict_proba(X)
        sklearn_proba_df = pd.DataFrame(
            sklearn_proba, columns=sklearn_pipeline.classes_
        )
        for class_label in sklearn_pipeline.classes_:
            np.testing.assert_allclose(
                sql_results[f"output_probability.{class_label}"].to_numpy(),
                sklearn_proba_df[class_label].values.flatten(),
                rtol=0.15,  # Increased tolerance for rounding differences in probabilities
                atol=0.15,  # Increased tolerance for rounding differences in probabilities
            )


class TestTreePostTransformations:
    def test_gradient_boosting_binary_classification_post_transform_bug(
        self, db_connection
    ):
        """Test that GBM binary classification applies post-transforms correctly."""
        conn, dialect = db_connection

        # Create mock binary classification data
        X, y = make_classification(n_samples=100, n_features=20, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        sklearn_pipeline = Pipeline(
            [
                (
                    "preprocess",
                    ColumnTransformer(
                        [("scaler", StandardScaler(with_std=False), [0, 1, 2])],
                        remainder="passthrough",
                    ),
                ),
                (
                    "gbm",
                    GradientBoostingClassifier(
                        max_depth=1, n_estimators=1, random_state=42
                    ),
                ),
            ]
        )

        sklearn_pipeline.fit(X_train, y_train)
        sklearn_proba = sklearn_pipeline.predict_proba(X_train)
        sklearn_predictions = sklearn_pipeline.predict(X_train)

        n_cols = len(X_train[0])
        nm_cols = [f"var_{i}" for i in range(n_cols)]
        features = {n: types.DoubleColumnType() for n in nm_cols}

        parsed_pipeline = orbital.parse_pipeline(sklearn_pipeline, features=features)
        sql = orbital.export_sql("data", parsed_pipeline, dialect=dialect)

        test_data = pd.DataFrame(X_train, columns=nm_cols)
        sql_results = execute_sql(sql, conn, dialect, test_data)

        # Probabilities should sum to 1
        prob_0 = sql_results["output_probability.0"].to_numpy()
        prob_1 = sql_results["output_probability.1"].to_numpy()
        prob_sums = prob_0 + prob_1
        np.testing.assert_allclose(
            prob_sums,
            1.0,
            rtol=1e-6,
            atol=1e-6,
            err_msg="Probabilities don't sum to 1.0 due to incorrect post-transform order in GBM binary classification",
        )

        # Compare individual class probabilities with sklearn
        sklearn_prob_0 = sklearn_proba[:, 0]  # Class 0 probabilities
        sklearn_prob_1 = sklearn_proba[:, 1]  # Class 1 probabilities
        np.testing.assert_allclose(
            prob_1,
            sklearn_prob_1,
            rtol=1e-4,
            atol=1e-4,
            err_msg="Class 1 probabilities don't match sklearn due to incorrect post-transform",
        )
        np.testing.assert_allclose(
            prob_0,
            sklearn_prob_0,
            rtol=1e-4,
            atol=1e-4,
            err_msg="Class 0 probabilities don't match sklearn due to incorrect post-transform order",
        )

        # Predictions should match sklearn
        sql_predictions = sql_results["output_label"].to_numpy()
        np.testing.assert_array_equal(
            sql_predictions,
            sklearn_predictions,
            err_msg="Predictions don't match sklearn due to incorrect probability calculations",
        )

    def test_random_forest_binary_classification_post_transform_check(
        self, db_connection
    ):
        """Test RandomForest binary classification works correctly because it uses post_transform="NONE"."""
        conn, dialect = db_connection

        # Create mock binary classification data
        X, y = make_classification(n_samples=100, n_features=20, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        sklearn_pipeline = Pipeline(
            [
                (
                    "preprocess",
                    ColumnTransformer(
                        [("scaler", StandardScaler(with_std=False), [0, 1, 2])],
                        remainder="passthrough",
                    ),
                ),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=10, max_depth=3, random_state=42
                    ),
                ),
            ]
        )

        sklearn_pipeline.fit(X_train, y_train)
        sklearn_proba = sklearn_pipeline.predict_proba(X_train)
        sklearn_predictions = sklearn_pipeline.predict(X_train)

        n_cols = len(X_train[0])
        nm_cols = [f"var_{i}" for i in range(n_cols)]
        features = {n: types.DoubleColumnType() for n in nm_cols}

        parsed_pipeline = orbital.parse_pipeline(sklearn_pipeline, features=features)
        sql = orbital.export_sql("data", parsed_pipeline, dialect=dialect)

        test_data = pd.DataFrame(X_train, columns=nm_cols)
        sql_results = execute_sql(sql, conn, dialect, test_data)

        # RandomForest should work correctly: probabilities should sum to 1
        prob_0 = sql_results["output_probability.0"].to_numpy()
        prob_1 = sql_results["output_probability.1"].to_numpy()
        prob_sums = prob_0 + prob_1
        np.testing.assert_allclose(
            prob_sums,
            1.0,
            rtol=1e-6,
            atol=1e-6,
            err_msg="RandomForest: Probabilities don't sum to 1.0 - unexpected bug!",
        )

        # RandomForest probabilities should match sklearn exactly
        sklearn_prob_0 = sklearn_proba[:, 0]  # Class 0 probabilities
        sklearn_prob_1 = sklearn_proba[:, 1]  # Class 1 probabilities
        np.testing.assert_allclose(
            prob_0,
            sklearn_prob_0,
            rtol=1e-4,
            atol=1e-4,
            err_msg="RandomForest: Class 0 probabilities don't match sklearn",
        )

        np.testing.assert_allclose(
            prob_1,
            sklearn_prob_1,
            rtol=1e-4,
            atol=1e-4,
            err_msg="RandomForest: Class 1 probabilities don't match sklearn",
        )

        # RandomForest predictions should match sklearn
        sql_predictions = sql_results["output_label"].to_numpy()
        np.testing.assert_array_equal(
            sql_predictions,
            sklearn_predictions,
            err_msg="RandomForest: Predictions don't match sklearn",
        )

    def test_decision_tree_binary_classification_post_transform_check(
        self, db_connection
    ):
        """Test DecisionTree binary classification works correctly because DecisionTree uses post_transform="NONE"."""
        conn, dialect = db_connection

        # Create mock binary classification data
        X, y = make_classification(n_samples=100, n_features=20, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        sklearn_pipeline = Pipeline(
            [
                (
                    "preprocess",
                    ColumnTransformer(
                        [("scaler", StandardScaler(with_std=False), [0, 1, 2])],
                        remainder="passthrough",
                    ),
                ),
                ("dt", DecisionTreeClassifier(max_depth=5, random_state=42)),
            ]
        )

        sklearn_pipeline.fit(X_train, y_train)
        sklearn_proba = sklearn_pipeline.predict_proba(X_train)
        sklearn_predictions = sklearn_pipeline.predict(X_train)

        n_cols = len(X_train[0])
        nm_cols = [f"var_{i}" for i in range(n_cols)]
        features = {n: types.DoubleColumnType() for n in nm_cols}

        parsed_pipeline = orbital.parse_pipeline(sklearn_pipeline, features=features)
        sql = orbital.export_sql("data", parsed_pipeline, dialect=dialect)

        test_data = pd.DataFrame(X_train, columns=nm_cols)
        sql_results = execute_sql(sql, conn, dialect, test_data)

        # DecisionTree should work correctly: probabilities should sum to 1
        prob_0 = sql_results["output_probability.0"].to_numpy()
        prob_1 = sql_results["output_probability.1"].to_numpy()
        prob_sums = prob_0 + prob_1
        np.testing.assert_allclose(
            prob_sums,
            1.0,
            rtol=1e-6,
            atol=1e-6,
            err_msg="DecisionTree: Probabilities don't sum to 1.0 - unexpected bug!",
        )

        # DecisionTree probabilities should match sklearn exactly
        sklearn_prob_0 = sklearn_proba[:, 0]  # Class 0 probabilities
        sklearn_prob_1 = sklearn_proba[:, 1]  # Class 1 probabilities
        np.testing.assert_allclose(
            prob_0,
            sklearn_prob_0,
            rtol=1e-4,
            atol=1e-4,
            err_msg="DecisionTree: Class 0 probabilities don't match sklearn",
        )

        np.testing.assert_allclose(
            prob_1,
            sklearn_prob_1,
            rtol=1e-4,
            atol=1e-4,
            err_msg="DecisionTree: Class 1 probabilities don't match sklearn",
        )

        # DecisionTree predictions should match sklearn
        sql_predictions = sql_results["output_label"].to_numpy()
        np.testing.assert_array_equal(
            sql_predictions,
            sklearn_predictions,
            err_msg="DecisionTree: Predictions don't match sklearn",
        )

    def test_gradient_boosting_multiclass_classification_post_transform_check(
        self, iris_data, db_connection
    ):
        """Test that GradientBoosting multi-class classification works correctly with SOFTMAX post-transform."""
        conn, dialect = db_connection

        # Use iris dataset for multi-class classification (3 classes)
        df, feature_names = iris_data

        # Create pipeline with GradientBoosting multi-class classifier
        sklearn_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "gbm",
                    GradientBoostingClassifier(
                        n_estimators=5, max_depth=3, random_state=42
                    ),
                ),
            ]
        )

        X = df[feature_names]
        y = df["target"]
        sklearn_pipeline.fit(X, y)
        sklearn_proba = sklearn_pipeline.predict_proba(X)
        sklearn_predictions = sklearn_pipeline.predict(X)

        features = {fname: types.FloatColumnType() for fname in feature_names}
        parsed_pipeline = orbital.parse_pipeline(sklearn_pipeline, features=features)
        sql = orbital.export_sql("data", parsed_pipeline, dialect=dialect)

        test_data = df[feature_names]
        sql_results = execute_sql(sql, conn, dialect, test_data)

        # Multi-class GBM with SOFTMAX should work correctly: probabilities should sum to 1
        prob_0 = sql_results["output_probability.0"].to_numpy()
        prob_1 = sql_results["output_probability.1"].to_numpy()
        prob_2 = sql_results["output_probability.2"].to_numpy()
        prob_sums = prob_0 + prob_1 + prob_2
        np.testing.assert_allclose(
            prob_sums,
            1.0,
            rtol=1e-4,
            atol=1e-4,
            err_msg="Multi-class GBM: Probabilities don't sum to 1.0 - SOFTMAX post-transform bug!",
        )

        # Multi-class GBM probabilities should match sklearn exactly
        sklearn_prob_0 = sklearn_proba[:, 0]  # Class 0 probabilities
        sklearn_prob_1 = sklearn_proba[:, 1]  # Class 1 probabilities
        sklearn_prob_2 = sklearn_proba[:, 2]  # Class 2 probabilities
        np.testing.assert_allclose(
            prob_0,
            sklearn_prob_0,
            rtol=1e-4,
            atol=1e-4,
            err_msg="Multi-class GBM: Class 0 probabilities don't match sklearn",
        )
        np.testing.assert_allclose(
            prob_1,
            sklearn_prob_1,
            rtol=1e-4,
            atol=1e-4,
            err_msg="Multi-class GBM: Class 1 probabilities don't match sklearn",
        )
        np.testing.assert_allclose(
            prob_2,
            sklearn_prob_2,
            rtol=1e-4,
            atol=1e-4,
            err_msg="Multi-class GBM: Class 2 probabilities don't match sklearn",
        )

        # Multi-class GBM predictions should match sklearn
        sql_predictions = sql_results["output_label"].to_numpy()
        np.testing.assert_array_equal(
            sql_predictions,
            sklearn_predictions,
            err_msg="Multi-class GBM: Predictions don't match sklearn",
        )
