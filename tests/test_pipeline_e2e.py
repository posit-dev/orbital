import sqlite3

import duckdb
import sqlalchemy
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_diabetes, load_iris
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import orbitalml
from orbitalml import types


class TestEndToEndPipelines:
    @pytest.fixture(scope="class")
    def iris_data(self):
        iris = load_iris()
        # Clean feature names to match what's used in the example
        feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        X = pd.DataFrame(iris.data, columns=feature_names)  # Use clean names directly
        y = pd.DataFrame(iris.target, columns=["target"])
        df = pd.concat([X, y], axis=1)
        return df, feature_names

    @pytest.fixture(scope="class")
    def diabetes_data(self):
        diabetes = load_diabetes()
        feature_names = diabetes.feature_names
        X = pd.DataFrame(diabetes.data, columns=feature_names)
        y = pd.DataFrame(diabetes.target, columns=["target"])
        df = pd.concat([X, y], axis=1)
        return df, feature_names

    @pytest.fixture(params=["duckdb", "sqlite", "postgres"])
    def db_connection(self, request):
        dialect = request.param
        if dialect == "duckdb":
            conn = duckdb.connect(":memory:")
            yield conn, dialect
            conn.close()
        elif dialect == "sqlite":
            conn = sqlite3.connect(":memory:")
            yield conn, dialect
            conn.close()
        elif dialect == "postgres":
            try:
                conn = sqlalchemy.create_engine("postgresql://orbitalmltestuser:orbitalmltestpassword@localhost:5432/orbitalmltestdb")
                with conn.connect() as testcon:
                    testcon.execute(sqlalchemy.text("SELECT 1"))  # Test connection
            except (sqlalchemy.exc.OperationalError, ImportError):
                pytest.skip("Postgres database not available")
            yield conn, dialect
            conn.dispose()

    def execute_sql(self, sql, conn, dialect, data):
        if dialect == "duckdb":
            conn.execute("CREATE TABLE data AS SELECT * FROM data")
            result = conn.execute(sql).fetchdf()
        elif dialect in ("sqlite", "postgres"):
            data.to_sql("data", conn, index=False, if_exists="replace")
            result = pd.read_sql(sql, conn)
        return result

    def test_simple_linear_regression(self, iris_data, db_connection):
        df, feature_names = iris_data
        conn, dialect = db_connection

        sklearn_pipeline = Pipeline(
            [("scaler", StandardScaler()), ("regression", LinearRegression())]
        )
        X = df[feature_names]
        y = df["target"]
        sklearn_pipeline.fit(X, y)
        sklearn_preds = sklearn_pipeline.predict(X)

        features = {fname: types.FloatColumnType() for fname in feature_names}
        parsed_pipeline = orbitalml.parse_pipeline(sklearn_pipeline, features=features)

        sql = orbitalml.export_sql("data", parsed_pipeline, dialect=dialect)

        sql_results = self.execute_sql(sql, conn, dialect, df)
        np.testing.assert_allclose(
            sql_results.values.flatten(), sklearn_preds.flatten(), rtol=1e-4, atol=1e-4
        )

    def test_simple_linear_with_projection(self, iris_data, db_connection):
        df, feature_names = iris_data
        conn, dialect = db_connection

        sklearn_pipeline = Pipeline(
            [("scaler", StandardScaler()), ("regression", LinearRegression())]
        )
        X = df[feature_names]
        y = df["target"]
        sklearn_pipeline.fit(X, y)
        sklearn_preds = sklearn_pipeline.predict(X)

        features = {fname: types.FloatColumnType() for fname in feature_names}
        parsed_pipeline = orbitalml.parse_pipeline(sklearn_pipeline, features=features)

        sql = orbitalml.export_sql("data", parsed_pipeline, projection=orbitalml.ResultsProjection(["sepal_length"]), dialect=dialect)

        sql_results = self.execute_sql(sql, conn, dialect, df)
        print(sql_results)
        assert set(sql_results.columns) == {"sepal_length", "variable.target_0"}
        np.testing.assert_allclose(
            sql_results["variable.target_0"].values.flatten(), sklearn_preds.flatten(), rtol=1e-4, atol=1e-4
        )

    def test_feature_selection_pipeline(self, diabetes_data, db_connection):
        df, feature_names = diabetes_data
        conn, dialect = db_connection

        sklearn_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("feature_selection", SelectKBest(f_regression, k=5)),
                ("regression", LinearRegression()),
            ]
        )
        X = df[feature_names]
        y = df["target"]
        sklearn_pipeline.fit(X, y)
        sklearn_preds = sklearn_pipeline.predict(X)

        features = {str(fname): types.FloatColumnType() for fname in feature_names}
        parsed_pipeline = orbitalml.parse_pipeline(sklearn_pipeline, features=features)

        sql = orbitalml.export_sql("data", parsed_pipeline, dialect=dialect)

        sql_results = self.execute_sql(sql, conn, dialect, df)
        np.testing.assert_allclose(
            sql_results.values.flatten(), sklearn_preds.flatten(), rtol=1e-4, atol=1e-4
        )

    def test_column_transformer_pipeline(self, iris_data, db_connection):
        df, feature_names = iris_data
        conn, dialect = db_connection

        df["cat_feature"] = np.random.choice(["A", "B", "C"], size=df.shape[0])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), feature_names),
                ("cat", OneHotEncoder(), ["cat_feature"]),
            ]
        )

        sklearn_pipeline = Pipeline(
            [("preprocessor", preprocessor), ("regression", LinearRegression())]
        )

        X = df[feature_names + ["cat_feature"]]
        y = df["target"]
        sklearn_pipeline.fit(X, y)
        sklearn_preds = sklearn_pipeline.predict(X)

        features = {fname: types.FloatColumnType() for fname in feature_names}
        features["cat_feature"] = types.StringColumnType()
        parsed_pipeline = orbitalml.parse_pipeline(sklearn_pipeline, features=features)

        sql = orbitalml.export_sql("data", parsed_pipeline, dialect=dialect)

        sql_results = self.execute_sql(sql, conn, dialect, df)
        np.testing.assert_allclose(
            sql_results.values.flatten(), sklearn_preds.flatten(), rtol=1e-4, atol=1e-4
        )

    def test_logistic_regression(self, iris_data, db_connection):
        df, feature_names = iris_data
        conn, dialect = db_connection

        binary_df = df[df["target"].isin([0, 1])].copy()

        sklearn_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(random_state=42)),
            ]
        )

        X = binary_df[feature_names]
        y = binary_df["target"]
        sklearn_pipeline.fit(X, y)
        sklearn_proba = sklearn_pipeline.predict_proba(X)

        features = {fname: types.FloatColumnType() for fname in feature_names}
        parsed_pipeline = orbitalml.parse_pipeline(sklearn_pipeline, features=features)

        sql = orbitalml.export_sql("data", parsed_pipeline, dialect=dialect)

        sql_results = self.execute_sql(sql, conn, dialect, binary_df)

        sklearn_proba_df = pd.DataFrame(
            sklearn_proba, columns=sklearn_pipeline.classes_, index=binary_df.index
        )

        for class_label in sklearn_pipeline.classes_:
            np.testing.assert_allclose(
                sql_results[f"output_probability.{class_label}"].values.flatten(),
                sklearn_proba_df[class_label].values.flatten(),
                rtol=1e-4,
                atol=1e-4,
            )

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
        parsed_pipeline = orbitalml.parse_pipeline(sklearn_pipeline, features=features)

        sql = orbitalml.export_sql("data", parsed_pipeline, dialect=dialect)
        sql_results = self.execute_sql(sql, conn, dialect, binary_df)

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
        parsed_pipeline = orbitalml.parse_pipeline(sklearn_pipeline, features=features)

        sql = orbitalml.export_sql("data", parsed_pipeline, dialect=dialect)

        sql_results = self.execute_sql(sql, conn, dialect, df)
        np.testing.assert_allclose(
            sql_results.values.flatten(),
            sklearn_preds.flatten(),
            rtol=1e-4,
            atol=1e-4,
        )

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
        parsed_pipeline = orbitalml.parse_pipeline(sklearn_pipeline, features=features)

        sql = orbitalml.export_sql("data", parsed_pipeline, dialect=dialect)
        sql_results = self.execute_sql(sql, conn, dialect, df)

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
        parsed_pipeline = orbitalml.parse_pipeline(sklearn_pipeline, features=features)

        sql = orbitalml.export_sql("data", parsed_pipeline, dialect=dialect)
        sql_results = self.execute_sql(sql, conn, dialect, df)

        np.testing.assert_allclose(
            sql_results["variable"].to_numpy(),
            sklearn_preds.flatten(),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_elasticnet(self, diabetes_data, db_connection):
        """Test an ElasticNet pipeline with preprocessing transformations."""
        df, feature_names = diabetes_data
        conn, dialect = db_connection

        sklearn_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),  # Standardize features
                ("normalizer", MinMaxScaler()),  # Scale to [0,1] range
                ("regressor", ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42)),
            ]
        )

        X = df[feature_names]
        y = df["target"]
        sklearn_pipeline.fit(X, y)
        sklearn_preds = sklearn_pipeline.predict(X)

        features = {str(fname): types.FloatColumnType() for fname in feature_names}
        parsed_pipeline = orbitalml.parse_pipeline(sklearn_pipeline, features=features)

        sql = orbitalml.export_sql("data", parsed_pipeline, dialect=dialect)
        sql_results = self.execute_sql(sql, conn, dialect, df)

        np.testing.assert_allclose(
            sql_results["variable.target_0"].to_numpy(),
            sklearn_preds.flatten(),
            rtol=1e-4,
            atol=1e-4,
        )

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
        parsed_pipeline = orbitalml.parse_pipeline(sklearn_pipeline, features=features)

        sql = orbitalml.export_sql("data", parsed_pipeline, dialect=dialect)
        sql_results = self.execute_sql(sql, conn, dialect, df)

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
        parsed_pipeline = orbitalml.parse_pipeline(sklearn_pipeline, features=features)

        # Test prediction
        sql = orbitalml.export_sql("data", parsed_pipeline, dialect=dialect)
        sql_results = self.execute_sql(sql, conn, dialect, binary_df)
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
