import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

import orbital
from orbital import types
from orbital_testing_helpers import execute_sql


class TestEndToEndPipelines:
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
        parsed_pipeline = orbital.parse_pipeline(sklearn_pipeline, features=features)

        sql = orbital.export_sql("data", parsed_pipeline, dialect=dialect)

        sql_results = execute_sql(sql, conn, dialect, df)
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
        parsed_pipeline = orbital.parse_pipeline(sklearn_pipeline, features=features)

        sql = orbital.export_sql(
            "data",
            parsed_pipeline,
            projection=orbital.ResultsProjection(["sepal_length"]),
            dialect=dialect,
        )

        sql_results = execute_sql(sql, conn, dialect, df)
        assert set(sql_results.columns) == {"sepal_length", "variable.target_0"}
        np.testing.assert_allclose(
            sql_results["variable.target_0"].values.flatten(),
            sklearn_preds.flatten(),
            rtol=1e-4,
            atol=1e-4,
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
        parsed_pipeline = orbital.parse_pipeline(sklearn_pipeline, features=features)

        sql = orbital.export_sql("data", parsed_pipeline, dialect=dialect)

        sql_results = execute_sql(sql, conn, dialect, df)
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
        parsed_pipeline = orbital.parse_pipeline(sklearn_pipeline, features=features)

        sql = orbital.export_sql("data", parsed_pipeline, dialect=dialect)

        sql_results = execute_sql(sql, conn, dialect, df)
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
        parsed_pipeline = orbital.parse_pipeline(sklearn_pipeline, features=features)

        sql = orbital.export_sql("data", parsed_pipeline, dialect=dialect)

        sql_results = execute_sql(sql, conn, dialect, binary_df)

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
        parsed_pipeline = orbital.parse_pipeline(sklearn_pipeline, features=features)

        sql = orbital.export_sql("data", parsed_pipeline, dialect=dialect)
        sql_results = execute_sql(sql, conn, dialect, df)

        np.testing.assert_allclose(
            sql_results["variable.target_0"].to_numpy(),
            sklearn_preds.flatten(),
            rtol=1e-4,
            atol=1e-4,
        )
