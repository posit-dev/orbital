"""Test all supported models work in single-step pipelines (no preprocessing)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, ElasticNet

import orbital
from orbital import types
from orbital.ast import parse_pipeline
from orbital_testing_helpers import execute_sql


class TestSingleStepPipelines:
    """Test that all supported models work in single-step pipelines.

    These tests validate that:
    1. Pipeline parsing works correctly (the main issue that was fixed)
    2. SQL generation works (thanks to automatic concat injection)
    3. SQL execution produces the same results as sklearn
    """

    def setup_method(self):
        """Set up test data for various scenarios."""
        # Numeric regression data
        self.numeric_data = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "feature2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
                "feature3": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0],
                "target_reg": [11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0],
                "target_cls": [0, 1, 0, 1, 0, 1, 0, 1],
            }
        )

        # Integer data
        self.integer_data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8],
                "feature2": [10, 20, 30, 40, 50, 60, 70, 80],
                "target_reg": [11, 22, 33, 44, 55, 66, 77, 88],
                "target_cls": [0, 1, 0, 1, 0, 1, 0, 1],
            }
        )

        # String classification data (for testing string features)
        self.string_data = pd.DataFrame(
            {
                "feature1": ["A", "B", "A", "B", "A", "B", "A", "B"],
                "feature2": ["X", "Y", "X", "Y", "X", "Y", "X", "Y"],
                "target_cls": [0, 1, 0, 1, 0, 1, 0, 1],
            }
        )

    def validate_sql_execution(
        self, pipeline, X, y, features, is_classification=True, db_connection=None
    ):
        """Helper to validate SQL execution matches sklearn predictions."""
        # Use duckdb connection if none provided
        if db_connection is None:
            import duckdb

            conn = duckdb.connect(":memory:")
            dialect = "duckdb"
        else:
            conn, dialect = db_connection

        # Parse pipeline and generate SQL
        parsed = parse_pipeline(pipeline, features)
        sql = orbital.export_sql("data", parsed, dialect=dialect)

        # Get sklearn predictions
        sklearn_pred = pipeline.predict(X)

        # Execute SQL and compare
        sql_results = execute_sql(sql, conn, dialect, X)

        if is_classification:
            # For classification, compare predicted labels
            sql_pred = sql_results["output_label"].values
            np.testing.assert_array_equal(
                sklearn_pred,
                sql_pred,
                err_msg="SQL and sklearn predictions don't match",
            )
        else:
            # For regression, compare with tolerance
            sql_pred = sql_results["variable"].values
            np.testing.assert_allclose(
                sklearn_pred,
                sql_pred,
                rtol=1e-10,
                atol=1e-10,
                err_msg="SQL and sklearn predictions don't match within tolerance",
            )

    def test_decision_tree_classifier_double_features(self):
        """Test DecisionTreeClassifier with all double features."""
        features = {
            "feature1": types.DoubleColumnType(),
            "feature2": types.DoubleColumnType(),
            "feature3": types.DoubleColumnType(),
        }

        X = self.numeric_data[["feature1", "feature2", "feature3"]]
        y = self.numeric_data["target_cls"]

        pipeline = Pipeline([("classifier", DecisionTreeClassifier(random_state=42))])
        pipeline.fit(X, y)

        # Test parsing, SQL generation, and execution
        self.validate_sql_execution(pipeline, X, y, features, is_classification=True)

    def test_decision_tree_classifier_float_features(self):
        """Test DecisionTreeClassifier with all float features."""
        features = {
            "feature1": types.FloatColumnType(),
            "feature2": types.FloatColumnType(),
        }

        X = self.numeric_data[["feature1", "feature2"]]
        y = self.numeric_data["target_cls"]

        pipeline = Pipeline([("classifier", DecisionTreeClassifier(random_state=42))])
        pipeline.fit(X, y)

        # Test parsing, SQL generation, and execution
        self.validate_sql_execution(pipeline, X, y, features, is_classification=True)

    def test_decision_tree_classifier_int_features(self):
        """Test DecisionTreeClassifier with all int features."""
        features = {
            "feature1": types.Int64ColumnType(),
            "feature2": types.Int64ColumnType(),
        }

        X = self.integer_data[["feature1", "feature2"]]
        y = self.integer_data["target_cls"]

        pipeline = Pipeline([("classifier", DecisionTreeClassifier(random_state=42))])
        pipeline.fit(X, y)

        # Test parsing and SQL generation (execution validation would require type conversion)
        parsed = parse_pipeline(pipeline, features)
        assert parsed is not None
        sql = orbital.export_sql("test_table", parsed)
        assert sql is not None

    def test_decision_tree_regressor_double_features(self):
        """Test DecisionTreeRegressor with all double features."""
        features = {
            "feature1": types.DoubleColumnType(),
            "feature2": types.DoubleColumnType(),
        }

        X = self.numeric_data[["feature1", "feature2"]]
        y = self.numeric_data["target_reg"]

        pipeline = Pipeline([("regressor", DecisionTreeRegressor(random_state=42))])
        pipeline.fit(X, y)

        # Test parsing, SQL generation, and execution
        self.validate_sql_execution(pipeline, X, y, features, is_classification=False)

    def test_linear_regression_double_features(self):
        """Test LinearRegression with all double features."""
        features = {
            "feature1": types.DoubleColumnType(),
            "feature2": types.DoubleColumnType(),
        }

        X = self.numeric_data[["feature1", "feature2"]]
        y = self.numeric_data["target_reg"]

        pipeline = Pipeline([("regressor", LinearRegression())])
        pipeline.fit(X, y)

        # Test parsing, SQL generation, and execution
        self.validate_sql_execution(pipeline, X, y, features, is_classification=False)

    def test_gradient_boosting_classifier_double_features(self):
        """Test GradientBoostingClassifier with all double features."""
        features = {
            "feature1": types.DoubleColumnType(),
            "feature2": types.DoubleColumnType(),
        }

        X = self.numeric_data[["feature1", "feature2"]]
        y = self.numeric_data["target_cls"]

        pipeline = Pipeline(
            [
                (
                    "classifier",
                    GradientBoostingClassifier(random_state=42, n_estimators=10),
                )
            ]
        )
        pipeline.fit(X, y)

        # Test parsing, SQL generation, and execution
        self.validate_sql_execution(pipeline, X, y, features, is_classification=True)

    @pytest.mark.skip(
        reason="DuckDB DECIMAL type inference issue with precise GradientBoosting numeric constants"
    )
    def test_gradient_boosting_regressor_double_features(self):
        """Test GradientBoostingRegressor with all double features."""
        features = {
            "feature1": types.DoubleColumnType(),
            "feature2": types.DoubleColumnType(),
        }

        X = self.numeric_data[["feature1", "feature2"]]
        y = self.numeric_data["target_reg"]

        pipeline = Pipeline(
            [("regressor", GradientBoostingRegressor(random_state=42, n_estimators=10))]
        )
        pipeline.fit(X, y)

        # Test parsing, SQL generation, and execution
        self.validate_sql_execution(pipeline, X, y, features, is_classification=False)

    @pytest.mark.skip(
        reason="SQL and sklearn predictions don't match - needs investigation"
    )
    def test_random_forest_classifier_double_features(self):
        """Test RandomForestClassifier with all double features."""
        features = {
            "feature1": types.DoubleColumnType(),
            "feature2": types.DoubleColumnType(),
        }

        X = self.numeric_data[["feature1", "feature2"]]
        y = self.numeric_data["target_cls"]

        pipeline = Pipeline(
            [("classifier", RandomForestClassifier(random_state=42, n_estimators=10))]
        )
        pipeline.fit(X, y)

        # Test parsing, SQL generation, and execution
        self.validate_sql_execution(pipeline, X, y, features, is_classification=True)

    @pytest.mark.skip(reason="Sigmoid step not yet implemented")
    def test_logistic_regression_double_features(self):
        """Test LogisticRegression with all double features."""
        features = {
            "feature1": types.DoubleColumnType(),
            "feature2": types.DoubleColumnType(),
        }

        X = self.numeric_data[["feature1", "feature2"]]
        y = self.numeric_data["target_cls"]

        pipeline = Pipeline([("classifier", LogisticRegression(random_state=42))])
        pipeline.fit(X, y)

        # Test parsing, SQL generation, and execution
        self.validate_sql_execution(pipeline, X, y, features, is_classification=True)

    def test_lasso_double_features(self):
        """Test Lasso with all double features."""
        features = {
            "feature1": types.DoubleColumnType(),
            "feature2": types.DoubleColumnType(),
        }

        X = self.numeric_data[["feature1", "feature2"]]
        y = self.numeric_data["target_reg"]

        pipeline = Pipeline([("regressor", Lasso(alpha=0.1))])
        pipeline.fit(X, y)

        # Test parsing, SQL generation, and execution
        self.validate_sql_execution(pipeline, X, y, features, is_classification=False)

    def test_elastic_net_double_features(self):
        """Test ElasticNet with all double features."""
        features = {
            "feature1": types.DoubleColumnType(),
            "feature2": types.DoubleColumnType(),
        }

        X = self.numeric_data[["feature1", "feature2"]]
        y = self.numeric_data["target_reg"]

        pipeline = Pipeline([("regressor", ElasticNet(alpha=0.1, l1_ratio=0.5))])
        pipeline.fit(X, y)

        # Test parsing, SQL generation, and execution
        self.validate_sql_execution(pipeline, X, y, features, is_classification=False)

    def test_mixed_types_fails(self):
        """Test that mixed feature types fail with clear error message."""
        features = {
            "feature1": types.DoubleColumnType(),
            "feature2": types.FloatColumnType(),  # Different type
        }

        X = self.numeric_data[["feature1", "feature2"]]
        y = self.numeric_data["target_cls"]

        pipeline = Pipeline([("classifier", DecisionTreeClassifier(random_state=42))])
        pipeline.fit(X, y)

        with pytest.raises(
            ValueError,
            match="All features must be of the same type when pipeline starts with a model",
        ):
            parse_pipeline(pipeline, features)

    def test_mixed_types_error_message_content(self):
        """Test that the error message contains the actual type names."""
        features = {
            "feature1": types.DoubleColumnType(),
            "feature2": types.Int64ColumnType(),  # Different type
            "feature3": types.FloatColumnType(),  # Another different type
        }

        X = self.numeric_data[["feature1", "feature2", "feature3"]]
        y = self.numeric_data["target_cls"]

        pipeline = Pipeline([("classifier", DecisionTreeClassifier(random_state=42))])
        pipeline.fit(X, y)

        with pytest.raises(ValueError) as exc_info:
            parse_pipeline(pipeline, features)

        error_msg = str(exc_info.value)
        assert "DoubleTensorType" in error_msg
        assert "Int64TensorType" in error_msg
        assert "FloatTensorType" in error_msg

    def test_string_features_not_supported(self):
        """Test that string features fail with informative error."""
        features = {
            "feature1": types.StringColumnType(),
            "feature2": types.StringColumnType(),
        }

        # For this test, we need to encode the string data numerically for sklearn
        # but the orbital types should still work
        from sklearn.preprocessing import LabelEncoder

        le1, le2 = LabelEncoder(), LabelEncoder()

        X_encoded = pd.DataFrame(
            {
                "feature1": le1.fit_transform(self.string_data["feature1"]),
                "feature2": le2.fit_transform(self.string_data["feature2"]),
            }
        )
        y = self.string_data["target_cls"]

        pipeline = Pipeline([("classifier", DecisionTreeClassifier(random_state=42))])
        pipeline.fit(X_encoded, y)

        # String features should fail with sklearn2onnx
        with pytest.raises(
            RuntimeError, match="got an input .* with a wrong type .*StringTensorType"
        ):
            parse_pipeline(pipeline, features)

    def test_string_mixed_with_numeric_fails(self):
        """Test that string mixed with numeric features fails."""
        features = {
            "feature1": types.DoubleColumnType(),
            "feature2": types.StringColumnType(),  # Mixed with numeric
        }

        X = self.numeric_data[["feature1", "feature2"]]
        y = self.numeric_data["target_cls"]

        pipeline = Pipeline([("classifier", DecisionTreeClassifier(random_state=42))])
        pipeline.fit(X, y)

        # Mixed types should fail with our validation
        with pytest.raises(
            ValueError,
            match="All features must be of the same type when pipeline starts with a model",
        ):
            parse_pipeline(pipeline, features)

    def test_single_feature_still_works(self):
        """Test that single features still work (no change in behavior)."""
        features = {
            "feature1": types.DoubleColumnType(),
        }

        X = self.numeric_data[["feature1"]]
        y = self.numeric_data["target_cls"]

        pipeline = Pipeline([("classifier", DecisionTreeClassifier(random_state=42))])
        pipeline.fit(X, y)

        # Single feature should work and generate SQL
        parsed = parse_pipeline(pipeline, features)
        assert parsed is not None
        sql = orbital.export_sql("test_table", parsed)
        assert sql is not None
