import numpy as np
import pandas as pd
import pytest
from google.protobuf.json_format import MessageToDict
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from orbital import ast, types


class TestPipelineParsing:
    DF_DATA = {
        "feature1": [1, 2, 3, np.nan, 5],
        "feature2": [np.nan, 1, 0, 3, 1],
        "feature3": [1.1, 2.1, 3.1, 4.1, np.nan],
    }
    DATA_TYPES = {
        "feature1": types.DoubleColumnType(),
        "feature2": types.DoubleColumnType(),
        "feature3": types.DoubleColumnType(),
    }

    def test_need_to_parse(self):
        with pytest.raises(NotImplementedError):
            ast.ParsedPipeline()

    def test_parse_pipeline(self):
        df = pd.DataFrame(self.DF_DATA)

        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )
        pipeline.fit(df)

        parsed = ast.parse_pipeline(pipeline, self.DATA_TYPES)
        assert parsed.features == self.DATA_TYPES
        assert parsed._model is not None

        model_graph = MessageToDict(parsed._model.graph)
        assert {i["name"] for i in model_graph["input"]} == {
            "feature1",
            "feature2",
            "feature3",
        }
        assert {n["name"] for n in model_graph["node"]} == {
            "Di_Div",
            "FeatureVectorizer",
            "Imputer",
            "N1",
            "Su_Sub",
        }

    def test_dump_load_pipeline(self, tmp_path):
        df = pd.DataFrame(self.DF_DATA)

        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )
        pipeline.fit(df)

        parsed = ast.parse_pipeline(pipeline, self.DATA_TYPES)
        filename = tmp_path / "test_dump_load_pipeline.dump"
        parsed.dump(filename)

        loaded = ast.ParsedPipeline.load(filename)
        assert loaded.features == parsed.features
        assert loaded._model is not None
        assert loaded._model.SerializeToString() == parsed._model.SerializeToString()

    def test_load_incompatible_version(self, tmp_path):
        import pickle

        header = {"version": 2, "features": {}}
        header_data = pickle.dumps(header)
        header_len = len(header_data).to_bytes(4, "big")

        filename = tmp_path / "test_load_incompatible_version.dump"
        with open(filename, "wb") as f:
            f.write(header_len)
            f.write(header_data)

        with pytest.raises(ast.UnsupportedFormatVersion):
            ast.ParsedPipeline.load(filename)

    def test_parse_pipeline_no_preprocessing(self):
        """Test parsing a pipeline with no preprocessing steps.

        This tests the fix for the issue where sklearn2onnx fails when there are
        no preprocessing steps and multiple input features.
        """
        # Create some test data
        X, y = make_classification(n_features=5, random_state=42)
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create a pipeline with only a model (no preprocessing)
        pipeline = Pipeline(
            [
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=3, max_depth=2, random_state=42
                    ),
                )
            ]
        )
        pipeline.fit(X_train, y_train)

        # Define features for all columns
        features = {f"feature_{i}": types.DoubleColumnType() for i in range(X.shape[1])}

        # This should not raise an exception (previously it would fail)
        parsed = ast.parse_pipeline(pipeline, features)

        assert parsed.features == features
        assert parsed._model is not None

        # The model should have been converted successfully
        model_graph = MessageToDict(parsed._model.graph)

        # With the fix, we should have individual feature inputs for SQL compatibility
        assert len(model_graph["input"]) == 5  # One input per feature
        input_names = {inp["name"] for inp in model_graph["input"]}
        expected_names = {f"feature_{i}" for i in range(5)}
        assert input_names == expected_names

        # Should have a Concat node to combine features and the tree ensemble
        node_types = {n["opType"] for n in model_graph["node"]}
        assert "Concat" in node_types  # Injected for SQL compatibility
        assert "TreeEnsembleClassifier" in node_types
