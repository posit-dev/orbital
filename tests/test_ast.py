import numpy as np
import pandas as pd
import pytest
from google.protobuf.json_format import MessageToDict
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from orbitalml import ast, types


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
