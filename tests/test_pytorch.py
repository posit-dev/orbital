import builtins

import duckdb
import numpy as np
import pandas as pd
import pytest

import orbital
from orbital import types

FEATURES = {
    "amount": types.DoubleColumnType(),
    "hour": types.DoubleColumnType(),
    "v1": types.DoubleColumnType(),
    "v2": types.DoubleColumnType(),
}


class TestParsePytorchModel:
    def test_without_torch(self, monkeypatch):
        """A clear error suggests the pytorch extra when torch is missing."""
        real_import = builtins.__import__

        def block_torch(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("No module named 'torch'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", block_torch)
        with pytest.raises(ImportError, match=r"orbital\[pytorch\]"):
            orbital.parse_pytorch_model(object(), FEATURES)

    def test_matches_torch_predictions(self):
        torch = pytest.importorskip("torch")
        torch.manual_seed(42)
        model = torch.nn.Sequential(
            torch.nn.Linear(len(FEATURES), 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
            torch.nn.Sigmoid(),
        )
        # Parsing must not flip the caller's model out of training mode.
        model.train()

        parsed = orbital.parse_pytorch_model(model, FEATURES)
        assert isinstance(parsed, orbital.ast.ParsedPipeline)
        assert parsed.features == FEATURES
        assert model.training

        sql = orbital.export_sql("transactions", parsed, dialect="duckdb")

        test_data = pd.DataFrame(
            {
                "amount": [50.0, 200.0, 10.0, 500.0],
                "hour": [1.0, 14.0, 2.0, 20.0],
                "v1": [0.5, 2.0, -1.0, 3.0],
                "v2": [0.5, 2.0, -1.0, 3.0],
            }
        )
        conn = duckdb.connect(":memory:")
        conn.register("transactions", test_data)
        sql_predictions = conn.sql(sql).df().iloc[:, 0].to_numpy()

        with torch.no_grad():
            torch_predictions = (
                model.eval()(torch.from_numpy(test_data.to_numpy(dtype=np.float32)))
                .numpy()
                .flatten()
            )

        np.testing.assert_allclose(sql_predictions, torch_predictions, atol=1e-5)

    def test_double_model_matches_torch_predictions(self):
        torch = pytest.importorskip("torch")
        torch.manual_seed(42)
        # float64 weights: the export dummy input must match the model dtype.
        model = torch.nn.Sequential(
            torch.nn.Linear(len(FEATURES), 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
            torch.nn.Sigmoid(),
        ).double()

        parsed = orbital.parse_pytorch_model(model, FEATURES)
        sql = orbital.export_sql("transactions", parsed, dialect="duckdb")

        test_data = pd.DataFrame(
            {
                "amount": [50.0, 200.0, 10.0, 500.0],
                "hour": [1.0, 14.0, 2.0, 20.0],
                "v1": [0.5, 2.0, -1.0, 3.0],
                "v2": [0.5, 2.0, -1.0, 3.0],
            }
        )
        conn = duckdb.connect(":memory:")
        conn.register("transactions", test_data)
        sql_predictions = conn.sql(sql).df().iloc[:, 0].to_numpy()

        with torch.no_grad():
            torch_predictions = (
                model(torch.from_numpy(test_data.to_numpy(dtype=np.float64)))
                .numpy()
                .flatten()
            )

        np.testing.assert_allclose(sql_predictions, torch_predictions, atol=1e-9)

    def test_all_passthrough_features(self):
        torch = pytest.importorskip("torch")
        model = torch.nn.Sequential(torch.nn.Linear(1, 1))

        with pytest.raises(ValueError, match="passthrough"):
            orbital.parse_pytorch_model(
                model, {"amount": types.DoubleColumnType(passthrough=True)}
            )

    def test_mixed_feature_types(self):
        torch = pytest.importorskip("torch")
        model = torch.nn.Sequential(torch.nn.Linear(2, 1))

        mixed_features = {
            "amount": types.DoubleColumnType(),
            "count": types.Int64ColumnType(),
        }
        with pytest.raises(ValueError, match="same type"):
            orbital.parse_pytorch_model(model, mixed_features)
