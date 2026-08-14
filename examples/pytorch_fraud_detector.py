"""Translate a PyTorch neural network into SQL.

A fraud detection network (4 inputs -> 16 -> 8 hidden neurons with ReLU
-> 1 sigmoid output) is trained in PyTorch and converted to a SQL query
that computes the same predictions directly inside DuckDB.

This example requires PyTorch: pip install orbital[pytorch]
"""

import os

import duckdb
import numpy as np
import pandas as pd
import torch

import orbital
import orbital.types

PRINT_SQL = int(os.environ.get("PRINT_SQL", "0"))
ASSERT = int(os.environ.get("ASSERT", "0"))
PREDICT_WITH_LIBRARY = int(os.environ.get("PREDICT_WITH_LIBRARY", "1"))

FEATURES = {
    "amount": orbital.types.DoubleColumnType(),
    "hour": orbital.types.DoubleColumnType(),
    "v1": orbital.types.DoubleColumnType(),
    "v2": orbital.types.DoubleColumnType(),
}

np.random.seed(42)
torch.manual_seed(42)

# Synthetic transactions: higher amounts and v1+v2 increase fraud probability.
num_samples = 1000
X_train = np.column_stack(
    [
        np.random.exponential(100, num_samples),
        np.random.uniform(0, 24, num_samples),
        np.random.randn(num_samples),
        np.random.randn(num_samples),
    ]
).astype(np.float32)
fraud_prob = 0.5 / (1 + np.exp(-X_train[:, 0] / 50)) + 0.3 / (
    1 + np.exp(-(X_train[:, 2] + X_train[:, 3] - 1))
)
y_train = (np.random.rand(num_samples) < fraud_prob).astype(np.float32)

model = torch.nn.Sequential(
    torch.nn.Linear(len(FEATURES), 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 1),
    torch.nn.Sigmoid(),
)

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
X_tensor = torch.from_numpy(X_train)
y_tensor = torch.from_numpy(y_train.reshape(-1, 1))
for epoch in range(100):
    optimizer.zero_grad()
    loss = criterion(model(X_tensor), y_tensor)
    loss.backward()
    optimizer.step()
print(f"Trained model, final loss: {loss.item():.4f}")

# Prepare the inputs outside the benchmarked function.
test_data = pd.DataFrame(
    {
        "amount": [50.0, 200.0, 10.0, 500.0],
        "hour": [1.0, 14.0, 2.0, 20.0],
        "v1": [0.5, 2.0, -1.0, 3.0],
        "v2": [0.5, 2.0, -1.0, 3.0],
    }
)
duckdb.register("transactions", test_data)


def main():
    pipeline = orbital.parse_pytorch_model(model, FEATURES)

    sql = orbital.export_sql("transactions", pipeline, dialect="duckdb")
    if PRINT_SQL:
        print("\nGenerated Query for DuckDB:")
        print(sql)

    sql_predictions = duckdb.sql(sql).df().iloc[:, 0].to_numpy()
    print("\nPrediction with SQL")
    print(sql_predictions)

    if PREDICT_WITH_LIBRARY:
        print("\nPrediction with PyTorch")
        with torch.no_grad():
            torch_predictions = (
                model(torch.from_numpy(test_data.to_numpy(dtype=np.float32)))
                .numpy()
                .flatten()
            )
        print(torch_predictions)

        if ASSERT:
            assert np.allclose(sql_predictions, torch_predictions, atol=1e-5), (
                "SQL and PyTorch predictions do not match"
            )
            print("\nSQL and PyTorch predictions match.")


if __name__ == "__main__":
    main()
