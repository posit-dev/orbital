"""Translate a PyTorch predictive-maintenance network into SQL.

A machine-health network (5 sensor readings -> 64 -> 32 hidden neurons with
ReLU + BatchNorm1d + Dropout -> 3-way Softmax) predicts one of three failure
modes {normal, bearing_wear, overheating}. Trained and converted to a SQL
query that computes the same predictions directly inside DuckDB.

BatchNorm1d and Dropout are training-time regularization layers: in eval
mode BatchNorm1d becomes a fixed per-channel affine transform (folded into
the preceding layer at export) and Dropout becomes a no-op, so they add no
extra translation work but prove real regularized architectures survive
conversion, not just toy Linear+activation stacks.

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
PREDICT_WITH_LIBRARY = int(os.environ.get("PREDICT_WITH_LIBRARY", "1")) or ASSERT

FEATURES = {
    "temperature": orbital.types.DoubleColumnType(),
    "vibration": orbital.types.DoubleColumnType(),
    "pressure": orbital.types.DoubleColumnType(),
    "rpm": orbital.types.DoubleColumnType(),
    "age": orbital.types.DoubleColumnType(),
}
FAILURE_MODES = ["normal", "bearing_wear", "overheating"]

np.random.seed(42)
torch.manual_seed(42)

# Synthetic sensor readings: overheating tracks temperature, bearing wear
# tracks vibration and age. Each sample is labelled by the highest of three
# noisy scores, so the classes overlap like real sensor data instead of
# being perfectly separable.
num_samples = 1500
temperature = np.random.normal(70, 15, num_samples)
vibration = np.random.exponential(1.0, num_samples)
pressure = np.random.normal(100, 20, num_samples)
rpm = np.random.normal(1800, 300, num_samples)
age = np.random.uniform(0, 10, num_samples)

score_noise = np.random.normal(0, 0.4, (num_samples, 3))
scores = (
    np.column_stack(
        [
            np.zeros(num_samples),  # normal: baseline score
            (temperature - 85) / 10,  # overheating: rises with temperature
            (vibration - 1.2) * 2 + (age - 5) / 5,  # bearing_wear: vibration + age
        ]
    )
    + score_noise
)
y_train = scores.argmax(axis=1)

X_train = np.column_stack([temperature, vibration, pressure, rpm, age]).astype(
    np.float32
)

model = torch.nn.Sequential(
    torch.nn.Linear(len(FEATURES), 64),
    torch.nn.BatchNorm1d(64),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(64, 32),
    torch.nn.BatchNorm1d(32),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(32, len(FAILURE_MODES)),
    torch.nn.Softmax(dim=1),
)

# The model's last layer is already a Softmax, so training needs the
# probability-based counterpart of cross entropy (NLLLoss on log
# probabilities) rather than CrossEntropyLoss, which expects raw logits
# and would apply a second softmax internally.
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
X_tensor = torch.from_numpy(X_train)
y_tensor = torch.from_numpy(y_train.astype(np.int64))
model.train()
for epoch in range(300):
    optimizer.zero_grad()
    output = model(X_tensor)
    loss = criterion(torch.log(output + 1e-12), y_tensor)
    loss.backward()
    optimizer.step()
print(f"Trained model, final loss: {loss.item():.4f}")

# Prepare the inputs outside the benchmarked function.
test_data = pd.DataFrame(
    {
        "temperature": [65.0, 95.0, 68.0, 72.0],
        "vibration": [0.8, 0.9, 3.2, 1.0],
        "pressure": [98.0, 105.0, 101.0, 99.0],
        "rpm": [1790.0, 1850.0, 1770.0, 1800.0],
        "age": [2.0, 3.0, 8.5, 4.0],
    }
)
duckdb.register("sensor_readings", test_data)


def main():
    pipeline = orbital.parse_pytorch_model(model, FEATURES)

    sql = orbital.export_sql("sensor_readings", pipeline, dialect="duckdb")
    if PRINT_SQL:
        print("\nGenerated Query for DuckDB:")
        print(sql)

    # The model has no ZipMap step (that is an sklearn-classifier concept),
    # so the three Softmax outputs surface as plain columns "softmax.out_0..2".
    sql_predictions = duckdb.sql(sql).df().to_numpy()
    print("\nPrediction with SQL")
    print(sql_predictions)

    if PREDICT_WITH_LIBRARY:
        print("\nPrediction with PyTorch")
        # The SQL was generated from the eval-mode graph (fixed BatchNorm1d
        # stats, Dropout disabled). Without eval() here, Dropout would
        # randomly zero activations and the two predictions would diverge.
        model.eval()
        with torch.no_grad():
            torch_predictions = model(
                torch.from_numpy(test_data.to_numpy(dtype=np.float32))
            ).numpy()
        print(torch_predictions)

        if ASSERT:
            assert np.allclose(sql_predictions, torch_predictions, atol=1e-5), (
                "SQL and PyTorch predictions do not match"
            )
            print("\nSQL and PyTorch predictions match.")


if __name__ == "__main__":
    main()
