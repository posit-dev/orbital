"""Translate a PyTorch demand-forecasting network into SQL.

A retail demand network (price, promo flag, day of week, prior-week sales
-> 64 -> 64 hidden neurons with ReLU, no output activation) predicts units
sold for the coming week. Trained and converted to a SQL query that
computes the same predictions directly inside DuckDB.

Unlike the fraud and maintenance classifiers, this is a plain regression
network: no Sigmoid/Softmax squashes the output, and there is currently no
other NN regression example in this repository.

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
    "price": orbital.types.DoubleColumnType(),
    "promo": orbital.types.DoubleColumnType(),
    "day_of_week": orbital.types.DoubleColumnType(),
    "prior_week_sales": orbital.types.DoubleColumnType(),
}

np.random.seed(42)
torch.manual_seed(42)

# Synthetic weekly sales: higher price reduces demand, promos and a strong
# prior week both increase it, weekends (day_of_week 5-6) taper it slightly.
num_samples = 1500
price = np.random.uniform(5, 50, num_samples)
promo = np.random.binomial(1, 0.3, num_samples).astype(np.float64)
day_of_week = np.random.randint(0, 7, num_samples).astype(np.float64)
prior_week_sales = np.random.uniform(50, 500, num_samples)
noise = np.random.normal(0, 10, num_samples)
units_sold = (
    150 - 1.8 * price + 60 * promo - 4 * day_of_week + 0.4 * prior_week_sales + noise
)
units_sold = np.clip(units_sold, 0, None)

X_train = np.column_stack([price, promo, day_of_week, prior_week_sales]).astype(
    np.float32
)
y_train = units_sold.astype(np.float32)

model = torch.nn.Sequential(
    torch.nn.Linear(len(FEATURES), 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 1),
)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
X_tensor = torch.from_numpy(X_train)
y_tensor = torch.from_numpy(y_train.reshape(-1, 1))
for epoch in range(300):
    optimizer.zero_grad()
    loss = criterion(model(X_tensor), y_tensor)
    loss.backward()
    optimizer.step()
print(f"Trained model, final loss: {loss.item():.4f}")

# Prepare the inputs outside the benchmarked function.
test_data = pd.DataFrame(
    {
        "price": [10.0, 45.0, 25.0, 15.0],
        "promo": [1.0, 0.0, 0.0, 1.0],
        "day_of_week": [5.0, 1.0, 3.0, 6.0],
        "prior_week_sales": [400.0, 80.0, 250.0, 300.0],
    }
)
duckdb.register("weekly_sales", test_data)


def main():
    pipeline = orbital.parse_pytorch_model(model, FEATURES)

    sql = orbital.export_sql("weekly_sales", pipeline, dialect="duckdb")
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
            assert np.allclose(sql_predictions, torch_predictions, atol=1e-4), (
                "SQL and PyTorch predictions do not match"
            )
            print("\nSQL and PyTorch predictions match.")


if __name__ == "__main__":
    main()
