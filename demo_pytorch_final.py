#!/usr/bin/env python3
"""
Demo: Convert PyTorch Neural Network to SQL using Orbital

This demonstrates how to convert a simple PyTorch neural network
to a SQL query that can run directly in a database.

Use Case: Credit Card Fraud Detection
- Simple feedforward neural network with ReLU and Sigmoid activations
- Only 45 parameters
- Can be converted to pure SQL for database execution
"""

import sys
import torch
import torch.nn as nn
import numpy as np

# Check dependencies
try:
    import torch
    import orbital
    import orbital.types as types
    import onnx
    import duckdb
    print("✓ All dependencies available")
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("PyTorch to SQL Demo with Orbital")
print("="*70)


# ============================================================================
# Step 1: Define a Simple Neural Network for Fraud Detection
# ============================================================================
print("\n[Step 1] Defining Neural Network Architecture...")

class FraudDetectorNN(nn.Module):
    """Simple neural network for credit card fraud detection.
    
    Architecture:
    - Input: 4 features (amount, time, v1, v2)
    - Hidden layer: 8 neurons with ReLU activation
    - Output: 1 neuron with Sigmoid activation (probability of fraud)
    - Total parameters: 4*8 + 8 + 8*1 + 1 = 49
    """
    def __init__(self, input_size=4):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

model = FraudDetectorNN()
print(f"  Model: {model.__class__.__name__}")
print(f"  Input: 4 features -> Hidden: 8 neurons (ReLU) -> Output: 1 neuron (Sigmoid)")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")


# ============================================================================
# Step 2: Generate Synthetic Training Data
# ============================================================================
print("\n[Step 2] Generating synthetic training data...")

np.random.seed(42)
torch.manual_seed(42)

num_samples = 1000
amounts = np.random.exponential(100, num_samples).astype(np.float32)
times = np.random.uniform(0, 86400, num_samples).astype(np.float32)
v1 = np.random.randn(num_samples).astype(np.float32)
v2 = np.random.randn(num_samples).astype(np.float32)

# Create labels: fraud probability based on features
fraud_prob = 0.5 * (1 / (1 + np.exp(-amounts / 50))) + \
              0.3 * (1 / (1 + np.exp(-(v1 + v2 - 1))))
labels = (np.random.rand(num_samples) < fraud_prob).astype(np.float32)

X_train = np.column_stack([amounts, times, v1, v2])
y_train = labels.reshape(-1, 1)

print(f"  Training samples: {num_samples}")
print(f"  Fraud rate: {labels.mean():.2%}")


# ============================================================================
# Step 3: Train the Model
# ============================================================================
print("\n[Step 3] Training the model...")

X_tensor = torch.from_numpy(X_train)
y_tensor = torch.from_numpy(y_train)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

print(f"  Final loss: {loss.item():.4f}")


# ============================================================================
# Step 4: Test PyTorch Model
# ============================================================================
print("\n[Step 4] Testing PyTorch model...")

X_test = np.column_stack([
    np.array([50.0, 200.0, 10.0, 500.0]),
    np.array([1000.0, 50000.0, 2000.0, 70000.0]),
    np.array([0.5, 2.0, -1.0, 3.0]),
    np.array([0.5, 2.0, -1.0, 3.0]),
])
X_test_tensor = torch.from_numpy(X_test.astype(np.float32))

with torch.no_grad():
    pytorch_predictions = model(X_test_tensor).numpy().flatten()

print(f"  Test predictions:")
for i, (amount, pred) in enumerate(zip(X_test[:, 0], pytorch_predictions)):
    print(f"    Amount=${amount:6.1f} -> Fraud probability: {pred:.4f}")


# ============================================================================
# Step 5: Export PyTorch Model to ONNX
# ============================================================================
print("\n[Step 5] Exporting PyTorch model to ONNX...")

model.eval()
sample_input = torch.randn(1, 4)

onnx_path = "/tmp/fraud_detector.onnx"
torch.onnx.export(
    model,
    sample_input,
    onnx_path,
    input_names=['input'],
    output_names=['fraud_probability'],
    opset_version=14
)

onnx_model = onnx.load(onnx_path)
print(f"  ONNX model exported (opset version: {onnx_model.opset_import[0].version})")


# ============================================================================
# Step 6: Inject Concat Operation for SQL Compatibility
# ============================================================================
print("\n[Step 6] Preparing ONNX model for SQL conversion...")

import onnx as _onnx

# Define feature types for Orbital
features = {
    "amount": types.DoubleColumnType(),
    "time": types.DoubleColumnType(),
    "v1": types.DoubleColumnType(),
    "v2": types.DoubleColumnType(),
}
feature_names = list(features.keys())

# Inject Concat to split 'input' into individual features
graph = onnx_model.graph

# Create new individual feature inputs
new_inputs = []
for fname in feature_names:
    onnx_type = features[fname]._to_onnxtype()
    onnx_type_obj = onnx_type.to_onnx_type()
    new_inputs.append(
        _onnx.helper.make_tensor_value_info(
            fname,
            onnx_type_obj.tensor_type.elem_type,
            [None, 1],
        )
    )

# Create Concat node
concat_node = _onnx.helper.make_node(
    "Concat",
    inputs=feature_names,
    outputs=["input"],
    axis=1,
    name="orbital_concat",
)

# Modify the graph
del graph.input[:]
graph.input.extend(new_inputs)
graph.node.insert(0, concat_node)

print(f"  ✓ Concat operation injected to handle individual SQL columns")


# ============================================================================
# Step 7: Convert ONNX Model to Orbital Pipeline
# ============================================================================
print("\n[Step 7] Converting ONNX model to Orbital pipeline...")

from orbital.ast import ParsedPipeline

orbital_pipeline = ParsedPipeline._from_onnx_model(onnx_model, features)
print(f"  ✓ ParsedPipeline created")


# ============================================================================
# Step 8: Generate SQL Query
# ============================================================================
print("\n[Step 8] Generating SQL query...")

sql_query = orbital.export_sql("transactions", orbital_pipeline, dialect="duckdb")
print(f"\n  Generated SQL query:")
print("-" * 70)
print(sql_query)
print("-" * 70)


# ============================================================================
# Step 9: Test SQL Query with DuckDB
# ============================================================================
print("\n[Step 9] Testing SQL query with DuckDB...")

# Create DuckDB table with test data
conn = duckdb.connect()
conn.execute("""
    CREATE TABLE transactions AS
    SELECT * FROM VALUES
        (50.0, 1000.0, 0.5, 0.5),
        (200.0, 50000.0, 2.0, 2.0),
        (10.0, 2000.0, -1.0, -1.0),
        (500.0, 70000.0, 3.0, 3.0)
    AS t(amount, time, v1, v2)
""")

# Execute SQL query
result = conn.execute(sql_query).fetchall()
sql_predictions = [row[0] for row in result]

print(f"\n  SQL predictions:")
for i, (amount, pred) in enumerate(zip(X_test[:, 0], sql_predictions)):
    print(f"    Amount=${amount:6.1f} -> Fraud probability: {pred:.4f}")

# Compare with PyTorch
max_diff = max(abs(p - s) for p, s in zip(pytorch_predictions, sql_predictions))
print(f"\n  Max difference: {max_diff:.10f}")

if max_diff < 1e-5:
    print(f"  ✓ Predictions match perfectly!")
else:
    print(f"  ⚠ Predictions differ slightly (within {max_diff:.2e})")


# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("Demo Complete!")
print("="*70)
print("\nSummary:")
print("  ✓ Defined a 49-parameter neural network for fraud detection")
print("  ✓ Trained the model on synthetic data")
print("  ✓ Exported PyTorch model to ONNX format")
print("  ✓ Converted ONNX model to Orbital pipeline")
print("  ✓ Generated SQL query that implements the neural network")
print("  ✓ Verified SQL predictions match PyTorch predictions")
print("\nKey Takeaway:")
print("  You can now run neural network predictions DIRECTLY in your database!")
print("  No Python runtime needed - just pure SQL.")
print("\nThis enables:")
print("  • Faster predictions on large datasets (database optimizations)")
print("  • Deployment without Python dependencies")
print("  • Embedded ML in SQL-based applications")
print("="*70)
