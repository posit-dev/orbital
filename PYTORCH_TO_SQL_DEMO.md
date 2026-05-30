# PyTorch to SQL Demo with Orbital

## Overview

This demo shows how to convert a PyTorch neural network to a SQL query using Orbital, enabling neural network predictions to run directly in a database without any Python runtime.

## Use Case: Credit Card Fraud Detection

**Why this is a great demo:**
- **Business Value**: Fraud detection is a critical real-world application
- **Wow Factor**: Running a neural network in pure SQL is unexpected and impressive
- **Small Model**: Only 49 parameters, perfect for demonstration
- **Non-linear**: Uses ReLU activation to show neural networks can capture complex patterns

## Architecture

### Neural Network
```python
class FraudDetectorNN(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(4, 8)   # 4 inputs → 8 hidden neurons
        self.relu = nn.ReLU()        # ReLU activation
        self.fc2 = nn.Linear(8, 1)    # 8 hidden → 1 output
        self.sigmoid = nn.Sigmoid()  # Probability output
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.sigmoid(self.fc2(x))
```

### Conversion Pipeline
1. **PyTorch Model** → Export to ONNX using `torch.onnx.export()`
2. **ONNX Model** → Inject Concat operation to split single input tensor into individual SQL columns
3. **ONNX with Concat** → Load into Orbital's `ParsedPipeline._from_onnx_model()`
4. **ParsedPipeline** → Use Orbital's translation infrastructure to generate SQL

## Implementation

### New Files Created

1. **`src/orbital/translation/steps/relu.py`** - ReLU activation translator
   - Translates ONNX ReLU to SQL `CASE WHEN x >= 0 THEN x ELSE 0 END`
   - Handles both single values and variable groups

2. **`src/orbital/translation/steps/sigmoid.py`** - Sigmoid activation translator
   - Translates ONNX Sigmoid to SQL `1 / (1 + EXP(-x))`
   - Handles both single values and variable groups

3. **`src/orbital/translation/steps/gemm.py`** - General Matrix Multiplication translator
   - Translates ONNX Gemm to SQL matrix multiplication
   - Handles weight matrices and bias vectors
   - Supports both transposed and non-transposed matrices
   - Works with PyTorch-exported ONNX models that use `raw_data`

### Modified Files

1. **`src/orbital/translate.py`**
   - Added imports for new translators: `ReLUTranslator`, `SigmoidTranslator`, `GemmTranslator`
   - Registered new translators in `TRANSLATORS` dict:
     - `"Relu": ReLUTranslator` (note: ONNX uses "Relu", not "ReLU")
     - `"Sigmoid": SigmoidTranslator`
     - `"Gemm": GemmTranslator`

## Running the Demo

```bash
# Install dependencies
uv pip install torch onnx duckdb

# Run the demo
uv run python demo_pytorch_final.py
```

## Key Features

### SQL Query Characteristics
- **Pure SQL**: No Python code, can run in any SQL database
- **Complex Expressions**: Nested CASE statements for ReLU, EXP for Sigmoid
- **Optimized**: Uses database's native mathematical functions

### Performance Benefits
- **Database Optimization**: Leverages database query optimizers
- **No Data Transfer**: Predictions happen where data lives
- **Batch Processing**: Can process millions of rows efficiently

### Deployment Benefits
- **No Python Runtime**: Just SQL
- **Portable**: Works with DuckDB, PostgreSQL, MySQL, etc.
- **Embeddable**: Can be part of larger SQL queries

## Example SQL Output

```sql
SELECT 1 / (EXP(-(CASE WHEN (amount * w1 + time * w2 + v1 * w3 + v2 * w4 + bias) >= 0 
                    THEN amount * w1 + time * w2 + v1 * w3 + v2 * w4 + bias 
                    ELSE 0 END * w5 + ...)) + 1) 
AS fraud_probability 
FROM transactions
```

## Limitations

- Currently supports feedforward networks with ReLU and Sigmoid activations
- Weight matrices must be constant (from ONNX initializers)
- Input must be tabular data (not images, text, etc.)
- Model size limited by SQL query complexity

## Future Enhancements

- Support more activation functions (Tanh, LeakyReLU, etc.)
- Support more layer types (Dropout, BatchNorm, etc.)
- Optimize SQL generation for specific databases
- Add support for more complex architectures

## Files Modified

- `src/orbital/translate.py` - Added translator registrations
- `src/orbital/translation/steps/relu.py` - NEW
- `src/orbital/translation/steps/sigmoid.py` - NEW  
- `src/orbital/translation/steps/gemm.py` - NEW
- `demo_pytorch_final.py` - NEW (demo script)

## Conclusion

This demo successfully shows that small neural networks can be converted to SQL and executed directly in databases. The key insight is that Orbital already uses ONNX as an intermediate format, so we can leverage PyTorch's ONNX export capability and add the missing translators for neural network operations.
