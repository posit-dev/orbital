"""Translator for Gemm (General Matrix Multiplication) operation."""

import typing

import ibis

from ..translator import Translator
from ..variables import ValueVariablesGroup


class GemmTranslator(Translator):
    """Processes a Gemm node and updates the variables with the output expression.
    
    Gemm implements: output = alpha * (A @ B) + beta * C
    
    For neural network linear layers:
    - A is the input (shape: [batch, in_features])
    - B is the weight matrix (shape: [in_features, out_features])
    - C is the bias vector (shape: [out_features])
    - alpha is typically 1.0
    - beta is typically 1.0
    - transA is typically 0 (no transpose)
    - transB is typically 1 (transpose B)
    
    So the operation becomes: output = A @ B^T + C
    """

    def process(self) -> None:
        """Performs the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx__Gemm.html
        
        # Get attributes
        alpha = self._attributes.get("alpha", 1.0)
        beta = self._attributes.get("beta", 1.0)
        transA = self._attributes.get("transA", 0)
        transB = self._attributes.get("transB", 0)
        
        # For typical neural network usage:
        # - transA = 0 (input not transposed)
        # - transB = 1 (weight matrix is transposed, so we don't need to transpose it again)
        # - alpha = 1.0
        # - beta = 1.0
        
        # Input A (first input)
        A = self._variables.consume(self._inputs[0])
        
        # Input B (second input) - weight matrix
        B = self._variables.get_initializer(self._inputs[1])
        if B is None:
            raise ValueError(
                f"Weight matrix (second input '{self._inputs[1]}') not found in initializers."
            )
        
        # Get weight values - handle both float_data and raw_data
        B_values = self._variables.get_initializer_value(self._inputs[1])
        
        if B_values is None or not isinstance(B_values, (list, tuple)) or len(B_values) == 0:
            # Try to get values from raw_data using onnx.numpy_helper
            # This is needed for PyTorch-exported ONNX models which use raw_data
            import onnx.numpy_helper
            B_array = onnx.numpy_helper.to_array(B)
            B_values = B_array.flatten().tolist()
        
        if not isinstance(B_values, (list, tuple)) or len(B_values) == 0:
            raise NotImplementedError(
                "Gemm: Second input (weight matrix) must be a constant list."
            )
        
        # Input C (third input, optional) - bias vector
        C = None
        C_values = None
        if len(self._inputs) > 2:
            C = self._variables.get_initializer(self._inputs[2])
            if C is None:
                raise ValueError(
                    f"Bias vector (third input '{self._inputs[2]}') not found in initializers."
                )
            C_values = self._variables.get_initializer_value(self._inputs[2])
            
            if C_values is None or not isinstance(C_values, (list, tuple)) or len(C_values) == 0:
                # Try to get values from raw_data using onnx.numpy_helper
                # This is needed for PyTorch-exported ONNX models which use raw_data
                import onnx.numpy_helper
                C_array = onnx.numpy_helper.to_array(C)
                C_values = C_array.flatten().tolist()
            
            if not isinstance(C_values, (list, tuple)) or len(C_values) == 0:
                raise NotImplementedError(
                    "Gemm: Third input (bias vector) must be a constant list."
                )
        
        # Get dimensions
        B_dims = list(B.dims)
        
        # For typical case:
        # - When transB=1: B has shape [out_features, in_features] and we use B^T
        # - When transB=0: B has shape [in_features, out_features]
        if transB == 1:
            # B is stored as [out_features, in_features]
            # B^T has shape [in_features, out_features]
            in_features = B_dims[1]
            out_features = B_dims[0]
        else:
            # B is stored as [in_features, out_features]
            in_features = B_dims[0]
            out_features = B_dims[1]
        
        # Process based on input A type
        from ..variables import VariablesGroup
        
        if isinstance(A, (dict, VariablesGroup)):
            # A is a dict of columns (multiple input features)
            A_exprs: list[ibis.expr.types.NumericValue] = list(A.values())
            num_input_features = len(A_exprs)
            
            if num_input_features != in_features:
                raise ValueError(
                    f"Mismatch: input has {num_input_features} features "
                    f"but weight matrix expects {in_features}"
                )
            

            
            # Compute matrix multiplication: A @ B
            # For each output feature j: sum over i of A[i] * B[i][j]
            result_list: list[ibis.expr.types.NumericValue] = []
            
            for j in range(out_features):
                # Compute sum for output feature j
                terms = []
                for i in range(in_features):
                    # Get weight for input i, output j
                    if transB == 1:
                        # B is stored as [out_features, in_features]
                        weight_idx = j * in_features + i
                    else:
                        # B is stored as [in_features, out_features]
                        weight_idx = i * out_features + j
                    
                    weight = B_values[weight_idx]
                    terms.append(self._optimizer.fold_operation(A_exprs[i] * weight))
                
                # Sum all terms for this output
                matmul_result = sum(self._optimizer.fold_contiguous_sum(terms))
                
                # Apply alpha scaling
                if alpha != 1.0:
                    matmul_result = self._optimizer.fold_operation(matmul_result * alpha)
                
                # Add bias if present
                if C is not None and beta != 0:
                    bias = C_values[j] if isinstance(C_values, (list, tuple)) else C_values
                    bias_term = self._optimizer.fold_operation(bias * beta)
                    matmul_result = self._optimizer.fold_operation(matmul_result + bias_term)
                
                result_list.append(matmul_result)
            
            # Return result
            if out_features == 1:
                self.set_output(result_list[0])
            else:
                result = ValueVariablesGroup(
                    {f"out_{j}": result_list[j] for j in range(out_features)}
                )
                self.set_output(result)
                
        else:
            # A is a single expression
            A = typing.cast(ibis.expr.types.NumericValue, A)
            
            # For a single input, B should be a vector
            if len(B_dims) == 1:
                # B is a vector of length out_features
                out_features = B_dims[0]
                
                result_list = []
                for j in range(out_features):
                    weight = B_values[j]
                    term = self._optimizer.fold_operation(A * weight)
                    
                    if alpha != 1.0:
                        term = self._optimizer.fold_operation(term * alpha)
                    
                    if C is not None and beta != 0:
                        bias = C_values[j] if isinstance(C_values, (list, tuple)) else C_values
                        bias_term = self._optimizer.fold_operation(bias * beta)
                        term = self._optimizer.fold_operation(term + bias_term)
                    
                    result_list.append(term)
                
                if out_features == 1:
                    self.set_output(result_list[0])
                else:
                    result = ValueVariablesGroup(
                        {f"out_{j}": result_list[j] for j in range(out_features)}
                    )
                    self.set_output(result)
            else:
                raise NotImplementedError(
                    "Gemm with single input and matrix second operand is not yet supported"
                )
