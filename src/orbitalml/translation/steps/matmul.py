"""Implementation of the LabelEncoder operator."""

import typing

import ibis

from ..translator import Translator
from ..variables import ValueVariablesGroup


class MatMulTranslator(Translator):
    """Processes a MatMul node and updates the variables with the output expression.

    This class is responsible for handling the matrix multiplication operation
    in the translation process. It takes two inputs: the first operand and the
    second operand (coefficient tensor).
    The first operand can be a column group or a single column,
    while the second operand must be a constant value.

    When the second operand is a single value, all columns of the column
    group are multiplied by that value. If the second operand is instead
    a list, each column of the column group is multiplied by the
    corresponding value in the list.
    """

    def process(self) -> None:
        """Performs the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx__MatMul.html

        coef_tensor = self._variables.get_initializer(self.inputs[1])
        if coef_tensor is None:
            raise ValueError(
                "Coefficient tensor (second input) not found in initializers."
            )
        coef_shape = list(coef_tensor.dims)
        if len(coef_shape) not in (1, 2):
            raise ValueError(
                "MatMul with coefficient tensor rank > 2 is not supported."
            )

        coef = self._variables.get_initializer_value(self.inputs[1])
        if coef is None or not isinstance(coef, (list, tuple)):
            raise NotImplementedError(
                "MatMul: Second input (divisor) must be a constant list."
            )
        coef_type_check = coef[0]
        if not isinstance(coef_type_check, (int, float)):
            raise ValueError("MatMul: The second operand must be a numeric value.")

        first_operand = self._variables.consume(self.inputs[0])
        operand_type_check = first_operand
        if isinstance(operand_type_check, dict):
            operand_type_check = list(operand_type_check.values())[0]
        if not isinstance(operand_type_check, ibis.expr.types.NumericValue):
            raise ValueError(
                "MatMul: The first operand must be a numeric column or a column group of numerics."
            )

        # Case 1: left operand is a dict (multiple columns)
        if isinstance(first_operand, dict):
            left_exprs: list[ibis.expr.types.NumericValue] = list(
                first_operand.values()
            )
            num_features = len(left_exprs)
            if len(coef_shape) == 1:
                # Coefficient vector: expected shape (num_features,)
                if num_features != coef_shape[0]:
                    raise ValueError(
                        "Mismatch: number of features and coefficient vector length"
                    )
                result = sum(
                    self._optimizer.fold_contiguous_sum(
                        [
                            self._optimizer.fold_operation(left_exprs[i] * coef[i])
                            for i in range(num_features)
                        ]
                    )
                )
                self.set_output(result)
            elif len(coef_shape) == 2:
                # Coefficient matrix: expected shape (num_features, output_dim)
                if num_features != coef_shape[0]:
                    raise ValueError(
                        "Mismatch: number of features and coefficient matrix rows"
                    )
                output_dim = coef_shape[1]
                result_list: list[ibis.expr.types.NumericValue] = [
                    sum(
                        self._optimizer.fold_contiguous_sum(
                            [
                                self._optimizer.fold_operation(
                                    left_exprs[i] * coef[i * output_dim + j]
                                )
                                for i in range(num_features)
                            ]
                        )
                    )
                    for j in range(output_dim)
                ]
                if output_dim == 1:
                    result = result_list[0]
                else:
                    # Return a dict of output expressions if there are multiple output columns.
                    result = ValueVariablesGroup(
                        {f"out_{j}": result_list[j] for j in range(output_dim)}
                    )
                self.set_output(result)
            else:
                raise NotImplementedError(
                    "MatMul with coefficient tensor rank > 2 is not supported"
                )
        else:
            first_operand = typing.cast(ibis.expr.types.NumericValue, first_operand)
            # Case 2: left operand is a single expression.
            if len(coef_shape) == 1:
                # Expect a single coefficient.
                if coef_shape[0] != 1:
                    raise ValueError(
                        "Expected coefficient vector of length 1 for single operand"
                    )
                self.set_output(self._optimizer.fold_operation(first_operand * coef[0]))
            elif len(coef_shape) == 2:
                # Two possible shapes: [1, N] or [N, 1]
                if coef_shape[0] == 1:
                    output_dim = coef_shape[1]
                    result_list = [
                        self._optimizer.fold_operation(first_operand * coef[j])
                        for j in range(output_dim)
                    ]
                    if output_dim == 1:
                        result = result_list[0]
                        self.set_output(result_list[0])
                    else:
                        result = ValueVariablesGroup(
                            {f"out_{j}": result_list[j] for j in range(output_dim)}
                        )
                    self.set_output(result)
                elif coef_shape[1] == 1:
                    # This case implies the left operand is a vector of length matching coef_shape[0],
                    # but a single expression cannot be indexed. We mark this as not supported.
                    raise NotImplementedError(
                        "MatMul with left operand as single column and coefficient matrix shape [N,1] is not supported"
                    )
                else:
                    raise NotImplementedError(
                        "Unexpected coefficient shape for single operand"
                    )
            else:
                raise NotImplementedError(
                    "MatMul with coefficient tensor rank > 2 is not supported"
                )
