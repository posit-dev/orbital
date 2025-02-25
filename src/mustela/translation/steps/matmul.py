from ..translator import Translator


class MatMulTranslator(Translator):
    def process(self):
        assert len(self.inputs) == 2, "MatMul node must have exactly 2 inputs."

        coef_tensor = self._variables.get_initializer(self.inputs[1])
        assert coef_tensor, (
            "Coefficient tensor (second input) not found in initializers."
        )

        coef = self._variables.get_initializer_value(self.inputs[1])
        coef_shape = list(coef_tensor.dims)
        first_operand = self._variables.consume(self.inputs[0])

        assert len(coef_shape) in (1, 2), (
            "MatMul with coefficient tensor rank > 2 is not supported."
        )

        # Case 1: left operand is a dict (multiple columns)
        if isinstance(first_operand, dict):
            left_exprs = list(first_operand.values())
            num_features = len(left_exprs)
            if len(coef_shape) == 1:
                # Coefficient vector: expected shape (num_features,)
                if num_features != coef_shape[0]:
                    raise ValueError(
                        "Mismatch: number of features and coefficient vector length"
                    )
                result = sum(
                    self.optimizer.fold_contiguous_sum(
                        self.optimizer.fold_operation(left_exprs[i] * coef[i])
                        for i in range(num_features)
                    )
                )
                self._variables[self._output_name] = result
            elif len(coef_shape) == 2:
                # Coefficient matrix: expected shape (num_features, output_dim)
                if num_features != coef_shape[0]:
                    raise ValueError(
                        "Mismatch: number of features and coefficient matrix rows"
                    )
                output_dim = coef_shape[1]
                result_list = [
                    sum(
                        self._optimizer.fold_contiguous_sum(
                            self._optimizer.fold_operation(
                                left_exprs[i] * coef[i * output_dim + j]
                            )
                            for i in range(num_features)
                        )
                    )
                    for j in range(output_dim)
                ]
                if output_dim == 1:
                    self._variables[self._output_name] = result_list[0]
                else:
                    # Return a dict of output expressions if there are multiple output columns.
                    self._variables[self._output_name] = {
                        f"out_{j}": result_list[j] for j in range(output_dim)
                    }
            else:
                raise NotImplementedError(
                    "MatMul with coefficient tensor rank > 2 is not supported"
                )
        else:
            # Case 2: left operand is a single expression.
            if len(coef_shape) == 1:
                # Expect a single coefficient.
                if coef_shape[0] != 1:
                    raise ValueError(
                        "Expected coefficient vector of length 1 for single operand"
                    )
                self._variables[self._output_name] = self._optimizer.fold_operation(
                    first_operand * coef[0]
                )
            elif len(coef_shape) == 2:
                # Two possible shapes: [1, N] or [N, 1]
                if coef_shape[0] == 1:
                    output_dim = coef_shape[1]
                    result_list = [
                        self._optimizer.fold_operation(first_operand * coef[j])
                        for j in range(output_dim)
                    ]
                    if output_dim == 1:
                        self._variables[self._output_name] = result_list[0]
                    else:
                        self._variables[self._output_name] = {
                            f"out_{j}": result_list[j] for j in range(output_dim)
                        }
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