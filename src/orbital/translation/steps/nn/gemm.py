"""Translator for Gemm (General Matrix Multiplication) operation."""

import typing

import ibis
from onnx import numpy_helper

from ...translator import Translator
from ...variables import ValueVariablesGroup, VariablesGroup


class GemmTranslator(Translator):
    """Processes a Gemm node and updates the variables with the output expression.

    Gemm implements output = alpha * (A @ B) + beta * C, which is how
    neural network linear layers (like torch.nn.Linear) are exported
    to ONNX: A is the input data, B the weight matrix and C the bias vector.
    """

    def process(self) -> None:
        """Performs the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx__Gemm.html
        alpha = typing.cast(float, self._attributes.get("alpha", 1.0))
        beta = typing.cast(float, self._attributes.get("beta", 1.0))
        if self._attributes.get("transA", 0):
            raise NotImplementedError("Gemm: transA=1 is not supported")

        weights_proto = self._variables.get_initializer(self._inputs[1])
        if weights_proto is None:
            raise NotImplementedError(
                "Gemm: second input (weight matrix) must be a constant initializer"
            )
        # numpy_helper handles both float_data and raw_data storage,
        # PyTorch-exported models use raw_data.
        weights = numpy_helper.to_array(weights_proto)
        if self._attributes.get("transB", 0):
            weights = weights.T
        in_features, out_features = weights.shape

        bias = None
        if len(self._inputs) > 2:
            bias_proto = self._variables.get_initializer(self._inputs[2])
            if bias_proto is None:
                raise NotImplementedError(
                    "Gemm: third input (bias vector) must be a constant initializer"
                )
            bias = numpy_helper.to_array(bias_proto)

        first_operand = self._variables.consume(self._inputs[0])
        if isinstance(first_operand, VariablesGroup):
            input_exprs: list[ibis.expr.types.NumericValue] = list(
                first_operand.values()
            )
        else:
            input_exprs = [typing.cast(ibis.expr.types.NumericValue, first_operand)]
        if len(input_exprs) != in_features:
            raise ValueError(
                f"Gemm: input has {len(input_exprs)} features "
                f"but weight matrix expects {in_features}"
            )

        result_list: list[ibis.expr.types.NumericValue] = []
        for j in range(out_features):
            # alpha scales the matrix product, so it can be folded into each weight.
            output = sum(
                self._optimizer.fold_contiguous_sum(
                    [
                        self._optimizer.fold_operation(
                            input_exprs[i] * float(weights[i, j] * alpha)
                        )
                        for i in range(in_features)
                    ]
                )
            )
            if bias is not None and beta != 0:
                output = self._optimizer.fold_operation(output + float(bias[j] * beta))
            result_list.append(output)

        if out_features == 1:
            self.set_output(result_list[0])
        else:
            self.set_output(
                ValueVariablesGroup(
                    {f"out_{j}": result_list[j] for j in range(out_features)}
                )
            )
