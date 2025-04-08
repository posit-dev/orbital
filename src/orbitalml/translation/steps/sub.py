"""Implementation of the Sub operator."""

import typing

import ibis

from orbitalml.translation.variables import (
    NumericVariablesGroup,
    ValueVariablesGroup,
    VariablesGroup,
)

from ..translator import Translator


class SubTranslator(Translator):
    """Processes a Sub node and updates the variables with the output expression.

    Given the node to translate, the variables and constants available for
    the translation context, generates a query expression that processes
    the input variables and produces a new output variable that computes
    based on the Sub operation.
    """

    def process(self) -> None:
        """Performs the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx__Sub.html
        assert len(self._inputs) == 2, "The Sub node must have exactly 2 inputs."

        first_operand = self._variables.consume(self._inputs[0])
        second_operand = self._variables.get_initializer_value(self._inputs[1])
        if second_operand is None or not isinstance(second_operand, (list, tuple)):
            raise NotImplementedError(
                "Sub: Second input (divisor) must be a constant list."
            )

        type_check_var = first_operand
        if isinstance(type_check_var, dict):
            type_check_var = next(iter(type_check_var.values()), None)
        if not isinstance(type_check_var, ibis.expr.types.NumericValue):
            raise ValueError("Sub: The first operand must be a numeric value.")

        sub_values = list(second_operand)
        if isinstance(first_operand, VariablesGroup):
            first_operand = NumericVariablesGroup(first_operand)
            struct_fields = list(first_operand.keys())
            assert len(sub_values) == len(struct_fields), (
                f"The number of values in the initializer ({len(sub_values)}) must match the number of fields ({len(struct_fields)}"
            )
            self.set_output(
                ValueVariablesGroup(
                    {
                        field: (
                            self._optimizer.fold_operation(
                                first_operand[field] - sub_values[i]
                            )
                        )
                        for i, field in enumerate(struct_fields)
                    }
                )
            )
        else:
            if len(sub_values) != 1:
                raise ValueError(
                    "When the first operand is a single column, the second operand must contain exactly 1 value"
                )
            first_operand = typing.cast(ibis.expr.types.NumericValue, first_operand)
            self.set_output(
                self._optimizer.fold_operation(first_operand - sub_values[0])
            )
