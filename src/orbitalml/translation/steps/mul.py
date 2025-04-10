"""Translate an Mul operation to the equivalent query expression."""

import typing

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, ValueVariablesGroup, VariablesGroup


class MulTranslator(Translator):
    """Processes an Mul node and updates the variables with the output expression.

    Given the node to translate, the variables and constants available for
    the translation context, generates a query expression that processes
    the input variables and produces a new output variable that computes
    based on the Mul operation.
    """

    def process(self) -> None:
        """Performs the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx__Mul.html

        first_operand = self._variables.consume(self._inputs[0])
        second_operand = self._variables.get_initializer_value(self._inputs[1])
        if second_operand is None or not isinstance(second_operand, (list, tuple)):
            raise NotImplementedError(
                "Mul: Second input (divisor) must be a constant list."
            )

        type_check_var = first_operand
        if isinstance(type_check_var, VariablesGroup):
            type_check_var = next(iter(type_check_var.values()), None)
        if not isinstance(type_check_var, ibis.expr.types.NumericValue):
            raise ValueError("Mul: The first operand must be a numeric value.")

        add_values = list(second_operand)
        if isinstance(first_operand, VariablesGroup):
            first_operand = NumericVariablesGroup(first_operand)
            struct_fields = list(first_operand.keys())
            if len(add_values) != len(struct_fields):
                # TODO: Implement dividing by a single value,
                #       see Div implementation.
                raise ValueError(
                    "When the first operand is a group of columns, the second operand must contain the same number of values"
                )
            self.set_output(
                ValueVariablesGroup(
                    {
                        field: (
                            self._optimizer.fold_operation(
                                first_operand[field] * add_values[i]
                            )
                        )
                        for i, field in enumerate(struct_fields)
                    }
                )
            )
        else:
            if len(add_values) != 1:
                raise ValueError(
                    "When the first operand is a single column, the second operand must contain exactly 1 value"
                )
            first_operand = typing.cast(ibis.expr.types.NumericValue, first_operand)
            self.set_output(
                self._optimizer.fold_operation(first_operand * add_values[0])
            )
