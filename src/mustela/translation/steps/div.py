"""Defines the translation step for the Div operation."""

import typing

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, ValueVariablesGroup, VariablesGroup


class DivTranslator(Translator):
    """Processes a Div node and updates the variables with the output expression.

    This class is responsible for handling the division operation in the
    translation process. It takes two inputs: the first operand and the second
    operand (divisor).

    The first operand can be a column group or a single column,
    while the second operand must be a constant value.

    When the second operand is a single value, all columns of the column
    group are divided for that value. If the second operand is instead
    a list, each column of the column group is divided for the corresponding
    value in the list.
    """

    def process(self) -> None:
        """Performs the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx__Div.html

        first_operand = self._variables.consume(self.inputs[0])
        second_arg = self._variables.get_initializer_value(self.inputs[1])
        if second_arg is None or not isinstance(second_arg, (list, tuple)):
            raise NotImplementedError(
                "Div: Second input (divisor) must be a constant list."
            )

        if isinstance(first_operand, VariablesGroup):
            first_operand = NumericVariablesGroup(first_operand)
            struct_fields = list(first_operand.keys())
            for value in first_operand.values():
                if not isinstance(value, ibis.expr.types.NumericValue):
                    raise ValueError("Div: The first operand must be a numeric value.")

            first_operand = typing.cast(
                dict[str, ibis.expr.types.NumericValue], first_operand
            )
            if len(second_arg) == 1:
                second_arg = second_arg[0]
                if not isinstance(second_arg, (int, float)):
                    raise ValueError("Div: The second operand must be a numeric value.")
                self.set_output(
                    ValueVariablesGroup(
                        {
                            field: (
                                self._optimizer.fold_operation(
                                    first_operand[field] / ibis.literal(second_arg)
                                )
                            )
                            for field in struct_fields
                        }
                    )
                )
            else:
                if len(second_arg) != len(first_operand):
                    raise ValueError(
                        "The number of elements in the second operand must match the number of columns in the first operand."
                    )
                self.set_output(
                    ValueVariablesGroup(
                        {
                            field: (
                                self._optimizer.fold_operation(
                                    first_operand[field] / second_arg[i]
                                )
                            )
                            for i, field in enumerate(struct_fields)
                        }
                    )
                )
        else:
            if not isinstance(first_operand, ibis.expr.types.NumericValue):
                raise ValueError("Div: The first operand must be a numeric value.")
            if len(second_arg) != 1:
                raise ValueError(
                    "when first operand is a single column, second operand must contain only one value."
                )

            first_operand = typing.cast(ibis.expr.types.NumericValue, first_operand)
            self.set_output(self._optimizer.fold_operation(first_operand / second_arg))
