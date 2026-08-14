"""Defines the translation step for the Div operation."""

from ..translator import Translator
from ..variables import ValueVariablesGroup


class DivTranslator(Translator):
    """Processes a Div node and updates the variables with the output expression.

    Both operands are treated symmetrically: each one can be a group of columns,
    a single column, a constant scalar or a constant list of values.

    When any of the two operands is a group of columns, the result is a group of
    columns too and it borrows its column names from the first operand that is a
    group, as those names end up being the names of the resulting SQL columns.
    That group also dictates the width of the result: any other operand must
    either provide exactly as many values, or a single value that is divided
    (or divided by) every column of the group.

    When neither operand is a group, both must be single values and the result
    is a single column.
    """

    def process(self) -> None:
        """Performs the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx__Div.html

        left_keys, left_values = self._variables.consume_operand_values(self.inputs[0])
        right_keys, right_values = self._variables.consume_operand_values(
            self.inputs[1]
        )

        # The first operand that is a group dictates the width of the result
        # and, at the very end, the names of the resulting columns.
        keys = left_keys if left_keys is not None else right_keys
        if keys is None:
            # Simple case, no columns group involved, so we just divide the two values.
            if len(left_values) != 1 or len(right_values) != 1:
                raise ValueError(
                    "Div: when no operand is a group of columns, each operand must contain only one value."
                )
            self.set_output(
                self._optimizer.fold_operations(left_values[0] / right_values[0])
            )
            return

        for values in (left_values, right_values):
            if len(values) not in (1, len(keys)):
                raise ValueError(
                    "Div: the number of values of each operand must match the number of columns of the resulting group."
                )

        # A single value is shared by all columns of the resulting group.
        if len(left_values) == 1:
            left_values = left_values * len(keys)
        if len(right_values) == 1:
            right_values = right_values * len(keys)

        self.set_output(
            ValueVariablesGroup(
                {
                    key: self._optimizer.fold_operations(left_value / right_value)
                    for key, left_value, right_value in zip(
                        keys, left_values, right_values
                    )
                }
            )
        )
