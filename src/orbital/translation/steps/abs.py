"""Implementation of the Abs operator."""

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, ValueVariablesGroup, VariablesGroup


class AbsTranslator(Translator):
    """Processes an Abs node and updates the variables with the output expression.

    The operation computes the elementwise absolute value::

        Abs(x) = |x|

    When the input is a column group, the absolute value is computed
    independently for each column in the group.
    """

    def process(self) -> None:
        """Performs the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx__Abs.html
        data = self._variables.consume(self._inputs[0])
        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            self.set_output(
                ValueVariablesGroup(
                    {
                        name: self._optimizer.fold_operation(value.abs())
                        for name, value in data.items()
                    }
                )
            )
        elif isinstance(data, ibis.expr.types.NumericValue):
            self.set_output(self._optimizer.fold_operation(data.abs()))
        else:
            raise ValueError(
                "Abs: The first operand must be a numeric column or a column group of numerics."
            )
