"""Implementation of the Relu operator."""

import ibis

from ...translator import Translator
from ...variables import NumericVariablesGroup, ValueVariablesGroup, VariablesGroup


class ReLUTranslator(Translator):
    """Processes a Relu node and updates the variables with the output expression.

    The operation computes the elementwise rectified linear function::

        Relu(x) = max(0, x)

    which translates to a SQL CASE expression that returns zero
    when the value is negative and the value itself otherwise.

    When the input is a column group, the rectification is computed
    independently for each column in the group.
    """

    def process(self) -> None:
        """Performs the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx__Relu.html
        data = self._variables.consume(self._inputs[0])
        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            self.set_output(
                ValueVariablesGroup(
                    {
                        # NULL < 0 is not-true, so NULLs fall through to the
                        # else branch and propagate instead of becoming 0.
                        name: self._optimizer.fold_case(
                            ibis.cases((value < 0, 0), else_=value)
                        )
                        for name, value in data.items()
                    }
                )
            )
        elif isinstance(data, ibis.expr.types.NumericValue):
            self.set_output(
                self._optimizer.fold_case(ibis.cases((data < 0, 0), else_=data))
            )
        else:
            raise ValueError(
                "Relu: The first operand must be a numeric column or a column group of numerics."
            )
