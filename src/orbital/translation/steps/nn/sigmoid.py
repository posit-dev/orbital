"""Implementation of the Sigmoid operator."""

import typing

import ibis

from ...transformations import apply_post_transform
from ...translator import Translator
from ...variables import NumericVariablesGroup, VariablesGroup


class SigmoidTranslator(Translator):
    """Processes a Sigmoid node and updates the variables with the output expression.

    The operation computes the elementwise logistic function of the input::

        Sigmoid(x) = 1 / (1 + exp(-x))

    When the input is a column group, the sigmoid is computed
    independently for each column in the group.
    """

    def process(self) -> None:
        """Performs the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx__Sigmoid.html
        data = self._variables.consume(self._inputs[0])
        if not isinstance(data, (ibis.expr.types.NumericValue, dict)):
            raise ValueError(
                "Sigmoid: The first operand must be a numeric column or a column group of numerics."
            )

        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
        else:
            data = typing.cast(ibis.expr.types.NumericValue, data)
        self.set_output(apply_post_transform(data, "LOGISTIC"))
