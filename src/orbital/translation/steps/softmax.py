"""Implementation of the Softmax operator."""

import typing

import ibis

from ..transformations import apply_post_transform
from ..translator import Translator
from ..variables import NumericVariablesGroup, VariablesGroup


class SoftmaxTranslator(Translator):
    """Processes a Softmax node and updates the variables with the output expression.

    The operation computes the normalized exponential of the input::

        Softmax = Exp(input) / Sum(Exp(input))

    Currently the Softmax operation is supported only for axis=-1 or axis=1,
    which means for the a column group means that the softmax is computed
    independently for each column in the group.
    """

    def process(self) -> None:
        """Performs the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx__Softmax.html
        data = self._variables.consume(self.inputs[0])
        if not isinstance(data, (ibis.expr.types.NumericValue, dict)):
            raise ValueError(
                "Softmax: The first operand must be a numeric column or a column group of numerics."
            )

        axis = self._attributes.get("axis", -1)
        if axis not in (-1, 1):
            raise ValueError(
                "SoftmaxTranslator supports only axis=-1 or axis=1 for group of columns"
            )

        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
        else:
            data = typing.cast(ibis.expr.types.NumericValue, data)
        self.set_output(apply_post_transform(data, "SOFTMAX"))
