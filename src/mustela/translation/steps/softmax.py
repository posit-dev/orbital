"""Implementation of the Softmax operator."""

import typing

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, ValueVariablesGroup, VariablesGroup


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
            data = typing.cast(
                ibis.expr.types.NumericValue, ibis.expr.types.NumericValue
            )
        self.set_output(self.compute_softmax(self, data))

    @classmethod
    def compute_softmax(
        cls,
        translator: Translator,
        data: ibis.expr.types.NumericValue | VariablesGroup,
    ) -> ibis.Expr | VariablesGroup:
        """Computes the actual softmax operation over a column or column group."""
        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            max_value = ibis.greatest(*data.values()).name(
                translator.variable_unique_short_alias("sfmx")
            )
            translator.preserve(max_value)

            # Compute, for each column, the exponent
            exp_dict = {k: (v - max_value).exp() for k, v in data.items()}

            # Sum all column exponents
            sum_exp = sum(exp_dict.values())

            # Multi columns case: softmax = exp(column_exp) / (exponents_sum)
            return ValueVariablesGroup({k: exp_dict[k] / sum_exp for k in data.keys()})
        elif isinstance(data, ibis.Expr):
            # Single column case: softmax(x) = exp(x) / exp(x) = 1
            return ibis.literal(1.0)
        else:
            raise TypeError(
                f"Softmax: expected a column group or a single column. Got {type(data)}"
            )
