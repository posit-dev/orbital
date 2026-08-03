"""Implementation of the Tanh operator."""

import ibis

from ...translator import Translator
from ...variables import NumericVariablesGroup, ValueVariablesGroup, VariablesGroup


class TanhTranslator(Translator):
    """Processes a Tanh node and updates the variables with the output expression.

    The operation computes the elementwise hyperbolic tangent of the input::

        Tanh(x) = tanh(x)

    When the input is a column group, the hyperbolic tangent is computed
    independently for each column in the group.
    """

    def process(self) -> None:
        """Performs the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx__Tanh.html
        data = self._variables.consume(self._inputs[0])

        # Ibis exposes no tanh(), so the database native tanh is called through
        # a builtin UDF. This emits TANH(x) verbatim, which every engine we test
        # provides, but it will fail on engines that lack the function.
        # Expressing it as 2/(1+exp(-2x))-1 instead would only need exp(), but
        # it overflows: PostgreSQL rejects large |x| with a range error.
        # Drop the UDF once ibis gains native tanh support.
        if isinstance(data, VariablesGroup):
            data = NumericVariablesGroup(data)
            self.set_output(
                ValueVariablesGroup(
                    {name: _tanh(value) for name, value in data.items()}
                )
            )
        elif isinstance(data, ibis.expr.types.NumericValue):
            self.set_output(_tanh(data))
        else:
            raise ValueError(
                "Tanh: The first operand must be a numeric column or a column group of numerics."
            )


@ibis.udf.scalar.builtin(name="tanh")
def _tanh(value: float) -> float:  # type: ignore[empty-body]
    """Emits a call to the database native tanh function.

    Ibis builds the SQL call from the signature, the body is never executed.
    """
