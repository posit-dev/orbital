"""Translator for Sigmoid activation function."""

import ibis

from ...translator import Translator
from ...variables import ValueVariablesGroup, VariablesGroup


class SigmoidTranslator(Translator):
    """Translate ONNX Sigmoid operation to SQL expression.

    Sigmoid(x) = 1 / (1 + exp(-x))
    """

    def process(self) -> None:
        """Process Sigmoid node and generate SQL translation."""
        input_var = self._variables.consume(self._inputs[0])

        # Handle both single values and variable groups
        if isinstance(input_var, VariablesGroup):
            # Apply Sigmoid to each element in the group
            result = ValueVariablesGroup()
            for key, value in input_var.items():
                # SQL: 1 / (1 + EXP(-x))
                output = 1 / (1 + (-value).exp())
                result[key] = output
            self.set_output(result)
        elif isinstance(input_var, ibis.expr.types.NumericValue):
            # Single numeric value
            # SQL: 1 / (1 + EXP(-x))
            output = 1 / (1 + (-input_var).exp())
            self.set_output(output)
        else:
            raise ValueError(
                f"Sigmoid: Expected a numeric value or variable group, got {type(input_var)}"
            )
