"""Implementation of the Imputer operator."""

import ibis

from ..translator import Translator
from ..variables import ValueVariablesGroup, VariablesGroup


class ImputerTranslator(Translator):
    """Processes an Imputer node and updates the variables with the output expression.

    The imputer node replaces missing values in the input expression with
    another value. Currently the only supported value is a float, which is
    used to replace all missing values in the input expression.
    """

    def process(self) -> None:
        """Performs the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx_aionnxml_Imputer.html

        imputed_values = self._attributes["imputed_value_floats"]
        if not isinstance(imputed_values, (tuple, list)):
            raise ValueError("Imputer: imputed_value must be a list or tuple of floats")

        expr = self._variables.consume(self.inputs[0])
        if isinstance(expr, VariablesGroup):
            keys = list(expr.keys())
            if len(keys) != len(imputed_values):
                raise ValueError(
                    "Imputer: number of imputed values does not match number of columns"
                )
            new_expr = ValueVariablesGroup()
            for i, key in enumerate(keys):
                new_expr[key] = ibis.coalesce(expr[key], imputed_values[i])
            self.set_output(new_expr)
        else:
            self.set_output(ibis.coalesce(expr, imputed_values[0]))
