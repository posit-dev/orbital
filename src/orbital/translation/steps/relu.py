"""Translator for ReLU activation function."""

from ..translator import Translator
from ..variables import VariablesGroup, ValueVariablesGroup
import ibis


class ReLUTranslator(Translator):
    """Translate ONNX ReLU operation to SQL CASE expression.
    
    ReLU(x) = max(0, x) = CASE WHEN x >= 0 THEN x ELSE 0 END
    """

    def process(self) -> None:
        """Process ReLU node and generate SQL translation."""
        input_var = self._variables.consume(self._inputs[0])
        
        # Handle both single values and variable groups
        if isinstance(input_var, VariablesGroup):
            # Apply ReLU to each element in the group
            result = ValueVariablesGroup()
            for key, value in input_var.items():
                # SQL: CASE WHEN x >= 0 THEN x ELSE 0 END
                output = ibis.cases(
                    (value >= 0, value),
                    else_=0
                )
                result[key] = output
            self.set_output(result)
        elif isinstance(input_var, ibis.expr.types.NumericValue):
            # Single numeric value
            # SQL: CASE WHEN x >= 0 THEN x ELSE 0 END
            output = ibis.cases(
                (input_var >= 0, input_var),
                else_=0
            )
            self.set_output(output)
        else:
            raise ValueError(
                f"ReLU: Expected a numeric value or variable group, got {type(input_var)}"
            )
