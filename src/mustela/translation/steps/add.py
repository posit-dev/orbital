"""Translate an Add operation to the equivalent query expression."""

from ..translator import Translator


class AddTranslator(Translator):
    """Processes an Add node and updates the variables with the output expression.

    Given the node to translate, the variables and constants available for
    the translation context, generates a query expression that processes
    the input variables and produces a new output variable that computes
    based on the Add operation.
    """

    def process(self) -> None:
        """Performs the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx__Add.html
        if len(self._inputs) != 2:
            raise ValueError("The Add step must have exactly 2 operands")

        first_operand = self._variables.consume(self._inputs[0])
        second_operand = self._variables.get_initializer_value(self._inputs[1])
        if second_operand is None:
            raise ValueError("Add currently expects a constants as its second operand")

        add_values = second_operand
        if isinstance(first_operand, dict):
            struct_fields = list(first_operand.keys())
            if len(add_values) != len(struct_fields):
                raise ValueError(
                    "When the first operand is a group of columns, the second operand must contain the same number of values"
                )
            self._variables[self._output_name] = {
                field: (
                    self._optimizer.fold_operation(first_operand[field] + add_values[i])
                )
                for i, field in enumerate(struct_fields)
            }
        else:
            if len(add_values) != 1:
                raise ValueError(
                    "When the first operand is a single column, the second operand must contain exactly 1 value"
                )
            self._variables[self._output_name] = self._optimizer.fold_operation(
                first_operand + add_values[0]
            )
