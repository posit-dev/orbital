"""Implementation of the Scaler operator."""

import typing

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, ValueVariablesGroup, VariablesGroup


class ScalerTranslator(Translator):
    """Processes a Scaler node and updates variables with the scaled expression.

    The Scaler operator applies a scaling and offset to the input:
    Y = (X - offset) * scale

    The scaler operation is not always used, for more complex pipelines
    usually a combination of Sub and Mul operations is used.
    """

    def process(self) -> None:
        """Performs the translation and sets the output variable."""
        # https://onnx.ai/onnx/operators/onnx_aionnxml_Scaler.html
        scale = typing.cast(list[float], self._attributes["scale"])
        offset = typing.cast(list[float], self._attributes["offset"])

        if len(self._inputs) != 1:
            raise ValueError("Scaler node must have exactly 1 input.")

        input_operand = self._variables.consume(self._inputs[0])

        type_check_var = input_operand
        if isinstance(type_check_var, dict):
            type_check_var = next(iter(type_check_var.values()), None)

        if not isinstance(type_check_var, ibis.expr.types.NumericValue):
            raise ValueError("Scaler: The input operand must be numeric.")

        if isinstance(input_operand, VariablesGroup):
            input_operand = NumericVariablesGroup(input_operand)

            # If the attributes are len=1,
            # it means to apply the same value to all inputs.
            num_fields = len(input_operand)
            if len(offset) == 1:
                offset = offset * num_fields
            if len(scale) == 1:
                scale = scale * num_fields

            if len(offset) != num_fields or len(scale) != num_fields:
                raise ValueError(
                    "Scaler: offset and scale lists must match the number of input fields."
                )

            self.set_output(
                ValueVariablesGroup(
                    {
                        field: self._optimizer.fold_operation(
                            (val - offset[i]) * scale[i]
                        )
                        for i, (field, val) in enumerate(input_operand.items())
                    }
                )
            )
        else:
            input_operand = typing.cast(ibis.expr.types.NumericValue, input_operand)
            self.set_output(
                self._optimizer.fold_operation((input_operand - offset[0]) * scale[0])
            )
