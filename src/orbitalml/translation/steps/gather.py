"""Defines the translation step for the Gather operation."""

from orbitalml.translation.variables import VariablesGroup

from ..translator import Translator


class GatherTranslator(Translator):
    """Processes a Gather node and updates the variables with the output expression.

    The gather operations is meant to pick a specific value out of a column or
    column group.

    The first operand can be a column group or a single column,
    while the second operand must be a constant value.

    When the first operand is a column, the second operand must be 0 as
    there is only one column.

    The operation could in theory be used to pick a specific row of columns
    by setting axis=0, but this is not supported in the current implementation.
    """

    def process(self) -> None:
        """Performs the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx__Gather.html

        axis = self._attributes.get("axis", 0)
        if axis != 1:
            raise NotImplementedError(
                f"Gather: axis {axis} not supported, only selecting columns (axis=1) is supported"
            )

        expr = self._variables.consume(self.inputs[0])
        idx = self._variables.get_initializer_value(self.inputs[1])
        if not isinstance(idx, (tuple, list)) or len(idx) != 1:
            raise NotImplementedError(
                "Gather second operand must a list of one element"
            )

        idx = idx[0]  # TODO: Support gathering multiple columns
        if not isinstance(idx, int):
            raise ValueError("Gather: index must be an integer constant")

        if isinstance(expr, VariablesGroup):
            keys = list(expr.keys())
            if idx < 0 or idx >= len(keys):
                raise IndexError("Gather: index out of bounds")
            self.set_output(expr[keys[idx]])
        else:
            # Assume that if it's a single column by virtue of the fact that we only
            # support axis=1, then the index must be 0.
            if idx != 0:
                raise NotImplementedError(
                    f"Gather: index {idx} not supported for single columns"
                )
            self.set_output(expr)
