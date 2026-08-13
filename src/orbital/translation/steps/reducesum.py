"""Implementation of the ReduceSum operator."""

from ..translator import Translator
from ..variables import NumericVariablesGroup, VariablesGroup


class ReduceSumTranslator(Translator):
    """Processes a ReduceSum node and updates the variables with the output expression.

    The operation sums together all the columns of a group::

        ReduceSum([x0, x1, ..., xn]) = x0 + x1 + ... + xn

    Only the reduction over the features axis (``1`` or ``-1``) is supported,
    as reducing over the rows would collapse the whole table into a single
    row that can't be mixed with the other per-row expressions.
    For the same reason the input must be a group of columns, a single
    column has nothing to be summed with.

    The axes are expected as a constant second input, which is how they
    are exported since opset 13.
    """

    def process(self) -> None:
        """Performs the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx__ReduceSum.html
        if len(self._inputs) != 2:
            raise NotImplementedError(
                "ReduceSum: the axes must be provided as the second input."
            )

        data = self._variables.consume(self._inputs[0])
        if not isinstance(data, VariablesGroup):
            raise NotImplementedError(
                "ReduceSum can only be applied to a group of columns"
            )

        axes = self._variables.get_initializer_value(self._inputs[1])
        if axes not in ([1], [-1]):
            raise NotImplementedError(
                f"ReduceSum: only reduction over the features axis is supported, got axes={axes}"
            )

        # keepdims is irrelevant here, a [N] and a [N, 1] tensor are both
        # a single column in the columns model used by orbital.
        columns = list(NumericVariablesGroup(data).values())
        self.set_output(sum(self._optimizer.fold_contiguous_sum(columns)))
