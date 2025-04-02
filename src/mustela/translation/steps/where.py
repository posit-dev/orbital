"""Implementation of the Where operator."""

import itertools

import ibis

from ..translator import Translator
from ..variables import ValueVariablesGroup, VariablesGroup


class WhereTranslator(Translator):
    """Processes a Where node and updates the variables with the output expression.

    The where operation is expected to return ether its first or second input
    depending on a condition variable. When the variable is true, the first
    input is returned, otherwise the second input is returned.

    The condition variable will usually be a column computed through an expression
    that represents a boolean predicate.

    The first and second inputs can be either a single column or a group of columns.
    If any of the two is a group of columns, a new group of column is produced
    as the result.
    If both are single columns, the result is a single column.
    """

    def process(self) -> None:
        """Performs the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx__Where.html
        condition_expr = self._variables.consume(self.inputs[0])
        true_expr = self._variables.consume(self.inputs[1])
        false_expr = self._variables.consume(self.inputs[2])

        if isinstance(condition_expr, VariablesGroup):
            raise NotImplementedError(
                "Where: The condition expression can't be a group of columns. Must be a single column."
            )

        if isinstance(true_expr, VariablesGroup) and isinstance(
            false_expr, VariablesGroup
        ):
            true_values = list(true_expr.values())
            false_values = list(false_expr.values())
            if len(true_values) != len(false_values):
                raise ValueError(
                    "Where: The number of values in the true and false expressions must match."
                )
            result = ValueVariablesGroup()
            for true_val, false_val, idx in zip(
                true_values, false_values, itertools.count()
            ):
                result[f"c{idx}"] = self._optimizer.fold_case(
                    ibis.case().when(condition_expr, true_val).else_(false_val).end()
                )
        elif isinstance(true_expr, VariablesGroup) and not isinstance(
            false_expr, VariablesGroup
        ):
            result = ValueVariablesGroup()
            for idx, true_val in enumerate(true_expr.values()):
                result[f"c{idx}"] = self._optimizer.fold_case(
                    ibis.case().when(condition_expr, true_val).else_(false_expr).end()
                )
        elif not isinstance(true_expr, VariablesGroup) and isinstance(
            false_expr, VariablesGroup
        ):
            result = ValueVariablesGroup()
            for idx, false_val in enumerate(false_expr.values()):
                result[f"c{idx}"] = self._optimizer.fold_case(
                    ibis.case().when(condition_expr, true_expr).else_(false_val).end()
                )
        else:
            result = self._optimizer.fold_case(
                ibis.case().when(condition_expr, true_expr).else_(false_expr).end()
            )

        self.set_output(result)
