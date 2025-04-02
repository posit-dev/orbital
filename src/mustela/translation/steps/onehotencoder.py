"""Implementation of the OneHotEncoder operator."""

import typing

import ibis

from ..translator import Translator
from ..variables import ValueVariablesGroup


class OneHotEncoderTranslator(Translator):
    """Processes a MatMul node and updates the variables with the output expression.

    Given a categorical variable, this class creates a new group of columns,
    with one column for each category. The values of the column are 1.0
    if the original column value is equal to the category, and 0.0 otherwise.

    It supports only strings for categories and emits floats as column
    values.
    """

    def process(self) -> None:
        """Performs the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx_aionnxml_OneHotEncoder.html
        cats = typing.cast(list[str], self._attributes.get("cats_strings"))
        if not isinstance(cats, list):
            # We currently only support string values for categories
            raise ValueError("OneHotEncoder: attribute cats_strings not found")

        input_expr = self._variables.consume(self.inputs[0])
        if not isinstance(input_expr, ibis.Value):
            raise ValueError("OneHotEncoder: input expression not found")

        casted_variables = [
            ibis.ifelse(input_expr == cat, 1, 0)
            .cast("float64")
            .name(self.variable_unique_short_alias("oh"))
            for cat in cats
        ]

        # OneHot encoded features are usually consumed multiple times
        # by subsequent operations, so preserving them makes sense.
        casted_variables = self.preserve(*casted_variables)
        self.set_output(
            ValueVariablesGroup(
                {cat: casted_variables[i] for i, cat in enumerate(cats)}
            )
        )
