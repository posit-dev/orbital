"""Implementation of the ZipMap operator."""

import typing

import ibis

from ..translator import Translator
from ..variables import ValueVariablesGroup, VariablesGroup


class ZipMapTranslator(Translator):
    """Processes a ZipMap node and updates the variables with the output expression.

    The ZipMap operator is used to map values from one variable to another set
    of values. It is usually meant to map numeric values to categories.

    If the input is a group of columns, all columns in the group
    will be remappped according to the class labels.
    """

    def process(self) -> None:
        """Performs the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx_aionnxml_ZipMap.html
        data = self._variables.consume(self.inputs[0])

        int_labels = typing.cast(
            list[int] | None, self._attributes.get("classlabels_int64s")
        )
        string_labels = typing.cast(
            list[str] | None, self._attributes.get("classlabels_strings")
        )
        if string_labels is not None:
            labels = string_labels
        elif int_labels is not None:
            labels = [str(i) for i in int_labels]
        else:
            raise ValueError("ZipMap: required mapping attributes not found.")

        if isinstance(data, VariablesGroup):
            if len(labels) != len(data):
                raise ValueError("ZipMap: The number of labels and columns must match.")
            result = ValueVariablesGroup(
                {label: value for label, value in zip(labels, data.values())}
            )
        elif isinstance(data, ibis.Expr):
            if len(labels) != 1:
                raise ValueError("ZipMap: The number of labels and columns must match.")
            result = ValueVariablesGroup({label: data for label in labels})
        else:
            raise ValueError(
                f"ZipMap: expected a column group or a single column. Got {type(data)}"
            )

        self.set_output(result)
