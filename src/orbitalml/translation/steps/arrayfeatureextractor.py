""""""

import typing

import ibis.expr.types

from ..translator import Translator
from ..variables import ValueVariablesGroup, VariablesGroup


class ArrayFeatureExtractorTranslator(Translator):
    """Processes an ArrayFeatureExtractor node and updates the variables with the output expression.

    ArrayFeatureExtractor can be considered the opposit of :class:`ConactTranslator`, as
    in most cases it will be used to pick one or more features out of a group of column
    previously concatenated, or to pick a specific feature out of the result of an ArgMax operation.

    The provided indices always refer to the **last** axis of the input tensor.
    If the input is a 2D tensor, the last axis is the column axis. So an index
    of ``0`` would mean the first column. If the input is a 1D tensor instead the
    last axis is the row axis. So an index of ``0`` would mean the first row.

    This could be confusing because axis are inverted between tensors and orbitalml column groups.
    In the case of Tensors, index=0 means row=0, while instead in orbitalml
    column groups (by virtue of being a group of columns), index=0 means
    the first column.

    We have to consider that the indices we receive, in case of column groups,
    are actually column indices, not row indices as in case of a tensor,
    the last index would be the column index. In case of single columns,
    instead the index is the index of a row like it would be with a 1D tensor.
    """

    def process(self) -> None:
        """Performs the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx_aionnxml_ArrayFeatureExtractor.html

        data = self._variables.consume(self.inputs[0])
        indices = self._variables.consume(self.inputs[1])

        if isinstance(data, VariablesGroup):
            # We are selecting a set of columns out of a column group

            # This expects that dictionaries are sorted by insertion order
            # AND that all values of the dictionary are columns.
            data_keys: list[str] = list(data.keys())
            data_values: list[ibis.Expr] = list(data.values())

            if not isinstance(indices, (list, tuple)):
                raise ValueError(
                    "ArrayFeatureExtractor expects a list of indices as input."
                )

            indices = typing.cast(list[int], indices)
            if len(indices) > len(data_keys):
                raise ValueError(
                    "Indices requested are more than the available numer of columns."
                )

            # Pick only the columns that are in the list of indicies.
            result = ValueVariablesGroup(
                {data_keys[i]: data_values[i] for i in indices}
            )
        elif isinstance(data, (tuple, list)):
            # We are selecting values out of a list of values
            # This is usually used to select "classes" out of a list of
            # possible values based on the variables that represents those classes.
            if not isinstance(indices, ibis.expr.types.Column):
                raise ValueError(
                    "ArrayFeatureExtractor expects a column as indices when picking from a group of values."
                )

            case_expr = ibis.case()
            for i, col in enumerate(data):
                case_expr = case_expr.when(indices == i, col)
            result = case_expr.else_(ibis.null()).end()
        else:
            raise NotImplementedError(
                "ArrayFeatureExtractor only supports column groups or lists of constants as input."
            )

        self.set_output(result)
