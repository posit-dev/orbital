""""""

import ibis.expr.types

from ..translator import Translator


class ArrayFeatureExtractorTranslator(Translator):
    """Processes an ArrayFeatureExtractor node and updates the variables with the output expression.
    
    ArrayFeatureExtractor can be considered the opposit of :class:`ConactTranslator`, as
    in most cases it will be used to pick one or more features out of a group of column
    previously concatenated, or to pick a specific feature out of the result of an ArgMax operation.

    The provided indices always refer to the **last** axis of the input tensor.
    If the input is a 2D tensor, the last axis is the column axis. So an index
    of ``0`` would mean the first column. If the input is a 1D tensor instead the
    last axis is the row axis. So an index of ``0`` would mean the first row.

    This could be confusing because axis are inverted between tensors and mustela column groups.
    In the case of Tensors, axis=0 means row=0, while instead of mustela
    column groups (by virtue of being a group of columns), axis=0 means
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

        if not isinstance(data, dict):
            # TODO: Implement support for selecting rows from a 1D tensor
            raise NotImplementedError("ArrayFeatureExtractor only supports column groups as inputs")

        # This expects that dictionaries are sorted by insertion order
        # AND that all values of the dictionary are columns.
        data_keys = list(data.keys())
        data = list(data.values())

        if isinstance(indices, (list, tuple)):
            if data_keys is None:
                raise ValueError("ArrayFeatureExtractor expects a group of columns as input when receiving a list of indices")
            if len(indices) > len(data_keys):
                raise ValueError("Indices requested are more than the available numer of columns.")
            # Pick only the columns that are in the list of indicies.
            result = {data_keys[i]: data[i] for i in indices}
        elif isinstance(indices, ibis.expr.types.Column):
            # The indices that we need to pick are contained in
            # another column of the table.
            case_expr = ibis.case()
            for i, col in enumerate(data):
                case_expr = case_expr.when(indices == i, col)
            result = case_expr.else_(data[0]).end()
        else:
            raise ValueError(f"Index Type not supported: {type(indices)}: {indices}")

        self.set_output(result)
