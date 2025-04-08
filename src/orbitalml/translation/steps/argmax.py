"""Defines the translation step for the ArgMax operation."""

import ibis

from ..translator import Translator


class ArgMaxTranslator(Translator):
    """Processes an ArgMax node and updates the variables with the output expression.

    Given the node to translate, the variables and constants available for
    the translation context, generates a query expression that processes
    the input variables and produces a new output variable that computes
    based on the ArgMax operation.

    The ArgMax implementation is currently limited to emitting a variable
    that represents the index of the column with the maximum value in a group
    of columns. It is not possible to compute the max of a set of rows
    thus axis must be 1 and keepdims must be 1.

    As it computes the maximum out of a set of columns, argmax expects
    a columns group as its input.

    The limitation is due to the fact that we can't mix variables with
    a different amount of rows and MAX(col) would end up producing a single row.
    This is usually ok, as ArgMax is primarily used to pick the values
    with the maximum value out of the features analyzed by the model,
    and thus it is only required to produce a new value for each entry
    on which to perform a prediction/classification (row).
    """

    def process(self) -> None:
        """Performs the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx__ArgMax.html
        data = self._variables.consume(self.inputs[0])
        axis = self._attributes.get("axis", 1)
        keepdims = self._attributes.get("keepdims", 1)
        select_last_index = self._attributes.get("select_last_index", 0)

        if not isinstance(data, dict):
            # if it's a single column, we can't really do much with it
            # as there aren't other comlumns to compare with.
            raise NotImplementedError(
                "ArgMaxTranslator can only be applied to a group of columns"
            )

        if axis != 1:
            # For axis=0 we would want to return the index of the row
            # with the maximum value, but we don't have a row identifier
            raise NotImplementedError("ArgMaxTranslator only supports axis=1")
        if keepdims != 1:
            raise NotImplementedError(
                "ArgMaxTranslator only supports retaining original dimensions"
            )

        keys = list(data.keys())

        # Generate a CASE THEN ELSE expression to find
        # which out of all the columns has the maximum value.
        case_expr = ibis.case()
        for idx, key in enumerate(keys):
            cond = None
            # Compare the current column with all other columns
            for j, other in enumerate(keys):
                if j == idx:
                    # Do not compare to yourself.
                    continue
                # When select_last_index is True
                # We use '>', otherwise '>=' so that we can pick the first occurrence.
                cmp_expr = (
                    data[key] > data[other]
                    if select_last_index
                    else data[key] >= data[other]
                )
                cond = cmp_expr if cond is None else cond & cmp_expr
            case_expr = case_expr.when(cond, idx)
        argmax_expr = case_expr.else_(0).end()

        self.set_output(argmax_expr)
