"""Implementation of the Reshape operator."""

from ..translator import Translator


class ReshapeTranslator(Translator):
    """Processes a Reshape node and updates the variables with the output expression.

    Reshape is currently a noop operation, it only supports cases where
    it doesn't have to change the data shape.
    That is generally not possible to support columns of different length in
    the same expressions/table so we can't really change the shape of a column
    as it implies changing its length.
    """

    def process(self) -> None:
        """Performs the translation and set the output variable."""

        # https://onnx.ai/onnx/operators/onnx__Reshape.html
        first_operand = self._variables.consume(self.inputs[0])
        if isinstance(first_operand, dict):
            first_operand_len = len(first_operand)
        else:
            first_operand_len = 1

        shape = self._variables.get_initializer_value(self.inputs[1])
        if not isinstance(shape, list) or not isinstance(shape[0], int):
            # Reshape explicitly requires ints.
            raise NotImplementedError("Reshape: requires integer values for the shape.")

        if shape[0] != -1:
            # We don't support changing the numer of rows
            raise NotImplementedError("Reshape can't change the number of rows")

        if len(shape) == 1 and first_operand_len == 1:
            # We can reshape a single column to a single column
            # nothing has changed.
            pass
        elif len(shape) == 2 and shape[1] == first_operand_len:
            # We can reshape a group of columns into the same
            # number of columns, nothing has changed.
            pass
        else:
            raise ValueError(f"Reshape shape={shape} not supported")

        # At this point we should have a single column containing the
        # result of the whole expression, so there should really be nothing to reshape.
        self.set_output(first_operand)
