from ..translator import Translator


class ReshapeTranslator(Translator):
    def process(self):
        first_operand = self._variables.consume(self.inputs[0])
        if isinstance(first_operand, dict):
            first_operand_len = len(first_operand)
        else:
            first_operand_len = 1

        shape = self._variables.get_initializer_value(self.inputs[1])
        if shape[0] != -1:
            # We don't support changing the numer of rows
            raise NotImplementedError("Reshape can't change the number of rows")

        if len(shape) == 1 and first_operand_len == 1:
            # We can reshape a single column to a single column
            # nothing has changed.
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