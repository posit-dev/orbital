from ..translator import Translator


class ReshapeTranslator(Translator):
    def process(self):
        first_operand = self._variables.consume(self.inputs[0])
        if isinstance(first_operand, dict):
            first_operand_len = len(first_operand)
        else:
            first_operand_len = 1

        shape = self._variables.get_initializer_value(self.inputs[1])
        assert shape[1] == first_operand_len and shape[0] == -1, (
            "Reshaping is only supported when it doesn't change the shape"
        )

        # At this point we should have a single column containing the
        # result of the whole expression, so there should really be nothing to reshape.
        self.set_output(first_operand)