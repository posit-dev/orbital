from ..translator import Translator


class AddTranslator(Translator):
    def process(self):            
        # TODO: Fix by copying Sub design
        intercept = self._variables.get_initializer_value(self.inputs[1])[0]
        first_operand = self._variables.consume(self.inputs[0])

        if isinstance(first_operand, dict):
            struct_fields = list(first_operand.keys())
            self.set_output({
                field: (self._optimizer.fold_operation(first_operand[field] + intercept))
                for i, field in enumerate(struct_fields)
            })
        else:
            self.set_output(self._optimizer.fold_operation(
                first_operand + intercept
            ))
