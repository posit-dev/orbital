from ..translator import Translator


class DivTranslator(Translator):
    def process(self):            
        first_operand = self._variables.consume(self.inputs[0])

        second_arg = self._variables.get_initializer_value(self.inputs[1])
        assert second_arg, (
            "Second input (divisor) must be a constant."
        )

        if isinstance(first_operand, dict):
            struct_fields = list(first_operand.keys())
            if len(second_arg) == 1:
                self.set_output({
                    field: (
                        self._optimizer.fold_operation(first_operand[field] / second_arg)
                    )
                    for i, field in enumerate(struct_fields)
                })
            else:
                assert len(second_arg) == len(first_operand)
                self.set_output({
                    field: (
                        self._optimizer.fold_operation(
                            first_operand[field] / second_arg[i]
                        )
                    )
                    for i, field in enumerate(struct_fields)
                })
        else:
            assert len(second_arg) == 1
            self.set_output(self._optimizer.fold_operation(
                first_operand / second_arg
            ))
