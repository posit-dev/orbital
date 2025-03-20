from ..translator import Translator


class AddTranslator(Translator):
    def process(self):
        assert len(self._inputs) == 2, "The Add step must have exactly 2 inputs."

        first_operand = self._variables.consume(self._inputs[0])
        second_operand = self._variables.get_initializer_value(self._inputs[1])
        assert second_operand is not None, (
            "The second input (initializer) is not present in initializers."
        )

        add_values = second_operand
        if isinstance(first_operand, dict):
            struct_fields = list(first_operand.keys())
            assert len(add_values) == len(struct_fields), (
                f"The number of values in the initializer ({len(add_values)}) must match the number of fields ({len(struct_fields)}"
            )
            self._variables[self._output_name] = {
                field: (
                    self._optimizer.fold_operation(first_operand[field] + add_values[i])
                )
                for i, field in enumerate(struct_fields)
            }
        else:
            assert len(add_values) == 1, (
                "The second input (initializer) must contain exactly 1 value when the first operand is not concatenated."
            )
            self._variables[self._output_name] = self._optimizer.fold_operation(
                first_operand - add_values[0]
            )

    def _process(self):            
        # TODO: Fix by copying Sub design
        intercept = self._variables.get_initializer_value(self.inputs[1])[0]
        first_operand = self._variables.consume(self.inputs[0])

        if isinstance(first_operand, dict):
            struct_fields = list(first_operand.keys())
            self.set_output({
                field: first_operand[field] + intercept
                for i, field in enumerate(struct_fields)
            })
        else:
            self.set_output(self._optimizer.fold_operation(
                first_operand + intercept
            ))
