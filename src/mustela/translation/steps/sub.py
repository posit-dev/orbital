from ..translator import Translator


class SubTranslator(Translator):
    def process(self):
        assert len(self._inputs) == 2, "The Sub node must have exactly 2 inputs."

        first_operand = self._variables.consume(self._inputs[0])
        second_operand = self._variables.get_initializer_value(self._inputs[1])
        assert second_operand is not None, (
            "The second input (initializer) is not present in initializers."
        )

        sub_values = second_operand
        if isinstance(first_operand, dict):
            struct_fields = list(first_operand.keys())
            assert len(sub_values) == len(struct_fields), (
                f"The number of values in the initializer ({len(sub_values)}) must match the number of fields ({len(struct_fields)}"
            )
            self._variables[self._output_name] = {
                field: (
                    self._optimizer.fold_operation(first_operand[field] - sub_values[i])
                )
                for i, field in enumerate(struct_fields)
            }
        else:
            assert len(sub_values) == 1, (
                "The second input (initializer) must contain exactly 1 value when the first operand is not concatenated."
            )
            self._variables[self._output_name] = self._optimizer.fold_operation(
                first_operand - sub_values[0]
            )