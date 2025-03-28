"""Implementation of the LabelEncoder operator."""

import ibis

from ..translator import Translator


class LabelEncoderTranslator(Translator):
    """Processes a LabelEncoder node and updates the variables with the output expression.

    LabelEncoder is used to map values from one variable to values of another one.
    It is usually meant to map numeric values to categories
    """

    def process(self) -> None:
        """Performs the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx_aionnxml_LabelEncoder.html

        input_values = self._variables.consume(self.inputs[0])

        # Automatically find attributes that start with "keys_", "values_", and "default_"
        mapping_keys = next(
            (
                attr_value
                for attr_name, attr_value in self._attributes.items()
                if attr_name.startswith("keys_")
            ),
            None,
        )
        mapping_values = next(
            (
                attr_value
                for attr_name, attr_value in self._attributes.items()
                if attr_name.startswith("values_")
            ),
            None,
        )
        default = next(
            (
                attr_value
                for attr_name, attr_value in self._attributes.items()
                if attr_name.startswith("default_")
            ),
            None,
        )
        if mapping_keys is None or mapping_values is None:
            raise ValueError("LabelEncoder: required mapping attributes not found.")
        if not isinstance(mapping_values, (tuple, list)) or not isinstance(
            mapping_keys, (tuple, list)
        ):
            raise ValueError("LabelEncoder: mapping must be a list of keys and values")

        if default is None:
            value_sample = mapping_values[0]
            if isinstance(value_sample, int):
                default = -1
            elif isinstance(value_sample, str):
                default = "_Unused"
            elif isinstance(value_sample, float):
                default = -0.0
            else:
                raise ValueError(
                    f"LabelEncoder: unsupported values attribute type: {mapping_values}"
                )

        case_expr = ibis.case()
        for k, v in zip(mapping_keys, mapping_values):
            case_expr = case_expr.when(input_values == k, v)
        case_expr = case_expr.else_(default).end()

        if not isinstance(case_expr, ibis.Value):
            raise NotImplementedError("Deferred case expression not supported")
        case_expr = self._optimizer.fold_case(case_expr)

        self.set_output(case_expr)
