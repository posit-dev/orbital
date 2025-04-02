"""Implementation of the LinearRegression operator."""

import typing

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, ValueVariablesGroup, VariablesGroup


class LinearRegressorTranslator(Translator):
    """Processes a LinearRegression node and updates variables with the predicted expression.

    The LinearRegression operator computes predictions as:
    Y = X * coefficients + intercept

    For more complex pipelines the LinearRegression operator is not always used,
    usually a combination of Mul and Add operations is used.
    """

    def process(self) -> None:
        """Performs the translation and sets the output variable."""
        # https://onnx.ai/onnx/operators/onnx_aionnxml_LinearRegressor.html
        coefficients = typing.cast(list[float], self._attributes["coefficients"])
        intercepts = typing.cast(list[float], self._attributes.get("intercepts", [0.0]))
        targets = typing.cast(int, self._attributes.get("targets", 1))

        post_transform = self._attributes.get("post_transform", "NONE")
        if post_transform != "NONE":
            raise NotImplementedError("Post transform is not implemented.")

        if len(intercepts) not in [0, targets]:
            raise ValueError(
                "LinearRegressor: intercepts length must match targets or be empty."
            )

        if len(self._inputs) != 1:
            raise ValueError("LinearRegressor node must have exactly 1 input.")

        input_operand = self._variables.consume(self._inputs[0])

        if isinstance(input_operand, VariablesGroup):
            input_operand = NumericVariablesGroup(input_operand)
            num_features = len(input_operand)

            if len(coefficients) != targets * num_features:
                raise ValueError(
                    "Coefficients length must equal targets number of input fields."
                )

            results = {}
            fields = list(input_operand.values())

            for target_idx in range(targets):
                start = target_idx * num_features
                end = start + num_features
                coef_slice = coefficients[start:end]

                intercept = intercepts[target_idx] if intercepts else 0.0

                prediction = ibis.literal(intercept)
                for val, coef in zip(fields, coef_slice):
                    prediction += val * coef

                # TODO: apply post_transform here if needed

                results[f"target_{target_idx}"] = self._optimizer.fold_operation(
                    prediction
                )

            self.set_output(ValueVariablesGroup(results))

        else:
            input_operand = typing.cast(ibis.expr.types.NumericValue, input_operand)

            if targets != 1 or len(coefficients) != 1:
                raise ValueError(
                    "Single column input expects exactly one target and one coefficient."
                )

            intercept = intercepts[0] if intercepts else 0.0
            prediction = (input_operand * coefficients[0]) + intercept

            # TODO: apply post_transform here if needed

            self.set_output(self._optimizer.fold_operation(prediction))
