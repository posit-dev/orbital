"""Implementation of the LinearClassifier operator."""

import typing

import ibis

from ..translator import Translator
from ..variables import NumericVariablesGroup, ValueVariablesGroup, VariablesGroup


class LinearClassifierTranslator(Translator):
    """Processes a LinearClassifier node and updates variables with the classification results.

    The LinearClassifier operator computes classification outputs as:
    Scores = X * coefficients + intercepts

    For more complex pipelines the LinearClassifier operator is not always used,
    usually a combination of Mul and Add operations is used.
    """

    def process(self) -> None:
        """Performs the translation and sets the output variables Y (predictions) and Z (scores)."""
        # https://onnx.ai/onnx/operators/onnx_aionnxml_LinearClassifier.html
        coefficients = typing.cast(list[float], self._attributes["coefficients"])
        intercepts = typing.cast(list[float], self._attributes.get("intercepts", []))
        multi_class = typing.cast(int, self._attributes.get("multi_class", 0))
        post_transform = typing.cast(
            str, self._attributes.get("post_transform", "NONE")
        )

        if multi_class != 0:
            raise NotImplementedError("Multi-class classification is not implemented.")

        classlabels: list[str] | list[int] | None = typing.cast(
            list[int] | None, self._attributes.get("classlabels_ints")
        ) or typing.cast(list[str] | None, self._attributes.get("classlabels_strings"))

        if classlabels is None:
            raise ValueError(
                "LinearClassifier: classlabels_ints or classlabels_strings must be defined."
            )

        if len(self._inputs) != 1:
            raise ValueError("LinearClassifier node must have exactly 1 input.")

        input_operand = self._variables.consume(self._inputs[0])

        # Standardize input_operand to a columns group,
        # so that we can reuse a single implementation.
        if not isinstance(input_operand, VariablesGroup):
            input_operand = ValueVariablesGroup({"feature": input_operand})

        num_features = len(input_operand)
        num_classes = len(classlabels)

        if len(coefficients) != num_classes * num_features:
            raise ValueError(
                "Coefficients length must equal number of classes × number of input fields."
            )

        fieldsgroup = NumericVariablesGroup(input_operand)
        fields = list(fieldsgroup.values())
        scores = []

        for class_idx in range(num_classes):
            start = class_idx * num_features
            end = start + num_features
            coef_slice = coefficients[start:end]
            intercept = intercepts[class_idx] if intercepts else 0.0

            score = ibis.literal(intercept)
            for val, coef in zip(fields, coef_slice):
                score += val * coef

            score = self._apply_post_transform(score, post_transform)
            scores.append(self._optimizer.fold_operation(score))

        scores_struct = ValueVariablesGroup(
            {str(label): score for label, score in zip(classlabels, scores)}
        )

        max_score = ibis.greatest(*scores_struct.values())
        predictions = ibis.case()
        for label, score in scores_struct.items():
            predictions = predictions.when(score == max_score, label)
        predictions = predictions.end()

        self.set_output(predictions, index=0)
        self.set_output(scores_struct, index=1)

    @classmethod
    def _apply_post_transform(
        cls, score: ibis.expr.types.NumericValue, transform: str
    ) -> ibis.expr.types.NumericValue:
        # TODO: Move to a dedicated set of post-transform
        #       functions together with SOFTMAX
        if transform == "LOGISTIC":
            return 1 / (1 + (-score).exp())
        elif transform == "NONE":
            return score
        else:
            # TODO: apply more post_transform here if needed
            raise NotImplementedError(
                f"Post transform '{transform}' is not implemented."
            )
