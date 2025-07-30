"""Post-transformation functions for ONNX operators.

This module provides post-transformation functions that can be applied to
both individual numeric values and groups of numeric variables.

Example usage:
    # For a single numeric value
    score = ibis.literal(2.0)
    transformed_score = apply_post_transform(score, "LOGISTIC")
    # Result: Divide(1, Add(Exp(Negate(2.0)), 1)): 1 / Exp(Negate(2.0)) + 1

    # For a group of variables
    scores_group = NumericVariablesGroup({
        "class_0": ibis.literal(1.0),
        "class_1": ibis.literal(-0.5)
    })
    transformed_group = apply_post_transform(scores_group, "LOGISTIC")
    # Result: NumericVariablesGroup({
    #     "class_0": Divide(1, Add(Exp(Negate(1.0)), 1)),
    #     "class_1": Divide(1, Add(Exp(Negate(-0.5)), 1))
    # })
    # Mathematical values: class_0: 1.0 → 0.731059, class_1: -0.5 → 0.377541

    # SOFTMAX transformation (only for groups)
    softmax_group = apply_post_transform(scores_group, "SOFTMAX")
    # Result: NumericVariablesGroup with normalized probabilities that sum to 1.0
"""

import abc
import typing

import ibis

from .variables import NumericVariablesGroup, ValueVariablesGroup


def apply_post_transform(
    data: typing.Union[
        ibis.expr.types.NumericValue, NumericVariablesGroup, ValueVariablesGroup
    ],
    transform: str,
) -> typing.Union[ibis.expr.types.NumericValue, NumericVariablesGroup]:
    """Apply post-transformation to a single numeric value or a group of numeric variables.

    :param data: Either a single NumericValue, NumericVariablesGroup, or ValueVariablesGroup containing NumericValues
    :param transform: Name of the transformation to apply (e.g., "LOGISTIC", "NONE", "SOFTMAX")
    :return: The transformed data in the same format as the input
    """
    transform_class = TRANSFORM_CLASSES.get(transform)
    if not transform_class:
        raise NotImplementedError(f"Post transform '{transform}' is not implemented.")

    if isinstance(data, ValueVariablesGroup):
        # Convert ValueVariablesGroup to NumericVariablesGroup
        # this is mostly for convenience, it will raise TypeError if any value is not numeric
        data = NumericVariablesGroup(data)

    if transform_class is IdentityTransform:
        # Identity transform does not change the data
        # We can avoid applying it at all.
        return data

    transformer = transform_class()

    if isinstance(data, NumericVariablesGroup):
        # Apply transformation to the group of variables
        return transformer.transform_variables_group(data)
    elif isinstance(data, ibis.expr.types.NumericValue):
        # Apply transformation to single value
        return transformer.transform_numeric(data)
    else:
        raise TypeError(
            f"Expected NumericValue or NumericVariablesGroup, got {type(data)}"
        )


class PostTransform(abc.ABC):
    """Base class for post-transformation operations."""

    @abc.abstractmethod
    def transform_numeric(
        self, value: ibis.expr.types.NumericValue
    ) -> ibis.expr.types.NumericValue:
        """Transform a single numeric value.

        :param value: The numeric value to transform
        :return: The transformed numeric value
        """
        pass

    def transform_variables_group(
        self, group: NumericVariablesGroup
    ) -> NumericVariablesGroup:
        """Transform a group of numeric variables.

        Default implementation applies transform_numeric to each variable in the group.
        Subclasses can override this for transformations that need to consider all variables together.

        :param group: The group of numeric variables to transform
        :return: The transformed group of numeric variables
        """
        return NumericVariablesGroup(
            {name: self.transform_numeric(value) for name, value in group.items()}
        )


class LogisticTransform(PostTransform):
    """Logistic (sigmoid) transformation: 1 / (1 + exp(-x))."""

    def transform_numeric(
        self, value: ibis.expr.types.NumericValue
    ) -> ibis.expr.types.NumericValue:
        """Apply logistic transformation to a single value."""
        return 1 / (1 + (-value).exp())


class SoftmaxTransform(PostTransform):
    """Softmax transformation: exp(x_i) / sum(exp(x_j)) for all j."""

    def transform_numeric(
        self, value: ibis.expr.types.NumericValue
    ) -> ibis.expr.types.NumericValue:
        """Apply softmax to a single value: exp(x) / exp(x) = 1.0."""
        return ibis.literal(1.0)

    def transform_variables_group(
        self, group: NumericVariablesGroup
    ) -> NumericVariablesGroup:
        """Apply softmax transformation across all variables in the group.

        Uses numerical stability: subtract max before exponentiation.
        """
        # Use numerical stability: subtract max before exponentiation
        max_value = ibis.greatest(*group.values())

        # Compute exp(x_i - max) for each variable
        exp_dict = {name: (value - max_value).exp() for name, value in group.items()}

        # Sum all exponentials
        sum_exp = sum(exp_dict.values())

        # Compute softmax: exp(x_i - max) / sum(exp(x_j - max))
        return NumericVariablesGroup(
            {name: exp_val / sum_exp for name, exp_val in exp_dict.items()}
        )


class NormalizeTransform(PostTransform):
    """Normalization transformation: x / TOTAL where TOTAL is the sum of all variables."""

    def transform_numeric(
        self, value: ibis.expr.types.NumericValue
    ) -> ibis.expr.types.NumericValue:
        """Apply normalization to a single value."""
        return 1.0

    def transform_variables_group(
        self, group: NumericVariablesGroup
    ) -> NumericVariablesGroup:
        """Apply L1 normalization across all variables in the group."""
        sum_votes = sum(group.values())
        return NumericVariablesGroup(
            {name: value / sum_votes for name, value in group.items()}
        )


class IdentityTransform(PostTransform):
    """Identity transformation (no change).

    This primarily only acts as a guard
    against applying unnecessary transformations.
    It still provides a working implementation,
    but `apply_post_transform` will ignore it.
    """

    def transform_numeric(
        self, value: ibis.expr.types.NumericValue
    ) -> ibis.expr.types.NumericValue:
        """Return the value unchanged."""
        return value


# Mapping of transformation names to their corresponding classes
TRANSFORM_CLASSES: dict[str, type[PostTransform]] = {
    "LOGISTIC": LogisticTransform,
    "NONE": IdentityTransform,
    "SOFTMAX": SoftmaxTransform,
    # Make sure you prefix ORBITAL specific transforms with ORBITAL_
    # to avoid conflicts with ONNX transforms.
    "ORBITAL_NORMALIZE": NormalizeTransform,
}
