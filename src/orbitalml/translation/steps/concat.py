"""Translator for Concat and FeatureVectorizer operations."""

import typing

import ibis

from ..translator import Translator
from ..variables import ValueVariablesGroup, VariablesGroup


class ConcatTranslator(Translator):
    """Concatenate multiple columns into a single group of columns.

    In tensor terms, this is meant to create a new tensor by concatenating
    the inputs along a given axis. In most cases, this is used to
    concatenate multiple features into a single one, thus its purpose
    is usually to create a column group from separate columns.

    This means that the most common use case is axis=1,
    which means concatenating over the columns (by virtue of
    column/rows in tensors being flipped over column groups),
    and thus only axis=1 case is supported.
    """

    def process(self) -> None:
        """Performs the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx__Concat.html

        # Currently only support concatenating over columns,
        # we can't concatenate rows.
        if self._attributes["axis"] not in (1, -1):
            # -1 means last axis, which for 2D entities is equal axis=1
            raise NotImplementedError(
                "Concat currently only supports concatenating over columns (axis=1 or -1)."
            )
        self.set_output(self._concatenate_columns(self))

    @classmethod
    def _concatenate_columns(cls, translator: Translator) -> VariablesGroup:
        """Implement actual operation of concatenating columns.

        This is used by both Concat and FeatureVectorizer translators,
        as they both need to concatenate columns.
        """
        result = ValueVariablesGroup()

        for col in translator.inputs:
            feature = translator._variables.consume(col)
            if isinstance(feature, dict):
                # When the feature is a dictionary,  it means that it was previously
                # concatenated with other features. In pure ONNX terms it would be
                # a tensor, so when we concatenate it we should just merge all the values
                # like we would do when concatenating two tensors.
                for key in feature:
                    varname = col + "." + key
                    result[varname] = feature[key]
            elif isinstance(feature, ibis.Expr):
                result[col] = feature
            else:
                raise ValueError(
                    f"Concat: expected a column group or a single column. Got {type(feature)}"
                )

        return result


class FeatureVectorizerTranslator(Translator):
    """Concatenate multiple columns into a single group of columns.

    This is similar to Concat, but it is a simplified version
    that always only acts on columns, and does not support
    concatenating over rows. While Concat can in theory
    support rows concatenation, even though orbitalml doesn't implement it.
    """

    def process(self) -> None:
        """Performs the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx_aionnxml_FeatureVectorizer.html

        # We can support this by doing the same as Concat,
        # in most cases it's sufficient
        ninputdimensions = typing.cast(list[int], self._attributes["inputdimensions"])

        if len(ninputdimensions) != len(self._inputs):
            raise ValueError(
                "Number of input dimensions should be equal to number of inputs."
            )

        # Validate that dimensions are actually correct,
        # as inputdimensions is meant to provide the number of columns of each variable
        for input_idx, colname in enumerate(self.inputs):
            dimensions = ninputdimensions[input_idx]
            feature = self._variables.peek_variable(colname)
            if isinstance(feature, dict):
                if len(feature) != dimensions:
                    raise ValueError(
                        f"Number of columns in input {colname} should be equal to the number of dimensions, got {len(feature)} != {dimensions}"
                    )
            else:
                if dimensions != 1:
                    raise ValueError(
                        f"When merging over individual columns, the dimension should be 1, got {dimensions} for {colname}"
                    )

        self.set_output(ConcatTranslator._concatenate_columns(self))
