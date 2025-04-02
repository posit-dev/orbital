"""Translators for Cast and CastLike operations"""

import typing

import ibis
import onnx

from ..translator import Translator
from ..variables import ValueVariablesGroup, VariablesGroup

ONNX_TYPES_TO_IBIS: dict[int, ibis.expr.datatypes.DataType] = {
    onnx.TensorProto.FLOAT: ibis.expr.datatypes.float32,  # 1: FLOAT
    onnx.TensorProto.DOUBLE: ibis.expr.datatypes.float64,  # 11: DOUBLE
    onnx.TensorProto.STRING: ibis.expr.datatypes.string,  # 8: STRING
    onnx.TensorProto.INT64: ibis.expr.datatypes.int64,  # 7: INT64
    onnx.TensorProto.BOOL: ibis.expr.datatypes.boolean,  # 9: BOOL
}


class CastTranslator(Translator):
    """Processes a Cast node and updates the variables with the output expression.

    Cast operation is used to convert a variable from one type to another one
    provided by the attribute `to`.
    """

    def process(self) -> None:
        """Performs the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx__Cast.html
        expr = self._variables.consume(self.inputs[0])
        to_type: int = typing.cast(int, self._attributes["to"])
        if to_type not in ONNX_TYPES_TO_IBIS:
            raise NotImplementedError(f"Cast: type {to_type} not supported")

        target_type = ONNX_TYPES_TO_IBIS[to_type]
        if isinstance(expr, VariablesGroup):
            casted = ValueVariablesGroup(
                {
                    k: self._optimizer.fold_cast(expr.as_value(k).cast(target_type))
                    for k in expr
                }
            )
            self.set_output(casted)
        elif isinstance(expr, ibis.Value):
            self.set_output(self._optimizer.fold_cast(expr.cast(target_type)))
        else:
            raise ValueError(
                f"Cast: expected a column group or a single column. Got {type(expr)}"
            )


class CastLikeTranslator(Translator):
    """Processes a CastLike node and updates the variables with the output expression.

    CastLike operation is used to convert a variable from one type to
    the same type of another variable, thus uniforming the two
    """

    def process(self) -> None:
        """Performs the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx__CastLike.html

        # Cast a variable to have the same type of another variable.
        # For the moment provide a very minimal implementation,
        # in most cases this is used to cast concatenated features to the same type
        # of another feature.
        expr = self._variables.consume(self.inputs[0])
        like_expr = self._variables.consume(self.inputs[1])

        # Assert that the first input is a dict (multiple concatenated columns).
        if not isinstance(expr, VariablesGroup):
            # TODO: Support single variables as well.
            #       This should be fairly straightforward to implement,
            #       but there hasn't been the need for it yet.
            raise NotImplementedError(
                "CastLike currently only supports casting a group of columns."
            )

        # Assert that the second input is a single expression.
        if isinstance(like_expr, VariablesGroup):
            raise NotImplementedError(
                "CastLike currently only supports casting to a single column type, not a group."
            )

        if not isinstance(like_expr, ibis.Value):
            raise ValueError(
                f"CastLike: expected a single column. Got {type(like_expr)}"
            )

        # Get the target type from the second input.
        target_type: ibis.DataType = like_expr.type()

        # Now cast each field in the dictionary to the target type.
        casted = ValueVariablesGroup(
            {
                key: self._optimizer.fold_cast(expr.as_value(key).cast(target_type))
                for key in expr
            }
        )
        self.set_output(casted)
