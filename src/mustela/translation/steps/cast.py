"""Translators for Cast and CastLike operations"""

import onnx

from ..translator import Translator

ONNX_TYPES_TO_IBIS = {
    onnx.TensorProto.FLOAT: "float32",  # 1: FLOAT
    onnx.TensorProto.DOUBLE: "float64",  # 11: DOUBLE
    onnx.TensorProto.STRING: "string",  # 8: STRING
    onnx.TensorProto.INT64: "int64",  # 7: INT64
    onnx.TensorProto.BOOL: "bool",  # 9: BOOL
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
        to_type = self._attributes["to"]
        if to_type in ONNX_TYPES_TO_IBIS:
            target_type = ONNX_TYPES_TO_IBIS[to_type]
            if isinstance(expr, dict):
                casted = {
                    k: self._optimizer.fold_cast(expr[k].cast(target_type))
                    for k in expr
                }
                self.set_output(casted)
            else:
                self.set_output(self._optimizer.fold_cast(expr.cast(target_type)))
        else:
            raise NotImplementedError(f"Cast: type {to_type} not supported")


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
        if not isinstance(expr, dict):
            # TODO: Support single variables as well.
            #       This should be fairly straightforward to implement,
            #       but there hasn't been the need for it yet.
            raise NotImplementedError("CastLike currently only supports casting a group of columns.")

        # Assert that the second input is a single expression.
        if isinstance(like_expr, dict):
            raise NotImplementedError("CastLike currently only supports casting to a single column type, not a group.")

        assert hasattr(like_expr, "type"), (
            "CastLike: second input must have a 'type' attribute."
        )

        # Get the target type from the second input.
        target_type = like_expr.type()

        # Now cast each field in the dictionary to the target type.
        casted = {
            key: self._optimizer.fold_cast(expr[key].cast(target_type)) for key in expr
        }
        self.set_output(casted)
