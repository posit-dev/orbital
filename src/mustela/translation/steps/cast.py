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
    def process(self):
        expr = self._variables.consume(self.inputs[0])
        to_type = self._attributes["to"]
        if to_type in ONNX_TYPES_TO_IBIS:
            target_type = ONNX_TYPES_TO_IBIS[to_type]
            if isinstance(expr, dict):
                casted = {
                    k: self._optimizer.fold_cast(expr[k].cast(target_type)) for k in expr
                }
                self.set_output(casted)
            else:
                self.set_output(self._optimizer.fold_cast(
                    expr.cast(target_type)
                ))
        else:
            raise NotImplementedError(f"Cast: type {to_type} not supported")


class CastLikeTranslator(Translator):
    def process(self):
        # Cast a variable to have the same type of another variable.
        # For the moment provide a very minimal implementation,
        # in most cases this is used to cast concatenated features to the same type
        # of another feature.
        expr = self._variables.consume(self.inputs[0])
        like_expr = self._variables.consume(self.inputs[1])

        # Assert that the first input is a dict (multiple concatenated columns).
        # TODO: Support single variables as well.
        assert isinstance(expr, dict), (
            "CastLike: first input must be a dict of expressions."
        )

        # Assert that the second input is a single expression.
        assert not isinstance(like_expr, dict), (
            "CastLike: second input must be a single expression."
        )
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
