import ibis

from ..translator import Translator


class WhereTranslator(Translator):
    # https://onnx.ai/onnx/operators/onnx__Where.html

    def process(self):
        condition_expr = self._variables.consume(self.inputs[0])
        true_expr = self._variables.consume(self.inputs[1])
        false_expr = self._variables.consume(self.inputs[2])
        assert isinstance(false_expr, dict), false_expr

        result = {}
        for col_name, false_val in false_expr.items():
            result[col_name] = self._optimizer.fold_case(
                ibis.case().when(condition_expr, true_expr).else_(false_val).end()
            )
        self.set_output(result)
