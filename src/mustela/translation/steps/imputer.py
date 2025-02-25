import ibis

from ..translator import Translator


class ImputerTranslator(Translator):        
    # https://onnx.ai/onnx/operators/onnx_aionnxml_Imputer.html

    def process(self):
        imputed_values = self._attributes["imputed_value_floats"]
        expr = self._variables.consume(self.inputs[0])
        if isinstance(expr, dict):
            keys = list(expr.keys())
            if len(keys) != len(imputed_values):
                raise ValueError(
                    "Imputer: number of imputed values does not match number of columns"
                )
            new_expr = {}
            for i, key in enumerate(keys):
                new_expr[key] = ibis.coalesce(expr[key], imputed_values[i])
            self.set_output(new_expr)
        else:
            self.set_output(ibis.coalesce(
                expr, imputed_values[0]
            ))
