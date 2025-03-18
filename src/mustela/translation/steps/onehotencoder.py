import ibis

from ..translator import Translator


class OneHotEncoderTranslator(Translator):        
    # https://onnx.ai/onnx/operators/onnx_aionnxml_OneHotEncoder.html
    
    def process(self):
        cats = self._attributes.get("cats_strings")
        if cats is None:
            # We currently only support string values for categories
            raise ValueError("OneHotEncoder: attribute cats_strings not found")

        input_expr = self._variables.consume(self.inputs[0])
        result = {
            cat: (input_expr == cat).cast("float64")
            for cat in cats
        }

        self.set_output(result)