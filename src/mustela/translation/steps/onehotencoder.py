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

        casted_variables = [self._optimizer.fold_cast((input_expr == cat).cast("float64")).name(self.variable_unique_short_alias("onehot")) for cat in cats]
        
        # OneHot encoded features are usually consumed multiple times 
        # by subsequent operations, so preserving them makes sense.
        casted_variables = self.preserve(*casted_variables)
        self.set_output({
            cat: casted_variables[i]
            for i, cat in enumerate(cats)
        })
