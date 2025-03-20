import ibis

from ..translator import Translator


class SoftmaxTranslator(Translator):
    # https://onnx.ai/onnx/operators/onnx__Softmax.html   

    def process(self):
        data = self._variables.consume(self.inputs[0])
        axis = self._attributes.get("axis", -1)
        
        if axis not in (-1, 1):
            raise ValueError("SoftmaxTranslator supports only axis=-1 or axis=1 for group of columns")
        
        if isinstance(data, dict):
            # Compute, for each column, the exponent
            exp_dict = {k: v.exp() for k, v in data.items()}
            
            # Sum all column exponents
            sum_exp = None
            for expr in exp_dict.values():
                sum_exp = expr if sum_exp is None else sum_exp + expr
            
            # Multi columns case: softmax = exp(column_exp) / (exponents_sum)
            softmax_result = {k: exp_dict[k] / sum_exp for k in data.keys()}
        else:
            # Single column case: softmax(x) = exp(x) / exp(x) = 1
            softmax_result = ibis.exp(data) / ibis.exp(data)
        
        self.set_output(softmax_result)