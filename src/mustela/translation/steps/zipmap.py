from ..translator import Translator


class ZipMapTranslator(Translator):        
    # https://onnx.ai/onnx/operators/onnx_aionnxml_ZipMap.html

    def process(self):
            expr = self._variables.consume(self.inputs[0])
            labels = self._attributes.get("classlabels_strings")
            if labels:
                if isinstance(expr, dict):
                    zipped = {}
                    keys = list(expr.keys())
                    for i, label in enumerate(labels):
                        if i < len(keys):
                            zipped[label] = expr[keys[i]]
                    self.set_output(zipped)
                else:
                    self.set_output({
                        label: expr for label in labels
                    })
            else:
                # int64 class labels,
                # for the moment assume our data is aldeary numeric
                self.set_output(expr)