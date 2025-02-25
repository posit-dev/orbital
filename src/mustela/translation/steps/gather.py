from ..translator import Translator


class GatherTranslator(Translator):
    # https://onnx.ai/onnx/operators/onnx__Gather.html

    def process(self):
            expr = self._variables.consume(self.inputs[0])
            idx = self._variables.get_initializer_value(self.inputs[1])[0]
            if not isinstance(idx, int):
                raise ValueError("Gather: index must be an integer constant")
            
            if isinstance(expr, dict):
                keys = list(expr.keys())
                if idx < 0 or idx >= len(keys):
                    raise IndexError("Gather: index out of bounds")
                self.set_output(expr[keys[idx]])
            else:
                # Assume that if it's a single column then the index just points to the first value
                # which will be the column itself.
                if idx == 0:
                    self.set_output(expr)
                else:
                    raise NotImplementedError(
                        f"Gather: index {idx} not supported for non-dict expression of type {type(expr)}"
                    )
