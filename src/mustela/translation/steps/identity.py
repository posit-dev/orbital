from ..translator import Translator


class IdentityTranslator(Translator):
    # https://onnx.ai/onnx/operators/onnx__Identity.html

    def process(self):
        self.set_output(self._variables.consume(self._inputs[0]))
