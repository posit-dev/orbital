"""Implementation of the Identity operator."""

from ..translator import Translator


class IdentityTranslator(Translator):
    """Processes an Identity node and updates the variables with the output expression.

    The identity node is a no-op, it simply passes the input to the output,
    it is meant to copy the input into the output, but as there could be
    multiple references to the same expression, it doesn't actually need
    to perform a copy.
    """

    def process(self) -> None:
        """Performs the translation and set the output variable."""
        # https://onnx.ai/onnx/operators/onnx__Identity.html

        self.set_output(self._variables.consume(self._inputs[0]))
