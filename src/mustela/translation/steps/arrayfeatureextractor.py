from ..translator import Translator


class ArrayFeatureExtractorTranslator(Translator):
    # https://onnx.ai/onnx/operators/onnx_aionnxml_ArrayFeatureExtractor.html

    def process(self):      
            # Given an array of features, grab only one of them
            # This probably is used to extract a single feature from a list of features
            # Previously made by Concat
            data = self._variables.consume(self.inputs[0])
            indices = self._variables.consume(self.inputs[1])
            if hasattr(indices, "__iter__") and not isinstance(indices, (str, bytes)):
                indices = list(indices)

            data_keys = None
            if isinstance(data, dict):
                # This expects that dictionaries are sorted by insertion order
                # AND that all values of the dictionary are featues with dim_value: 1
                # TODO: Implement a class for Concatenaed values
                #       that implements support based on dimensions
                data_keys = list(data.keys())
                data = list(data.values())

            if isinstance(indices, (list, tuple)):
                # We only work with dictionaries of faturename: feature
                # So when we are expected to output a list of features
                # we should output a dictionary of features as they are just sorted.
                result = {data_keys[i]: data[i] for i in indices}
            elif isinstance(indices, int):
                result = data[indices]
            else:
                raise ValueError(
                    f"Index Type not supported: {type(indices)}: {indices}"
                )

            self.set_output(result)