from ..translator import Translator


class ConcatTranslator(Translator):
    # https://onnx.ai/onnx/operators/onnx__Concat.html

    def process(self):
        # Currently only support concatenating over columns,
        # we can't concatenate rows.
        assert self._attributes["axis"] in (1, -1)
        self._concatenate_columns(self)

    @classmethod
    def _concatenate_columns(cls, translator):
        result = {}
        for col in translator.inputs:
            feature = translator._variables.consume(col)
            if isinstance(feature, dict):
                # When the feature is a dictionary,  it means that it was previously
                # concatenated with other features. In pure ONNX terms it would be
                # a tensor, so when we concatenate it we should just merge all the values
                # like we would do when concatenating two tensors.
                for key in feature:
                    varname = col + "." + key
                    result[varname] = feature[key]
            else:
                result[col] = feature

        translator._variables[translator._output_name] = result


class FeatureVectorizerTranslator(Translator):
    # https://onnx.ai/onnx/operators/onnx_aionnxml_FeatureVectorizer.html

    def process(self):
        # We can support this by doing the same as Concat,
        # in most cases it's sufficient
        ninputdimensions = self._attributes["inputdimensions"]
        assert len(ninputdimensions) == len(self._inputs), (
            f"Only supported when concatenating, got {len(ninputdimensions)} over {len(self._inputs)} features"
        )
        assert set(ninputdimensions) == {1}, (
            f"Only supported when all input dimensions are 1, got {ninputdimensions}"
        )
        ConcatTranslator._concatenate_columns(self)