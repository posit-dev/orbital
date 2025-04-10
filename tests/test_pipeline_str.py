import onnx

from orbitalml import types
from orbitalml.ast import ParsedPipeline

BASIC_FEATURES = {
    "X": types.FloatColumnType(),
    "W": types.FloatColumnType(),
    "B": types.FloatColumnType(),
}
BASIC_MODEL = onnx.helper.make_model(
    onnx.parser.parse_graph("""
    agraph 
    (float[N, 128] X, float[128,10] W, float[10] B) => (float[N] C) < float Z = {123.0}, float[1] Q = {456.0} >
    {
        T = MatMul <alpha: float = 0.5> (X, W)
        S = Add(T, Z)
        C = Softmax(S)
    }
""")
)


class TestParsedPipelineRepr:
    def test_repr(self):
        parsed_pipeline = ParsedPipeline._from_onnx_model(BASIC_MODEL, BASIC_FEATURES)
        assert (
            str(parsed_pipeline)
            == """\
ParsedPipeline(
  features={
    X: FloatColumnType()
    W: FloatColumnType()
    B: FloatColumnType()
  },
  steps=[
    T=MatMul(
      inputs: X, W,
      attributes: 
        alpha=0.5
    )
    S=Add(
      inputs: T, Z=123.0,
      attributes: 
    )
    C=Softmax(
      inputs: S,
      attributes: 
    )
  ],
)
"""
        )
