import ibis
import onnx

from orbitalml.translation.translator import Translator
from orbitalml.translation.variables import GraphVariables

BASIC_TABLE = ibis.memtable(
    {
        "X": [1.0, 2.0, 3.0],
        "W": [4.0, 5.0, 6.0],
        "B": [7.0, 8.0, 9.0],
    }
)
BASIC_MODEL = onnx.parser.parse_graph("""
    agraph 
    (float[N, 128] X, float[128,10] W, float[10] B) => (float[N] C) < float Z = {123.0}, float[1] Q = {456.0} >
    {
        T = MatMul <alpha: float = 0.5> (X, W)
        S = Add(T, B)
        C = Softmax(S)
    }
""")


class FakeTranslator(Translator):
    def process(self):
        pass


class TestGraphVariables:
    def test_creation(self):
        variables = GraphVariables(BASIC_TABLE, BASIC_MODEL)
        assert set(variables._variables.keys()) == {"X", "W", "B"}
        assert variables._consumed == set()
        assert variables._initializers_values == {"Q": [456.0], "Z": 123.0}


class TestTranslator:
    def test_creation(self):
        variables = GraphVariables(BASIC_TABLE, BASIC_MODEL)
        translator = FakeTranslator(None, BASIC_MODEL.node[0], variables, None)
        assert translator._attributes == {"alpha": 0.5}
