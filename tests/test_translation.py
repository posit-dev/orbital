import ibis
import onnx
import pytest

from orbital.translation.options import TranslationOptions
from orbital.translation.translator import Translator
from orbital.translation.variables import GraphVariables

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
        translator = FakeTranslator(
            None, BASIC_MODEL.node[0], variables, None, TranslationOptions()
        )
        assert translator._attributes == {"alpha": 0.5}

    def test_preserve_materializes_expression_as_column(self):
        """Preserved variables become table columns that can be reused."""
        variables = GraphVariables(BASIC_TABLE, BASIC_MODEL)
        translator = FakeTranslator(
            BASIC_TABLE, BASIC_MODEL.node[0], variables, None, TranslationOptions()
        )

        (preserved,) = translator.preserve((BASIC_TABLE["X"] * 2).name("doubled"))

        # Reusing the returned expression refers to the projected column,
        # instead of repeating the expression that computed it.
        assert preserved.get_name() == "doubled"
        assert "doubled" in translator.mutated_table.columns
        backend = ibis.duckdb.connect()
        assert list(backend.execute(preserved)) == [2.0, 4.0, 6.0]

    def test_preserve_rejects_name_already_in_table(self):
        """Preserving a name already in the table is rejected."""
        variables = GraphVariables(BASIC_TABLE, BASIC_MODEL)
        translator = FakeTranslator(
            BASIC_TABLE, BASIC_MODEL.node[0], variables, None, TranslationOptions()
        )

        with pytest.raises(
            ValueError, match="Preserve variable already exists in the table: X"
        ):
            translator.preserve(BASIC_TABLE["X"])
