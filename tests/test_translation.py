import ibis
import onnx
import pytest

import orbital
from orbital import types
from orbital.ast import ParsedPipeline
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


class TestTranslate:
    def test_deep_network_sql_scaling(self):
        """SQL for a multi-layer network stays small and correct."""
        import time

        import duckdb
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(42)
        layer_sizes = [20, 64, 64, 1]
        feature_names = [f"f{i}" for i in range(layer_sizes[0])]

        nodes = [
            onnx.helper.make_node(
                "Concat", inputs=feature_names, outputs=["input"], axis=1
            )
        ]
        initializers = []
        layers = []
        current = "input"
        for idx in range(len(layer_sizes) - 1):
            n_in, n_out = layer_sizes[idx], layer_sizes[idx + 1]
            weights = rng.normal(scale=0.5, size=(n_in, n_out))
            bias = rng.normal(scale=0.5, size=(1, n_out))
            layers.append((weights, bias))
            # The initializers are read from the typed proto fields,
            # onnx.numpy_helper.from_array would store them as raw bytes
            # and they would be seen as empty constants.
            initializers.append(
                onnx.helper.make_tensor(
                    f"w{idx}", onnx.TensorProto.DOUBLE, [n_in, n_out], weights.flatten()
                )
            )
            initializers.append(
                onnx.helper.make_tensor(
                    f"b{idx}", onnx.TensorProto.DOUBLE, [1, n_out], bias.flatten()
                )
            )
            nodes.append(
                onnx.helper.make_node("MatMul", [current, f"w{idx}"], [f"matmul{idx}"])
            )
            nodes.append(
                onnx.helper.make_node("Add", [f"matmul{idx}", f"b{idx}"], [f"add{idx}"])
            )
            current = f"add{idx}"
            if idx < len(layer_sizes) - 2:
                nodes.append(onnx.helper.make_node("Relu", [current], [f"relu{idx}"]))
                current = f"relu{idx}"
        nodes.append(onnx.helper.make_node("Sigmoid", [current], ["probability"]))

        graph = onnx.helper.make_graph(
            nodes,
            "mlp",
            [
                onnx.helper.make_tensor_value_info(
                    name, onnx.TensorProto.DOUBLE, [None, 1]
                )
                for name in feature_names
            ],
            [
                onnx.helper.make_tensor_value_info(
                    "probability", onnx.TensorProto.DOUBLE, [None, 1]
                )
            ],
            initializer=initializers,
        )
        parsed = ParsedPipeline._from_onnx_model(
            onnx.helper.make_model(graph),
            {name: types.DoubleColumnType() for name in feature_names},
        )

        start = time.perf_counter()
        sql = orbital.export_sql("data", parsed, dialect="duckdb")
        generation_time = time.perf_counter() - start
        # ~600KB / ~5s when the layers are projected, 12.4MB / ~364s when inlined.
        assert len(sql) < 1024 * 1024
        # Guards the memo in _estimate_sql_size: losing it or scoping it per value
        # keeps the emitted SQL byte-identical while generation time explodes,
        # which nothing else in the suite would notice.
        assert generation_time < 120

        data = pd.DataFrame(
            rng.normal(size=(10, layer_sizes[0])), columns=feature_names
        )
        conn = duckdb.connect(":memory:")
        conn.register("data", data)
        sql_predictions = conn.sql(sql).df().iloc[:, 0].to_numpy()

        expected = data.to_numpy()
        for idx, (weights, bias) in enumerate(layers):
            expected = expected @ weights + bias
            if idx < len(layers) - 1:
                expected = np.maximum(expected, 0)
        expected = 1.0 / (1.0 + np.exp(-expected[:, 0]))
        np.testing.assert_allclose(sql_predictions, expected, atol=1e-8)

    def test_small_expressions_are_not_projected(self):
        """Expressions cheap enough to inline don't gain a column of their own."""
        graph = onnx.parser.parse_graph("""
            agraph (double[N,1] a, double[N,1] b, double[N,1] c, double[N,1] d) => (double[N,3] output)
            <double[4,3] weights = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2},
             double[3] bias = {0.1,0.2,0.3}>
            {
                merged = Concat <axis: int = 1> (a, b, c, d)
                multiplied = MatMul(merged, weights)
                biased = Add(multiplied, bias)
                output = Relu(biased)
            }
        """)
        parsed = ParsedPipeline._from_onnx_model(
            onnx.helper.make_model(graph),
            {name: types.DoubleColumnType() for name in "abcd"},
        )
        table = ibis.memtable({name: [1.0, 2.0] for name in "abcd"})

        # Omitting the projection exposes the temporary columns too.
        query = orbital.translate(table, parsed, orbital.ResultsProjection.omit())
        assert query.columns == (
            "a",
            "b",
            "c",
            "d",
            "output.out_0",
            "output.out_1",
            "output.out_2",
        )

    def test_variable_used_by_two_nodes_projects_once(self):
        """A variable shared by two nodes doesn't leak into the results."""
        graph = onnx.parser.parse_graph("""
            agraph (double[N,1] a, double[N,1] b, double[N,1] c, double[N,1] d) => (double[N,2] left, double[N,2] right)
            <double[4,3] weights = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2},
             double[3] bias = {0.1,0.2,0.3},
             double[3,2] out_weights = {0.5,0.6,0.7,0.8,0.9,1.0}>
            {
                merged = Concat <axis: int = 1> (a, b, c, d)
                multiplied = MatMul(merged, weights)
                biased = Add(multiplied, bias)
                activated = Relu(biased)
                shared = MatMul(activated, out_weights)
                left = Relu(shared)
                right = Sigmoid(shared)
            }
        """)
        parsed = ParsedPipeline._from_onnx_model(
            onnx.helper.make_model(graph),
            {name: types.DoubleColumnType() for name in "abcd"},
        )
        table = ibis.memtable({name: [1.0, 2.0] for name in "abcd"})

        # Only "shared" crosses the inline threshold: each of its 2 neurons sums
        # the 3 neurons of "activated" (136 nodes each, vs 42 for "activated" and
        # 19 for "biased"). So it is projected as columns before its consumers.
        with_temporaries = orbital.translate(
            table, parsed, orbital.ResultsProjection.omit()
        )
        projected = [c for c in with_temporaries.columns if c.startswith("prs_")]
        assert len(projected) == 2, (
            f"the 2 neurons of shared must be projected once: {with_temporaries.columns}"
        )

        # Projecting "shared" must not make it reappear as a pipeline result.
        assert orbital.translate(table, parsed).columns == (
            "left.out_0",
            "left.out_1",
            "right.out_0",
            "right.out_1",
        )
