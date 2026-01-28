"""Test individual pipeline steps/translators."""

import onnx
import ibis
import pytest

from orbital.translate import TRANSLATORS
from orbital.translation.steps.softmax import SoftmaxTranslator
from orbital.translation.steps.imputer import ImputerTranslator
from orbital.translation.steps.argmax import ArgMaxTranslator
from orbital.translation.steps.add import AddTranslator
from orbital.translation.steps.sub import SubTranslator
from orbital.translation.steps.mul import MulTranslator
from orbital.translation.steps.div import DivTranslator
from orbital.translation.steps.identity import IdentityTranslator
from orbital.translation.steps.reshape import ReshapeTranslator
from orbital.translation.variables import (
    GraphVariables,
    NumericVariablesGroup,
    ValueVariablesGroup,
)
from orbital.translation.optimizer import Optimizer
from orbital.translation.options import TranslationOptions


class TestStepCoverage:
    """Verify that all registered steps have corresponding test classes."""

    @pytest.mark.xfail(reason="Not all steps have tests yet, see #11 sub-issues")
    def test_all_registered_steps_have_tests(self):
        """Every step registered in TRANSLATORS must have a test class.

        Test classes must follow the naming convention Test{OperationName}Translator.
        This test ensures we don't forget to add tests when implementing new steps.
        """
        import sys

        module = sys.modules[__name__]
        existing_test_classes = {
            name
            for name in dir(module)
            if name.startswith("Test") and isinstance(getattr(module, name), type)
        }

        missing_tests = []
        for operation in sorted(TRANSLATORS.keys()):
            expected_test_class = f"Test{operation}Translator"
            if expected_test_class not in existing_test_classes:
                missing_tests.append((operation, expected_test_class))

        if missing_tests:
            missing_list = "\n".join(
                f"  - {op}: {test_class}" for op, test_class in missing_tests
            )
            pytest.fail(
                f"The following {len(missing_tests)} steps are missing test classes:\n"
                f"{missing_list}\n\n"
                f"Add a test class for each step to test_pipeline_steps.py"
            )


class TestSoftmaxTranslator:
    optimizer = Optimizer(enabled=False)

    def test_softmax_translator_single_input(self):
        """Test SoftmaxTranslator with a single numeric input."""
        table = ibis.memtable({"input": [2.0, 3.0, 4.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = Softmax(input)
            }
        """)

        variables = GraphVariables(table, model)

        translator = SoftmaxTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        # For single input, softmax should return 1.0
        backend = ibis.duckdb.connect()
        computed_value = backend.execute(result)
        assert computed_value == 1.0

    def test_softmax_translator_group_input(self):
        """Test SoftmaxTranslator with a group of numeric inputs."""
        multi_table = ibis.memtable(
            {
                "class_0": [1.0, 2.0, 3.0],
                "class_1": [0.5, 1.5, 2.5],
                "class_2": [2.0, 3.0, 4.0],
            }
        )

        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = Softmax(input)
            }
        """)

        # Use dummy table for GraphVariables since we override the input
        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)

        variables["input"] = NumericVariablesGroup(
            {
                "class_0": multi_table["class_0"],
                "class_1": multi_table["class_1"],
                "class_2": multi_table["class_2"],
            }
        )

        translator = SoftmaxTranslator(
            multi_table,
            model.node[0],
            variables,
            self.optimizer,
            TranslationOptions(),
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        # Should return a NumericVariablesGroup
        assert isinstance(result, NumericVariablesGroup)
        assert len(result) == 3
        assert "class_0" in result
        assert "class_1" in result
        assert "class_2" in result

        # Test that softmax values sum to 1.0 for each row
        backend = ibis.duckdb.connect()

        # backend.execute() returns a pandas Series, so we take the first element
        values = [
            backend.execute(result[class_name])[0]
            for class_name in ["class_0", "class_1", "class_2"]
        ]

        # Verify they sum to approximately 1.0
        total_sum = sum(values)
        assert abs(total_sum - 1.0) < 1e-10, (
            f"Softmax values should sum to 1.0, got {total_sum}"
        )

    def test_softmax_translator_invalid_axis(self):
        """Test that SoftmaxTranslator raises error for unsupported axis."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = Softmax <axis: int = 0> (input)
            }
        """)

        variables = GraphVariables(table, model)

        translator = SoftmaxTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            ValueError, match="SoftmaxTranslator supports only axis=-1 or axis=1"
        ):
            translator.process()

    def test_softmax_translator_invalid_input_type(self):
        """Test that SoftmaxTranslator raises error for invalid input type."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = Softmax(input)
            }
        """)

        variables = GraphVariables(table, model)

        # Intentionally set invalid input type to test error handling
        variables["input"] = "invalid_string_input"  # type: ignore[assignment]

        translator = SoftmaxTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            ValueError, match="Softmax: The first operand must be a numeric column"
        ):
            translator.process()

    def test_softmax_uses_apply_post_transform(self):
        """Test that SoftmaxTranslator uses the apply_post_transform function."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = Softmax(input)
            }
        """)

        variables = GraphVariables(table, model)

        variables["input"] = NumericVariablesGroup(
            {
                "class_0": ibis.literal(1.0),
                "class_1": ibis.literal(2.0),
            }
        )

        translator = SoftmaxTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, NumericVariablesGroup)


class TestImputerTranslator:
    optimizer = Optimizer(enabled=False)

    def test_imputer_single_column(self):
        """Test ImputerTranslator with a single column input."""
        table = ibis.memtable({"input": [1.0, None, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = ai.onnx.ml.Imputer <imputed_value_floats: floats = [2.0]> (input)
            }
        """)

        variables = GraphVariables(table, model)
        translator = ImputerTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        computed = backend.execute(result)
        assert list(computed) == [1.0, 2.0, 3.0]

    def test_imputer_group_columns(self):
        """Test ImputerTranslator with a group of columns."""
        table = ibis.memtable(
            {
                "col_a": [1.0, None, 3.0],
                "col_b": [None, 5.0, 6.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = ai.onnx.ml.Imputer <imputed_value_floats: floats = [10.0, 20.0]> (input)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = ValueVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = ImputerTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["col_a"])) == [1.0, 10.0, 3.0]
        assert list(backend.execute(result["col_b"])) == [20.0, 5.0, 6.0]

    def test_imputer_invalid_imputed_value_type(self):
        """Test ImputerTranslator raises error for non-list imputed values."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = ai.onnx.ml.Imputer <imputed_value_floats: floats = [2.0]> (input)
            }
        """)

        variables = GraphVariables(table, model)

        # Override attributes to test validation
        node = model.node[0]
        translator = ImputerTranslator(
            table, node, variables, self.optimizer, TranslationOptions()
        )
        translator._attributes["imputed_value_floats"] = 2.0  # Invalid: not a list

        with pytest.raises(ValueError, match="imputed_value must be a list or tuple"):
            translator.process()

    def test_imputer_mismatched_column_count(self):
        """Test ImputerTranslator raises error when column count doesn't match."""
        table = ibis.memtable(
            {
                "col_a": [1.0, None, 3.0],
                "col_b": [None, 5.0, 6.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = ai.onnx.ml.Imputer <imputed_value_floats: floats = [10.0]> (input)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = ValueVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = ImputerTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="number of imputed values does not match"):
            translator.process()


class TestArgMaxTranslator:
    optimizer = Optimizer(enabled=False)

    def test_argmax_group_columns(self):
        """Test ArgMaxTranslator with a group of columns."""
        table = ibis.memtable(
            {
                "class_0": [1.0, 5.0, 2.0],
                "class_1": [3.0, 2.0, 8.0],
                "class_2": [2.0, 1.0, 3.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] data) => (int64[N] output) {
                output = ArgMax <axis: int = 1, keepdims: int = 1> (data)
            }
        """)

        variables = GraphVariables(ibis.memtable({"data": [1.0]}), model)
        variables["data"] = NumericVariablesGroup(
            {
                "class_0": table["class_0"],
                "class_1": table["class_1"],
                "class_2": table["class_2"],
            }
        )

        translator = ArgMaxTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        computed = list(backend.execute(result))
        # Row 0: max is class_1 (3.0) -> index 1
        # Row 1: max is class_0 (5.0) -> index 0
        # Row 2: max is class_1 (8.0) -> index 1
        assert computed == [1, 0, 1]

    def test_argmax_single_column(self):
        """Test ArgMaxTranslator raises error for single column input."""
        table = ibis.memtable({"data": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] data) => (int64[N] output) {
                output = ArgMax <axis: int = 1> (data)
            }
        """)

        variables = GraphVariables(table, model)

        translator = ArgMaxTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(NotImplementedError, match="can only be applied to a group"):
            translator.process()

    def test_argmax_unsupported_axis(self):
        """Test ArgMaxTranslator raises error for axis != 1."""
        table = ibis.memtable({"data": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] data) => (int64[N] output) {
                output = ArgMax <axis: int = 0, keepdims: int = 1> (data)
            }
        """)

        variables = GraphVariables(ibis.memtable({"data": [1.0]}), model)
        variables["data"] = NumericVariablesGroup(
            {
                "class_0": table["data"],
                "class_1": table["data"],
            }
        )

        translator = ArgMaxTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(NotImplementedError, match="only supports axis=1"):
            translator.process()

    def test_argmax_unsupported_keepdims(self):
        """Test ArgMaxTranslator raises error for keepdims != 1."""
        table = ibis.memtable({"data": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] data) => (int64[N] output) {
                output = ArgMax <axis: int = 1, keepdims: int = 0> (data)
            }
        """)

        variables = GraphVariables(ibis.memtable({"data": [1.0]}), model)
        variables["data"] = NumericVariablesGroup(
            {
                "class_0": table["data"],
                "class_1": table["data"],
            }
        )

        translator = ArgMaxTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            NotImplementedError, match="only supports retaining original"
        ):
            translator.process()

    def test_argmax_empty_group(self):
        """Test ArgMaxTranslator raises error for empty group."""
        table = ibis.memtable({"data": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] data) => (int64[N] output) {
                output = ArgMax <axis: int = 1, keepdims: int = 1> (data)
            }
        """)

        variables = GraphVariables(ibis.memtable({"data": [1.0]}), model)
        variables["data"] = NumericVariablesGroup({})

        translator = ArgMaxTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="requires at least one column"):
            translator.process()

    def test_argmax_single_key_in_group(self):
        """Test ArgMaxTranslator with single key returns that value."""
        table = ibis.memtable({"class_0": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] data) => (int64[N] output) {
                output = ArgMax <axis: int = 1, keepdims: int = 1> (data)
            }
        """)

        variables = GraphVariables(ibis.memtable({"data": [1.0]}), model)
        variables["data"] = NumericVariablesGroup(
            {
                "class_0": table["class_0"],
            }
        )

        translator = ArgMaxTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables

    def test_argmax_select_last_index(self):
        """Test ArgMaxTranslator with select_last_index=1."""
        table = ibis.memtable(
            {
                "class_0": [3.0, 3.0],
                "class_1": [3.0, 3.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] data) => (int64[N] output) {
                output = ArgMax <axis: int = 1, keepdims: int = 1, select_last_index: int = 1> (data)
            }
        """)

        variables = GraphVariables(ibis.memtable({"data": [1.0]}), model)
        variables["data"] = NumericVariablesGroup(
            {
                "class_0": table["class_0"],
                "class_1": table["class_1"],
            }
        )

        translator = ArgMaxTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        # Just verify the translator processes without error and exercises the code path
        assert "output" in variables


class TestAddTranslator:
    optimizer = Optimizer(enabled=False)

    def test_add_single_column(self):
        """Test AddTranslator with a single column input."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[1] add_value = {5.0}>
            {
                output = Add(input, add_value)
            }
        """)

        variables = GraphVariables(table, model)
        translator = AddTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        computed = list(backend.execute(result))
        assert computed == [6.0, 7.0, 8.0]

    def test_add_group_columns(self):
        """Test AddTranslator with a group of columns."""
        table = ibis.memtable(
            {
                "col_a": [1.0, 2.0, 3.0],
                "col_b": [10.0, 20.0, 30.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[2] add_values = {5.0, 100.0}>
            {
                output = Add(input, add_values)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = AddTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["col_a"])) == [6.0, 7.0, 8.0]
        assert list(backend.execute(result["col_b"])) == [110.0, 120.0, 130.0]

    def test_add_invalid_non_numeric(self):
        """Test AddTranslator raises error for non-numeric operand."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[1] add_value = {5.0}>
            {
                output = Add(input, add_value)
            }
        """)

        variables = GraphVariables(table, model)
        variables["input"] = "not_a_numeric_value"  # type: ignore[assignment]

        translator = AddTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="first operand must be a numeric value"):
            translator.process()

    def test_add_mismatched_column_count(self):
        """Test AddTranslator raises error when column count doesn't match."""
        table = ibis.memtable(
            {
                "col_a": [1.0, 2.0, 3.0],
                "col_b": [10.0, 20.0, 30.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[1] add_values = {5.0}>
            {
                output = Add(input, add_values)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = AddTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="same number of values"):
            translator.process()

    def test_add_single_column_requires_single_value(self):
        """Test AddTranslator raises error when single column given multiple values."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[2] add_values = {5.0, 10.0}>
            {
                output = Add(input, add_values)
            }
        """)

        variables = GraphVariables(table, model)

        translator = AddTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="must contain exactly 1 value"):
            translator.process()

    def test_add_second_operand_not_constant(self):
        """Test AddTranslator raises error when second operand is not a constant."""
        table = ibis.memtable(
            {
                "input": [1.0, 2.0, 3.0],
                "other": [5.0, 5.0, 5.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input, float[N] other) => (float[N] output) {
                output = Add(input, other)
            }
        """)

        variables = GraphVariables(table, model)

        translator = AddTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(NotImplementedError, match="must be a constant list"):
            translator.process()


class TestSubTranslator:
    optimizer = Optimizer(enabled=False)

    def test_sub_single_column(self):
        """Test SubTranslator with a single column input."""
        table = ibis.memtable({"input": [10.0, 20.0, 30.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[1] sub_value = {3.0}>
            {
                output = Sub(input, sub_value)
            }
        """)

        variables = GraphVariables(table, model)
        translator = SubTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        computed = list(backend.execute(result))
        assert computed == [7.0, 17.0, 27.0]

    def test_sub_group_columns(self):
        """Test SubTranslator with a group of columns."""
        table = ibis.memtable(
            {
                "col_a": [10.0, 20.0, 30.0],
                "col_b": [100.0, 200.0, 300.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[2] sub_values = {3.0, 50.0}>
            {
                output = Sub(input, sub_values)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = SubTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["col_a"])) == [7.0, 17.0, 27.0]
        assert list(backend.execute(result["col_b"])) == [50.0, 150.0, 250.0]

    def test_sub_invalid_non_numeric(self):
        """Test SubTranslator raises error for non-numeric operand."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[1] sub_value = {5.0}>
            {
                output = Sub(input, sub_value)
            }
        """)

        variables = GraphVariables(table, model)
        variables["input"] = "not_a_numeric_value"  # type: ignore[assignment]

        translator = SubTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="first operand must be a numeric value"):
            translator.process()

    def test_sub_single_column_requires_single_value(self):
        """Test SubTranslator raises error when single column given multiple values."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[2] sub_values = {5.0, 10.0}>
            {
                output = Sub(input, sub_values)
            }
        """)

        variables = GraphVariables(table, model)

        translator = SubTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="must contain exactly 1 value"):
            translator.process()

    def test_sub_mismatched_column_count(self):
        """Test SubTranslator raises error when column count doesn't match."""
        table = ibis.memtable(
            {
                "col_a": [1.0, 2.0, 3.0],
                "col_b": [10.0, 20.0, 30.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[1] sub_values = {5.0}>
            {
                output = Sub(input, sub_values)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = SubTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(AssertionError, match="must match the number of fields"):
            translator.process()

    def test_sub_second_operand_not_constant(self):
        """Test SubTranslator raises error when second operand is not a constant."""
        table = ibis.memtable(
            {
                "input": [1.0, 2.0, 3.0],
                "other": [5.0, 5.0, 5.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input, float[N] other) => (float[N] output) {
                output = Sub(input, other)
            }
        """)

        variables = GraphVariables(table, model)

        translator = SubTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(NotImplementedError, match="must be a constant list"):
            translator.process()


class TestMulTranslator:
    optimizer = Optimizer(enabled=False)

    def test_mul_single_column(self):
        """Test MulTranslator with a single column input."""
        table = ibis.memtable({"input": [2.0, 3.0, 4.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[1] mul_value = {5.0}>
            {
                output = Mul(input, mul_value)
            }
        """)

        variables = GraphVariables(table, model)
        translator = MulTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        computed = list(backend.execute(result))
        assert computed == [10.0, 15.0, 20.0]

    def test_mul_group_columns(self):
        """Test MulTranslator with a group of columns."""
        table = ibis.memtable(
            {
                "col_a": [2.0, 3.0, 4.0],
                "col_b": [10.0, 20.0, 30.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[2] mul_values = {3.0, 2.0}>
            {
                output = Mul(input, mul_values)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = MulTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["col_a"])) == [6.0, 9.0, 12.0]
        assert list(backend.execute(result["col_b"])) == [20.0, 40.0, 60.0]

    def test_mul_invalid_non_numeric(self):
        """Test MulTranslator raises error for non-numeric operand."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[1] mul_value = {5.0}>
            {
                output = Mul(input, mul_value)
            }
        """)

        variables = GraphVariables(table, model)
        variables["input"] = "not_a_numeric_value"  # type: ignore[assignment]

        translator = MulTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="first operand must be a numeric value"):
            translator.process()

    def test_mul_mismatched_column_count(self):
        """Test MulTranslator raises error when column count doesn't match."""
        table = ibis.memtable(
            {
                "col_a": [1.0, 2.0, 3.0],
                "col_b": [10.0, 20.0, 30.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[1] mul_values = {5.0}>
            {
                output = Mul(input, mul_values)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = MulTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="same number of values"):
            translator.process()

    def test_mul_single_column_requires_single_value(self):
        """Test MulTranslator raises error when single column given multiple values."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[2] mul_values = {5.0, 10.0}>
            {
                output = Mul(input, mul_values)
            }
        """)

        variables = GraphVariables(table, model)

        translator = MulTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="must contain exactly 1 value"):
            translator.process()

    def test_mul_second_operand_not_constant(self):
        """Test MulTranslator raises error when second operand is not a constant."""
        table = ibis.memtable(
            {
                "input": [1.0, 2.0, 3.0],
                "other": [5.0, 5.0, 5.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input, float[N] other) => (float[N] output) {
                output = Mul(input, other)
            }
        """)

        variables = GraphVariables(table, model)

        translator = MulTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(NotImplementedError, match="must be a constant list"):
            translator.process()


class TestDivTranslator:
    optimizer = Optimizer(enabled=False)

    def test_div_single_column(self):
        """Test DivTranslator with a single column input."""
        table = ibis.memtable({"input": [10.0, 20.0, 30.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[1] div_value = {2.0}>
            {
                output = Div(input, div_value)
            }
        """)

        variables = GraphVariables(table, model)
        translator = DivTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        backend = ibis.duckdb.connect()
        computed = list(backend.execute(result))
        assert computed == [5.0, 10.0, 15.0]

    def test_div_group_columns_with_matching_values(self):
        """Test DivTranslator with group of columns and matching divisor values."""
        table = ibis.memtable(
            {
                "col_a": [10.0, 20.0, 30.0],
                "col_b": [100.0, 200.0, 300.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[2] div_values = {2.0, 10.0}>
            {
                output = Div(input, div_values)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = DivTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["col_a"])) == [5.0, 10.0, 15.0]
        assert list(backend.execute(result["col_b"])) == [10.0, 20.0, 30.0]

    def test_div_group_columns_broadcast_single_value(self):
        """Test DivTranslator with group of columns and single divisor (broadcast)."""
        table = ibis.memtable(
            {
                "col_a": [10.0, 20.0, 30.0],
                "col_b": [100.0, 200.0, 300.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[1] div_value = {10.0}>
            {
                output = Div(input, div_value)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = DivTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")
        assert isinstance(result, ValueVariablesGroup)

        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["col_a"])) == [1.0, 2.0, 3.0]
        assert list(backend.execute(result["col_b"])) == [10.0, 20.0, 30.0]

    def test_div_invalid_non_numeric_single(self):
        """Test DivTranslator raises error for non-numeric single operand."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[1] div_value = {5.0}>
            {
                output = Div(input, div_value)
            }
        """)

        variables = GraphVariables(table, model)
        variables["input"] = "not_a_numeric_value"  # type: ignore[assignment]

        translator = DivTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="first operand must be a numeric value"):
            translator.process()

    def test_div_mismatched_column_count(self):
        """Test DivTranslator raises error when column count doesn't match values."""
        table = ibis.memtable(
            {
                "col_a": [1.0, 2.0, 3.0],
                "col_b": [10.0, 20.0, 30.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[3] div_values = {5.0, 10.0, 15.0}>
            {
                output = Div(input, div_values)
            }
        """)

        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
            }
        )

        translator = DivTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="must match the number of columns"):
            translator.process()

    def test_div_single_column_requires_single_value(self):
        """Test DivTranslator raises error when single column given multiple values."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[2] div_values = {5.0, 10.0}>
            {
                output = Div(input, div_values)
            }
        """)

        variables = GraphVariables(table, model)

        translator = DivTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="must contain only one value"):
            translator.process()

    def test_div_second_operand_not_constant(self):
        """Test DivTranslator raises error when second operand is not a constant."""
        table = ibis.memtable(
            {
                "input": [1.0, 2.0, 3.0],
                "other": [5.0, 5.0, 5.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input, float[N] other) => (float[N] output) {
                output = Div(input, other)
            }
        """)

        variables = GraphVariables(table, model)

        translator = DivTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(NotImplementedError, match="must be a constant list"):
            translator.process()


class TestIdentityTranslator:
    optimizer = Optimizer(enabled=False)

    def test_identity_single_column_passthrough(self):
        """Test IdentityTranslator passes a single column through unchanged."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = Identity(input)
            }
        """)

        variables = GraphVariables(table, model)
        translator = IdentityTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        # Verify output equals input
        backend = ibis.duckdb.connect()
        input_values = list(backend.execute(table["input"]))
        output_values = list(backend.execute(result))
        assert output_values == input_values

    def test_identity_group_columns_passthrough(self):
        """Test IdentityTranslator passes a NumericVariablesGroup through unchanged."""
        table = ibis.memtable(
            {
                "col_a": [1.0, 2.0, 3.0],
                "col_b": [4.0, 5.0, 6.0],
                "col_c": [7.0, 8.0, 9.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output) {
                output = Identity(input)
            }
        """)

        # Use dummy table for GraphVariables since we override the input
        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
                "col_c": table["col_c"],
            }
        )

        translator = IdentityTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        # Should return the same NumericVariablesGroup
        assert isinstance(result, NumericVariablesGroup)
        assert len(result) == 3
        assert "col_a" in result
        assert "col_b" in result
        assert "col_c" in result

        # Verify all columns preserved with same values
        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["col_a"])) == [1.0, 2.0, 3.0]
        assert list(backend.execute(result["col_b"])) == [4.0, 5.0, 6.0]
        assert list(backend.execute(result["col_c"])) == [7.0, 8.0, 9.0]


class TestReshapeTranslator:
    optimizer = Optimizer(enabled=False)

    def test_reshape_single_column_to_single_column(self):
        """Test ReshapeTranslator with shape=[-1] on single column (passes through)."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <int64[1] shape = {-1}>
            {
                output = Reshape(input, shape)
            }
        """)

        variables = GraphVariables(table, model)
        translator = ReshapeTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        # Verify output equals input (passthrough)
        backend = ibis.duckdb.connect()
        input_values = list(backend.execute(table["input"]))
        output_values = list(backend.execute(result))
        assert output_values == input_values

    def test_reshape_group_to_same_size(self):
        """Test ReshapeTranslator with shape=[-1, N] on N columns (passes through)."""
        table = ibis.memtable(
            {
                "col_a": [1.0, 2.0, 3.0],
                "col_b": [4.0, 5.0, 6.0],
                "col_c": [7.0, 8.0, 9.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <int64[2] shape = {-1, 3}>
            {
                output = Reshape(input, shape)
            }
        """)

        # Use dummy table for GraphVariables since we override the input
        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
                "col_c": table["col_c"],
            }
        )

        translator = ReshapeTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )
        translator.process()

        assert "output" in variables
        result = variables.peek_variable("output")

        # Should return the same NumericVariablesGroup (passthrough)
        assert isinstance(result, NumericVariablesGroup)
        assert len(result) == 3

        # Verify all columns preserved with same values
        backend = ibis.duckdb.connect()
        assert list(backend.execute(result["col_a"])) == [1.0, 2.0, 3.0]
        assert list(backend.execute(result["col_b"])) == [4.0, 5.0, 6.0]
        assert list(backend.execute(result["col_c"])) == [7.0, 8.0, 9.0]

    def test_reshape_requires_integer_shape(self):
        """Test ReshapeTranslator raises error when shape is not integers."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <float[1] shape = {-1.0}>
            {
                output = Reshape(input, shape)
            }
        """)

        variables = GraphVariables(table, model)
        translator = ReshapeTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            NotImplementedError, match="requires integer values for the shape"
        ):
            translator.process()

    def test_reshape_cannot_change_row_count(self):
        """Test ReshapeTranslator raises error when shape[0] != -1."""
        table = ibis.memtable({"input": [1.0, 2.0, 3.0]})
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <int64[2] shape = {3, 1}>
            {
                output = Reshape(input, shape)
            }
        """)

        variables = GraphVariables(table, model)
        translator = ReshapeTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(
            NotImplementedError, match="Reshape can't change the number of rows"
        ):
            translator.process()

    def test_reshape_unsupported_shape(self):
        """Test ReshapeTranslator raises error for unsupported shape combinations."""
        # Test case: shape=[-1, 2] but input has 3 columns
        table = ibis.memtable(
            {
                "col_a": [1.0, 2.0, 3.0],
                "col_b": [4.0, 5.0, 6.0],
                "col_c": [7.0, 8.0, 9.0],
            }
        )
        model = onnx.parser.parse_graph("""
            agraph (float[N] input) => (float[N] output)
            <int64[2] shape = {-1, 2}>
            {
                output = Reshape(input, shape)
            }
        """)

        # Use dummy table for GraphVariables since we override the input
        variables = GraphVariables(ibis.memtable({"input": [1.0]}), model)
        variables["input"] = NumericVariablesGroup(
            {
                "col_a": table["col_a"],
                "col_b": table["col_b"],
                "col_c": table["col_c"],
            }
        )

        translator = ReshapeTranslator(
            table, model.node[0], variables, self.optimizer, TranslationOptions()
        )

        with pytest.raises(ValueError, match="Reshape shape=\\[-1, 2\\] not supported"):
            translator.process()
