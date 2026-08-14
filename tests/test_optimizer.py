import ibis
import onnx
import pytest
from ibis.expr.operations import Literal

from orbital.translation.optimizer import Optimizer, _OptimizedOps
from orbital.translation.options import TranslationOptions
from orbital.translation.translator import Translator
from orbital.translation.variables import GraphVariables


class TestOptimizerFold:
    optimizer = Optimizer()

    def test_fold_sum_only_literals(self):
        result = self.optimizer.fold_operations(
            ibis.literal(1) + ibis.literal(2) + ibis.literal(3)
        )
        assert isinstance(result.op(), Literal)
        assert result.op().value == pytest.approx(6)

    def test_fold_sum_merges_only_constants_adjacent_in_the_tree(self):
        # Folding never reorders the tree, so the two literals of
        # "1 + column + 2" cannot be merged, while those of "1 + 2 + column"
        # are adjacent and become a single literal.
        column = ibis.memtable({"value": [1.0, 2.0, 3.0]})["value"]
        around_column = self.optimizer.fold_operations(
            ibis.literal(1) + column + ibis.literal(2)
        )
        assert around_column.equals(column + ibis.literal(1) + ibis.literal(2))
        adjacent = self.optimizer.fold_operations(
            ibis.literal(1) + ibis.literal(2) + column
        )
        assert adjacent.equals(column + ibis.literal(3))

    def test_fold_sum_keeps_identity_with_other_terms(self):
        column = ibis.memtable({"value": [1.0, 2.0, 3.0]})["value"]
        result = self.optimizer.fold_operations(ibis.literal(0) + column)
        assert result.equals(column)

    def test_optimized_ops_recognize_an_ibis_literal_as_constant(self):
        # fold_operations never hits this directly: its own recursion unwraps
        # a Literal node to a plain Python value before _OptimizedOps sees it.
        # _OptimizedOps must still handle an ibis.literal(...) operand on its
        # own terms, since that is part of its documented contract.
        column = ibis.memtable({"value": [1.0, 2.0, 3.0]})["value"]
        assert _OptimizedOps.add(ibis.literal(0), column).equals(column)
        assert _OptimizedOps.mul(ibis.literal(1), column).equals(column)

    def test_fold_operations_removes_zero_buried_in_the_innermost_node(self):
        # python's sum() seeds the accumulation with 0, which ibis turns into
        # a real "+ 0" node at the very bottom of the chain.
        table = ibis.memtable({"a": [1.0], "b": [2.0]})
        expr = sum([table["a"] * 2.0, table["b"] * 3.0])
        result = self.optimizer.fold_operations(expr)
        assert result.equals(table["a"] * 2.0 + table["b"] * 3.0)

    def test_fold_product_returns_zero_when_zero_present(self):
        column = ibis.memtable({"value": [1.0, 2.0, 3.0]})["value"]
        result = self.optimizer.fold_operations(column * ibis.literal(0))
        assert result.op().value == 0

    def test_fold_product_only_literals(self):
        result = self.optimizer.fold_operations(ibis.literal(2) * ibis.literal(3))
        assert result.op().value == pytest.approx(6)

    def test_fold_product_keeps_identity_with_other_terms(self):
        column = ibis.memtable({"value": [1.0, 2.0, 3.0]})["value"]
        assert self.optimizer.fold_operations(ibis.literal(1) * column).equals(column)
        assert self.optimizer.fold_operations(column * ibis.literal(1)).equals(column)

    def test_fold_division_by_identity(self):
        column = ibis.memtable({"value": [1.0, 2.0, 3.0]})["value"]
        assert self.optimizer.fold_operations(column / ibis.literal(1)).equals(column)
        result = self.optimizer.fold_operations(ibis.literal(6) / ibis.literal(3))
        assert result.op().value == pytest.approx(2)

    def test_fold_cast_merges_nested_casts(self):
        table = ibis.memtable({"a": [1.0]})
        expr = table["a"].cast("float32").cast("string")
        folded = self.optimizer.fold_cast(expr)
        assert isinstance(folded.op(), ibis.expr.operations.Cast)
        assert not isinstance(folded.op().arg, ibis.expr.operations.Cast)
        assert folded.type().is_string()

    def test_fold_cast_literal_int(self):
        expr = ibis.literal(3.7).cast("int64")
        result = self.optimizer.fold_cast(expr)
        assert result.op().value == 3

    def test_fold_cast_literal_float(self):
        expr = ibis.literal(5).cast("float64")
        result = self.optimizer.fold_cast(expr)
        assert result.op().value == 5.0

    def test_fold_cast_literal_string(self):
        expr = ibis.literal(42).cast("string")
        result = self.optimizer.fold_cast(expr)
        assert result.op().value == "42"

    def test_fold_cast_literal_boolean(self):
        expr = ibis.literal(1).cast("boolean")
        result = self.optimizer.fold_cast(expr)
        assert result.op().value is True

    def test_fold_cast_nested_literals(self):
        expr = ibis.literal(7).cast("float64").cast("string")
        result = self.optimizer.fold_cast(expr)
        assert result.op().value == "7"

    def test_fold_operations_subtract_left(self):
        expr = ibis.literal(0) - ibis.literal(5)
        result = self.optimizer.fold_operations(expr)
        assert result.execute() == -5

    def test_fold_operations_subtract_left_column(self):
        table = ibis.memtable({"value": [1.0, 2.0, 3.0]})
        expr = ibis.literal(0) - table["value"]
        result = self.optimizer.fold_operations(expr)
        assert result.execute().tolist() == pytest.approx([-1.0, -2.0, -3.0])

    def test_fold_operations_subtract_right(self):
        column = ibis.memtable({"value": [1.0, 2.0, 3.0]})["value"]
        result = self.optimizer.fold_operations(column - ibis.literal(0))
        assert result.equals(column)

    def test_fold_operation_unary_negate(self):
        expr = -ibis.literal(10)
        result = self.optimizer.fold_operations(expr)
        # Folding must reduce the expression to a precomputed literal,
        # not return the unreduced operation tree.
        assert isinstance(result.op(), Literal)
        assert result.op().value == -10

    def test_fold_operation_unary_not(self):
        expr = ~ibis.literal(True)
        result = self.optimizer.fold_operations(expr)
        # Folding must reduce the expression to a precomputed literal,
        # not return the unreduced operation tree.
        assert isinstance(result.op(), Literal)
        assert result.op().value is False

    def test_fold_operation_unary_on_non_constant_is_left_alone(self):
        # The unary functions are pure python ones, handing them an ibis
        # expression would raise instead of folding anything.
        expr = ~ibis.memtable({"flag": [True]})["flag"]
        result = self.optimizer.fold_operations(expr)
        assert result.equals(expr)

    def test_ensure_expr_passthrough_for_expr(self):
        column = ibis.memtable({"a": [1.0]})["a"]
        result = self.optimizer._ensure_expr(column)
        assert result is column

    def test_ensure_expr_wraps_literal_op_node(self):
        op_node = ibis.literal(5).op()
        result = self.optimizer._ensure_expr(op_node)
        assert isinstance(result, ibis.Expr)
        assert result.execute() == 5

    def test_ensure_expr_raises_for_unsupported_type(self):
        with pytest.raises(TypeError):
            self.optimizer._ensure_expr(object())

    def test_fold_case_deferred_raises(self):
        with pytest.raises(NotImplementedError):
            self.optimizer.fold_case(ibis._.x)

    def test_fold_case_disabled_returns_unchanged(self):
        disabled = Optimizer(enabled=False)
        expr = ibis.cases((ibis.literal(True), ibis.literal(1)), else_=ibis.literal(0))
        result = disabled.fold_case(expr)
        assert result is expr

    def test_fold_case_all_results_same_value_folds_to_value(self):
        # Every result and the default are literal 7, so the case can never
        # produce anything else - it should collapse to a plain literal.
        expr = ibis.cases(
            (ibis.literal(True), ibis.literal(7)),
            (ibis.literal(False), ibis.literal(7)),
            else_=ibis.literal(7),
        )
        result = self.optimizer.fold_case(expr)
        assert result.op().value == 7

    def test_fold_case_literal_true_condition_returns_result(self):
        expr = ibis.cases((ibis.literal(True), ibis.literal(1)), else_=ibis.literal(0))
        result = self.optimizer.fold_case(expr)
        assert result.op().value == 1

    def test_fold_case_literal_false_condition_returns_default(self):
        expr = ibis.cases((ibis.literal(False), ibis.literal(1)), else_=ibis.literal(0))
        result = self.optimizer.fold_case(expr)
        assert result.op().value == 0

    def test_fold_case_boolean_ifelse_on_non_literal_condition_not_folded(self):
        # Folding a single IF/ELSE returning 1 or 0 into a boolean cast is
        # deliberately disabled (see the FIXME in fold_case: it doesn't work
        # on postgresql), so a non-literal condition must come back
        # unchanged rather than being rewritten.
        table = ibis.memtable({"x": [1.0]})
        condition = table["x"] > 0
        expr = ibis.cases((condition, ibis.literal(1)), else_=ibis.literal(0))
        result = self.optimizer.fold_case(expr)
        assert result is expr

    def test_fold_case_multiple_cases_returns_unchanged(self):
        expr = ibis.cases(
            (ibis.literal(True), ibis.literal("a")),
            (ibis.literal(False), ibis.literal("b")),
            else_=ibis.literal("c"),
        )
        result = self.optimizer.fold_case(expr)
        assert result is expr

    def test_fold_cast_disabled_returns_unchanged(self):
        disabled = Optimizer(enabled=False)
        expr = ibis.literal(5).cast("float64")
        result = disabled.fold_cast(expr)
        assert result is expr

    def test_fold_cast_non_cast_expr_returns_unchanged(self):
        # Ibis itself drops a cast to a column's own type, so fold_cast
        # never even sees a Cast node here - it should just hand the
        # expression back.
        table = ibis.memtable({"a": [1.0]})
        column = table["a"]
        not_a_cast = column.cast("float64")
        assert not isinstance(not_a_cast.op(), ibis.expr.operations.Cast)
        result = self.optimizer.fold_cast(not_a_cast)
        assert result is not_a_cast

    def test_fold_cast_unsupported_literal_type_raises(self):
        expr = ibis.literal(5).cast("date")
        with pytest.raises(NotImplementedError):
            self.optimizer.fold_cast(expr)

    def test_fold_cast_already_target_type_unwraps_manual_cast(self):
        # Build a Cast node directly (bypassing ibis's own .cast(), which
        # already drops a same-type cast on its own) so fold_cast has to be
        # the one to notice the arg is already the target type.
        table = ibis.memtable({"a": [1.0]})
        column = table["a"]
        manual_cast = ibis.expr.operations.Cast(column.op(), to=column.type()).to_expr()
        result = self.optimizer.fold_cast(manual_cast)
        assert result.equals(column)

    def test_fold_operations_disabled_returns_unchanged(self):
        disabled = Optimizer(enabled=False)
        expr = ibis.literal(0) + ibis.literal(2) + ibis.literal(3)
        result = disabled.fold_operations(expr)
        assert result is expr

    def test_fold_operations_multiply_left_zero(self):
        expr = ibis.literal(0) * ibis.literal(7)
        result = self.optimizer.fold_operations(expr)
        assert result.op().value == 0

    def test_fold_operations_multiply_right_zero(self):
        expr = ibis.literal(7) * ibis.literal(0)
        result = self.optimizer.fold_operations(expr)
        assert result.op().value == 0

    def test_fold_operations_add_right_zero(self):
        table = ibis.memtable({"value": [1.0, 2.0]})
        column = table["value"]
        expr = column + ibis.literal(0)
        result = self.optimizer.fold_operations(expr)
        assert result.equals(column)

    def test_fold_operations_no_zero_operand_returns_unchanged(self):
        table = ibis.memtable({"value": [1.0, 2.0]})
        column = table["value"]
        expr = column - ibis.literal(4)
        result = self.optimizer.fold_operations(expr)
        assert result.equals(expr)

    def test_fold_operations_python_scalar_wrapped_as_literal(self):
        # fold_operations can be handed a plain Python value when a previous
        # fold already reduced a subtree to a scalar; it still has to come
        # back out as an ibis expression.
        result = self.optimizer.fold_operations(5)
        assert isinstance(result, ibis.Expr)
        assert result.op().value == 5

    def test_fold_operations_binary_literal_folding(self):
        expr = ibis.literal(2) + ibis.literal(3)
        result = self.optimizer.fold_operations(expr)
        assert isinstance(result.op(), Literal)
        assert result.op().value == 5

    def test_fold_operations_comparison_of_mixed_operands_is_rebuilt(self):
        # Comparisons have no algebraic shortcut: with two constants python
        # computes them, with a column the node is rebuilt as it was.
        column = ibis.memtable({"value": [1.0, 2.0]})["value"]
        constants = self.optimizer.fold_operations(ibis.literal(2) == ibis.literal(3))
        assert constants.op().value is False
        mixed = self.optimizer.fold_operations(column == ibis.literal(2))
        assert mixed.equals(column == ibis.literal(2))

    def test_fold_operations_unfoldable_op_returns_unchanged(self):
        # IsNull isn't in BINARY_OPS or UNARY_OPS, so even with an
        # all-literal input there's nothing fold_operations can precompute.
        expr = ibis.literal(5).isnull()
        result = self.optimizer.fold_operations(expr)
        assert result.equals(expr)

    def test_fold_operations_does_not_descend_into_columns(self):
        # A column reference leads to the table, its schema and namespace,
        # which hold no arithmetic and would be pointlessly expensive to walk.
        column = ibis.memtable({"value": [1.0, 2.0]})["value"]
        result = self.optimizer.fold_operations(column)
        assert result.equals(column)


class TestOptimizerPreserve:
    # T is referenced by the second node, Y is a result of the pipeline.
    graph = onnx.parser.parse_graph("""
        agraph (double[N] X) => (double[N] Y)
        {
            T = Identity(X)
            Y = Identity(T)
        }
    """)

    class IncrementTranslator(Translator):
        """Emits a single output the optimizer can preserve."""

        def process(self):
            self.set_output(self._table["X"] + 1.0)

    class PassThroughTranslator(Translator):
        """Emits an output that is already a column of the table."""

        def process(self):
            self.set_output(self._table["X"])

    def _process(self, optimizer, node_index, translator_class=IncrementTranslator):
        table = ibis.memtable({"X": [1.0, 2.0]})
        variables = GraphVariables(table, self.graph)
        translator = translator_class(
            table,
            self.graph.node[node_index],
            variables,
            optimizer,
            TranslationOptions(),
        )
        translator.process()
        optimizer.preserve_referenced_outputs(translator, variables)
        return translator, variables

    def test_referenced_output_is_preserved(self):
        translator, _ = self._process(Optimizer(), node_index=0)
        assert [c for c in translator.mutated_table.columns if c.startswith("prs_")]

    def test_output_nothing_references_is_not_preserved(self):
        translator, _ = self._process(Optimizer(), node_index=1)
        assert translator.mutated_table.columns == ("X",)

    def test_output_that_is_already_a_column_is_not_preserved(self):
        translator, variables = self._process(
            Optimizer(), node_index=0, translator_class=self.PassThroughTranslator
        )
        assert translator.mutated_table.columns == ("X",)
        # The output still points at the original column, it was not re-aliased.
        assert variables.peek_variable("T").get_name() == "X"

    def test_preserve_disabled_leaves_the_table_untouched(self):
        translator, _ = self._process(Optimizer(enabled=False), node_index=0)
        assert translator.mutated_table.columns == ("X",)
