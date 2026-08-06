import operator

import ibis
import pytest
from ibis.expr.operations import Literal

from orbital.translation.optimizer import Optimizer


class TestOptimizerFold:
    optimizer = Optimizer()

    def test_fold_sum_only_literals(self):
        result = self.optimizer.fold_contiguous_sum(
            [ibis.literal(1), ibis.literal(2), ibis.literal(3)]
        )
        assert len(result) == 1
        assert result[0].execute() == pytest.approx(6)

    def test_fold_sum_with_column_and_literals(self):
        table = ibis.memtable({"value": [1.0, 2.0, 3.0]})
        column = table["value"]
        result = self.optimizer.fold_contiguous_sum(
            [ibis.literal(1), column, ibis.literal(2)]
        )
        assert len(result) == 2
        assert result[0] is column
        assert result[1].execute() == pytest.approx(3)

    def test_fold_sum_preserves_non_scalar_order(self):
        table = ibis.memtable({"a": [1.0], "b": [2.0]})
        a_col = table["a"]
        b_col = table["b"]
        result = self.optimizer.fold_contiguous_sum([a_col, ibis.literal(5), b_col])
        assert len(result) == 3
        assert result[0] is a_col
        assert result[1] is b_col
        assert result[2].execute() == pytest.approx(5)

    def test_fold_sum_keeps_identity_with_other_terms(self):
        table = ibis.memtable({"value": [1.0, 2.0, 3.0]})
        column = table["value"]
        result = self.optimizer.fold_contiguous_sum([ibis.literal(0), column])
        assert len(result) == 2
        assert result[0] is column
        assert result[1].execute() == pytest.approx(0)

    def test_fold_product_returns_zero_when_zero_present(self):
        table = ibis.memtable({"value": [1.0, 2.0, 3.0]})
        column = table["value"]
        result = self.optimizer.fold_contiguous_product(
            [column, ibis.literal(0), ibis.literal(5)]
        )
        assert len(result) == 2
        assert result[0] is column
        assert result[1].execute() == pytest.approx(0)

    def test_fold_product_with_column_and_literals(self):
        table = ibis.memtable({"value": [1.0, 2.0, 3.0]})
        column = table["value"]
        result = self.optimizer.fold_contiguous_product(
            [ibis.literal(2), column, ibis.literal(3)]
        )
        assert len(result) == 2
        assert result[0] is column
        assert result[1].execute() == pytest.approx(6)

    def test_fold_product_preserves_non_scalar_order(self):
        table = ibis.memtable({"a": [1.0], "b": [2.0]})
        a_col = table["a"]
        b_col = table["b"]
        result = self.optimizer.fold_contiguous_product([a_col, ibis.literal(4), b_col])
        assert len(result) == 3
        assert result[0] is a_col
        assert result[1] is b_col
        assert result[2].execute() == pytest.approx(4)

    def test_fold_product_keeps_identity_with_other_terms(self):
        table = ibis.memtable({"value": [1.0, 2.0, 3.0]})
        column = table["value"]
        result = self.optimizer.fold_contiguous_product([ibis.literal(1), column])
        assert len(result) == 2
        assert result[0] is column
        assert result[1].execute() == pytest.approx(1)

    def test_fold_unsupported_operator_raises(self):
        with pytest.raises(NotImplementedError):
            self.optimizer._fold_associative_op_contiguous(
                [ibis.literal(1), ibis.literal(2)], operator.sub
            )

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

    def test_fold_zeros_subtract_left(self):
        expr = ibis.literal(0) - ibis.literal(5)
        result = self.optimizer.fold_zeros(expr)
        assert result.op().value == 5

    def test_fold_zeros_subtract_right(self):
        expr = ibis.literal(5) - ibis.literal(0)
        result = self.optimizer.fold_zeros(expr)
        assert result.op().value == 5

    def test_fold_operation_unary_negate(self):
        expr = -ibis.literal(10)
        result = self.optimizer.fold_operation(expr)
        # Folding must reduce the expression to a precomputed literal,
        # not return the unreduced operation tree.
        assert isinstance(result.op(), Literal)
        assert result.op().value == -10

    def test_fold_operation_unary_not(self):
        expr = ~ibis.literal(True)
        result = self.optimizer.fold_operation(expr)
        # Folding must reduce the expression to a precomputed literal,
        # not return the unreduced operation tree.
        assert isinstance(result.op(), Literal)
        assert result.op().value is False

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

    def test_fold_contiguous_sum_disabled_returns_unchanged(self):
        disabled = Optimizer(enabled=False)
        a, b = ibis.literal(1), ibis.literal(2)
        result = disabled.fold_contiguous_sum([a, b])
        assert len(result) == 2
        assert result[0] is a
        assert result[1] is b

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

    def test_fold_zeros_disabled_returns_unchanged(self):
        disabled = Optimizer(enabled=False)
        expr = ibis.literal(0) * ibis.literal(7)
        result = disabled.fold_zeros(expr)
        assert result is expr

    def test_fold_zeros_multiply_left_zero(self):
        expr = ibis.literal(0) * ibis.literal(7)
        result = self.optimizer.fold_zeros(expr)
        assert result.op().value == 0

    def test_fold_zeros_multiply_right_zero(self):
        expr = ibis.literal(7) * ibis.literal(0)
        result = self.optimizer.fold_zeros(expr)
        assert result.op().value == 0

    def test_fold_zeros_add_left_zero(self):
        table = ibis.memtable({"value": [1.0, 2.0]})
        column = table["value"]
        expr = ibis.literal(0) + column
        result = self.optimizer.fold_zeros(expr)
        assert result.equals(column)

    def test_fold_zeros_add_right_zero(self):
        table = ibis.memtable({"value": [1.0, 2.0]})
        column = table["value"]
        expr = column + ibis.literal(0)
        result = self.optimizer.fold_zeros(expr)
        assert result.equals(column)

    def test_fold_zeros_no_zero_operand_returns_unchanged(self):
        expr = ibis.literal(3) - ibis.literal(4)
        result = self.optimizer.fold_zeros(expr)
        assert result is expr

    def test_fold_operation_disabled_returns_unchanged(self):
        disabled = Optimizer(enabled=False)
        expr = ibis.literal(2) + ibis.literal(3)
        result = disabled.fold_operation(expr)
        assert result is expr

    def test_fold_operation_python_scalar_wrapped_as_literal(self):
        # fold_operation can be handed a plain Python value when a previous
        # fold already reduced a subtree to a scalar; it still has to come
        # back out as an ibis expression.
        result = self.optimizer.fold_operation(5)
        assert isinstance(result, ibis.Expr)
        assert result.op().value == 5

    def test_fold_operation_mixed_literal_and_column_delegates_to_fold_zeros(self):
        table = ibis.memtable({"value": [1.0, 2.0]})
        column = table["value"]
        expr = column * ibis.literal(0)
        result = self.optimizer.fold_operation(expr)
        assert result.op().value == 0

    def test_fold_operation_binary_literal_folding(self):
        expr = ibis.literal(2) + ibis.literal(3)
        result = self.optimizer.fold_operation(expr)
        assert isinstance(result.op(), Literal)
        assert result.op().value == 5

    def test_fold_operation_unfoldable_op_returns_unchanged(self):
        # IsNull isn't in BINARY_OPS or UNARY_OPS, so even with an
        # all-literal input there's nothing fold_operation can precompute.
        expr = ibis.literal(5).isnull()
        result = self.optimizer.fold_operation(expr)
        assert result is expr
