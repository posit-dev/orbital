import operator

import ibis
import pytest

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
