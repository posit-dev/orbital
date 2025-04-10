"""Implement optiomizations to the Ibis expression tree.

Primarily it takes care of folding constant expressions
and removing unnecessary casts.
"""

import functools
import itertools
import operator
import typing

import ibis
import ibis.expr.datatypes as dt
from ibis.expr.operations import (
    Abs,
    Add,
    And,
    Binary,
    Ceil,
    Divide,
    Equals,
    Floor,
    FloorDivide,
    Greater,
    GreaterEqual,
    IdenticalTo,
    Less,
    LessEqual,
    Literal,
    Modulus,
    Multiply,
    Negate,
    Not,
    NotEquals,
    Or,
    Subtract,
    Unary,
    Xor,
)
from ibis.expr.types import NumericScalar


class Optimizer:
    """Optimizer for Ibis expressions.

    This class is responsible for applying a set of optimization
    processes to Ibis expressionsto remove unecessary operations and
    reduce query complexity.
    """

    BINARY_OPS: dict[type[Binary], typing.Callable] = {
        # Mathematical Operators
        Add: operator.add,
        Subtract: operator.sub,
        Multiply: operator.mul,
        Divide: operator.truediv,
        FloorDivide: operator.floordiv,
        Modulus: operator.mod,
        # Logical Operators
        Equals: operator.eq,
        NotEquals: operator.ne,
        Greater: operator.gt,
        GreaterEqual: operator.ge,
        Less: operator.lt,
        LessEqual: operator.le,
        IdenticalTo: operator.eq,
        # Binary Operators
        And: operator.and_,
        Or: operator.or_,
        Xor: operator.xor,
    }

    UNARY_OPS: dict[type[Unary], typing.Callable] = {
        Negate: operator.neg,
        Abs: operator.abs,
        Ceil: lambda x: float(operator.methodcaller("ceil")(x)),  # Se necessario
        Floor: lambda x: float(operator.methodcaller("floor")(x)),
        Not: operator.not_,
    }

    def __init__(self, enabled: bool = True) -> None:
        """
        :param enabled: Whether to enable the optimizer.
                        When disabled, the optimizer will
                        return the expression unchanged.
        """
        self.ENABLED = enabled

    def _ensure_expr(self, value: ibis.Expr) -> ibis.Expr:
        """Ensure that the value is an Ibis expression.

        Literal objects need to be converted back to an
        Ibis expression to be used in the query.
        """
        if isinstance(value, Literal):
            return ibis.literal(value.value)
        return value

    def _fold_associative_op_contiguous(
        self, lst: list[ibis.expr.types.NumericValue], pyop: typing.Callable
    ) -> list[ibis.expr.types.NumericValue]:
        """Precompute an operation applied on multiple elements.

        Given a list of expressions and a binary operation,
        this function will precompute the operation on all
        constant expressions in the list returning a new list
        of expressions with the folded constants.
        """
        if self.ENABLED is False:
            return list(lst)

        lst = list(lst)

        result = []
        for is_number, group in itertools.groupby(
            lst, key=lambda x: isinstance(x, NumericScalar)
        ):
            if is_number:
                values = [scalar.execute() for scalar in group]
                folded_value = ibis.literal(functools.reduce(pyop, values, 0))
                result.append(folded_value)
            else:
                result.extend(group)
        return result

    def fold_contiguous_sum(
        self, lst: list[ibis.expr.types.NumericValue]
    ) -> list[ibis.expr.types.NumericValue]:
        """Precompute constants in a list of sums"""
        return self._fold_associative_op_contiguous(lst, operator.add)

    def fold_contiguous_product(
        self, lst: list[ibis.expr.types.NumericValue]
    ) -> list[ibis.expr.types.NumericValue]:
        """Precompute constants in a list of multiplications"""
        return self._fold_associative_op_contiguous(lst, operator.mul)

    def fold_case(self, expr: ibis.Value | ibis.Deferred) -> ibis.Value:
        """Apply different folding strategies to CASE WHHEN expressions.

        - If the CASE is a constant, it will evalute it immediately.
        - If the CASE is a IF ELSE statement returning 1 or 0,
          it will be converted to a boolean expression.
        - When the results and the default are the same, just return
          the default value.
        """
        if not isinstance(expr, ibis.Value):
            raise NotImplementedError("Deferred case expressions are not supported")

        if self.ENABLED is False:
            return expr

        op = expr.op()

        results_are_literals = all(
            isinstance(c, Literal) for c in itertools.chain([op.default], op.results)
        )
        possible_values = (
            set(itertools.chain([op.default.value], [c.value for c in op.results]))
            if results_are_literals
            else set()
        )

        if results_are_literals and len(possible_values) == 1:
            # All results and the default are literals with the same value.
            # It doesn't make any sense to have the case as it will always
            # lead to the same result.
            return self._ensure_expr(possible_values.pop())
        elif len(op.cases) == 1 and isinstance(op.cases[0], Literal):
            # It's only a IF ELSE statement, we can check the case
            # and eventually drop it if it's a constant.
            if op.cases[0].value:
                return op.results[0].to_expr()
            else:
                return op.default.to_expr()
        elif len(op.cases) == 1 and results_are_literals and possible_values == {1, 0}:
            # results are 1 or 0, we can fold it to a boolean expression.
            # FIXME: This doesn't work on postgresql so we need to disable it for the moment.
            return expr
            if op.results[0].value == 1:
                return (op.cases[0].to_expr()).cast("float64")
            else:
                return (~(op.cases[0].to_expr())).cast("float64")

        return expr

    def fold_cast(self, expr: ibis.Value) -> ibis.Value:
        """Given a cast expression, precompute it if possible."""
        if self.ENABLED is False:
            return expr

        op_instance = expr.op()
        if not isinstance(op_instance, ibis.expr.operations.Cast):
            # Not a cast, ignore
            # This can happen when a Field (a column) is casted to a type
            # and the Column is already of the same type.
            # Ibis seems to optimize this case and remove the cast.
            return expr

        target_type = op_instance.to
        arg = op_instance.arg

        if isinstance(arg, Literal):
            value = arg.value
            if target_type == dt.int64:
                return ibis.literal(int(value))
            elif target_type == dt.float64:
                return ibis.literal(float(value))
            elif target_type == dt.string:
                return ibis.literal(str(value))
            elif target_type == dt.boolean:
                return ibis.literal(bool(value))
            else:
                raise NotImplementedError(
                    f"Literal Cast to {target_type} not supported"
                )
        elif arg.dtype() == target_type:
            # The expression is already of the target type
            # No need to cast it again.
            return expr

        return expr

    def fold_zeros(self, expr: ibis.Expr) -> ibis.Expr:
        """Given a binary expression, precompute the result if it contains zeros.

        Operations like x + 0, x * 0, x - 0 etc can be folded in just x or 0
        without the need to compute the operation.
        """
        if self.ENABLED is False:
            return expr

        op = expr.op()
        inputs = op.args
        op_class = type(op)

        if op_class == Multiply:
            left_val = inputs[0].value if isinstance(inputs[0], Literal) else None
            right_val = inputs[1].value if isinstance(inputs[1], Literal) else None
            if left_val == 0 or right_val == 0:
                return ibis.literal(0)
        elif op_class == Add:
            left_val = inputs[0].value if isinstance(inputs[0], Literal) else None
            right_val = inputs[1].value if isinstance(inputs[1], Literal) else None
            if left_val == 0:
                return inputs[1].to_expr()
            elif right_val == 0:
                return inputs[0].to_expr()
        elif op_class == Subtract:
            left_val = inputs[0].value if isinstance(inputs[0], Literal) else None
            right_val = inputs[1].value if isinstance(inputs[1], Literal) else None
            if left_val == 0:
                return inputs[1].to_expr()
            elif right_val == 0:
                return inputs[0].to_expr()

        return expr

    def fold_operation(self, expr: ibis.Expr) -> ibis.Expr:
        """Given a node (an Ibis expression) fold constant expressions.

        If all node immediate children are constant (i.e. NumericScalar),
        compute the operation in Python and return a literal with the result.

        Otherwise, simply return the expression unchanged.

        This function assumes that constant folding has already been applied
        to the children.
        """
        if self.ENABLED is False:
            return expr

        if isinstance(expr, (int, float, str, bool)):
            # In some cases the operation has been computed in python.
            # For example when we try to compute * between a ONNX literal
            # and a previously folded expression.
            # In those case return a literal so we guarantee we always
            # return an Ibis expression
            return ibis.literal(expr)

        op = expr.op()
        inputs = op.args

        if not all(isinstance(child, Literal) for child in inputs):
            # We can only fold operations where all children are literals.
            # At least we can remove zeros if they exist.
            return self.fold_zeros(expr)

        op_class = type(op)
        if op_class in self.BINARY_OPS:
            left_val = inputs[0].value
            right_val = inputs[1].value
            result = self.BINARY_OPS[typing.cast(type[Binary], op_class)](
                left_val, right_val
            )
            return self._ensure_expr(result)
        elif op_class in self.UNARY_OPS and len(inputs) == 1:
            result = self.UNARY_OPS[typing.cast(type[Unary], op_class)](inputs[0].value)
            return self._ensure_expr(result)
        else:
            # No possible folding
            return expr

    def _debug(self, expr: ibis.Expr, show_args: bool = True) -> str:
        """Given an expression, return a string representation for debugging."""
        if isinstance(expr, Literal):
            return repr(expr.value)
        elif show_args is False:
            return type(expr).__name__
        elif not hasattr(expr, "args"):
            return f"{type(expr).__name__}(<unknown>)"
        else:
            return f"{type(expr).__name__}({', '.join([self._debug(a, show_args=False) for a in expr.args])})"
