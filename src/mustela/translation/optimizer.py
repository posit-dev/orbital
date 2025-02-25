import functools
import itertools
import operator

import ibis
import ibis.expr.datatypes as dt
from ibis.expr.operations import (
    Abs,
    Add,
    And,
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
    Xor,
)
from ibis.expr.types import NumericScalar


class Optimizer:
    BINARY_OPS = {
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

    UNARY_OPS = {
        Negate: operator.neg,
        Abs: operator.abs,
        Ceil: lambda x: float(operator.methodcaller("ceil")(x)),  # Se necessario
        Floor: lambda x: float(operator.methodcaller("floor")(x)),
        Not: operator.not_,
    }

    def __init__(self, enabled=True):
        self.ENABLED = enabled

    def debug_folding(self, expr):
        def _recurse(_expr):
            if isinstance(_expr, dict):
                return {k: _recurse(v) for k, v in _expr.items() if _recurse(v)}

            if hasattr(expr, "op"):
                expr_op = _expr.op()
                if isinstance(expr_op, Literal):
                    # Avoid printing literals, there is no folding necessary
                    return ""
                return self._debug(_expr.op())
            if isinstance(_expr, NumericScalar):
                return repr(_expr)
            else:
                return ""  # f"Unknown folding: {type(_expr)}"

        res = _recurse(expr)
        if res:
            print("Possible folding", res)

    def _ensure_expr(self, value):
        """Ensure that the value is an Ibis expression.

        Literal objects need to be converted back to an
        Ibis expression to be used in the query.
        """
        if isinstance(value, Literal):
            return ibis.literal(value.value)
        return value

    def _fold_associative_op_contiguous(self, lst, pyop):
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

    def fold_contiguous_sum(self, lst):
        return self._fold_associative_op_contiguous(lst, operator.add)

    def fold_contiguous_product(self, lst):
        return self._fold_associative_op_contiguous(lst, operator.mul)

    def fold_case(self, expr):
        if self.ENABLED is False:
            return expr

        op = expr.op()

        if all(isinstance(c, Literal) for c in op.cases):
            # print("Can fold case", [type(c).__name__ for c in op.cases])
            for cond, res in zip(op.cases, op.results):
                # print("\t", cond.value, res.value)
                if cond.value:
                    return self._ensure_expr(res)
            return self._ensure_expr(op.default)
        elif all(
            isinstance(c, Literal) for c in itertools.chain([op.default], op.results)
        ):
            # print("Want to fold case", expr, [op.default.value], [c.value for c in op.results])
            values = set(
                itertools.chain([op.default.value], [c.value for c in op.results])
            )
            if len(values) == 1:
                # print("Folding case to", dir(expr))
                return self._ensure_expr(values.pop())

        # print("Unable to fold", [type(c).__name__ for c in op.cases])
        return expr

    def fold_cast(self, expr):
        if self.ENABLED is False:
            return expr

        # TODO: When the expr is already of the target type do nothing
        op_instance = expr.op()
        target_type = op_instance.to
        arg = op_instance.arg
        if isinstance(arg, Literal):
            value = arg.value
            # print("Casting", value, "to", target_type, type(target_type), (target_type == dt.boolean))
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
        return expr

    def fold_zeros(self, expr):
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

        return expr

    def fold_operation(self, expr):
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

        # print(f"\t Trying to fold {type(op).__name__}({', '.join([self._debug(i) for i in inputs])})")
        if not all(isinstance(child, Literal) for child in inputs):
            # We can only fold operations where all children are literals.
            return self.fold_zeros(expr)

        op_class = type(op)
        if op_class in self.BINARY_OPS:
            left_val = inputs[0].value
            right_val = inputs[1].value
            result = self.BINARY_OPS[op_class](left_val, right_val)
            return self._ensure_expr(result)
        elif op_class in self.UNARY_OPS and len(inputs) == 1:
            result = self.UNARY_OPS[op_class](inputs[0].value)
            return self._ensure_expr(result)
        else:
            # No possible folding
            return expr

    def _debug(self, expr, show_args=True):
        if isinstance(expr, Literal):
            return repr(expr.value)
        elif show_args is False:
            return type(expr).__name__
        elif not hasattr(expr, "args"):
            return f"{type(expr).__name__}(<unknown>)"
        else:
            return f"{type(expr).__name__}({', '.join([self._debug(a, show_args=False) for a in expr.args])})"
