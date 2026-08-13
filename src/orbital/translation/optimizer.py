"""Implement optiomizations to the Ibis expression tree.

Primarily it takes care of folding constant expressions,
removing unnecessary casts and preserving the expressions
that are too big to be repeated inline.
"""

import itertools
import logging
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
    Field,
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
    Node,
    Not,
    NotEquals,
    Or,
    Relation,
    Subtract,
    Unary,
    Xor,
)
from ibis.expr.types import NumericScalar

from .variables import GraphVariables, VariablesGroup

if typing.TYPE_CHECKING:
    # Translators own the Optimizer, so the import can only happen
    # when type checking to avoid a circular import at runtime.
    from .translator import Translator

log = logging.getLogger(__name__)

# Alias to avoid long references when checking for nested casts.
CastOp = ibis.expr.operations.Cast


class Optimizer:
    """Optimizer for Ibis expressions.

    This class is responsible for applying a set of optimization
    processes to Ibis expressionsto remove unecessary operations and
    reduce query complexity.
    """

    # Expressions bigger than this get preserved before being consumed.
    # Measured over all examples, no example regresses in the 59..81 window:
    # at 58 cheap expressions gain pointless columns (+151 bytes on the
    # decision tree classifier pipeline), at 82 layered models explode again
    # (sklearn MLP with two 32 neuron layers goes from 72.9KB to 877KB).
    PRESERVE_THRESHOLD = 70

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

    def preserve_oversized_expressions(
        self, translator: "Translator", variables: GraphVariables
    ) -> None:
        """Preserve the node inputs that are too big to be repeated inline.

        An expression that is not preserved is re-emitted in the SQL text once
        per consumer, so layered pipelines (a neural network reads every value
        of the previous layer in every neuron of the next one) grow
        exponentially with depth. Preserving the oversized inputs first lets
        the node reference plain columns instead.

        :param translator: Translator whose node is about to be processed.
        :param variables: Variables of the graph being translated.
        """
        if self.ENABLED is False:
            return

        sizes: dict[Node, int] = {}
        for name in translator.inputs:
            variable = variables.peek_variable(name)
            if variable is None:
                continue  # Initializers are constants, they are never preserved.
            group = variable if isinstance(variable, VariablesGroup) else None
            candidates = list(group.values()) if group is not None else [variable]
            if not all(isinstance(c, ibis.expr.types.Value) for c in candidates):
                continue
            values = typing.cast(list[ibis.expr.types.Value], candidates)
            biggest = max(self._estimate_sql_size(v.op(), sizes) for v in values)
            if biggest <= self.PRESERVE_THRESHOLD:
                continue
            log.debug(
                f"Preserving {name} ({biggest} nodes) before {translator.operation}"
            )
            preserved = translator.preserve(
                *(
                    value.name(translator.variable_unique_short_alias("prs"))
                    for value in values
                )
            )
            # Assigning marks the variable as unconsumed again, which would add a
            # stray column to the final projection. Safe only because this runs
            # before process() and every translator consumes all of its inputs.
            # The group type is preserved as it carries validation of the subvariables,
            # and the original order is kept as later steps address groups by position.
            variables[name] = (
                type(group)(dict(zip(group.keys(), preserved)))
                if group is not None
                else preserved[0]
            )

    def _estimate_sql_size(self, op: Node, sizes: dict[Node, int]) -> int:
        """Count the nodes the SQL compiler has to emit for an inlined expression.

        The descent stops at fields and relations because they compile to a
        column or table reference of constant length, no matter how big the
        expression that produced them was. Counting through them would instead
        report the size of the whole pipeline built so far.

        :param op: Root of the ibis operations DAG to measure.
        :param sizes: Memo of already measured operations, shared between calls.
        """

        def inlined_children(op: Node) -> tuple[Node, ...]:
            if isinstance(op, (Field, Relation)):
                return ()
            return op.__children__

        # Iterative post-order visit: expressions are deeper than the Python stack.
        visited: set[Node] = set()
        pending = [(op, False)]
        while pending:
            current, expanded = pending.pop()
            if expanded:
                sizes[current] = 1 + sum(sizes[c] for c in inlined_children(current))
            elif current not in sizes and current not in visited:
                visited.add(current)
                pending.append((current, True))
                pending.extend((child, False) for child in inlined_children(current))
        return sizes[op]

    def _ensure_expr(self, value: typing.Any) -> ibis.Expr:
        """Ensure that the value is an Ibis expression.

        Literal operation nodes and bare Python scalars are wrapped in
        ``ibis.literal`` so downstream code can always treat the result as
        an ``ibis.Expr`` (e.g. to call ``.name(...)`` on it).
        """
        if isinstance(value, ibis.Expr):
            return value
        if isinstance(value, Literal):
            return ibis.literal(value.value)
        if isinstance(value, (int, float, bool, str, bytes)) or value is None:
            return ibis.literal(value)
        raise TypeError(
            f"Optimizer._ensure_expr: unsupported value type {type(value).__name__!r} "
            f"({value!r}); expected ibis.Expr, ibis Literal, or a Python scalar."
        )

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

        if pyop not in {operator.add, operator.mul}:
            raise NotImplementedError(
                "Only addition and multiplication folding are supported."
            )

        resulting_items = []
        items = list(lst)
        folded_value = None

        while items:
            expr = items.pop(0)
            if isinstance(expr, NumericScalar):
                value = expr.op().value
                if folded_value is None:
                    folded_value = value
                else:
                    folded_value = pyop(folded_value, value)
            else:
                resulting_items.append(expr)

        if folded_value is not None:
            resulting_items.append(ibis.literal(folded_value))

        return resulting_items

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

    def fold_case(self, expr: typing.Union[ibis.Value, ibis.Deferred]) -> ibis.Value:
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
        if not isinstance(op_instance, CastOp):
            # Not a cast, ignore
            # This can happen when a Field (a column) is casted to a type
            # and the Column is already of the same type.
            # Ibis seems to optimize this case and remove the cast.
            return expr

        target_type = op_instance.to
        arg_op = op_instance.arg

        # Collapse nested casts so we only apply the outermost cast once.
        while isinstance(arg_op, CastOp):
            arg_op = arg_op.arg

        if isinstance(arg_op, Literal):
            value = arg_op.value
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

        arg_expr = arg_op.to_expr()
        if arg_expr.type() == target_type:
            # The expression is already of the target type
            # No need to cast it again.
            return arg_expr

        return arg_expr.cast(target_type)

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
                return -inputs[1].to_expr()
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

    def _debug(  # pragma: no cover
        self, expr: ibis.Expr, show_args: bool = True
    ) -> str:
        """Given an expression, return a string representation for debugging.

        Only used on demand while developing the optimizer, so it is
        intentionally left uncovered by the test suite.
        """
        if isinstance(expr, Literal):
            return repr(expr.value)
        elif show_args is False:
            return type(expr).__name__
        elif not hasattr(expr, "args"):
            return f"{type(expr).__name__}(<unknown>)"
        else:
            return f"{type(expr).__name__}({', '.join([self._debug(a, show_args=False) for a in expr.args])})"
