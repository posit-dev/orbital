"""Implement optiomizations to the Ibis expression tree.

Primarily it takes care of folding constant expressions,
removing unnecessary casts and preserving the expressions
that other nodes reference.
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

    BINARY_OPS: dict[type[Binary], typing.Callable] = {
        # Mathematical Operators
        # _OptimizedOps is defined below Optimizer (Zen: public before
        # private), so it is invoked through a lambda here: the lambda body
        # is only evaluated on call, once the module has finished loading
        # and _OptimizedOps exists, rather than at this class body's
        # evaluation time.
        Add: lambda x, y: _OptimizedOps.add(x, y),
        Subtract: lambda x, y: _OptimizedOps.sub(x, y),
        Multiply: lambda x, y: _OptimizedOps.mul(x, y),
        Divide: lambda x, y: _OptimizedOps.div(x, y),
        FloorDivide: operator.floordiv,
        Modulus: operator.mod,
        # Logical Operators
        Equals: operator.eq,
        NotEquals: operator.ne,
        Greater: operator.gt,
        GreaterEqual: operator.ge,
        Less: operator.lt,
        LessEqual: operator.le,
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

    def preserve_referenced_outputs(
        self, translator: "Translator", variables: GraphVariables
    ) -> None:
        """Preserve the outputs of a node that other nodes reference.

        An expression that is not preserved is re-emitted in the SQL text once
        per reference, so layered pipelines (a neural network reads every value
        of the previous layer in every neuron of the next one) grow
        exponentially with depth. Preserving the referenced outputs as columns
        makes the nodes that follow reference them by name instead.

        :param translator: Translator whose node was just processed.
        :param variables: Variables of the graph being translated.
        """
        if self.ENABLED is False:
            return

        for name in translator.outputs:
            if variables.references(name) == 0:
                # Results of the pipeline are emitted once by the final
                # projection, so a column of their own would only add noise.
                continue
            variable = variables.peek_variable(name)
            group = variable if isinstance(variable, VariablesGroup) else None
            candidates = list(group.values()) if group is not None else [variable]
            if not all(isinstance(c, ibis.expr.types.Value) for c in candidates):
                continue
            values = typing.cast(list[ibis.expr.types.Value], candidates)
            if all(isinstance(v.op(), Field) for v in values):
                # A value that is already a column needs no column of its own,
                # re-aliasing it would only add noise to the query.
                continue
            log.debug(f"Preserving {name} emitted by {translator.operation}")
            preserved = translator.preserve(
                *(
                    value.name(translator.variable_unique_short_alias("prs"))
                    for value in values
                )
            )
            # The group type is preserved as it carries validation of the subvariables,
            # and the original order is kept as later steps address groups by position.
            variables[name] = (
                type(group)(dict(zip(group.keys(), preserved)))
                if group is not None
                else preserved[0]
            )

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

    def fold_operations(self, expr: ibis.Expr) -> ibis.Expr:
        """Given an Ibis expression, fold the constant parts of its whole tree.

        The tree is folded bottom-up, so a constant subtree is precomputed and
        the operations that combine it with the rest of the expression are
        dropped when they cannot change the result (``x + 0``, ``x * 1``, ...).
        Only adjacent constants get merged: the tree is never reordered, so
        ``column + 2 + 3`` keeps both literals while ``2 + 3 + column`` folds
        them into one.

        :param expr: The expression to fold.
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

        return self._ensure_expr(self._fold_node(expr.op()))

    def _fold_node(self, op: ibis.expr.operations.Node) -> typing.Any:
        """Fold a node, returning a Python constant when the whole subtree is constant.

        Returning a plain Python value for constant subtrees is what allows the
        operations in ``BINARY_OPS`` to serve both purposes: given two constants
        they compute the result, given a mix of constant and expression they
        rebuild the node.

        :param op: The operation node to fold.
        """
        if isinstance(op, Literal):
            return op.value

        op_class = type(op)
        if op_class in self.BINARY_OPS:
            return self.BINARY_OPS[typing.cast(type[Binary], op_class)](
                self._fold_node(op.args[0]), self._fold_node(op.args[1])
            )
        elif op_class in self.UNARY_OPS:
            arg = self._fold_node(op.args[0])
            if not isinstance(arg, ibis.Expr):
                return self.UNARY_OPS[typing.cast(type[Unary], op_class)](arg)
            # The unary functions are pure Python ones (``not``, ``ceil``, ...)
            # that raise when they are handed an Ibis expression, so the node is
            # kept as it was, forfeiting whatever was folded below it.
            return op.to_expr()

        # Any other node is left untouched without descending into it:
        # a column reference leads to tables, schemas and namespaces, which
        # hold nothing to fold and can amount to millions of nodes.
        return op.to_expr()

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


class _OptimizedOps:
    """Binary operations that fold constants and drop operands that do not change the result.

    They replace the plain ``operator`` equivalents in ``Optimizer.BINARY_OPS``
    for the operators that have an algebraic shortcut. Operands are either
    Python constants, when a whole subtree was folded, or Ibis expressions.
    """

    @staticmethod
    def add(x: typing.Any, y: typing.Any) -> typing.Any:
        xv, yv = _OptimizedOps._constant(x), _OptimizedOps._constant(y)
        if xv is not None and yv is not None:
            return xv + yv
        if xv == 0:
            # Adding zero cannot change the other operand.
            return y
        if yv == 0:
            return x
        return x + y

    @staticmethod
    def sub(x: typing.Any, y: typing.Any) -> typing.Any:
        xv, yv = _OptimizedOps._constant(x), _OptimizedOps._constant(y)
        if xv is not None and yv is not None:
            return xv - yv
        if yv == 0:
            return x
        if xv == 0:
            return -y
        return x - y

    @staticmethod
    def mul(x: typing.Any, y: typing.Any) -> typing.Any:
        xv, yv = _OptimizedOps._constant(x), _OptimizedOps._constant(y)
        if xv is not None and yv is not None:
            return xv * yv
        if xv == 0 or yv == 0:
            return 0
        if xv == 1:
            return y
        if yv == 1:
            return x
        return x * y

    @staticmethod
    def div(x: typing.Any, y: typing.Any) -> typing.Any:
        xv, yv = _OptimizedOps._constant(x), _OptimizedOps._constant(y)
        if xv is not None and yv is not None:
            return xv / yv
        if yv == 1:
            return x
        return x / y

    @staticmethod
    def _constant(operand: typing.Any) -> typing.Any:
        """Python value of a constant operand, ``None`` when it is not constant.

        Comparing an operand itself against a number is not an option, as the
        truth value of an Ibis expression is undefined and raises.

        :param operand: A Python value or an Ibis expression.
        """
        if isinstance(operand, (int, float, bool)):
            return operand
        if isinstance(operand, ibis.Expr) and isinstance(operand.op(), Literal):
            return operand.op().value
        return None
