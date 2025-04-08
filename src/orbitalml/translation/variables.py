"""Define the variables and group of variables used in the translation process."""

import typing

import ibis
import onnx

from .._utils import onnx as onnx_utils
from .._utils.onnx import VariableTypes

VariablesGroupVarT = typing.TypeVar("VariablesGroupVarT", bound=ibis.Expr)


class VariablesGroup(dict[str, VariablesGroupVarT], typing.Generic[VariablesGroupVarT]):
    """A group of variables that can be used to represent a set of expressions.

    This is used to represent a group of columns in a table,
    the group will act as a single entity on which expressions will
    be applied.

    If an expression is applied to the group, it will be applied to all
    columns in the group.
    """

    VAR_TYPE = ibis.Expr

    def __init__(self, vargroup: dict | None = None) -> None:
        """
        :param vargroup: A dictionary of names and expressions that are part of the group.
        """
        if vargroup is not None:
            for expr in vargroup.values():
                if not isinstance(expr, self.VAR_TYPE):
                    raise TypeError(f"Expected {self.VAR_TYPE} value, got {type(expr)}")
            args = [vargroup]
        else:
            args = []

        super().__init__(*args)

    def __setitem__(self, key: str, value: VariablesGroupVarT, /) -> None:
        if not isinstance(value, self.VAR_TYPE):
            raise TypeError(f"Expected {self.VAR_TYPE} value, got {type(value)}")
        return super().__setitem__(key, value)

    def as_value(self, name: str) -> ibis.Value:
        """Return a subvariable as a Value.

        Values are expressions on top of which operations
        like comparions, mathematical operations, etc. can be applied.
        """
        value = self[name]
        if not isinstance(value, ibis.Value):
            raise TypeError(f"Expected value, got {type(value)}")
        return value

    def values_value(self) -> list[ibis.Value]:
        """Return all subvariables as a list of Values."""
        values = list(self.values())
        for value in values:
            if not isinstance(value, ibis.Value):
                raise TypeError(f"Expected value, got {type(value)}")
        return typing.cast(list[ibis.Value], values)


class ValueVariablesGroup(VariablesGroup[ibis.expr.types.Value]):
    """A group of value variables that can be used to represent a set of values.

    This is used to represent a group of columns in a table,
    the group will act as a single entity on which expressions will
    be applied.

    If an expression is applied to the group, it will be applied to all
    columns in the group.
    """

    VAR_TYPE = ibis.expr.types.Value


class NumericVariablesGroup(VariablesGroup[ibis.expr.types.NumericValue]):
    """A group of numeric variables that can be used to represent a set of numeric values.

    This is used to represent a group of numeric columns in a table,
    steps that expect to be able to perform mathematical operations
    over a variables group will create a NumericVariablesGroup
    from it, so that it is guaranteed that all subvariables are numeric.
    """

    VAR_TYPE = ibis.expr.types.NumericValue


class GraphVariables:
    """A class to manage the variables used in the translation process.

    This class is responsible for managing the variables and constants
    used in the translation process. It keeps track of the variables
    that have been consumed and the variables that are still available.

    When a variable is consumed it will be hidden from the list of
    available variables. This makes sure that the remaining variables
    that were not consumed are only the variables that should appear
    in the output (as they were set with no one consuming them afterward).

    This class also manages constants (initializers) that are used in the translation
    process. When consuming a variable, it could be both a constant or a variable.
    But if its a constant it won't actually be consumed as constants never
    appear in the output and thus it will be available for other nodes that
    need it.
    """

    def __init__(self, table: ibis.Table, graph: onnx.GraphProto) -> None:
        """
        :param table: The table the variables came from.
        :param graph: The pipeline graph requiring the variables and providing the constants.
        """
        self._initializers: dict[str, onnx.TensorProto] = {
            init.name: init for init in graph.initializer
        }
        self._initializers_values: dict[str, VariableTypes] = {
            init.name: onnx_utils.get_initializer_data(init)
            for init in graph.initializer
        }
        self._variables: dict[str, ibis.Expr | VariablesGroup] = {
            inp.name: table[inp.name] for inp in graph.input
        }
        self._consumed: set[str] = set()
        self._uniqueid: int = 0

    def consume(self, name: str) -> ibis.Expr | VariableTypes | VariablesGroup:
        """Consume a variable or a constant.

        Return a python value for constants and an Expression
        or VariablesGroup for variables.

        When a variable is consumed it will be hidden from the list of
        remaining variables.
        """
        constant_value = self._initializers_values.get(name)
        if constant_value is not None:
            return constant_value

        self._consumed.add(name)
        return self._variables[name]

    def peek_variable(
        self, name: str, default: None | ibis.Expr = None
    ) -> ibis.Expr | VariablesGroup | None:
        """Peek a variable without consuming it."""
        return self._variables.get(name, default)

    def get_initializer(
        self, name: str, default: None | onnx.TensorProto = None
    ) -> onnx.TensorProto | None:
        """Get an initializer by name."""
        return self._initializers.get(name, default)

    def get_initializer_value(
        self, name: str, default: None | VariableTypes = None
    ) -> VariableTypes | None:
        """Get a constant value."""
        return self._initializers_values.get(name, default)

    def keys(self) -> list[str]:
        """Name of all the variables that were not consumed."""
        return [f for f in self._variables if f not in self._consumed]

    def __setitem__(self, key: str, value: ibis.Expr | VariablesGroup, /) -> None:
        self._variables[key] = value
        self._consumed.discard(key)

    def __contains__(self, key: str) -> bool:
        return key in self._variables and key not in self._consumed

    def __len__(self) -> int:
        return len(self.keys())

    def nested_len(self) -> int:
        """Get total amount of variables and subvariables"""
        total = 0
        for name in self._variables:
            if name not in self._consumed:
                var = self._variables[name]
                if isinstance(var, VariablesGroup):
                    total += len(var)
                else:
                    total += 1
        return total

    def remaining(self) -> dict[str, ibis.Expr | VariablesGroup]:
        """Return the variables that were not consumed."""
        return {
            name: self._variables[name]
            for name in self._variables
            if name not in self._consumed
        }

    def generate_unique_shortname(self) -> str:
        """Generate a unique short name for a variable."""
        self._uniqueid += 1
        return f"v{self._uniqueid}"
