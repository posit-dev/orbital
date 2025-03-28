import typing

import ibis
import onnx

from .._utils import onnx as onnx_utils


class VariablesGroup(dict[str, ibis.Expr]):
    def __init__(self, vargroup: dict|None = None):
        if vargroup is not None:
            for expr in vargroup.values():
                if not isinstance(expr, ibis.Expr):
                    raise TypeError(f"Expected numeric value, got {type(expr)}")
            args = (vargroup, )
        else:
            args = ()
        
        super().__init__(*args)

    def as_value(self, name:str) -> ibis.Value:
        value = self[name]
        if not isinstance(value, ibis.Value):
            raise TypeError(f"Expected value, got {type(value)}")
        return value

    def values_value(self) -> list[ibis.Value]:
        values = list(self.values())
        for value in values:
            if not isinstance(value, ibis.Value):
                raise TypeError(f"Expected value, got {type(value)}")
        return typing.cast(list[ibis.Value], values)


class NumericVariablesGroup(VariablesGroup):
    def __init__(self, vargroup: VariablesGroup):
        for expr in vargroup.values():
            if not isinstance(expr, ibis.expr.types.NumericValue):
                raise TypeError(f"Expected numeric value, got {type(expr)}")
        super().__init__(vargroup)

    def __setitem__(self, key: str, value: ibis.expr.types.NumericValue, /) -> None:
        if not isinstance(value, ibis.expr.types.NumericValue):
            raise TypeError(f"Expected numeric value, got {type(value)}")
        return super().__setitem__(key, value)
    
    def __getitem__(self, key: str, /) -> ibis.expr.types.NumericValue:
        return super().__getitem__(key)


class GraphVariables:
    def __init__(self, table: ibis.Table, graph: onnx.GraphProto):
        self._initializers: dict[str, onnx.TensorProto] = {
            init.name: init for init in graph.initializer
        }
        self._initializers_values: dict[str, onnx_utils.VariableTypes] = {
            init.name: onnx_utils.get_initializer_data(init)
            for init in graph.initializer
        }
        self._variables: dict[str, ibis.Expr|VariablesGroup] = {
            inp.name: table[inp.name] for inp in graph.input
        }
        self._consumed: set[str] = set()
        self._uniqueid: int = 0

    def consume(self, name: str) -> ibis.Expr | onnx_utils.VariableTypes | VariablesGroup:
        constant_value = self._initializers_values.get(name)
        if constant_value is not None:
            return constant_value
        
        self._consumed.add(name)
        return self._variables[name]

    def peek_variable(self, name, default=None) -> ibis.Expr|VariablesGroup | None:
        return self._variables.get(name, default)

    def get_initializer(self, name, default=None) -> onnx.TensorProto | None:
        return self._initializers.get(name, default)

    def get_initializer_value(self, name, default=None) -> onnx_utils.VariableTypes | None:
        return self._initializers_values.get(name, default)

    def keys(self):
        return [f for f in self._variables if f not in self._consumed]

    def __setitem__(self, key, value):
        self._variables[key] = value
        self._consumed.discard(key)

    def __contains__(self, key):
        return key in self._variables and key not in self._consumed

    def __len__(self):
        return len(self.keys())

    def nested_len(self) -> int:
        total = 0
        for name in self._variables:
            if name not in self._consumed:
                var = self._variables[name]
                if isinstance(var, VariablesGroup):
                    total += len(var)
                else:
                    total += 1
        return total

    def remaining(self):
        return {
            name: self._variables[name]
            for name in self._variables
            if name not in self._consumed
        }

    def generate_unique_shortname(self):
        """
        Generate a unique short name for a variable.
        """
        self._uniqueid += 1
        return f"v{self._uniqueid}"
