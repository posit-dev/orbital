import ibis
import onnx

from .._utils import onnx as onnx_utils


class GraphVariables:
    def __init__(self, table: ibis.Table, graph: onnx.GraphProto):
        self._initializers: dict[str, onnx.TensorProto] = {
            init.name: init for init in graph.initializer
        }
        self._initializers_values: dict[str, onnx_utils.VariableTypes] = {
            init.name: onnx_utils.get_initializer_data(init)
            for init in graph.initializer
        }
        self._variables: dict[str, ibis.Column] = {
            inp.name: table[inp.name] for inp in graph.input
        }
        self._consumed: set[str] = set()
        self._uniqueid: int = 0

    def consume(self, name: str) -> ibis.Expr | onnx_utils.VariableTypes:
        if name in self._initializers:
            return self.get_initializer_value(name)

        self._consumed.add(name)
        return self._variables[name]

    def peek_variable(self, name, default=None):
        return self._variables.get(name, default)

    def get_initializer(self, name, default=None):
        return self._initializers.get(name, default)

    def get_initializer_value(self, name, default=None):
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

    def nested_len(self):
        total = 0
        for name in self._variables:
            if name not in self._consumed:
                if isinstance(self._variables[name], dict):
                    total += len(self._variables[name])
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
