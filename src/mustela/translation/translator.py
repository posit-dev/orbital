import abc

import onnx

from .._utils import onnx as onnx_utils
from .optimizer import Optimizer
from .variables import GraphVariables


class Translator(abc.ABC):
    def __init__(self, node: onnx.NodeProto, variables: GraphVariables, optimizer: Optimizer=None):
        self._variables = variables
        self._node = node
        self._optimizer = optimizer
        self._inputs = node.input
        self._outputs = node.output
        self._output_name = node.output[0]  # most nodes have a single output, this is convenient.
        self._attributes = {attr.name: onnx_utils.get_attr_value(attr) for attr in  node.attribute}

    @abc.abstractmethod
    def process(self) -> None:
        pass

    @property
    def operation(self):
        return self._node.op_type

    @property
    def inputs(self):
        return [str(i) for i in self._inputs]

    @property
    def outputs(self):
        return [str(o) for o in self._outputs]

    def alias_output(self, value):
        if isinstance(value, dict):
            new_dict = {}
            for name, expr in value.items():
                cache_name = f"{self._output_name}_{name}"
                new_dict[name] = expr
            value = self._variables.alias(new_dict)
        else:            
            cache_name = f"{self._output_name}"
            value = self._variables.alias({cache_name: value})[cache_name]
        self.set_output(value)

    def set_output(self, value):
        if len(self.outputs) > 1:
            raise ValueError("Translator has more than one output")
        self._variables[self._output_name] = value

    def preserve(self, *variables):
        for v in variables:
            if v.get_name() in self._variables._table.columns:
                raise ValueError(
                    "Preserve variable already exists in the table: "
                    f"{v.get_name()}"
                )

        mutate_args = {
            v.get_name(): v
            for v in variables
        }
        self._variables._table = self._variables._table.mutate(**mutate_args)
        for v in mutate_args.keys():
            self._variables[v] = None
        
    def variable_unique_short_alias(self, prefix=None):
        shortname = self._variables.generate_unique_shortname()
        if prefix:
            shortname = f"{prefix}_{shortname}"
        return shortname