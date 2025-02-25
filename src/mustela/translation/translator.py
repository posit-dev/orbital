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

    def set_output(self, value):
        if len(self.outputs) > 1:
            raise ValueError("Translator has more than one output")
        self._variables[self._output_name] = value
