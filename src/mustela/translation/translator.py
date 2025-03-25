import abc

import ibis
import onnx

from .._utils import onnx as onnx_utils
from .optimizer import Optimizer
from .variables import GraphVariables


class Translator(abc.ABC):
    def __init__(
        self,
        table: ibis.Table,
        node: onnx.NodeProto,
        variables: GraphVariables,
        optimizer: Optimizer,
    ):
        self._table = table
        self._variables = variables
        self._node = node
        self._optimizer = optimizer
        self._inputs = node.input
        self._outputs = node.output
        self._output_name = node.output[
            0
        ]  # most nodes have a single output, this is convenient.
        self._attributes = {
            attr.name: onnx_utils.get_attr_value(attr) for attr in node.attribute
        }

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

    @property
    def mutated_table(self):
        return self._table

    def set_output(self, value):
        if len(self.outputs) > 1:
            raise ValueError("Translator has more than one output")
        self._variables[self._output_name] = value

    def preserve(self, *variables):
        for v in variables:
            if v.get_name() in self._table.columns:
                raise ValueError(
                    f"Preserve variable already exists in the table: {v.get_name()}"
                )

        mutate_args = {v.get_name(): v for v in variables}
        self._table = self._table.mutate(**mutate_args)

        # TODO: Should probably update self._variables too
        # in case the same variable is used in multiple places
        # but this is not a common case, and it's complex because
        # we don't know the variable name (!= column_name)
        # so we'll leave it for now.
        return [self._table[cname] for cname in mutate_args]

    def variable_unique_short_alias(self, prefix=None):
        shortname = self._variables.generate_unique_shortname()
        if prefix:
            shortname = f"{prefix}_{shortname}"
        return shortname
