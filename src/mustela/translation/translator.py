"""Base class for the translators of each pipeline step."""

import abc

import ibis
import onnx

from .._utils import onnx as onnx_utils
from .optimizer import Optimizer
from .variables import GraphVariables, VariablesGroup


class Translator(abc.ABC):
    """Base class for all translators.

    This class is responsible for translating pipeline steps into Ibis expressions.
    """

    def __init__(
        self,
        table: ibis.Table,
        node: onnx.NodeProto,
        variables: GraphVariables,
        optimizer: Optimizer,
    ) -> None:
        """
        :param table: The table the generated query should target.
        :param node: The pipeline node to be translated.
        :param variables: The variables used during the translation process.
        :param optimizer: The optimizer used for the translation.
        """
        self._table = table
        self._variables = variables
        self._node = node
        self._optimizer = optimizer
        self._inputs = node.input
        self._outputs = node.output
        self._attributes = {
            attr.name: onnx_utils.get_attr_value(attr) for attr in node.attribute
        }

    @abc.abstractmethod
    def process(self) -> None:
        """Performs the translation and set the output variable."""
        pass

    @property
    def operation(self) -> str:
        """What is the operation being translated"""
        return self._node.op_type

    @property
    def inputs(self) -> list[str]:
        """The input variables for this node"""
        return [str(i) for i in self._inputs]

    @property
    def outputs(self) -> list[str]:
        """The expected output variables the node should emit"""
        return [str(o) for o in self._outputs]

    @property
    def mutated_table(self) -> ibis.Table:
        """The table as it is being mutated by the translator.

        This is required for the translator to be able too set
        temporary variables that are not part of the final output.

        For example when an expression is used many times,
        the translator can create a temporary column in the
        SQL query to avoid recomputing the same expression.
        That leads to new columns being added to the table.
        """
        return self._table

    def set_output(
        self,
        value: ibis.Deferred | ibis.Expr | VariablesGroup | onnx_utils.VariableTypes,
        index: int = 0,
    ) -> None:
        """Set the output variable for the translator.

        This is only allowed if the translator has a single output.
        Otherwise the node is expected to explicitly set every variable.
        """
        if not isinstance(value, (ibis.Expr, VariablesGroup)):
            value = ibis.literal(value)
        self._variables[self.outputs[index]] = value

    def preserve(self, *variables) -> list[ibis.Expr]:
        """Preserve the given variables in the table.

        This causes the variables to be projected in the table,
        so that future expressions can use them instead of
        repeating the expression.
        """
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

    def variable_unique_short_alias(self, prefix: str | None = None) -> str:
        """Generate a unique short name for a variable.

        This is generally used to generate names for temporary variables
        that are used in the translation process.

        The names are as short as possible to minimize the
        SQL query length.
        """
        shortname = self._variables.generate_unique_shortname()
        if prefix:
            shortname = f"{prefix}_{shortname}"
        return shortname
