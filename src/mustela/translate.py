"""Translate a pipeline into an Ibis expression."""

import logging
import typing

import ibis

from .ast import ParsedPipeline
from .translation.optimizer import Optimizer
from .translation.steps.add import AddTranslator
from .translation.steps.argmax import ArgMaxTranslator
from .translation.steps.arrayfeatureextractor import ArrayFeatureExtractorTranslator
from .translation.steps.cast import CastLikeTranslator, CastTranslator
from .translation.steps.concat import ConcatTranslator, FeatureVectorizerTranslator
from .translation.steps.div import DivTranslator
from .translation.steps.gather import GatherTranslator
from .translation.steps.identity import IdentityTranslator
from .translation.steps.imputer import ImputerTranslator
from .translation.steps.labelencoder import LabelEncoderTranslator
from .translation.steps.linearclass import LinearClassifierTranslator
from .translation.steps.linearreg import LinearRegressorTranslator
from .translation.steps.matmul import MatMulTranslator
from .translation.steps.mul import MulTranslator
from .translation.steps.onehotencoder import OneHotEncoderTranslator
from .translation.steps.reshape import ReshapeTranslator
from .translation.steps.scaler import ScalerTranslator
from .translation.steps.softmax import SoftmaxTranslator
from .translation.steps.sub import SubTranslator
from .translation.steps.trees import (
    TreeEnsembleClassifierTranslator,
    TreeEnsembleRegressorTranslator,
)
from .translation.steps.where import WhereTranslator
from .translation.steps.zipmap import ZipMapTranslator
from .translation.translator import Translator
from .translation.variables import GraphVariables

# This is a mapping of ONNX operations to their respective translators
# It could be implemented via some form of autodiscovery and
# registration, but explicit mapping avoids effects at a distance and
# makes it easier to understand the translation process.
TRANSLATORS: dict[str, type[Translator]] = {
    "Cast": CastTranslator,
    "CastLike": CastLikeTranslator,
    "Concat": ConcatTranslator,
    "FeatureVectorizer": FeatureVectorizerTranslator,
    "Sub": SubTranslator,
    "MatMul": MatMulTranslator,
    "Add": AddTranslator,
    "Div": DivTranslator,
    "Mul": MulTranslator,
    "Reshape": ReshapeTranslator,
    "Scaler": ScalerTranslator,
    "Gather": GatherTranslator,
    "ArrayFeatureExtractor": ArrayFeatureExtractorTranslator,
    "Identity": IdentityTranslator,
    "Imputer": ImputerTranslator,
    "LabelEncoder": LabelEncoderTranslator,
    "OneHotEncoder": OneHotEncoderTranslator,
    "Where": WhereTranslator,
    "ZipMap": ZipMapTranslator,
    "ArgMax": ArgMaxTranslator,
    "Softmax": SoftmaxTranslator,
    "TreeEnsembleClassifier": TreeEnsembleClassifierTranslator,
    "TreeEnsembleRegressor": TreeEnsembleRegressorTranslator,
    "LinearRegressor": LinearRegressorTranslator,
    "LinearClassifier": LinearClassifierTranslator,
}

log = logging.getLogger(__name__)

# This is primarily for development purposes.
# It's disabled by default because it implies
# a significant cost of executing queries on each step.
LOG_DATA = False
LOG_SQL = False


def translate(table: ibis.Table, pipeline: ParsedPipeline) -> ibis.Table:
    """Translate a pipeline into an Ibis expression.

    This function takes a pipeline and a table and translates the pipeline
    into an Ibis expression applied to the table.

    It is possible to further chain operations on the result
    to allow post processing of the prediction.
    """
    optimizer = Optimizer(enabled=True)
    features = {colname: table[colname] for colname in table.columns}
    variables = GraphVariables(features, pipeline._model.graph)
    nodes = list(pipeline._model.graph.node)
    for node in nodes:
        op_type = node.op_type
        if op_type not in TRANSLATORS:
            raise NotImplementedError(f"Translation for {op_type} not implemented")
        translator = TRANSLATORS[op_type](table, node, variables, optimizer)  # type: ignore[abstract]
        _log_debug_start(translator, variables)
        translator.process()
        table = translator.mutated_table  # Translator might return a new table.
        _log_debug_end(translator, variables)
    return _projection_results(table, variables)


def _projection_results(table: ibis.Table, variables: GraphVariables) -> ibis.Table:
    # As we pop out the variables as we use them
    # the remaining ones are the values resulting from all
    # graph branches.
    final_projections = {}
    for key, value in variables.remaining().items():
        if isinstance(value, dict):
            for field in value:
                colkey = key + "." + field
                colvalue = value[field]
                if isinstance(colvalue, ibis.expr.types.StructColumn):
                    raise NotImplementedError(f"StructColumn not supported: {colvalue}")
                final_projections[colkey] = colvalue
        else:
            final_projections[key] = value
    return table.mutate(**final_projections).select(final_projections.keys())


def _log_debug_start(translator: Translator, variables: GraphVariables) -> None:
    debug_inputs = {}
    node = translator._node
    for inp in translator._inputs:
        value: typing.Any = None
        if (feature_value := translator._variables.peek_variable(inp)) is not None:
            value = type(feature_value)
        elif initializer := translator._variables.get_initializer_value(inp):
            value = initializer
        else:
            raise ValueError(
                f"Unknow input: {inp} for node {node.name}({translator.__class__.__name__})"
            )
        debug_inputs[inp] = value
    log.debug(
        f"Node: {node.name}, Op: {node.op_type}, Attributes: {translator._attributes}, Inputs: {debug_inputs}"
    )
    if LOG_DATA:
        print("Input Data", flush=True)
        print(
            _projection_results(translator.mutated_table, variables).execute(),
            flush=True,
        )
        print("", flush=True)


def _log_debug_end(translator: Translator, variables: GraphVariables) -> None:
    variables = translator._variables
    output_vars = {
        name: type(variables.peek_variable(name)) for name in translator.outputs
    }
    log.debug(
        f"\tOutput: {output_vars} TOTAL: {variables.nested_len()}/{len(variables)}"
    )

    if LOG_DATA:
        print("\tOutput Data", flush=True)
        print(
            _projection_results(translator.mutated_table, variables).execute(),
            flush=True,
        )
        print("", flush=True)
    if LOG_SQL:
        print("\tSQL Expressions", flush=True)
        print(
            ibis.duckdb.connect().compile(
                (_projection_results(translator.mutated_table, variables))
            ),
            flush=True,
        )
