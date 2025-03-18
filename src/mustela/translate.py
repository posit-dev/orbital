import logging

import ibis

from .ast import ParsedPipeline
from .translation.optimizer import Optimizer
from .translation.steps.add import AddTranslator
from .translation.steps.arrayfeatureextractor import ArrayFeatureExtractorTranslator
from .translation.steps.cast import CastLikeTranslator, CastTranslator
from .translation.steps.concat import ConcatTranslator, FeatureVectorizerTranslator
from .translation.steps.div import DivTranslator
from .translation.steps.gather import GatherTranslator
from .translation.steps.identity import IdentityTranslator
from .translation.steps.imputer import ImputerTranslator
from .translation.steps.labelencoder import LabelEncoderTranslator
from .translation.steps.matmul import MatMulTranslator
from .translation.steps.onehotencoder import OneHotEncoderTranslator
from .translation.steps.reshape import ReshapeTranslator
from .translation.steps.sub import SubTranslator
from .translation.steps.trees import (
    TreeEnsembleClassifierTranslator,
    TreeEnsembleRegressorTranslator,
)
from .translation.steps.where import WhereTranslator
from .translation.steps.zipmap import ZipMapTranslator
from .translation.variables import GraphVariables

# This is a mapping of ONNX operations to their respective translators
# It could be implemented via some form of autodiscovery and
# registration, but explicit mapping avoids effects at a distance and 
# makes it easier to understand the translation process.
TRANSLATORS = {
    "Cast": CastTranslator,
    "CastLike": CastLikeTranslator,
    "Concat": ConcatTranslator,
    "FeatureVectorizer": FeatureVectorizerTranslator,
    "Sub": SubTranslator,
    "MatMul": MatMulTranslator,
    "Add": AddTranslator,
    "Div": DivTranslator,
    "Reshape": ReshapeTranslator,
    "Gather": GatherTranslator,
    "ArrayFeatureExtractor": ArrayFeatureExtractorTranslator,
    "Identity": IdentityTranslator,
    "Imputer": ImputerTranslator,
    "LabelEncoder": LabelEncoderTranslator,
    "OneHotEncoder": OneHotEncoderTranslator,
    "Where": WhereTranslator,
    "ZipMap": ZipMapTranslator,
    "TreeEnsembleClassifier": TreeEnsembleClassifierTranslator,
    "TreeEnsembleRegressor": TreeEnsembleRegressorTranslator,
}

log = logging.getLogger(__name__)


def translate(table: ibis.Table, pipeline: ParsedPipeline) -> ibis.Table:
    optimizer = Optimizer(enabled=True)
    variables = GraphVariables(table, pipeline._model.graph)
    nodes = {node.name: node for node in pipeline._model.graph.node}
    for node_name, node in nodes.items():
        op_type = node.op_type
        if op_type not in TRANSLATORS:
            raise NotImplementedError(f"Translation for {op_type} not implemented")
        translator = TRANSLATORS[op_type](node, variables, optimizer)
        _log_debug_start(translator)
        translator.process()
        _log_debug_end(translator)
    return _projection_results(table, variables)


def translate_sqlglot(table: ibis.Table, pipeline: ParsedPipeline):
    ibis_expr = translate(table, pipeline)
    # sqlglot_schema = table.schema().to_sqlglot(dialect="duckdb")
        
    import ibis.backends.sql.compilers as sc
    sqlglot_expr = sc.duckdb.compiler.to_sqlglot(ibis_expr.unbind())

    sqlglot_catalog = {
        table.get_name(): table
    }
    from ibis.expr.sql import Catalog
    catalog = Catalog(sqlglot_catalog)

    import sqlglot.optimizer
    sqlglot_expr = sqlglot.optimizer.optimize(sqlglot_expr, schema=catalog.to_sqlglot())

    return sqlglot_expr.sql(dialect="duckdb")

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
                    # This happens with tree regressor probabilities
                    # Probably need to fix. It's a concatenated column
                    # that containes a concatenated column
                    # This should never happen
                    colkey = colkey + "." + field
                    colvalue = colvalue[field]
                final_projections[colkey] = colvalue
        else:
            final_projections[key] = value
    return table.select(**final_projections)


def _log_debug_start(translator):
    debug_inputs = {}
    node = translator._node
    for inp in translator._inputs:
        value = None
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

def _log_debug_end(translator):
    variables = translator._variables
    output_vars = {name: type(variables.peek_variable(name)) for name in translator.outputs}
    log.debug(f"\tOutput: {output_vars} TOTAL: {variables.nested_len()}/{len(variables)}")
