"""Translate scikit-learn models to an intermediate represetation.

The IR is what will be processed to generate the SQL queries.
"""

import logging
import pickle

import onnx as _onnx
import skl2onnx as _skl2o
import sklearn.pipeline

from ._utils import repr_pipeline
from .types import ColumnType, FeaturesTypes

log = logging.getLogger(__name__)


class ParsedPipeline:
    """An intermediate representation of a scikit-learn pipeline.

    This object can be converted to a SQL query and run on a database.
    In can also be saved and loaded back in binary format to the sake
    of model distribution. Even though distributing the SQL query
    is usually more convenient.
    """

    _model: _onnx.ModelProto  # type: ignore[assignment]
    features: FeaturesTypes  # type: ignore[assignment]

    def __init__(self) -> None:
        """ParsedPipeline objects can only be created by the parse_pipeline function."""

        raise NotImplementedError(
            "parse_pipeline must be used to create a ParsedPipeline object."
        )

    @classmethod
    def _from_onnx_model(
        cls, model: _onnx.ModelProto, features: FeaturesTypes
    ) -> "ParsedPipeline":
        """Create a ParsedPipeline from an ONNX model.

        This is considered an internal implementation detail
        as ONNX should never be exposed to the user.
        """
        self = super().__new__(cls)
        self._model = model
        self.features = self._validate_features(features)
        return self

    @classmethod
    def _validate_features(cls, features: FeaturesTypes) -> FeaturesTypes:
        """Validate the features of the pipeline.

        This checks that the features provided are compatible
        with what a SQL query can handle.
        """
        for name in features:
            if "." in name:
                raise ValueError(
                    f"Feature names cannot contain '.' characters: {name}, replace with '_'"
                )

        for ftype in features.values():
            if not isinstance(ftype, ColumnType):
                raise TypeError(f"Feature types must be ColumnType objects: {ftype}")

        return features

    def dump(self, filename: str) -> None:
        """Dump the parsed pipeline to a file."""
        # While the ONNX model is in protobuf format, and thus
        # it would make sense to use protobuf to serialize the
        # headers too. Using pickle avoids the need to define
        # a new protobuf schema for the headers and compile .proto files.
        header = {"version": 1, "features": self.features}
        header_data = pickle.dumps(header)
        header_len = len(header_data).to_bytes(4, "big")
        with open(filename, "wb") as f:
            f.write(header_len)
            f.write(header_data)
            f.write(self._model.SerializeToString())

    @classmethod
    def load(cls, filename: str) -> "ParsedPipeline":
        """Load a parsed pipeline from a file."""
        with open(filename, "rb") as f:
            header_len = int.from_bytes(f.read(4), "big")
            header_data = f.read(header_len)
            header = pickle.loads(header_data)
            if header["version"] != 1:
                # Currently there is only version 1
                raise UnsupportedFormatVersion("Unsupported format version.")
            model = _onnx.load_model(f)
        return cls._from_onnx_model(model, header["features"])

    def __str__(self) -> str:
        """Generate a string representation of the pipeline."""
        return str(repr_pipeline.ParsedPipelineStr(self))


def _pipeline_starts_with_model(pipeline: sklearn.pipeline.Pipeline) -> bool:
    """Check if the pipeline starts with a ML model that expects a single input tensor.

    Models (as opposed to transformers) expect a concatenated feature vector as input.
    This is different from transformers which can work with individual feature columns.
    """
    if not pipeline.steps:
        return False

    # Get the first step's estimator
    first_step_name, first_estimator = pipeline.steps[0]

    # Import classes for model identification
    # We do this inside the function to avoid top-level imports
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        RandomForestClassifier,
    )
    from sklearn.linear_model import (
        ElasticNet,
        Lasso,
        LinearRegression,
        LogisticRegression,
    )
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    # List of model classes that expect concatenated input
    model_classes = (
        LinearRegression,
        LogisticRegression,
        Lasso,
        ElasticNet,
        DecisionTreeClassifier,
        DecisionTreeRegressor,
        RandomForestClassifier,
        GradientBoostingClassifier,
        GradientBoostingRegressor,
    )

    return isinstance(first_estimator, model_classes)


def parse_pipeline(
    pipeline: sklearn.pipeline.Pipeline, features: FeaturesTypes
) -> ParsedPipeline:
    """Parse a scikit-learn pipeline into an intermediate representation.

    ``features`` should be a mapping of column names that are the inputs of the
    pipeline to their types from the :module:`.types` module::

        {
            "column_name": types.DoubleColumnType(),
            "another_column": types.Int64ColumnType()
        }

    """
    non_passthrough_features = {
        fname: ftype for fname, ftype in features.items() if not ftype.is_passthrough
    }

    if not non_passthrough_features:
        raise ValueError(
            "All provided features are passthrough. "
            "The pipeline would not do anything useful."
        )

    # Check if pipeline starts with a model (which expects concatenated input)
    pipeline_starts_with_model = _pipeline_starts_with_model(pipeline)

    if pipeline_starts_with_model:
        # Pipeline starts with a model - create concatenated input tensor
        # Models expect a single feature vector "input"

        # All features must be of the same type for model input
        feature_onnx_types = {
            type(ftype._to_onnxtype()) for ftype in non_passthrough_features.values()
        }

        if len(feature_onnx_types) != 1:
            # Mixed types not allowed for model input
            type_names = [t.__name__ for t in feature_onnx_types]
            raise ValueError(
                f"All features must be of the same type when pipeline starts with a model. "
                f"Found mixed types: {', '.join(sorted(type_names))}. "
                f"Please ensure all features use the same ColumnType."
            )

        # All features have the same type, use it for concatenated input
        uniform_type = next(iter(feature_onnx_types))
        initial_types = [("input", uniform_type([None, len(non_passthrough_features)]))]
    else:
        # Pipeline starts with transformers - use individual feature inputs
        initial_types = [
            (fname, ftype._to_onnxtype())
            for fname, ftype in non_passthrough_features.items()
        ]

    onnx_model = _skl2o.to_onnx(pipeline, initial_types=initial_types)

    # Inject concat operation for SQL compatibility when starting with model
    if pipeline_starts_with_model:
        onnx_model = _inject_concat_for_sql_compatibility(
            onnx_model, non_passthrough_features
        )

    return ParsedPipeline._from_onnx_model(onnx_model, features)


def _inject_concat_for_sql_compatibility(
    onnx_model: _onnx.ModelProto, non_passthrough_features: FeaturesTypes
) -> _onnx.ModelProto:
    """Inject a Concat operation for pipelines starting with models to enable SQL generation.

    Pipelines starting with models create a single "input" tensor, but SQL generation expects
    individual feature columns. This function modifies the ONNX graph to:
    1. Replace the single "input" with individual feature inputs
    2. Add a Concat operation to combine them back into "input"

    This bridges the gap between SQL (individual columns) and models (concatenated input).
    """
    graph = onnx_model.graph

    # Verify this is a pipeline with "input" tensor
    input_names = [inp.name for inp in graph.input]
    if "input" not in input_names:
        # Not a model pipeline, return unchanged
        return onnx_model

    # Get the original input tensor info for reference
    original_input = next(inp for inp in graph.input if inp.name == "input")
    original_elem_type = original_input.type.tensor_type.elem_type

    # Create new individual feature inputs
    new_inputs = []
    feature_names = list(non_passthrough_features.keys())

    for fname in feature_names:
        # Create new input tensor for each feature with shape [None, 1]
        new_input = _onnx.helper.make_tensor_value_info(
            fname,
            original_elem_type,  # Use same element type as original
            [None, 1],  # Individual feature column
        )
        new_inputs.append(new_input)

    # Create Concat node to combine individual features into "input"
    concat_node = _onnx.helper.make_node(
        "Concat",
        inputs=feature_names,  # Individual feature columns
        outputs=["input"],  # Combined input expected by the model
        axis=1,  # Concatenate along feature axis (columns)
        name="sql_compat_concat",
    )

    # Modify the graph
    # 1. Replace graph inputs with individual features
    del graph.input[:]
    graph.input.extend(new_inputs)

    # 2. Insert concat node at the beginning of the graph
    graph.node.insert(0, concat_node)

    return onnx_model


class UnsupportedFormatVersion(Exception):
    """Format of loaded pipeline is not supported.

    This usually happens when trying to load a newer
    format version with an older version of the framework.
    """

    pass
