"""Translate scikit-learn models to an intermediate represetation.

The IR is what will be processed to generate the SQL queries.
"""

import logging
import pickle
from typing import TYPE_CHECKING

import onnx as _onnx

from ._utils import repr_pipeline
from .types import ColumnType, FeaturesTypes

if TYPE_CHECKING:
    import sklearn.pipeline
    import torch

log = logging.getLogger(__name__)


class ParsedPipeline:
    """An intermediate representation of a scikit-learn pipeline.

    This object can be converted to a SQL query and run on a database.
    It can also be saved and loaded back in binary format for the sake
    of model distribution. Even though distributing the SQL query
    is usually more convenient.
    """

    _model: _onnx.ModelProto  # type: ignore[assignment]
    features: FeaturesTypes  # type: ignore[assignment]

    def __init__(self) -> None:
        """[orbital.ast.ParsedPipeline][] objects can only be created by the [orbital.ast.parse_pipeline][] function."""

        raise NotImplementedError(
            "parse_pipeline must be used to create a ParsedPipeline object."
        )

    @classmethod
    def _from_onnx_model(
        cls,
        model: _onnx.ModelProto,
        features: FeaturesTypes,
    ) -> "ParsedPipeline":
        """Create a [orbital.ast.ParsedPipeline][] from an ONNX model.

        This is considered an internal implementation detail
        as ONNX should never be exposed to the user.

        Returns a new [orbital.ast.ParsedPipeline][] instance.

        :param model: The ONNX model proto to wrap
        :param features: Dictionary mapping feature names to their [orbital.types.ColumnType][] objects
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

        Returns the validated features dictionary.

        :param features: Dictionary mapping feature names to their [orbital.types.ColumnType][] objects
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
        """Dump the parsed pipeline to a file.

        :param filename: Path to the file where the pipeline will be saved
        """
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
        """Load a parsed pipeline from a file.

        Returns a [orbital.ast.ParsedPipeline][] object loaded from the specified file.

        :param filename: Path to the file containing the saved pipeline
        """
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


def parse_pipeline(
    pipeline: "sklearn.pipeline.Pipeline", features: FeaturesTypes
) -> ParsedPipeline:
    """Parse a scikit-learn pipeline into an intermediate representation.

    Returns a [orbital.ast.ParsedPipeline][] object that can be converted to SQL queries.

    Requires scikit-learn, which can be installed with the ``orbital[sklearn]`` extra.

    :param pipeline: The fitted scikit-learn pipeline to parse
    :param features: Mapping of column names to their [orbital.types.ColumnType][] objects from the [orbital.types][] module

    ``features`` should be a mapping of column names that are the inputs of the
    pipeline to their types from the [orbital.types][] module:

    ```
        {
            "column_name": types.DoubleColumnType(),
            "another_column": types.Int64ColumnType()
        }
    ```
    """
    try:
        from . import _sklearn
    except ImportError as err:
        raise ImportError(
            "scikit-learn is required to parse scikit-learn pipelines. "
            "Install it with: pip install orbital[sklearn]"
        ) from err
    return _sklearn.parse_pipeline(pipeline, features)


def parse_pytorch_model(
    model: "torch.nn.Module", features: FeaturesTypes
) -> ParsedPipeline:
    """Parse a PyTorch model into an intermediate representation.

    Returns a [orbital.ast.ParsedPipeline][] object that can be converted to SQL queries.

    Requires PyTorch, which can be installed with the ``orbital[pytorch]`` extra.

    :param model: The trained PyTorch model to parse
    :param features: Mapping of column names to their [orbital.types.ColumnType][] objects from the [orbital.types][] module

    ``features`` should be a mapping of column names that are the inputs of the
    model to their types from the [orbital.types][] module:

    ```
        {
            "column_name": types.DoubleColumnType(),
            "another_column": types.DoubleColumnType()
        }
    ```

    As the model consumes a single input tensor, all features must be
    of the same type.
    """
    try:
        from . import _pytorch
    except ImportError as err:
        raise ImportError(
            "PyTorch is required to parse PyTorch models. "
            "Install it with: pip install orbital[pytorch]"
        ) from err
    return _pytorch.parse_pytorch_model(model, features)


class EnsureConcatenatedInputs:
    """Handle ONNX input tensor requirements for scikit-learn pipelines and PyTorch models.

    Models (as opposed to transformers) consume a single "input" tensor, while
    SQL provides individual columns. This class provides the logic to:

    1. Verify that all features share a type, so they can be concatenated
    2. Inject a Concat operation into the ONNX graph for SQL compatibility
    """

    def __init__(self, features: FeaturesTypes) -> None:
        """Initialize with the features dictionary.

        :param features: Dictionary mapping feature names to their [orbital.types.ColumnType][] objects
        """
        self.features = features

    def uniform_type(self) -> ColumnType:
        """Return the type shared by all features.

        Models expect a single concatenated input tensor, so all features
        must be of the same type for this to work.
        """
        feature_types = {type(ftype) for ftype in self.features.values()}

        if len(feature_types) != 1:
            # Mixed types not allowed for model input
            type_names = [t.__name__ for t in feature_types]
            raise ValueError(
                f"All features must be of the same type when pipeline starts with a model. "
                f"Found mixed types: {', '.join(sorted(type_names))}. "
                f"Please ensure all features use the same ColumnType."
            )

        return next(iter(self.features.values()))

    def inject_concat_step(self, onnx_model: _onnx.ModelProto) -> _onnx.ModelProto:
        """Inject a Concat operation for pipelines starting with models to enable SQL generation.

        Pipelines starting with models create a single "input" tensor, but SQL generation expects
        individual feature columns. This function modifies the ONNX graph to:

        1. Replace the single "input" with individual feature inputs
        2. Add a Concat operation to combine them back into "input"

        This bridges the gap between SQL (individual columns) and models (concatenated input).

        Returns the modified ONNX model with injected Concat operation.

        :param onnx_model: The ONNX model to modify
        """
        graph = onnx_model.graph

        # Verify this is a pipeline with "input" tensor
        input_names = [inp.name for inp in graph.input]
        if "input" not in input_names:
            # Not a model pipeline, return unchanged
            return onnx_model

        feature_names = list(self.features.keys())

        # Create new individual feature inputs
        new_inputs = []

        for fname in feature_names:
            # Create new input tensor for each feature with shape [None, 1]
            new_inputs.append(
                _onnx.helper.make_tensor_value_info(
                    fname,
                    self.features[fname]._onnx_elem_type,
                    [None, 1],
                )
            )

        # Create Concat node to combine individual features into "input"
        concat_node = _onnx.helper.make_node(
            "Concat",
            inputs=feature_names,  # Individual feature columns
            outputs=["input"],  # Combined input expected by the model
            axis=1,  # Concatenate along feature axis (columns)
            name="orbital_concat",
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
