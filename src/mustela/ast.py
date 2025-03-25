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
    onnx_model = _skl2o.to_onnx(
        pipeline,
        initial_types=[
            (fname, ftype._to_onnxtype()) for fname, ftype in features.items()
        ],
    )
    return ParsedPipeline._from_onnx_model(onnx_model, features)


class UnsupportedFormatVersion(Exception):
    """Format of loaded pipeline is not supported.

    This usually happens when trying to load a newer
    format version with an older version of the framework.
    """

    pass
