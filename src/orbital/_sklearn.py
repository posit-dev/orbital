"""Parse scikit-learn pipelines through skl2onnx."""

import logging
import typing

import onnx as _onnx
import skl2onnx
import skl2onnx.common.data_types as _sl2o_types
import skl2onnx.convert
import sklearn.pipeline

from .ast import EnsureConcatenatedInputs, ParsedPipeline
from .types import ColumnType, FeaturesTypes

log = logging.getLogger(__name__)

TENSOR_TYPES: dict[int, type[_sl2o_types.DataType]] = {
    _onnx.TensorProto.FLOAT: _sl2o_types.FloatTensorType,
    _onnx.TensorProto.FLOAT16: _sl2o_types.Float16TensorType,
    _onnx.TensorProto.DOUBLE: _sl2o_types.DoubleTensorType,
    _onnx.TensorProto.STRING: _sl2o_types.StringTensorType,
    _onnx.TensorProto.INT64: _sl2o_types.Int64TensorType,
    _onnx.TensorProto.UINT64: _sl2o_types.UInt64TensorType,
    _onnx.TensorProto.INT32: _sl2o_types.Int32TensorType,
    _onnx.TensorProto.UINT32: _sl2o_types.UInt32TensorType,
    _onnx.TensorProto.INT16: _sl2o_types.Int16TensorType,
    _onnx.TensorProto.UINT16: _sl2o_types.UInt16TensorType,
    _onnx.TensorProto.INT8: _sl2o_types.Int8TensorType,
    _onnx.TensorProto.UINT8: _sl2o_types.UInt8TensorType,
    _onnx.TensorProto.BOOL: _sl2o_types.BooleanTensorType,
}
"""skl2onnx tensor type for each ONNX TensorProto element type."""


def parse_pipeline(
    pipeline: sklearn.pipeline.Pipeline, features: FeaturesTypes
) -> ParsedPipeline:
    """Parse a scikit-learn pipeline into a [orbital.ast.ParsedPipeline][].

    :param pipeline: The fitted scikit-learn pipeline to parse
    :param features: Mapping of column names to their [orbital.types.ColumnType][] objects
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
    concatenated_inputs = EnsureConcatenatedInputs(non_passthrough_features)
    requires_input_vector = pipeline_requires_input_vector(
        pipeline, non_passthrough_features
    )

    if requires_input_vector:
        # Models expect a single feature vector "input", so we need to adapt the user
        # features to a single concatenated input tensor.
        # Later, we'll inject a concat operation to ensure the SQL query does work
        # with individual columns.
        uniform_type = concatenated_inputs.uniform_type()
        initial_types = [
            (
                "input",
                TENSOR_TYPES[uniform_type._onnx_elem_type](
                    [None, len(non_passthrough_features)]
                ),
            )
        ]
    else:
        initial_types = [
            (fname, TENSOR_TYPES[ftype._onnx_elem_type]([None, 1]))
            for fname, ftype in non_passthrough_features.items()
        ]

    onnx_model = typing.cast(
        _onnx.ModelProto,
        skl2onnx.to_onnx(pipeline, initial_types=initial_types),  # type: ignore[arg-type]
    )

    if requires_input_vector:
        # Inject concat operation to create the "input" tensor when necessary.
        onnx_model = concatenated_inputs.inject_concat_step(onnx_model)

    return ParsedPipeline._from_onnx_model(onnx_model, features)


def pipeline_requires_input_vector(
    pipeline: sklearn.pipeline.Pipeline, features: FeaturesTypes
) -> bool:
    """Determine if pipeline requires concatenated inputs by testing operator compatibility.

    This directly tests whether the first operator in the pipeline can handle
    individual feature inputs by calling `infer_types`. If it fails, the operator
    requires concatenated inputs.

    Returns True if the pipeline requires concatenated inputs, False otherwise.

    :param pipeline: The scikit-learn pipeline to analyze
    :param features: Mapping of column names to their [orbital.types.ColumnType][] objects
    """
    individual_types = [
        (fname, TENSOR_TYPES[ftype._onnx_elem_type]([None, 1]))
        for fname, ftype in features.items()
    ]

    topology = skl2onnx.convert.parse_sklearn_model(
        pipeline, initial_types=individual_types
    )

    if len(features) <= 1:
        # The user provided only one feature, no need for concatenation
        return False

    # Get the first operator in the topology
    first_operator = next(topology.unordered_operator_iterator(), None)
    if not first_operator:
        return False

    # Test if the operator can handle the individual inputs we provided
    try:
        first_operator.infer_types()
        # If infer_types() succeeds, the operator accepts the inputs the user provided
        return False
    except RuntimeError as err:
        if "at most 1 input" in str(err):
            # If infer_types() fails with "at most 1 input", the operator needs concatenated inputs
            # This is the best we can do as SKL2ONNX doesn't tell us how many inputs it expects.
            # And the `check_input_and_output_numbers` function always throws a RuntimeError
            return True
        return False


def guess_datatypes(dataframe: typing.Any) -> FeaturesTypes:
    """Given a DataFrame, guess the [orbital.types.FeaturesTypes][] of its columns.

    :param dataframe: A Pandas, Polars or PyArrow dataframe
    """
    if hasattr(dataframe, "to_pandas"):
        # Easiest way to ensure compatibility with Polars, Pandas and PyArrow.
        dataframe = dataframe.to_pandas()

    try:
        dtypes = _sl2o_types.guess_data_type(dataframe)
    except (TypeError, NotImplementedError) as exc:
        log.debug(f"Unable to guess types from {repr(dataframe)}, exception: {exc}")
        raise ValueError("Unable to guess types of dataframe") from None

    typesmap: FeaturesTypes = {}
    for name, dtype in dtypes:
        try:
            if dtype.shape != [None, 1]:
                raise ValueError("Only columnar data is supported.")
            typesmap[name] = ColumnType._from_onnx_elem_type(
                dtype.to_onnx_type().tensor_type.elem_type
            )
        except (ValueError, TypeError, AttributeError) as exc:
            log.debug(
                f"Unable to convert to column type from {name}:{repr(dtype)}, exception: {exc}"
            )
            raise ValueError(f"Unsupported datatype for column {name}") from None
    return typesmap
