"""Data types of the features processed by models."""

import abc
import inspect
import logging
import typing

import ibis.expr.datatypes as ibis_types
import skl2onnx.common.data_types as _sl2o_types

log = logging.getLogger(__name__)


class ColumnType(abc.ABC):
    """A base class representing the type of a column of data."""

    @abc.abstractmethod
    def _to_onnxtype(self) -> _sl2o_types.DataType:  # pragma: no cover
        """Convert the ColumnType to an onnx type.

        This should be implemented by all specific types.
        """
        pass

    @abc.abstractmethod
    def _to_ibistype(self) -> ibis_types.DataType:
        """Convert the ColumnType to an ibis type.

        This should be implemented by all specific types.
        """
        pass

    @staticmethod
    def _from_onnxtype(onnxtype: _sl2o_types.DataType) -> "ColumnType":
        """Given an onnx type, guess the right ColumnType."""
        if onnxtype.shape != [None, 1]:
            raise ValueError("Only columnar data is supported.")

        for scls in ColumnType.__subclasses__():
            supported_type = inspect.signature(scls._to_onnxtype).return_annotation
            if supported_type == onnxtype.__class__:
                return scls()  # type: ignore[abstract]
        else:
            raise TypeError(f"Unsupported data type {onnxtype.__class__.__name__}")

    def __eq__(self, other: object) -> bool:
        return self.__class__ == other.__class__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


FeaturesTypes = typing.Dict[str, ColumnType]


def guess_datatypes(dataframe: typing.Any) -> FeaturesTypes:
    """Given a DataFrame, try to guess the types of each feature in it.

    This procudes a :class:`.FeaturesTypes` dictionary that can be used by
    parse_pipeline to generate the SQL queries from the sklearn pipeline.

    In most cases this shouldn't be necessary as the user should know
    on what data the pipeline was trained on, but it can be convenient
    when experimenting or writing tests.
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
            typesmap[name] = ColumnType._from_onnxtype(dtype)
        except (ValueError, TypeError, AttributeError) as exc:
            log.debug(
                f"Unable to convert to column type from {name}:{repr(dtype)}, exception: {exc}"
            )
            raise ValueError(f"Unsupported datatype for column {name}") from None
    return typesmap


class FloatColumnType(ColumnType):
    """Mark a column as containing float values"""

    def _to_onnxtype(self) -> _sl2o_types.FloatTensorType:
        return _sl2o_types.FloatTensorType(shape=[None, 1])

    def _to_ibistype(self) -> ibis_types.Float32:
        return ibis_types.Float32()


class Float16ColumnType(ColumnType):
    """Mark a column as containing 16bit float values"""

    def _to_onnxtype(self) -> _sl2o_types.Float16TensorType:
        return _sl2o_types.Float16TensorType(shape=[None, 1])

    def _to_ibistype(self) -> ibis_types.Float16:
        return ibis_types.Float16()


class DoubleColumnType(ColumnType):
    """Mark a column as containing double values"""

    def _to_onnxtype(self) -> _sl2o_types.DoubleTensorType:
        return _sl2o_types.DoubleTensorType(shape=[None, 1])

    def _to_ibistype(self) -> ibis_types.Float64:
        return ibis_types.Float64()


class StringColumnType(ColumnType):
    """Mark a column as containing string values"""

    def _to_onnxtype(self) -> _sl2o_types.StringTensorType:
        return _sl2o_types.StringTensorType(shape=[None, 1])

    def _to_ibistype(self) -> ibis_types.String:
        return ibis_types.String()


class Int64ColumnType(ColumnType):
    """Mark a column as containing signed 64bit integer values"""

    def _to_onnxtype(self) -> _sl2o_types.Int64TensorType:
        return _sl2o_types.Int64TensorType(shape=[None, 1])

    def _to_ibistype(self) -> ibis_types.Int64:
        return ibis_types.Int64()


class UInt64ColumnType(ColumnType):
    """Mark a column as containing unsigned 64bit integer values"""

    def _to_onnxtype(self) -> _sl2o_types.UInt64TensorType:
        return _sl2o_types.UInt64TensorType(shape=[None, 1])

    def _to_ibistype(self) -> ibis_types.UInt64:
        return ibis_types.UInt64()


class Int32ColumnType(ColumnType):
    """Mark a column as containing signed 32bit integer values"""

    def _to_onnxtype(self) -> _sl2o_types.Int32TensorType:
        return _sl2o_types.Int32TensorType(shape=[None, 1])

    def _to_ibistype(self) -> ibis_types.Int32:
        return ibis_types.Int32()


class UInt32ColumnType(ColumnType):
    """Mark a column as containing unsigned 32bit integer values"""

    def _to_onnxtype(self) -> _sl2o_types.UInt32TensorType:
        return _sl2o_types.UInt32TensorType(shape=[None, 1])

    def _to_ibistype(self) -> ibis_types.UInt32:
        return ibis_types.UInt32()


class Int16ColumnType(ColumnType):
    """Mark a column as containing signed 16bit integer values"""

    def _to_onnxtype(self) -> _sl2o_types.Int16TensorType:
        return _sl2o_types.Int16TensorType(shape=[None, 1])

    def _to_ibistype(self) -> ibis_types.Int16:
        return ibis_types.Int16()


class UInt16ColumnType(ColumnType):
    """Mark a column as containing unsigned 16bit integer values"""

    def _to_onnxtype(self) -> _sl2o_types.UInt16TensorType:
        return _sl2o_types.UInt16TensorType(shape=[None, 1])

    def _to_ibistype(self) -> ibis_types.UInt16:
        return ibis_types.UInt16()


class Int8ColumnType(ColumnType):
    """Mark a column as containing signed 8bit integer values"""

    def _to_onnxtype(self) -> _sl2o_types.Int8TensorType:
        return _sl2o_types.Int8TensorType(shape=[None, 1])

    def _to_ibistype(self) -> ibis_types.Int8:
        return ibis_types.Int8()


class UInt8ColumnType(ColumnType):
    """Mark a column as containing unsigned 8bit integer values"""

    def _to_onnxtype(self) -> _sl2o_types.UInt8TensorType:
        return _sl2o_types.UInt8TensorType(shape=[None, 1])

    def _to_ibistype(self) -> ibis_types.UInt8:
        return ibis_types.UInt8()


class BooleanColumnType(ColumnType):
    """Mark a column as containing boolean values"""

    def _to_onnxtype(self) -> _sl2o_types.BooleanTensorType:
        return _sl2o_types.BooleanTensorType(shape=[None, 1])

    def _to_ibistype(self) -> ibis_types.Boolean:
        return ibis_types.Boolean()
