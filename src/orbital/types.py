"""Data types of the features processed by models."""

import abc
import typing

import ibis.expr.datatypes as ibis_types
import onnx as _onnx


class ColumnType(abc.ABC):
    """A base class representing the type of a column of data."""

    _onnx_elem_type: int
    """ONNX TensorProto element type shared by every parser."""

    def __init__(self, passthrough: bool = False) -> None:
        """
        :param passthrough: If True, the column is ignored by the pipeline and is only available to SQL generator.
                            You will still need to project those columns for them to be included in the SQL query.
        """
        self.is_passthrough = passthrough

    @abc.abstractmethod
    def _to_ibistype(self) -> ibis_types.DataType:
        """Convert the ColumnType to an ibis type.

        This should be implemented by all specific types.
        """
        pass

    @staticmethod
    def _from_onnx_elem_type(elem_type: int) -> "ColumnType":
        """Given an ONNX TensorProto element type, find the right ColumnType."""
        for scls in ColumnType.__subclasses__():
            if scls._onnx_elem_type == elem_type:
                return scls()  # type: ignore[abstract]
        raise TypeError(f"Unsupported data type {elem_type}")

    def __eq__(self, other: object) -> bool:
        return self.__class__ == other.__class__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


FeaturesTypes = typing.Dict[str, ColumnType]
"""Mapping of feature names to their types."""


def guess_datatypes(dataframe: typing.Any) -> FeaturesTypes:
    """Given a DataFrame, try to guess the types of each feature in it.

    This procudes a [orbital.types.FeaturesTypes][] dictionary that can be used by
    parse_pipeline to generate the SQL queries from the sklearn pipeline.

    In most cases this shouldn't be necessary as the user should know
    on what data the pipeline was trained on, but it can be convenient
    when experimenting or writing tests.

    Requires scikit-learn, which can be installed with the ``orbital[sklearn]`` extra.
    """
    try:
        from . import _sklearn
    except ImportError as err:
        raise ImportError(
            "scikit-learn is required to guess datatypes. "
            "Install it with: pip install orbital[sklearn]"
        ) from err
    return _sklearn.guess_datatypes(dataframe)


class FloatColumnType(ColumnType):
    """Mark a column as containing float values"""

    _onnx_elem_type = _onnx.TensorProto.FLOAT

    def _to_ibistype(self) -> ibis_types.Float32:
        return ibis_types.Float32()


class Float16ColumnType(ColumnType):
    """Mark a column as containing 16bit float values"""

    _onnx_elem_type = _onnx.TensorProto.FLOAT16

    def _to_ibistype(self) -> ibis_types.Float16:
        return ibis_types.Float16()


class DoubleColumnType(ColumnType):
    """Mark a column as containing double values"""

    _onnx_elem_type = _onnx.TensorProto.DOUBLE

    def _to_ibistype(self) -> ibis_types.Float64:
        return ibis_types.Float64()


class StringColumnType(ColumnType):
    """Mark a column as containing string values"""

    _onnx_elem_type = _onnx.TensorProto.STRING

    def _to_ibistype(self) -> ibis_types.String:
        return ibis_types.String()


class Int64ColumnType(ColumnType):
    """Mark a column as containing signed 64bit integer values"""

    _onnx_elem_type = _onnx.TensorProto.INT64

    def _to_ibistype(self) -> ibis_types.Int64:
        return ibis_types.Int64()


class UInt64ColumnType(ColumnType):
    """Mark a column as containing unsigned 64bit integer values"""

    _onnx_elem_type = _onnx.TensorProto.UINT64

    def _to_ibistype(self) -> ibis_types.UInt64:
        return ibis_types.UInt64()


class Int32ColumnType(ColumnType):
    """Mark a column as containing signed 32bit integer values"""

    _onnx_elem_type = _onnx.TensorProto.INT32

    def _to_ibistype(self) -> ibis_types.Int32:
        return ibis_types.Int32()


class UInt32ColumnType(ColumnType):
    """Mark a column as containing unsigned 32bit integer values"""

    _onnx_elem_type = _onnx.TensorProto.UINT32

    def _to_ibistype(self) -> ibis_types.UInt32:
        return ibis_types.UInt32()


class Int16ColumnType(ColumnType):
    """Mark a column as containing signed 16bit integer values"""

    _onnx_elem_type = _onnx.TensorProto.INT16

    def _to_ibistype(self) -> ibis_types.Int16:
        return ibis_types.Int16()


class UInt16ColumnType(ColumnType):
    """Mark a column as containing unsigned 16bit integer values"""

    _onnx_elem_type = _onnx.TensorProto.UINT16

    def _to_ibistype(self) -> ibis_types.UInt16:
        return ibis_types.UInt16()


class Int8ColumnType(ColumnType):
    """Mark a column as containing signed 8bit integer values"""

    _onnx_elem_type = _onnx.TensorProto.INT8

    def _to_ibistype(self) -> ibis_types.Int8:
        return ibis_types.Int8()


class UInt8ColumnType(ColumnType):
    """Mark a column as containing unsigned 8bit integer values"""

    _onnx_elem_type = _onnx.TensorProto.UINT8

    def _to_ibistype(self) -> ibis_types.UInt8:
        return ibis_types.UInt8()


class BooleanColumnType(ColumnType):
    """Mark a column as containing boolean values"""

    _onnx_elem_type = _onnx.TensorProto.BOOL

    def _to_ibistype(self) -> ibis_types.Boolean:
        return ibis_types.Boolean()
