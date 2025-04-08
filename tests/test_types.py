from unittest import mock

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from orbitalml import types


class TestDataTypesGuessing:
    DF_DATA = {
        "names": ["Aldo", "Giovanni", "Giacomo"],
        "age": [66, 68, 68],
        "ratio": [66 / 66, 68 / 66, 68 / 66],
    }
    EXPECTED_TYPES = {
        "names": types.StringColumnType(),
        "age": types.Int64ColumnType(),
        "ratio": types.DoubleColumnType(),
    }

    def test_from_pandas(self):
        df = pd.DataFrame(self.DF_DATA)
        assert types.guess_datatypes(df) == self.EXPECTED_TYPES

    def test_from_polars(self):
        df = pl.DataFrame(self.DF_DATA)
        assert types.guess_datatypes(df) == self.EXPECTED_TYPES

    def test_from_pyarrow(self):
        df = pa.table(self.DF_DATA)
        assert types.guess_datatypes(df) == self.EXPECTED_TYPES

    def test_invalid_datatype(self):
        with pytest.raises(ValueError) as exc:
            types.guess_datatypes({"column": 5})
        assert exc.match("Unable to guess types of dataframe")

    def test_invalid_datatype_conversion(self):
        with mock.patch.object(
            types._sl2o_types, "guess_data_type", return_value=[("somecol", "invalid")]
        ):
            with pytest.raises(ValueError) as exc:
                types.guess_datatypes("Doesn't matter")
            assert exc.match("Unsupported datatype for column somecol")

    def test_alltypes(self):
        for t in [
            types.FloatColumnType,
            types.Float16ColumnType,
            types.DoubleColumnType,
            types.StringColumnType,
            types.Int64ColumnType,
            types.UInt64ColumnType,
            types.Int32ColumnType,
            types.UInt32ColumnType,
            types.Int16ColumnType,
            types.UInt16ColumnType,
            types.Int8ColumnType,
            types.UInt8ColumnType,
            types.BooleanColumnType,
        ]:
            onxtype = t()._to_onnxtype()
            assert isinstance(onxtype, types._sl2o_types.DataType)
            assert t._from_onnxtype(onxtype) == t()

    def test_only_support_column_types(self):
        with pytest.raises(ValueError) as exc:
            types.ColumnType._from_onnxtype(
                types._sl2o_types.FloatTensorType(shape=[1, 1])
            )
        assert exc.match("Only columnar data is supported")

    def test_invalid_datatype_shape(self):
        with pytest.raises(TypeError) as exc:
            types.ColumnType._from_onnxtype(mock.Mock(shape=[None, 1]))
        assert exc.match("Unsupported data type Mock")
