import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from orbital import types


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

    @pytest.mark.parametrize(
        "dtype, expected",
        [
            (np.bool_, types.BooleanColumnType()),
            (np.float16, types.Float16ColumnType()),
            (np.float32, types.FloatColumnType()),
            (np.int32, types.Int32ColumnType()),
            (np.uint8, types.UInt8ColumnType()),
            (object, types.StringColumnType()),
        ],
    )
    def test_numpy_dtypes(self, dtype, expected):
        df = pd.DataFrame({"c": np.array([0, 1]).astype(dtype)})
        assert types.guess_datatypes(df) == {"c": expected}

    def test_invalid_datatype(self):
        with pytest.raises(ValueError) as exc:
            types.guess_datatypes({"column": 5})
        assert exc.match("Unable to guess types of dataframe")

    def test_unsupported_column_dtype(self):
        # complex128 has no ColumnType counterpart.
        with pytest.raises(ValueError) as exc:
            types.guess_datatypes(pd.DataFrame({"c": np.array([1 + 2j, 3j])}))
        assert exc.match("Unsupported datatype for column c")

    def test_alltypes(self):
        pytest.importorskip("sklearn")
        from orbital import _sklearn

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
            onxtype = _sklearn.TENSOR_TYPES[t._onnx_elem_type]([None, 1])
            assert (
                types.ColumnType._from_onnx_elem_type(
                    onxtype.to_onnx_type().tensor_type.elem_type
                )
                == t()
            )
