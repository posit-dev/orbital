import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

pytest.importorskip("sklearn")

from orbital import _sklearn, types


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

    def test_unsupported_column_dtype(self):
        # complex128 has no ColumnType counterpart.
        with pytest.raises(ValueError) as exc:
            types.guess_datatypes(pd.DataFrame({"c": np.array([1 + 2j, 3j])}))
        assert exc.match("Unsupported datatype for column c")

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
            onxtype = _sklearn.TENSOR_TYPES[t._onnx_elem_type]([None, 1])
            assert (
                types.ColumnType._from_onnx_elem_type(
                    onxtype.to_onnx_type().tensor_type.elem_type
                )
                == t()
            )

    def test_only_support_column_types(self):
        # A 2D array is guessed as a single "input" tensor of shape (3, 2),
        # which is not columnar data.
        with pytest.raises(ValueError) as exc:
            types.guess_datatypes(np.zeros((3, 2)))
        assert exc.match("Unsupported datatype for column input")
