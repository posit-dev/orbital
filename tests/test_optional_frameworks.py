"""The sklearn-only entry points point to the extra when scikit-learn is missing."""

import importlib.util

import pytest

import orbital

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is not None, reason="scikit-learn is installed"
)


def test_parse_pipeline_without_sklearn():
    with pytest.raises(ImportError, match=r"orbital\[sklearn\]"):
        orbital.parse_pipeline(None, {})


def test_guess_datatypes_without_sklearn():
    with pytest.raises(ImportError, match=r"orbital\[sklearn\]"):
        orbital.types.guess_datatypes(None)
