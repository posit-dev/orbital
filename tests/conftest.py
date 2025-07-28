import logging
import sqlite3
import math
import sys

import duckdb
import sqlalchemy
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes, load_iris

PY39 = sys.version_info[:2] < (3, 10)


def pytest_configure(config):
    # Enable debug logging for the projec itself,
    # so that in case of errors during tests we have
    # additional debug information.
    specific_logger = logging.getLogger("orbital")
    specific_logger.setLevel(logging.DEBUG)

    # Use deterministic seed for reproducible test results
    np.random.seed(42)


# Shared fixtures for all test files
@pytest.fixture(scope="class")
def iris_data():
    """Load and prepare the iris dataset for testing."""
    iris = load_iris()
    # Clean feature names to match what's used in the example
    feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X = pd.DataFrame(iris.data, columns=feature_names)  # Use clean names directly
    y = pd.DataFrame(iris.target, columns=["target"])
    df = pd.concat([X, y], axis=1)
    return df, feature_names


@pytest.fixture(scope="class")
def diabetes_data():
    """Load and prepare the diabetes dataset for testing."""
    diabetes = load_diabetes()
    feature_names = diabetes.feature_names
    X = pd.DataFrame(diabetes.data, columns=feature_names)
    y = pd.DataFrame(diabetes.target, columns=["target"])
    df = pd.concat([X, y], axis=1)
    return df, feature_names


@pytest.fixture(params=["duckdb", "sqlite", "postgres"])
def db_connection(request):
    """Create database connections for testing SQL exports."""
    dialect = request.param
    if dialect == "duckdb":
        conn = duckdb.connect(":memory:")
        yield conn, dialect
        conn.close()
    elif dialect == "sqlite":
        conn = sqlite3.connect(":memory:")
        if PY39:
            # Python 3.9 sqlite is compiled without -DSQLITE_ENABLE_MATH_FUNCTIONS
            conn.create_function("exp", 1, math.exp)
        yield conn, dialect
        conn.close()
    elif dialect == "postgres":
        try:
            conn = sqlalchemy.create_engine(
                "postgresql://orbitaltestuser:orbitaltestpassword@localhost:5432/orbitaltestdb"
            )
            with conn.connect() as testcon:
                testcon.execute(sqlalchemy.text("SELECT 1"))  # Test connection
        except (sqlalchemy.exc.OperationalError, ImportError):
            pytest.skip("Postgres database not available")
        yield conn, dialect
        conn.dispose()
