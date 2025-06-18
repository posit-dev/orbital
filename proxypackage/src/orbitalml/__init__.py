"""OrbitalML - Machine Learning pipeline to SQL converter

OrbitalML is a proxy package for the orbital library that provides the same
functionality with an alternative import name. It translates scikit-learn
pipelines into SQL queries and Ibis expressions.

This package allows you to execute machine learning models on databases without
the need for a Python runtime environment.

Usage:
    import orbitalml as orbital
    # Use exactly like the orbital package
"""

# Import everything from orbital's __all__
from orbital import *

# Import submodules that are NOT in orbital's __all__
from orbital import ast, sql, translate, translation, types

# Optional: Add version info
__version__ = "1.0.0"

# Re-export everything
__all__ = [
    # From orbital.__all__
    "parse_pipeline",
    "translate",
    "export_sql",
    "ResultsProjection",
    # Additional submodules
    "types",
    "ast",
    "sql",
    "translation",
]
