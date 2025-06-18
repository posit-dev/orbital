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

import warnings

# Import everything from orbital's __all__
from orbital import *

# Import submodules that are NOT in orbital's __all__
from orbital import ast, sql, translate, translation, types
import orbital

# Issue deprecation warning
warnings.warn(
    "OrbitalML is a transitional proxy package. "
    "Please migrate to 'import orbital' instead of 'import orbitalml'. "
    "New projects should use the 'orbital' package directly.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything
__all__ = orbital.__all__
