"""OrbitalML sql module - proxy to orbital.sql"""

import warnings

# Issue deprecation warning
warnings.warn(
    "OrbitalML is a transitional proxy package. "
    "Please migrate to 'import orbital.sql' instead of 'import orbitalml.sql'. "
    "New projects should use the 'orbital' package directly.",
    DeprecationWarning,
    stacklevel=2,
)

# Import and re-export everything from orbital.sql
from orbital.sql import *

# Ensure __all__ is properly set if it exists
import orbital.sql

if hasattr(orbital.sql, "__all__"):
    __all__ = orbital.sql.__all__
