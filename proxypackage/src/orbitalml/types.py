"""OrbitalML types module - proxy to orbital.types"""

import warnings

# Issue deprecation warning
warnings.warn(
    "OrbitalML is a transitional proxy package. "
    "Please migrate to 'import orbital.types' instead of 'import orbitalml.types'. "
    "New projects should use the 'orbital' package directly.",
    DeprecationWarning,
    stacklevel=2,
)

# Import and re-export everything from orbital.types
from orbital.types import *

# Ensure __all__ is properly set if it exists
import orbital.types

if hasattr(orbital.types, "__all__"):
    __all__ = orbital.types.__all__
