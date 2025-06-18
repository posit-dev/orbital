"""OrbitalML translate module - proxy to orbital.translate"""

import warnings

# Issue deprecation warning
warnings.warn(
    "OrbitalML is a transitional proxy package. "
    "Please migrate to 'import orbital.translate' instead of 'import orbitalml.translate'. "
    "New projects should use the 'orbital' package directly.",
    DeprecationWarning,
    stacklevel=2,
)

# Import and re-export everything from orbital.translate
from orbital.translate import *

# Ensure __all__ is properly set if it exists
import orbital.translate

if hasattr(orbital.translate, "__all__"):
    __all__ = orbital.translate.__all__
