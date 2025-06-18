"""OrbitalML translation module - proxy to orbital.translation"""

import warnings

# Issue deprecation warning
warnings.warn(
    "OrbitalML is a transitional proxy package. "
    "Please migrate to 'import orbital.translation' instead of 'import orbitalml.translation'. "
    "New projects should use the 'orbital' package directly.",
    DeprecationWarning,
    stacklevel=2,
)

# Import and re-export everything from orbital.translation
from orbital.translation import *

# Ensure __all__ is properly set if it exists
import orbital.translation

if hasattr(orbital.translation, "__all__"):
    __all__ = orbital.translation.__all__
