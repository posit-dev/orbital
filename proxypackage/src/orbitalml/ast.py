"""OrbitalML ast module - proxy to orbital.ast"""

import warnings

# Issue deprecation warning
warnings.warn(
    "OrbitalML is a transitional proxy package. "
    "Please migrate to 'import orbital.ast' instead of 'import orbitalml.ast'. "
    "New projects should use the 'orbital' package directly.",
    DeprecationWarning,
    stacklevel=2,
)

# Import and re-export everything from orbital.ast
from orbital.ast import *

# Ensure __all__ is properly set if it exists
import orbital.ast

if hasattr(orbital.ast, "__all__"):
    __all__ = orbital.ast.__all__
