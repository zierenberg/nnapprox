"""Core abstractions for nnapprox.

This sub‑package contains the abstract base class that all back‑ends must
implement, as well as the custom exception hierarchy.
"""

# Re‑export the public API ----------------------------------------------------
from .base import BaseApproximator
from .exceptions import NNApproxError, BackendNotAvailableError, ModelNotFittedError
from .utils import Transform

__all__ = [
    "BaseApproximator",
    "NNApproxError",
    "BackendNotAvailableError",
    "ModelNotFittedError",
    "Transform"
]