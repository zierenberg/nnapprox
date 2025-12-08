"""Custom exceptions for nnapprox."""

class NNApproxError(Exception):
    """Base exception for nnapprox."""
    pass

class BackendNotAvailableError(NNApproxError):
    """Raised when a requested backend is not available."""
    pass

class ModelNotFittedError(NNApproxError):
    """Raised when trying to use a model that hasn't been fitted."""
    pass