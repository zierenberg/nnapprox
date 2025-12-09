"""Neural network approximation package with multiple backends."""

from importlib import metadata

# expose version without importing heavy back‑ends
__version__: str = metadata.version(__name__)
from .core import BaseApproximator
from .core.exceptions import (
    NNApproxError,
    BackendNotAvailableError,
    ModelNotFittedError,
)

# Lazy factory
def create_approximator(backend: str = "torch", **kwargs) -> BaseApproximator:
    """
    Create an approximator instance for the specified backend.
    Parameters
    ----------
    backend : str, default="torch"
        Backend to use (currently only 'torch' is supported)
    **kwargs
        Additional keyword arguments passed to the backend-specific approximator
        constructor.
    """
    backend = backend.lower()
    if backend == "torch":
        from .backends.torch import PyTorchApproximator
        return PyTorchApproximator(**kwargs)
    # if backend == "jax":
    #     raise BackendNotAvailableError("JAX backend is not available in this installation.")
    #     #from .backends.jax import JAXApproximator
    #     #return JAXApproximator(**kwargs)
    raise BackendNotAvailableError(f"Unsupported backend {backend!r}. Available: torch.")

# load approximator with a given backend
def load_approximator(path: str, backend: str = "torch") -> BaseApproximator:
    """
    Load a saved approximator from disk using the specified backend.
    
    Parameters
    ----------
    path : str
        Path to the saved model file
    backend : str, default="torch"
        Backend to use for loading ('torch', etc.)
        
    Returns
    -------
    BaseApproximator
        The loaded approximator instance
        
    Example
    -------
    >>> func = create_approximator(backend="torch")
    >>> func.save("model.pt")
    >>> func = load_approximator("model.pt", backend="torch")
    """
    if backend == "torch":
        from .backends.torch import load_torch_approximator
        return load_torch_approximator(path)
    else:
        raise BackendNotAvailableError(f"Unsupported backend: {backend!r}")

__all__ = [
    "BaseApproximator",
    "load_approximator",
    "NNApproxError",
    "BackendNotAvailableError",
    "ModelNotFittedError",
    "create_approximator",
    "__version__",
]