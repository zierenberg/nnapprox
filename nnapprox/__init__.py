





from .core.base import BaseApproximator
from .core.exceptions import BackendNotAvailableError, NNApproxError

__version__ = "0.1.1"
__all__ = ["BaseApproximator", "BackendNotAvailableError", "NNApproxError"]

# Lazy backend imports
def get_pytorch_approximator():
    """Get PyTorch approximator if available."""
    try:
        from .backends.pytorch import PyTorchApproximator
        return PyTorchApproximator
    except ImportError as e:
        raise BackendNotAvailableError(
            "PyTorch backend not available. Install with: pip install nnapprox[pytorch]"
        ) from e

def get_jax_approximator():
    """Get JAX approximator if available."""
    try:
        from .backends.jax import JAXApproximator
        return JAXApproximator
    except ImportError as e:
        raise BackendNotAvailableError(
            "JAX backend not available. Install with: pip install nnapprox[jax]"
        ) from e

# Convenience function
def create_approximator(backend="pytorch", **kwargs):
    """Create an approximator with the specified backend."""
    if backend.lower() == "pytorch":
        ApproximatorClass = get_pytorch_approximator()
    elif backend.lower() == "jax":
        ApproximatorClass = get_jax_approximator()
    else:
        raise ValueError(f"Unknown backend: {backend}. Available: 'pytorch', 'jax'")
    
    return ApproximatorClass(**kwargs)