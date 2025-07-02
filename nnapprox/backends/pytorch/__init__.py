"""PyTorch backend for nnapprox."""

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    
    from .models import PyTorchApproximator
    
    __all__ = ["PyTorchApproximator"]
    
except ImportError as e:
    raise ImportError(
        "PyTorch backend requires PyTorch. Install with: pip install nnapprox[pytorch]"
    ) from e