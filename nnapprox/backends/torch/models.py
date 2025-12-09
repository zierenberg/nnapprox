from __future__ import annotations
from typing import List, Type, Union

from ...core.exceptions import BackendNotAvailableError

try:
    import torch.nn as nn
except ImportError as _torch_import_error:   # pragma: no cover
    nn = None
    _TORCH_IMPORT_ERROR = _torch_import_error

class MLPModel(nn.Module):
    """Mulitlayer perceptron"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        activation: Union[nn.Module, type[nn.Module]] = nn.Tanh,
        dropout: float | None = None,
    ):
        if nn is None:                     # pragma: no cover
            raise BackendNotAvailableError(
                "PyTorch is required to build the MLPModel. "
                "Install with: pip install nnapprox[torch]"
            ) from _TORCH_IMPORT_ERROR

        super().__init__()
        # Store these for saving/loading
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dropout = dropout

        dims = [input_dim] + hidden_dims + [output_dim]

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            # No activation / dropout on the output layer
            if i < len(dims) - 2:
                if isinstance(activation, type):
                    layers.append(activation())
                else:
                    # assume user passed an instance
                    layers.append(activation)
                if dropout is not None:
                    layers.append(nn.Dropout(p=dropout))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x