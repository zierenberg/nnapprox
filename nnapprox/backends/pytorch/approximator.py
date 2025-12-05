from __future__ import annotations
import pickle
import numpy as np
import pandas as pd
from typing import Any, Mapping, Sequence, Callable, Type
from tqdm import tqdm
import inspect
import types

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError as exc:
    torch = nn = optim = None
    _TORCH_IMPORT_ERROR = exc

from ...core.base import BaseApproximator
from ...core.exceptions import ModelNotFittedError, BackendNotAvailableError, NNApproxError
from ...core.utils import Transform
from .models import MLPModel

class PyTorchApproximator(BaseApproximator):
    """PyTorch-backed neural network approximator."""

    def __init__(
        self,
        *,
        input: Sequence[str],
        output: Sequence[str],
        hidden_dims: Sequence[int] | None = None,
        activation: Type[nn.Module] = nn.Tanh,
        dropout: float = 0.0,
        verbose: bool = False,
        **model_kwargs: Any,
    ):
        if torch is None:
            raise BackendNotAvailableError("PyTorch backend not available.") from _TORCH_IMPORT_ERROR

        # Backend-independent init
        super().__init__(input=input, output=output, verbose=verbose)

        # Device
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        # Build model
        hidden_dims = list(hidden_dims) if hidden_dims is not None else [42, 42, 42]
        self.model = MLPModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            **model_kwargs,
        ).to(self.device)

        # Scalers
        from sklearn.preprocessing import MinMaxScaler
        act_range = (-1, 1) if isinstance(activation, nn.Tanh) else (0, 1)
        self._x_scaler = MinMaxScaler(feature_range=act_range)
        self._y_scaler = MinMaxScaler(feature_range=act_range)
        self.is_fitted = False

    def set_transform(
        self,
        label: str,
        *,
        transform_type: str | None = None,
        forward: Callable | None = None,
        inverse: Callable | None = None,
    ) -> None:
        """
        Register a forward / inverse transformation for a variable.

        *Pre‑defined* transforms are referenced by ``transform_type`` (e.g.
        ``"log"``).  For a custom pair supply ``forward`` and ``inverse``.
        """
        # Resolve which list we are editing
        if label in self.input_names:
            idx = self.input_names.index(label)
            target = self.input_transforms
        elif label in self.output_names:
            idx = self.output_names.index(label)
            target = self.output_transforms
        else:
            raise ValueError(f"{label!r} is not a known input or output name.")

        # predefined transform
        if transform_type:
            if forward is not None or inverse is not None:
                raise ValueError(
                    "Provide either `transform_type` **or** both `forward`/`inverse`, not both."
                )
            target[idx] = Transform.predefined(transform_type)
            return

        # custom transform
        if forward is None or inverse is None:
            raise ValueError("Both `forward` and `inverse` must be supplied for a custom transform.")

        # Verify that we can get a stable import path (helps with saving / loading)
        try:
            inspect.getsource(forward)
            inspect.getsource(inverse)
        except OSError as exc:
            raise ValueError(
                "Custom transform functions must be defined in a module (not interactively)."
            ) from exc

        target[idx] = Transform.custom(forward, inverse)

    # prepare data for training
    def _extract_arrays(self, data: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Return (X_raw, Y_raw) as plain NumPy arrays."""
        X = np.column_stack([np.asarray(data[name]) for name in self.input_names])
        Y = np.column_stack([np.asarray(data[name]) for name in self.output_names])
        return X, Y

    def prepare_data(self, data: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Apply transforms + scaling – returns arrays ready for training."""
        X_raw, Y_raw = self._extract_arrays(data)

        # Apply forward transforms
        for i, tr in enumerate(self.input_transforms):
            X_raw[:, i] = tr.forward(X_raw[:, i])
        for i, tr in enumerate(self.output_transforms):
            Y_raw[:, i] = tr.forward(Y_raw[:, i])

        # Scale to activation range
        X_scaled = self._x_scaler.fit_transform(X_raw)
        Y_scaled = self._y_scaler.fit_transform(Y_raw)

        return X_scaled, Y_scaled

    # training the model
    def fit(
        self,
        data: Mapping[str, Any],
        *,
        custom_loss=None,
        epochs: int = 10_000,
        lr: float = 1e-3,
        eps: float = 1e-5,
        amsgrad: bool = False,
        batch_size: int | None = None,
        verbose: bool | None = None,
        **optim_kwargs: Any,
    ) -> "PyTorchApproximator":
        """
        Train the model.
        """
        Xs, Ys = self.prepare_data(data)

        # convert to torch tensors
        X_tensor = torch.from_numpy(Xs).float().to(self.device)
        Y_tensor = torch.from_numpy(Ys).float().to(self.device)
        model_ = self.model.to(self.device)
        if self.verbose:
            print(f"Training data with input shape {X_tensor.shape} and output shape {Y_tensor.shape}.")

        # Adam and MSE Loss
        optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=eps, amsgrad=amsgrad)
        loss_fn = nn.MSELoss(reduction="mean")
        if custom_loss is None:
            def custom_loss(Y_pred, Y, X):
                return loss_fn(Y_pred, Y)

        training_loss = np.zeros(epochs)
        for epoch in tqdm(range(epochs), desc="Training"):
            # forward pass
            Y_pred = model_(X_tensor)
            loss = custom_loss(Y_pred, Y_tensor, X_tensor)
            # compute gradients
            loss.backward()
            # update parameters
            optimizer.step()
            # zero the parameter gradients
            optimizer.zero_grad()
            # for logging
            training_loss[epoch] = loss.item()
            
        # fetch model from device
        self.model = model_.to("cpu")
        # set training to false to enter evaluation mode
        self.model.train(False)
        self.is_fitted = True
        self.training_loss = training_loss

        if self.verbose:
            print("Training complete; left training mode for faster evaluation! To re-enter training mode call `func.model.train(True)`.")
        return self

    def _prepare_inputs(self, *args: Any) -> tuple[np.ndarray, int]:
        """
        Return a 2-D array (n_samples, n_features) and the number of samples.
        Works with:
            - Scalars
            - 1D arrays / lists / Series
            - pandas DataFrames
        """
        import pandas as pd
        import numpy as np

        # DataFrame shortcut
        if len(args) == 1 and isinstance(args[0], pd.DataFrame):
            df = args[0]
            missing = [c for c in self.input_names if c not in df.columns]
            if missing:
                raise ValueError(f"DataFrame missing columns: {missing}")
            raw = np.column_stack([np.asarray(df[col]).ravel() for col in self.input_names])
        else:
            if len(args) != self.input_dim:
                raise ValueError(f"Expected {self.input_dim} inputs, got {len(args)}")
            # convert everything to 1D arrays
            arrays = []
            for arg in args:
                arr = np.asarray(arg)
                if arr.ndim == 0:
                    arr = np.array([arr])  # scalar -> 1D
                elif arr.ndim > 1 and arr.shape[0] == 1:
                    arr = arr.ravel()
                elif arr.ndim > 1 and arr.shape[0] != 1:
                    arr = arr.reshape(-1)
                arrays.append(arr)
            raw = np.column_stack(arrays)

        # Broadcast scalars to the longest vector
        max_len = max(col.shape[0] for col in raw.T)
        for i, col in enumerate(raw.T):
            if col.shape[0] == 1 and max_len > 1:
                raw[:, i] = np.full(max_len, col[0])
            elif col.shape[0] not in (1, max_len):
                raise ValueError(
                    f"Input column {i} length {col.shape[0]} is incompatible with other inputs ({max_len})."
                )

        # Apply forward transforms
        for i, tr in enumerate(self.input_transforms):
            raw[:, i] = tr.forward(raw[:, i])

        return raw, max_len


    def predict(
        self,
        *args: Any,
        return_dataframe: bool = False,
    ) -> np.ndarray | pd.DataFrame:
        """Return predictions; optionally as a DataFrame that also contains the inputs."""
        if not self.is_fitted:
            raise ModelNotFittedError("Call `fit` before `predict`.")

        X_raw, n = self._prepare_inputs(*args)

        # Scale, run through the net, inverse‑scale
        X_scaled = self._x_scaler.transform(X_raw)
        with torch.no_grad():
            Y_scaled = self.model(torch.from_numpy(X_scaled).float()).cpu().numpy()
        Y = self._y_scaler.inverse_transform(Y_scaled)

        # Apply inverse output transforms
        for i, tr in enumerate(self.output_transforms):
            Y[:, i] = tr.inverse(Y[:, i])

        # If a single sample was requested, return scalars
        if n == 1:
            Y = Y.squeeze()

        if return_dataframe:
            # Build a tidy DataFrame that mirrors the original input format
            if len(args) == 1 and isinstance(args[0], pd.DataFrame):
                input_df = args[0][self.input_names].reset_index(drop=True)
            else:
                # Re‑create the input columns from the broadcasted arrays
                input_df = pd.DataFrame(
                    {name: X_raw[:, i] for i, name in enumerate(self.input_names)}
                )
            out_df = pd.DataFrame(Y, columns=self.output_names)
            return pd.concat([input_df, out_df], axis=1)

        return Y


    
    def save(self, path: str) -> None:
        """Save the complete model state to a file."""
        if not self.is_fitted:
            raise ModelNotFittedError("Cannot save an unfitted model.")
        
        # Serialize transforms - only save specs, not callable objects
        def _serialize_transform(tr):
            spec = tr.spec.copy()
            # For custom transforms, extract the code as bytes
            if spec["type"] == "custom":
                # Serialize the function bytecode
                import marshal
                spec["forward_code"] = marshal.dumps(tr.forward.__code__)
                spec["inverse_code"] = marshal.dumps(tr.inverse.__code__)
                spec["forward_name"] = tr.forward.__name__
                spec["inverse_name"] = tr.inverse.__name__
            return spec
        
        # Serialize activation properly
        if hasattr(self.model, 'activation'):
            act = self.model.activation
            if isinstance(act, type):
                # It's a class (like nn.Tanh), save its name directly
                activation_info = {"type": "class", "name": act.__name__}
            elif isinstance(act, nn.Module):
                # It's an instance, save the class name
                activation_info = {"type": "instance", "name": act.__class__.__name__}
            else:
                raise NNApproxError(f"Unknown activation type: {type(act)}")
        else:
            activation_info = {"type": "class", "name": "Tanh"}
        
        state = {
            # Model architecture
            "model_state": self.model.state_dict(),
            "hidden_dims": self.model.hidden_dims,
            "activation": activation_info,
            "dropout": self.model.dropout,
            
            # Input/output configuration
            "input_names": self.input_names,
            "output_names": self.output_names,
            
            # Store only transform specs (no lambda objects)
            "input_transforms": [_serialize_transform(tr) for tr in self.input_transforms],
            "output_transforms": [_serialize_transform(tr) for tr in self.output_transforms],
            
            # Scalers
            "x_scaler": self._x_scaler,
            "y_scaler": self._y_scaler,
            
            # Training metadata
            "training_loss": getattr(self, 'training_loss', None),
        }
        
        torch.save(state, path, pickle_protocol=pickle.HIGHEST_PROTOCOL)


    def load(self, path: str) -> "PyTorchApproximator":
        """Load a complete model state from a file."""
        # Load directly to CPU (models are stored on CPU after training)
        state = torch.load(path, map_location="cpu", weights_only=False)
        
        if "model_state" not in state:
            raise NNApproxError(f"Invalid checkpoint: missing 'model_state'.")

        # Restore input/output configuration
        self.input_names = state["input_names"]
        self.output_names = state["output_names"]
        self.input_dim = len(self.input_names)
        self.output_dim = len(self.output_names)
        
        # Reconstruct transforms from specs
        def _rebuild_transform(spec: dict) -> Transform:
            if spec["type"] == "predefined":
                # Reconstruct from name - this recreates the lambdas
                return Transform.predefined(spec["name"])
            elif spec["type"] == "identity":
                return Transform.predefined("identity")
            elif spec["type"] == "custom":
                # Reconstruct functions from bytecode
                import marshal
                forward_code = marshal.loads(spec["forward_code"])
                inverse_code = marshal.loads(spec["inverse_code"])
                
                # Create function objects from code objects
                forward_func = types.FunctionType(forward_code, globals(), spec["forward_name"])
                inverse_func = types.FunctionType(inverse_code, globals(), spec["inverse_name"])
                
                return Transform.custom(forward_func, inverse_func)
            else:
                raise NNApproxError(f"Unknown transform type: {spec['type']}")

        self.input_transforms = [_rebuild_transform(s) for s in state["input_transforms"]]
        self.output_transforms = [_rebuild_transform(s) for s in state["output_transforms"]]

        # Restore scalers
        self._x_scaler = state["x_scaler"]
        self._y_scaler = state["y_scaler"]

        # Map activation name back to class
        activation_map = {
            "Tanh": nn.Tanh,
            "ReLU": nn.ReLU,
            "Sigmoid": nn.Sigmoid,
            "LeakyReLU": nn.LeakyReLU,
            "ELU": nn.ELU,
            "GELU": nn.GELU,
            "Identity": nn.Identity,
        }
        
        # Handle both old format (string) and new format (dict)
        activation_info = state["activation"]
        if isinstance(activation_info, str):
            # Old format - just a string
            activation_name = activation_info
        elif isinstance(activation_info, dict):
            # New format - dict with type and name
            activation_name = activation_info["name"]
        else:
            raise NNApproxError(f"Unknown activation format: {type(activation_info)}")
        
        if activation_name not in activation_map:
            raise NNApproxError(f"Unknown activation: {activation_name}. Available: {list(activation_map.keys())}")
        activation_class = activation_map[activation_name]

        # Recreate model with correct architecture (on CPU)
        self.model = MLPModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=state["hidden_dims"],
            activation=activation_class,
            dropout=state["dropout"],
        )  # No .to(self.device) - keep on CPU
        
        # Load the trained weights
        self.model.load_state_dict(state["model_state"])
        self.model.eval()
        
        # Restore training metadata
        if "training_loss" in state and state["training_loss"] is not None:
            self.training_loss = state["training_loss"]
        
        self.is_fitted = True
        return self
    
# create a possibility to load backend-specific approximators from file
def load_torch_approximator(path: str) -> PyTorchApproximator:
    """Load a PyTorchApproximator from file."""
    # Create a minimal dummy instance
    approximator = PyTorchApproximator(
        input=["dummy_input"],  # Will be overwritten
        output=["dummy_output"],  # Will be overwritten
        verbose=False
    )
    # Load the real state
    approximator.load(path)
    return approximator
