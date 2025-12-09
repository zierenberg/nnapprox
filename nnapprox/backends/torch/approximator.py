from __future__ import annotations
import pickle
try:
    import cloudpickle
except ImportError:
    cloudpickle = None
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

from ...core.base import BaseApproximator
from ...core.exceptions import ModelNotFittedError, BackendNotAvailableError, NNApproxError
from ...core.utils import Transform
from .models import MLPModel

class PyTorchApproximator(BaseApproximator):
    """
    PyTorch-backed neural network function approximator.
    
    This class provides a flexible interface for approximating arbitrary functions
    using multilayer perceptrons (MLPs). It supports automatic scaling, custom
    transformations, and serialization of trained models.
    
    Parameters
    ----------
    input : Sequence[str]
        Names of input variables (features)
    output : Sequence[str]
        Names of output variables (targets)
    hidden_dims : Sequence[int], optional
        Number of neurons in each hidden layer. Default is [42, 42, 42]
    activation : Type[nn.Module], default=nn.Tanh
        Activation function class (e.g., nn.ReLU, nn.Tanh)
    dropout : float, default=0.0
        Dropout probability for regularization (0.0 = no dropout)
    verbose : bool, default=False
        Whether to print training progress
    **model_kwargs : Any
        Additional keyword arguments passed to the MLP model
        
    Attributes
    ----------
    model : MLPModel
        The underlying PyTorch neural network
    is_fitted : bool
        Whether the model has been trained
    input_names : list[str]
        Names of input variables
    output_names : list[str]
        Names of output variables
    training_loss : np.ndarray
        Training loss history after fitting
        
    Examples
    --------
    Basic usage with numpy arrays:
    
    >>> import nnapprox as nna
    >>> import numpy as np
    >>> 
    >>> # Generate training data
    >>> x1 = np.linspace(0, 10, 100)
    >>> x2 = np.linspace(0, 5, 100)
    >>> y = np.sin(x1) * np.cos(x2)
    >>> 
    >>> # Create approximator
    >>> func = nna.PyTorchApproximator(
    ...     input=['x1', 'x2'],
    ...     output=['y'],
    ...     hidden_dims=[64, 64]
    ... )
    >>> 
    >>> # Train
    >>> func.fit({'x1': x1, 'x2': x2, 'y': y}, epochs=1000)
    >>> 
    >>> # Predict
    >>> predictions = func(x1_new, x2_new)
    
    With pandas DataFrame:
    
    >>> import pandas as pd
    >>> df = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
    >>> func.fit(df, epochs=1000)
    >>> result_df = func.predict(df, return_dataframe=True)
    
    With transformations:
    
    >>> func.set_transform('x1', predefined='log')
    >>> func.set_transform('y', predefined='exp')
    >>> func.fit(data, epochs=1000)
    
    Notes
    -----
    - The model automatically scales inputs and outputs to the activation function range
    - After training, the model is moved to CPU for efficient inference
    - Custom transforms must be serializable (defined in modules, not interactively)
      or use cloudpickle for lambda functions
      
    See Also
    --------
    load_torch_approximator : Load a saved approximator
    create_approximator : Factory function for creating approximators
    """
    def __init__(
        self,
        *,
        input: Sequence[str] | None = None,
        output: Sequence[str] | None = None,
        hidden_dims: Sequence[int] | None = None,
        activation: Type[nn.Module] = nn.Tanh,
        dropout: float = 0.0,
        verbose: bool = False,
        **model_kwargs: Any,
    ):
        if torch is None:
            raise BackendNotAvailableError("PyTorch backend not available.")

        if input is None and output is None:
            # dummy approximator for API testing
            super().__init__(input=[], output=[], verbose=verbose)
            self.model = None
            return 

        if input is None or output is None:
            raise ValueError("Input and output variable names must be provided.")

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

        # Scalers (TODO needs to be replaced with internal routine to be abel to access the scaling parameters for custom loss functions, etc.)
        from sklearn.preprocessing import MinMaxScaler
        act_range = (-1, 1) #if isinstance(activation, nn.Tanh) else (0, 1)
        self._x_scaler = MinMaxScaler(feature_range=act_range)
        self._y_scaler = MinMaxScaler(feature_range=act_range)
        self.is_fitted = False       

    # prepare data for training
    def _extract_arrays(self, data: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Return (X_raw, Y_raw) as plain NumPy arrays."""
        X = np.column_stack([np.asarray(data[name]) for name in self.input_names])
        Y = np.column_stack([np.asarray(data[name]) for name in self.output_names])
        return X, Y

    def prepare_data(self, data: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Apply transforms + scaling – returns arrays ready for training."""
        X_raw, Y_raw = self._extract_arrays(data)

        # Apply forward transforms to input and then scale to dynamic range
        for i, tr in enumerate(self.input_transforms):
            X_raw[:, i] = tr.forward(X_raw[:, i])
        X_scaled = self._x_scaler.fit_transform(X_raw)

        # Apply forward transforms to output and then scale to dynamic range
        for i, tr in enumerate(self.output_transforms):
            Y_raw[:, i] = tr.forward(Y_raw[:, i])
        Y_scaled = self._y_scaler.fit_transform(Y_raw)

        return X_scaled, Y_scaled

    # training the model
    def fit(
        self,
        data: Mapping[str, Any],
        *,
        custom_loss = None,
        epochs: int = 10_000,
        lr: float = 1e-3,
        amsgrad: bool = False, # makes convergence more stable in some cases
        device: str = None,
        verbose: bool | None = None,
        **optim_kwargs: Any,
    ) -> "PyTorchApproximator":
        """
        Train the neural network approximator.
        
        Parameters
        ----------
        data : Mapping[str, Any]
            Dictionary or DataFrame containing training data. Keys/columns must
            include all input and output variable names.
        custom_loss : Callable, optional
            Custom loss function with signature `loss(y_pred, y_true, x)`.
            Default is MSE loss.
        epochs : int, default=10_000
            Number of training epochs
        lr : float, default=1e-3
            Learning rate for Adam optimizer
        amsgrad : bool, default=False
            Whether to use AMSGrad variant of Adam. Can improve convergence
            but may find worse local minima.
        verbose : bool, optional
            Override instance verbose setting
        device : str, optional
            Device to use for training ('cpu', 'cuda', 'mps'). Default uses
            the instance's device.
        **optim_kwargs : Any
            Additional arguments passed to Adam optimizer
            
        Returns
        -------
        self : PyTorchApproximator
            The fitted approximator (for method chaining)
            
        Examples
        --------
        Basic training:
        
        >>> func.fit(data, epochs=5000, lr=1e-3)
        
        With custom loss:
        
        >>> def physics_loss(y_pred, y_true, x):
        ...     mse = F.mse_loss(y_pred, y_true)
        ...     physics_penalty = torch.mean((y_pred - x[:, 0])**2)
        ...     return mse + 0.1 * physics_penalty
        >>> 
        >>> func.fit(data, custom_loss=physics_loss, epochs=10000)
        
        Notes
        -----
        - The model is automatically moved to CPU after training for inference
        - Training loss history is stored in `self.training_loss`
        - Use lower `lr` if training is unstable
        - Default `eps=1e-8` provides best accuracy
        
        See Also
        --------
        predict : Make predictions with the fitted model
        """
        Xs, Ys = self.prepare_data(data)

        if device is None:
            device = self.device
        else:
            #test that device is valid
            device = torch.device(device)

        # convert to torch tensors
        X_tensor = torch.from_numpy(Xs).float().to(device)
        Y_tensor = torch.from_numpy(Ys).float().to(device)
        model_ = self.model.to(device)
        if self.verbose:
            print(f"Training data with input shape {X_tensor.shape} and output shape {Y_tensor.shape}.")

        # Adam and MSE Loss
        optimizer = optim.Adam(self.model.parameters(), lr=lr, amsgrad=amsgrad)
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
        Convert inputs to a 2D array (n_samples, n_features) and return the number of samples.
        
        Supports:
            - Single DataFrame with all input columns
            - Multiple arrays/lists/scalars (one per input)
        """
        # Case 1: Single DataFrame
        if len(args) == 1 and isinstance(args[0], pd.DataFrame):
            df = args[0]
            missing = [col for col in self.input_names if col not in df.columns]
            if missing:
                raise ValueError(f"DataFrame missing columns: {missing}")
            raw = np.column_stack([df[col].values for col in self.input_names])
            n_samples = len(df)
        
        # Case 2: Individual arguments (arrays, lists, scalars)
        else:
            if len(args) != self.input_dim:
                raise ValueError(f"Expected {self.input_dim} inputs, got {len(args)}")
            
            # Convert all inputs to 1D arrays
            arrays = [np.atleast_1d(np.asarray(arg).ravel()) for arg in args]
            
            # Find the maximum length (ignore length-1 arrays which will be broadcast)
            lengths = [len(arr) for arr in arrays]
            non_scalar_lengths = [l for l in lengths if l > 1]
            n_samples = max(non_scalar_lengths) if non_scalar_lengths else 1
            
            # Broadcast scalars to match n_samples
            for i, arr in enumerate(arrays):
                if len(arr) == 1 and n_samples > 1:
                    arrays[i] = np.full(n_samples, arr[0])
                elif len(arr) != n_samples and len(arr) != 1:
                    raise ValueError(
                        f"Input {i} ({self.input_names[i]}) has length {len(arr)}, "
                        f"expected {n_samples} or 1"
                    )
            
            raw = np.column_stack(arrays)
        
        return raw, n_samples


    def predict(
        self,
        *args: Any,
        return_dataframe: bool = False,
    ) -> np.ndarray | pd.DataFrame:
        """
        Make predictions using the trained model.
        
        Supports multiple input formats for convenience:
        - Individual arrays/lists for each input
        - Mixed scalars and arrays (scalars are broadcast)
        - Single DataFrame containing all inputs
        
        Parameters
        ----------
        *args : array-like or DataFrame
            Either individual arrays/scalars for each input (in order of
            input_names), or a single DataFrame with all input columns
        return_dataframe : bool, default=False
            If True, return DataFrame with both inputs and predictions
            
        Returns
        -------
        predictions : np.ndarray or pd.DataFrame
            - If single sample: scalar (1 output) or 1D array (multiple outputs)
            - If multiple samples, single output: 1D array of shape (n,)
            - If multiple samples, multiple outputs: 2D array of shape (n, m)
            - If return_dataframe=True: DataFrame with inputs and outputs
            
        Raises
        ------
        ModelNotFittedError
            If predict is called before fit
        ValueError
            If wrong number of inputs provided or input lengths don't match
            
        Examples
        --------
        With individual arrays:
        
        >>> predictions = func.predict(x1, x2, x3)
        
        With DataFrame:
        
        >>> df_pred = func.predict(df)
        
        With mixed scalars and arrays (broadcasting):
        
        >>> # Fix x1=5, vary x2
        >>> predictions = func(5, np.linspace(0, 10, 100))
        
        Return as DataFrame:
        
        >>> result = func.predict(x1, x2, return_dataframe=True)
        >>> print(result.columns)  # ['x1', 'x2', 'y']
        
        Notes
        -----
        - Scalars are automatically broadcast to match array lengths
        - All input arrays must have the same length or be length-1
        - Transformations are automatically applied and inverted
        
        See Also
        --------
        __call__ : Alias for predict without return_dataframe option
        """
        if not self.is_fitted:
            raise ModelNotFittedError("Call `fit` before `predict`.")

        X_raw, n_samples = self._prepare_inputs(*args)

        # Transform and scale inputs
        for i, tr in enumerate(self.input_transforms):
            X_raw[:, i] = tr.forward(X_raw[:, i])
        X_scaled = self._x_scaler.transform(X_raw)
        
        # Run through network
        with torch.no_grad():
            Y_scaled = self.model(torch.from_numpy(X_scaled).float()).cpu().numpy()
        
        # Inverse scale outputs
        Y = self._y_scaler.inverse_transform(Y_scaled)
        # Apply inverse output transforms
        for i, tr in enumerate(self.output_transforms):
            Y[:, i] = tr.inverse(Y[:, i])

        # Remove unnecessary dimensions for single sample or vector output
        if n_samples == 1:
            # return 1D array or scalar
            Y = Y.squeeze()
        elif self.output_dim == 1:
            # return 1D array
            Y = Y.ravel()

        if return_dataframe:
            # Build DataFrame with inputs and outputs
            if len(args) == 1 and isinstance(args[0], pd.DataFrame):
                input_df = args[0][self.input_names].reset_index(drop=True)
            else:
                input_df = pd.DataFrame(X_raw, columns=self.input_names)
            
            output_df = pd.DataFrame(Y if n_samples > 1 else [Y], columns=self.output_names)
            return pd.concat([input_df, output_df], axis=1)

        return Y
    
    def save(self, path: str, verbose: bool = False) -> None:
        """
        Save the trained model to disk.
        
        Saves the complete model state including architecture, weights, scalers,
        and transformations. Models can be loaded with `load_approximator()`.
        
        Parameters
        ----------
        path : str
            Path to save the model file (typically with .pt extension)
            
        Raises
        ------
        ModelNotFittedError
            If trying to save an unfitted model
        NNApproxError
            If custom transforms cannot be serialized
            
        Examples
        --------
        >>> func.fit(data, epochs=5000)
        >>> func.save('my_model.pt')
        
        Notes
        -----
        - Custom transforms require cloudpickle: `pip install cloudpickle`
        - Models with custom transforms may not load across Python versions
        - Predefined transforms are always portable
        
        See Also
        --------
        load : Load a model
        load_torch_approximator : Load a model without creating an instance
        """
        if self.model is None:
            # dummy instance that is stored for API testing
            state = {"is_dummy": True}
            if verbose:
                print("Saving dummy PyTorchApproximator instance for API testing.")
            torch.save(state, path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
            return

        if not self.is_fitted:
            raise ModelNotFittedError("Cannot save an unfitted model.")
        
        # Check if we have custom transforms and need cloudpickle
        has_custom = any(
            tr.spec["type"] == "custom" 
            for tr in self.input_transforms + self.output_transforms
        )
        
        if has_custom and cloudpickle is None:
            raise NNApproxError(
                "Cannot save model with custom transforms: cloudpickle is required.\n"
                "Install with: pip install cloudpickle\n\n"
                "Alternatively, use predefined transforms: func.set_transform('x', transform_type='log')"
            )
        
        # Serialize transforms - only save specs, not callable objects
        def _serialize_transform(tr):
            spec = tr.spec.copy()
            # For custom transforms, we just pickle the function objects
            if spec["type"] == "custom":
                # cloudpickle made for this purpose
                spec["forward_func"] = cloudpickle.dumps(tr.forward)
                spec["inverse_func"] = cloudpickle.dumps(tr.inverse)
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


    def load(self, path: str, verbose: bool = False) -> "PyTorchApproximator":
        """
        Load a trained model from disk.
        
        Parameters
        ----------
        path : str
            Path to the saved model file
            
        Returns
        -------
        self : PyTorchApproximator
            The loaded approximator
            
        Raises
        ------
        NNApproxError
            If the file is invalid or custom transforms cannot be deserialized
            
        Examples
        --------
        >>> func = PyTorchApproximator(input=['x'], output=['y'])
        >>> func.load('my_model.pt')
        
        Notes
        -----
        - Input/output names from the saved model override the instance
        - Model is loaded to CPU for efficient inference
        - Custom transforms require the same Python environment
        
        See Also
        --------
        save : Save a model
        load_torch_approximator : Convenience function for loading
        """
        # Load directly to CPU
        state = torch.load(path, map_location="cpu", weights_only=False)
        
        if state.get("is_dummy", False): # returns True (value of is_dummy) or defaults to False
            # dummy instance for API testing
            if verbose:
                print("Loaded dummy PyTorchApproximator instance for API testing.")
            return self

        if "model_state" not in state:
            raise NNApproxError(f"Invalid checkpoint: missing 'model_state'.")
        
        # Check if cloudpickle is needed
        if state.get("uses_cloudpickle", False) and cloudpickle is None:
            raise NNApproxError(
                "This model was saved with custom transforms and requires cloudpickle to load.\n"
                "Install with: pip install cloudpickle"
            )

        # Restore input/output configuration
        self.input_names = state["input_names"]
        self.output_names = state["output_names"]
        self.input_dim = len(self.input_names)
        self.output_dim = len(self.output_names)
        
         # Reconstruct transforms from specs
        def _rebuild_transform(spec: dict) -> Transform:
            if spec["type"] == "predefined":
                return Transform.predefined(spec["name"])
            elif spec["type"] == "identity":
                return Transform.predefined("identity")
            elif spec["type"] == "custom":
                try:
                    forward_func = cloudpickle.loads(spec["forward_func"])
                    inverse_func = cloudpickle.loads(spec["inverse_func"])
                    return Transform.custom(forward_func, inverse_func)
                except Exception as e:
                    raise NNApproxError(
                        f"Failed to load custom transform: {e}\n"
                        f"This model may have been saved in a different Python version."
                    ) from e
            else:
                raise NNApproxError(f"Unknown transform type: {spec['type']}")

        self.input_transforms = [_rebuild_transform(s) for s in state["input_transforms"]]
        self.output_transforms = [_rebuild_transform(s) for s in state["output_transforms"]]

        # Restore scalers
        self._x_scaler = state["x_scaler"]
        self._y_scaler = state["y_scaler"]
        
        # Get activation class from saved info
        activation_info = state["activation"]
        activation_name = activation_info["name"]
        activation_class = getattr(nn, activation_name)

        # Recreate model with correct architecture (on CPU)
        self.model = MLPModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=state["hidden_dims"],
            activation=activation_class,
            dropout=state["dropout"],
        ) 
        
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
    """
    Load a saved PyTorchApproximator from file.
    
    This is a convenience function that creates a new approximator instance
    and loads the saved state into it.
    
    Parameters
    ----------
    path : str
        Path to the saved model file
        
    Returns
    -------
    approximator : PyTorchApproximator
        A fully initialized and fitted approximator ready for predictions
        
    Examples
    --------
    >>> import nnapprox as nna
    >>> 
    >>> # Load and use immediately
    >>> func = nna.load_approximator('model.pt', backend='torch')
    >>> predictions = func(x1, x2)
    
    >>> # Or use the backend-specific function
    >>> from nnapprox.backends.torch import load_torch_approximator
    >>> func = load_torch_approximator('model.pt')
    
    Notes
    -----
    This is equivalent to:
    
    >>> func = PyTorchApproximator(input=['dummy_input'], output=['dummy_output'])
    >>> func.load('model.pt')
    
    See Also
    --------
    PyTorchApproximator.load : Load into an existing instance
    PyTorchApproximator.save : Save a model
    """
    # Create a minimal dummy instance
    approximator = PyTorchApproximator(verbose=False) 
    approximator.load(path)
    return approximator
