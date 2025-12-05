import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import List, Optional, Union, Tuple, Any, Callable
import inspect
import pickle

from ...core.base import BaseApproximator
from ...core.exceptions import ModelNotFittedError

from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

class MLPModel(nn.Module):
    """Multi-layer perceptron model."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int], 
                 activation: str = "tanh", dropout: float = 0.0):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dropout = dropout
        
        self.layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation/dropout on output layer
                if activation.lower() == "relu":
                    self.layers.append(nn.ReLU())
                elif activation.lower() == "tanh":
                    self.layers.append(nn.Tanh())
                elif activation.lower() == "sigmoid":
                    self.layers.append(nn.Sigmoid())
                
                if dropout > 0:
                    self.layers.append(nn.Dropout(dropout))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class PyTorchApproximator(BaseApproximator):
    """PyTorch-based neural network approximator with versatile input handling."""
    
    _predefined_transforms = {
        'identity': {'forward': lambda x: x, 'inverse': lambda x: x},
        'log': {'forward': np.log, 'inverse': np.exp}, # natural logarithm
        'log10': {'forward': np.log10, 'inverse': lambda x: np.power(10, x)}, # base 10 logarithm
        'exp': {'forward': np.exp, 'inverse': np.log}, # exponential
    }

    def __init__(self, verbose=False, **kwargs):
        self.verbose = verbose
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() 
            else "mps" if torch.backends.mps.is_available() 
            else "cpu"
        )
        
        if "params" in kwargs:
            self.params = kwargs["params"]
        else:
            self.params = None
            
        if "filename" in kwargs:
            filename = kwargs["filename"]
            if self.verbose:
                print(f"Loading function approximation from {filename} and ignore additional arguments!")
            self.load(filename)
        else:
            self.init(**kwargs)

    def init(self, **kwargs):
        self.input_names = kwargs["input"]
        self.output_names = kwargs["output"]
        assert len(set(self.input_names).intersection(set(self.output_names))) == 0
        
        self.input_dim = len(self.input_names)
        self.output_dim = len(self.output_names)
        
        # Initialize with 'identity' transform for all inputs and outputs
        # Stores tuples: ('predefined', 'identity') or ('custom', forward_src, inverse_src)
        self.input_transform_specs: List[Tuple[str, ...]] = [('predefined', 'identity') for _ in range(self.input_dim)]
        self.output_transform_specs: List[Tuple[str, ...]] = [('predefined', 'identity') for _ in range(self.output_dim)]
        
        # Runtime storage for actual callable functions
        self.input_transforms: List[Callable] = [self._predefined_transforms['identity']['forward'] for _ in range(self.input_dim)]
        self.output_transforms: List[Callable] = [self._predefined_transforms['identity']['forward'] for _ in range(self.output_dim)]
        self.output_inverse_transforms: List[Callable] = [self._predefined_transforms['identity']['inverse'] for _ in range(self.output_dim)]

        # ranges for input that the model is trained on (initialize with None)
        self.input_range = [(None, None) for _ in range(self.input_dim)]

        if "model" in kwargs:
            self.model = kwargs["model"]
            assert self.model.input_dim == self.input_dim
            assert self.model.output_dim == self.output_dim
            # Extract model parameters for saving/loading
            self.hidden_dims = self.model.hidden_dims
            self.activation = self.model.activation
            self.dropout = self.model.dropout
        else:
            if self.verbose:
                print("Initializing default model (choose own model with kwargs `model`).")
            self.hidden_dims = kwargs.get('hidden_dims', [42, 42, 42])
            self.activation = kwargs.get('activation', 'tanh')
            self.dropout = kwargs.get('dropout', 0.0)
            
            self.model = MLPModel(
                input_dim=self.input_dim,
                hidden_dims=self.hidden_dims,
                output_dim=self.output_dim,
                activation=self.activation,
                dropout=self.dropout
            )
        
        if self.verbose:
            print(self.model)

        # Attention: scaler has to match the activation function:
        # tanh: [-1,1], sigmoid: [0,1]
        feature_range = (-1, 1) if self.activation.lower() == 'tanh' else (0, 1)
        self.X_scaler = MinMaxScaler(feature_range=feature_range)
        self.Y_scaler = MinMaxScaler(feature_range=feature_range)
        
        self.is_fitted = False

        if self.verbose:
            print(
                "Next steps:\n>> func.set_transform('input_var_name', 'log') # set a predefined transformation for an input \n>> func.set_transform('output_var_name', forward=lambda x: x**2, inverse=lambda x: np.sqrt(x)) # set custom transformations\n>> func.train() # train neural network model\n>> func.save(filename) # save function approximation to file"
            )

    @property
    def is_fitted(self):
        return getattr(self, '_is_fitted', False)
    
    @is_fitted.setter
    def is_fitted(self, value):
        self._is_fitted = value

    def set_transform(self, label: str,
                      transform_type: Optional[str] = None,
                      forward: Optional[Callable] = None,
                      inverse: Optional[Callable] = None):
        """
        Specifies a transformation for a given input or output label.
        
        Args:
            label (str): The name of the input or output variable.
            transform_type (str, optional): Name of a predefined transformation 
                                            (e.g., 'log', 'log10', 'exp', 'identity').
                                            If provided, 'forward' and 'inverse' are ignored.
            forward (Callable, optional): Custom forward transformation function (lambda).
                                          Required if transform_type is None.
            inverse (Callable, optional): Custom inverse transformation function (lambda).
                                          Required if transform_type is None.
        
        Raises:
            ValueError: If label not found or invalid combination of arguments.
        """
        
        target_list_specs = None
        target_list_fwd = None
        target_list_inv = None
        idx = -1

        if label in self.input_names:
            idx = self.input_names.index(label)
            target_list_specs = self.input_transform_specs
            target_list_fwd = self.input_transforms
        elif label in self.output_names:
            idx = self.output_names.index(label)
            target_list_specs = self.output_transform_specs
            target_list_fwd = self.output_transforms
            target_list_inv = self.output_inverse_transforms
        else:
            raise ValueError(f"Label '{label}' not found in input or output names.")

        if transform_type:
            if forward is not None or inverse is not None:
                raise ValueError("Cannot provide both 'transform_type' and custom 'forward'/'inverse' functions.")
            if transform_type not in self._predefined_transforms:
                raise ValueError(f"Unknown predefined transform type: '{transform_type}'. "
                                 f"Available: {list(self._predefined_transforms.keys())}")
            
            target_list_specs[idx] = ('predefined', transform_type)
            target_list_fwd[idx] = self._predefined_transforms[transform_type]['forward']
            if target_list_inv is not None: # Only outputs have inverse transforms stored separately for prediction
                target_list_inv[idx] = self._predefined_transforms[transform_type]['inverse']

        elif forward is not None and inverse is not None:
            # WARNING: Storing and re-evaluating lambda functions is not secure and can break
            # if the source code changes. Consider a more robust serialization for production.
            try:
                forward_src = inspect.getsource(forward)
                inverse_src = inspect.getsource(inverse)
            except OSError as e:
                raise ValueError(f"Could not get source for lambda function. Ensure it's defined "
                                 f"in a module (not interactively): {e}")

            target_list_specs[idx] = ('custom', forward_src, inverse_src)
            target_list_fwd[idx] = forward
            if target_list_inv is not None:
                target_list_inv[idx] = inverse
        else:
            raise ValueError("Must provide either 'transform_type' or both 'forward' and 'inverse' functions.")

    def prepare_data(self, dataframe: pd.DataFrame):
        """Prepare data from dataframe for training."""
        assert self.model.training

        # extract training data from dataframe into numpy arrays
        Xs = dataframe[self.input_names].values
        Ys = dataframe[self.output_names].values

        # specify input ranges used for training
        for i in range(self.input_dim):
            self.input_range[i] = (np.min(Xs[:, i]), np.max(Xs[:, i]))

        # apply transformations
        for i in range(self.input_dim):
            Xs[:, i] = self.input_transforms[i](Xs[:, i])
        for i in range(self.output_dim):
            # For Ys, we need to apply the forward transform before scaling and training
            # So, we use the forward transform for outputs
            Ys[:, i] = self.output_transforms[i](Ys[:, i])

        # rescale data into activation function range
        Xs = self.X_scaler.fit_transform(Xs)
        Ys = self.Y_scaler.fit_transform(Ys)

        return Xs, Ys
    
    def train(self, Xs: np.ndarray, Ys: np.ndarray, custom_loss=None, epochs=10000, lr=1e-3, eps=1e-5, amsgrad=False, device=None):
        """Trains the neural network."""
        # select suitable device for training
        if device is None:
            device = self.device
        if self.verbose:
            print("Device for training: {}".format(device))

        # convert to torch tensors
        X_tensor = torch.from_numpy(Xs).float().to(device)
        Y_tensor = torch.from_numpy(Ys).float().to(device)
        model_ = self.model.to(device)
        if self.verbose:
            print(f"Training data with input shape {X_tensor.shape} and output shape {Y_tensor.shape}.")

        # Adam and MSE Loss
        optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=eps, amsgrad=amsgrad)
        loss_fn = nn.MSELoss(reduction="mean")
        if custom_loss is None:
            def custom_loss(Y_pred, Y, X):
                return loss_fn(Y_pred, Y)

        history_loss = []
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
            history_loss.append(loss.item())
            
        # fetch model from device
        self.model = model_.to("cpu")
        # set training to false to enter evaluation mode
        self.model.train(False)
        self.is_fitted = True
        
        if self.verbose:
            print("Training complete; left training mode for faster evaluation! To re-enter training mode call `func.model.train(True)`.")

        return history_loss

    def save(self, filepath: str) -> None:
        """Save the trained model."""
        if not self.is_fitted:
            raise ModelNotFittedError("Cannot save unfitted model")
        
        # save all relevant objects for the function approximation (not training!)
        state = {
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dims': self.hidden_dims,
            'activation': self.activation,
            'dropout': self.dropout,
            'is_fitted': self.is_fitted,
            'input_names': self.input_names,
            'output_names': self.output_names,
            'input_transform_specs': self.input_transform_specs, # Store specifications for transforms
            'output_transform_specs': self.output_transform_specs, # Store specifications for transforms
            'X_scaler': self.X_scaler,
            'Y_scaler': self.Y_scaler,
            'input_range': self.input_range,
            'params': self.params,
        }
        torch.save(state, filepath, pickle_protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filepath: str) -> None:
        """Load a trained model."""
        state = torch.load(filepath, map_location=self.device)
        
        self.input_dim = state['input_dim']
        self.output_dim = state['output_dim']
        self.hidden_dims = state['hidden_dims']
        self.activation = state['activation']
        self.dropout = state['dropout']
        self.is_fitted = state['is_fitted']
        self.input_names = state['input_names']
        self.output_names = state['output_names']
        
        self.input_transform_specs = state['input_transform_specs']
        self.output_transform_specs = state['output_transform_specs']
        
        # Re-construct callable functions from specs
        self.input_transforms: List[Callable] = []
        for spec_type, *args in self.input_transform_specs:
            if spec_type == 'predefined':
                self.input_transforms.append(self._predefined_transforms[args[0]]['forward'])
            elif spec_type == 'custom':
                # WARNING: eval is not secure. Use with caution.
                self.input_transforms.append(eval(args[0]))
            else:
                raise ValueError(f"Unknown transform spec type: {spec_type}")

        self.output_transforms: List[Callable] = [] # Stores forward transforms for outputs (for prepare_data)
        self.output_inverse_transforms: List[Callable] = [] # Stores inverse transforms for outputs (for predict)
        for spec_type, *args in self.output_transform_specs:
            if spec_type == 'predefined':
                self.output_transforms.append(self._predefined_transforms[args[0]]['forward'])
                self.output_inverse_transforms.append(self._predefined_transforms[args[0]]['inverse'])
            elif spec_type == 'custom':
                # WARNING: eval is not secure. Use with caution.
                self.output_transforms.append(eval(args[0]))
                self.output_inverse_transforms.append(eval(args[1]))
            else:
                raise ValueError(f"Unknown transform spec type: {spec_type}")

        self.X_scaler = state['X_scaler']
        self.Y_scaler = state['Y_scaler']
        self.input_range = state['input_range']
        self.params = state.get('params', None) # Handle params for backward compatibility
        
        self.model = MLPModel(
            self.input_dim, self.output_dim, self.hidden_dims,
            self.activation, self.dropout
        ).to(self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.model.eval() # Set to eval mode after loading

    def _prepare_inputs(self, *args) -> Tuple[np.ndarray, int]:
        """
        Prepare inputs for prediction, handling various input formats:
        - Individual scalars/arrays: func(x1, x2, x3)
        - Mixed scalars/arrays: func(scalar, array, scalar)
        - DataFrame: func(dataframe)
        
        Returns:
            Tuple of (prepared_inputs_array, max_length_of_inputs)
        """
        if len(args) == 1 and isinstance(args[0], pd.DataFrame):
            # DataFrame input
            df = args[0]
            if not all(col in df.columns for col in self.input_names):
                missing = [col for col in self.input_names if col not in df.columns]
                raise ValueError(f"DataFrame missing required columns: {missing}")
            inputs = [df[name].values for name in self.input_names]
        else:
            # Individual arguments
            if len(args) != self.input_dim:
                raise ValueError(f"Expected {self.input_dim} input values, but got {len(args)}!")
            inputs = list(args)

        # Convert inputs to numpy arrays and find the maximum length
        max_length = 1
        processed_inputs = []
        
        for i, x in enumerate(inputs):
            if np.isscalar(x):
                x_array = np.array([x])
            else:
                x_array = np.asarray(x) # Use asarray for flexibility with lists/tuples
                if x_array.ndim == 0:
                    x_array = np.array([x_array.item()])
                elif x_array.ndim > 1:
                    raise ValueError(f"Input {i} must be scalar or 1D array, got shape {x_array.shape}")
            
            max_length = max(max_length, len(x_array))
            processed_inputs.append(x_array)
        
        # Broadcast all inputs to the same length
        for i, x_array in enumerate(processed_inputs):
            if len(x_array) == 1 and max_length > 1:
                # Broadcast scalar to array
                processed_inputs[i] = np.full(max_length, x_array[0])
            elif len(x_array) != max_length and len(x_array) != 1:
                raise ValueError(f"Input {i} has length {len(x_array)}, expected 1 or {max_length}")
            
            # Apply input transformation using the stored forward functions
            processed_inputs[i] = self.input_transforms[i](processed_inputs[i])
        
        # Stack into 2D array
        X = np.column_stack(processed_inputs)
        return X, max_length

    def predict(self, *args, return_dataframe: bool = False) -> Union[Tuple[np.ndarray, ...], pd.DataFrame]:
        """
        Make predictions with versatile input handling.
        
        Args:
            *args: Input data - can be scalars, arrays, or a single DataFrame.
                   If a DataFrame is provided, it should be the first and only positional argument.
            return_dataframe (bool): If True, return results as a DataFrame including input columns.
                                    Defaults to False.
            
        Returns:
            Tuple of output arrays or DataFrame with results.
            Each output array will be 1D if original input was scalar/1D, 
            or matching the broadcasted length if inputs were mixed or multi-element arrays.
        """
        if self.model.training:
            raise ValueError("FunctionApproximation is in training mode. Call `func.model.eval()` to switch to evaluation mode.")
        
        if not self.is_fitted:
            raise ModelNotFittedError("Model must be fitted before making predictions")

        original_inputs_for_df = None
        if len(args) == 1 and isinstance(args[0], pd.DataFrame):
            # Store original DataFrame for building output DataFrame
            original_inputs_for_df = args[0]
        
        X, max_length = self._prepare_inputs(*args)
        
        # Scale inputs
        X_scaled = self.X_scaler.transform(X)
        
        # Predict
        # with torch.no_grad():
            # Y_scaled = self.model(torch.from_numpy(X_scaled).float().to(self.device)).detach().cpu().numpy()  
        Y_scaled = self.model(torch.from_numpy(X_scaled).float()).detach().numpy()
        
        # Inverse scale outputs
        Y = self.Y_scaler.inverse_transform(Y_scaled)
        
        # Apply inverse output transformations and prepare results
        outputs = []
        for i in range(self.output_dim):
            y_i = self.output_inverse_transforms[i](Y[:, i]) # Use inverse transforms for outputs
            # If input was scalar and output is single value, return scalar
            if max_length == 1:
                y_i = y_i[0] if hasattr(y_i, '__getitem__') else y_i
            outputs.append(y_i)
        
        if return_dataframe:
            # Prepare input columns using the original (untransformed) input
            input_data_for_df = {}
            if original_inputs_for_df is not None:
                for name in self.input_names:
                    input_data_for_df[name] = original_inputs_for_df[name].values
            else:
                # Reconstruct inputs if not from DataFrame
                reconstructed_args = []
                for i, arg_val in enumerate(args):
                    if np.isscalar(arg_val):
                        reconstructed_args.append(np.full(max_length, arg_val))
                    else:
                        reconstructed_args.append(np.asarray(arg_val) if np.asarray(arg_val).ndim > 0 else np.full(max_length, arg_val.item()))
                
                for i, name in enumerate(self.input_names):
                    input_data_for_df[name] = reconstructed_args[i]

            # Prepare output columns
            output_data_for_df = {}
            for name, output_val in zip(self.output_names, outputs):
                if max_length == 1 and not np.isscalar(output_val): # Ensure scalar output is wrapped for DF if needed
                    output_data_for_df[name] = [output_val.item()] if hasattr(output_val, 'item') else [output_val]
                else:
                    output_data_for_df[name] = output_val

            # Combine input and output columns
            data = {**input_data_for_df, **output_data_for_df}
            return pd.DataFrame(data)
        
        return tuple(outputs)
    
    def __call__(self, *args, **kwargs):
        """
        Versatile call interface supporting multiple input formats:
        
        Examples:
            # All scalars
            y1, y2 = func(1.0, 2.0, 3.0)
            
            # Mixed scalar and array
            y1, y2 = func(1.0, [2.0, 2.1, 2.2], 3.0)
            
            # All arrays
            y1, y2 = func([1.0, 1.1], [2.0, 2.1], [3.0, 3.1])
            
            # DataFrame input, tuple output
            y1, y2 = func(input_df)
            
            # DataFrame input, DataFrame output
            results_df = func(input_df, return_dataframe=True)
        """
        return self.predict(*args, **kwargs)