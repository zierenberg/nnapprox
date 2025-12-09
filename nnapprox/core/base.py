from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Sequence, Union
import numpy as np
import pandas as pd
from ..core.utils import Transform
from ..core.exceptions import NNApproxError, ModelNotFittedError

class BaseApproximator(ABC):
    """
    Minimal interface that all backends must implement.

    Concrete classes **must** implement:
        * ``fit`` – train the model
        * ``predict`` – return predictions for arbitrary input formats
        * ``save`` / ``load`` – persistence

    The ``__call__`` method is provided here so that every subclass can be used
    like a plain Python function:

    >>> y = func(x1, x2)         # scalar call
    >>> y_arr = func(df)         # DataFrame call
    >>> df_out = func(df, return_dataframe=True)

    """

    def __init__(
        self,
        *,
        input: Sequence[str],
        output: Sequence[str],
        verbose: bool = False,
    ):
        # Meta information
        self.input_names = list(input)
        self.output_names = list(output)
        self.input_dim = len(self.input_names)
        self.output_dim = len(self.output_names)

        if set(self.input_names) & set(self.output_names):
            raise NNApproxError("Input and output names must be disjoint.")

        self.verbose = verbose

        # Transform placeholders
        self.input_transforms = [Transform.predefined("identity") for _ in range(self.input_dim)]
        self.output_transforms = [Transform.predefined("identity") for _ in range(self.output_dim)]

        # Fit flag
        self.is_fitted = False

    # general method that all backends should support
    def set_transform(
        self,
        label: str,
        *,
        predefined: str | None = None,
        forward: Any = None,
        inverse: Any = None,
    ) -> None:
        """
        Register a transformation for an input or output variable.
        
        Transformations are applied before training/prediction and inverted after
        prediction. This is useful for handling variables with different scales
        or non-linear relationships.
        
        Parameters
        ----------
        label : str
            Name of the variable (must be in input_names or output_names)
        predefined : str, optional
            Predefined transform name. Options: 'log', 'log10', 'sqrt', 'square',
            'identity'. Cannot be used with forward/inverse.
        forward : Callable, optional
            Custom forward transformation function. Must be used with inverse.
        inverse : Callable, optional
            Custom inverse transformation function. Must be used with forward.
        
        Raises
        ------
        ValueError
            If label is not a known input or output name, or if both predefined
            and forward/inverse are provided
        
        Examples
        --------
        Using predefined transforms:
        
        >>> func.set_transform('x', predefined='log')
        >>> func.set_transform('y', predefined='sqrt')
        
        Using custom transforms (must be defined in a module)
        
        >>> # In my_transforms.py:
        >>> # def cube(x): return x**3
        >>> # def cube_root(x): return x**(1/3)
        >>> 
        >>> from my_transforms import cube, cube_root
        >>> func.set_transform('x', forward=cube, inverse=cube_root)
        
        Notes
        -----
        Custom transforms defined as lambdas or in notebooks require cloudpickle
        for serialization and may not be portable across Python versions.
        
        See Also
        --------
        Transform.predefined : Available predefined transforms
        
        """
        if label in self.input_names:
            idx = self.input_names.index(label)
            target = self.input_transforms
        elif label in self.output_names:
            idx = self.output_names.index(label)
            target = self.output_transforms
        else:
            raise NNApproxError(f"{label!r} is not a known input or output name.")

        # Predefined transform
        if predefined:
            if forward is not None or inverse is not None:
                raise NNApproxError(
                    "Provide either `predefined` **or** both `forward`/`inverse`, not both."
                )
            target[idx] = Transform.predefined(predefined)
            return

        # Custom transform
        if forward is None or inverse is None:
            raise NNApproxError("Both `forward` and `inverse` must be supplied for a custom transform.")

        # Ensure functions are defined in a module
        if not callable(forward) or not callable(inverse):
            raise NNApproxError("Custom transform functions must be callable.")

        target[idx] = Transform.custom(forward, inverse)

    # make the approximator callable
    def __call__(self, *args: Any, **kwargs: Any):
        return self.predict(*args, **kwargs)

    def score(self, X: Any, y: Any) -> float:
        """Return the coefficient of determination R² of the prediction."""
        y_pred = self.predict(X)
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]
        y = np.asarray(y)
        y_pred = np.asarray(y_pred)
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return 1.0 - ss_res / ss_tot

    #backend must implement these methods
    @abstractmethod
    def fit(self, data: Any, **kwargs) -> BaseApproximator:
        ...

    @abstractmethod
    def predict(self, *args: Any, return_dataframe: bool = False, **kwargs) -> Any:
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @abstractmethod
    def load(self, path: str) -> BaseApproximator:
        ...