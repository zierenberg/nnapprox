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

    >>> y = approximator(x1, x2)         # scalar call
    >>> y_arr = approximator(df)         # DataFrame call
    >>> df_out = approximator(df, return_dataframe=True)
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
        transform_type: str | None = None,
        forward: Any = None,
        inverse: Any = None,
    ) -> None:
        """
        Register a forward / inverse transformation for a variable.

        *Pre‑defined* transforms are referenced by ``transform_type`` (e.g.
        ``"log"``).  For a custom pair supply ``forward`` and ``inverse``.
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
        if transform_type:
            if forward is not None or inverse is not None:
                raise NNApproxError(
                    "Provide either `transform_type` **or** both `forward`/`inverse`, not both."
                )
            target[idx] = Transform.predefined(transform_type)
            return

        # Custom transform
        if forward is None or inverse is None:
            raise NNApproxError("Both `forward` and `inverse` must be supplied for a custom transform.")

        # Ensure functions are defined in a module
        import inspect
        try:
            inspect.getsource(forward)
            inspect.getsource(inverse)
        except OSError as exc:
            raise NNApproxError("Custom transform functions must be defined in a module.") from exc

        target[idx] = Transform.custom(forward, inverse)

    # helper data extraction method
    def _extract_arrays(self, data: dict) -> tuple[np.ndarray, np.ndarray]:
        """Return X, Y as plain NumPy arrays."""
        X = np.column_stack([np.asarray(data[name]) for name in self.input_names])
        Y = np.column_stack([np.asarray(data[name]) for name in self.output_names])
        return X, Y

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