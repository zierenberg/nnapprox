from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np

class BaseApproximator(ABC):
    """Abstract base class for neural network approximators."""
    
    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_fitted = False
        self._model = None
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'BaseApproximator':
        """Fit the approximator to training data."""
        pass
     
    @abstractmethod
    def save(self, filepath: str) -> None:
        """Save the trained model."""
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> None:
        """Load a trained model."""
        pass
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the coefficient of determination R^2."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    

