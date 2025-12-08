import numpy as np
from typing import Any, Callable

class Transform:
    """Container for a forward / inverse pair and a serialisable spec."""
    __slots__ = ("forward", "inverse", "spec")

    def __init__(
        self,
        forward: Callable[[Any], Any],
        inverse: Callable[[Any], Any],
        spec: dict,
    ) -> None:
        self.forward = forward
        self.inverse = inverse
        self.spec = spec

    @classmethod
    def predefined(cls, name: str) -> "Transform":
        """Factory for the four built‑in transforms."""
        predefined = {
            "identity": (lambda x: x, lambda x: x),
            "log": (np.log, np.exp),
            "log10": (np.log10, lambda x: np.power(10, x)),
            "exp": (np.exp, np.log),
        }
        if name not in predefined:
            raise ValueError(f"Unknown predefined transform {name!r}")
        fwd, inv = predefined[name]
        return cls(fwd, inv, {"type": "predefined", "name": name})

    @classmethod
    def custom(cls, forward: Callable, inverse: Callable) -> "Transform":
        """Factory for user‑supplied callables."""
        # Store importable location so the spec can be re‑created later.
        spec = {
            "type": "custom",
            "forward_path": f"{forward.__module__}.{forward.__name__}",
            "inverse_path": f"{inverse.__module__}.{inverse.__name__}",
        }
        return cls(forward, inverse, spec)