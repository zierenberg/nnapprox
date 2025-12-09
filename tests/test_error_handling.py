import nnapprox as nna
import numpy as np
import pytest



from nnapprox.core.exceptions import BackendNotAvailableError

def test_missing_torch(monkeypatch):
    # Simulate torch import failure
    module = "nnapprox.backends.torch.approximator"
    monkeypatch.setattr(f"{module}.torch", None)
    monkeypatch.setattr(f"{module}.nn", None)
    monkeypatch.setattr(f"{module}.optim", None)

    with pytest.raises(nna.BackendNotAvailableError):
        nna.create_approximator(
            input=["x"],
            output=["y"],
            backend="torch",
        )

def test_create_approximator_errors():
    with pytest.raises(BackendNotAvailableError):
        nna.create_approximator(
            input=["x"],
            output=["y"],
            backend="nonexistent_backend"
        )  

def test_save_and_load_model_errors():
    with pytest.raises(BackendNotAvailableError):
        nna.load_approximator("/dev/null/", backend="nonexistent_backend")


def test_missing_cloudpickle(monkeypatch):
    # Simulate cloudpickle import failure
    monkeypatch.setattr("nnapprox.backends.torch.approximator.cloudpickle", None)

    with pytest.raises(nna.NNApproxError):
        func = nna.create_approximator(
            input=["x"],
            output=["y"],
            backend="torch",
        )
        func.save("/dev/null/")