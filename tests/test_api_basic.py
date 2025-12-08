import nnapprox as nna
import numpy as np
import pytest

def test_create_approximator_basic():
    # minimal valid call
    func = nna.create_approximator(
        input=["x"],
        output=["y"],
        backend="pytorch"
    )
    assert func is not None

def test_create_approximator_invalid_backend():
    with pytest.raises(Exception):
        nna.create_approximator(
            input=["x"],
            output=["y"],
            backend="nonexistent_backend"
        )

def test_fit_accepts_minimal_data():
    func = nna.create_approximator(
        input=["x"],
        output=["y"],
        backend="pytorch"
    )
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 2.0])
    func.fit({"x": x, "y": y}, epochs=1)

def test_predict_returns_numpy_array():
    func = nna.create_approximator(
        input=["x"],
        output=["y"],
        backend="pytorch"
    )

    # minimal training to allow calling predict
    func.fit({"x": np.array([0, 1]), "y": np.array([0, 1])}, epochs=1)

    out = func(np.array([0.5, 1.5]))
    assert isinstance(out, np.ndarray)

import tempfile
import os

def test_save_and_load_model_api():
    func = nna.create_approximator(
        input=["x"],
        output=["y"],
        backend="pytorch"
    )
    func.fit({"x": np.array([0, 1]), "y": np.array([0, 1])}, epochs=1)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.nna")
        func.save(path)

        func2 = nna.load_approximator(path, backend="pytorch")
        assert func2 is not None
        assert callable(func2)