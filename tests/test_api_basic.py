import nnapprox as nna
import numpy as np
import pytest

from nnapprox.core.exceptions import BackendNotAvailableError

def test_create_approximator_basic():
    # minimal valid call tp torch backend
    func = nna.create_approximator(
        input=["x"],
        output=["y"],
        backend="torch"
    )
    assert func is not None  

def test_transform_setting_api():
    func = nna.create_approximator(
        input=["x1", "x2", "x3", "x4"],
        output=["y"],
        backend="torch"
    )

    # set default transforms for inputs
    func.set_transform('x3', predefined='identity')
    func.set_transform('x2', predefined='log10')
    func.set_transform('x1', predefined='log')
    func.set_transform('x4', predefined='exp')

    # set custom transform for output
    func.set_transform('y',
        forward=lambda y: y**3,
        inverse=lambda y: y**(1/3)
    )

    # test error handling
    with pytest.raises(nna.NNApproxError): # test wrong API (both predefined and custom)
        func.set_transform('x1', predefined='log', forward=lambda x: x, inverse=lambda x: x)
    with pytest.raises(nna.NNApproxError): # test unknown input name
        func.set_transform('x0', predefined='log')

def test_fit_accepts_minimal_data():
    func = nna.create_approximator(
        input=["x"],
        output=["y"],
        backend="torch"
    )
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 2.0])
    func.fit({"x": x, "y": y}, epochs=1)

def test_predict_returns_numpy_array():
    func = nna.create_approximator(
        input=["x"],
        output=["y"],
        backend="torch"
    )

    # minimal training to allow calling predict
    func.fit({"x": np.array([0, 1]), "y": np.array([0, 1])}, epochs=1)

    out = func(np.array([0.5, 1.5]))
    assert isinstance(out, np.ndarray)

def test_score_method():
    func = nna.create_approximator(
        input=["x"],
        output=["y"],
        backend="torch"
    )

    # minimal training to allow calling predict
    func.fit({"x": np.array([0, 1]), "y": np.array([0, 1])}, epochs=1)

    x_test = np.array([0.2, 0.4, 0.6, 0.8])
    y_test = np.array([0.2, 0.4, 0.6, 0.8])
    r2 = func.score(x_test, y_test)
    assert isinstance(r2, float)

import tempfile
import os

def test_save_and_load_model_api():
    func = nna.create_approximator(
        input=["x"],
        output=["y"],
        backend="torch"
    )
    func.fit({"x": np.array([0, 1]), "y": np.array([0, 1])}, epochs=1)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.nna")
        func.save(path)

        func2 = nna.load_approximator(path, backend="torch")
        assert func2 is not None
        assert callable(func2)

        with pytest.raises(BackendNotAvailableError):
            nna.load_approximator(path, backend="nonexistent_backend")