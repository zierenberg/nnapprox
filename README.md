# nnapprox (Neural Network Approximation)

[![Docs](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://zierenberg.github.io/nnapprox/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![codecov](https://codecov.io/gh/zierenberg/nnapprox/graph/badge.svg?token=23I9509JXH)](https://codecov.io/gh/zierenberg/nnapprox)
[![CI](https://github.com/zierenberg/nnapprox/actions/workflows/docs.yml/badge.svg)](https://github.com/zierenberg/nnapprox/actions)

Neural network function approximation using PyTorch (Future backends planned). Approximate arbitrary functions with automatic scaling, transformations, and serialization.

**[Full Documentation](https://zierenberg.github.io/nnapprox/)**

---


# Installation
Apple M1: Install pytorch with metal support via conda as described in the [Apple Instructions](https://developer.apple.com/metal/pytorch/)

```bash
pip install nnapprox
```

## Quick Start

```python
import nnapprox as nna
import numpy as np

# Generate data
x = np.linspace(0, 10, 100)
y = np.sin(x) * np.exp(-x/10)

# Train
func = nna.create_approximator(
    input=['x'],
    output=['y'],
    backend='torch'
)
func.fit({'x': x, 'y': y}, epochs=5000)

# Predict
y_pred = func(x)

# Save/load
func.save('model.nna')
func2 = nna.load_approximator('model.nna', backend='torch')
```

## Features

- **Simple API**: Train in 3 lines of code
- **Flexible inputs**: NumPy arrays, pandas DataFrames, scalars, or mixed types
- **Transformations**: Built-in (log, sqrt) or custom transforms
- **GPU support**: Automatic acceleration when available
- **Serialization**: Save/load complete model state

## Multi-dimensional Example

```python
import pandas as pd

x1, x2 = np.meshgrid(np.linspace(0,10,101), np.linspace(0,10,101))
x1 = x1.flatten()
x2 = x2.flatten()
def y_true(x1, x2):
    return 2*x1 + 3*x2**2 + 1
data = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y_true(x1, x2)})

func = nna.create_approximator(
    input=['x1', 'x2'],
    output=['y'],
    backend='torch'
)
func.fit(data, epochs=3000)
predictions = func(data, return_dataframe=True)
```

## Transformations

```python
# Predefined transforms
func.set_transform('x1', predefined='log')
func.set_transform('y', predefined='exp')

# Custom transforms
def forward_fn(x):
    return x**3
def inverse_fn(y):
    return y**(1/3)
func.set_transform('x1', forward=forward_fn, inverse=inverse_fn)
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.9
- NumPy >= 1.20
- pandas >= 1.3
- scikit-learn >= 1.0

Optional: `cloudpickle` for lambda function serialization

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## License

MIT License - see [LICENSE](LICENSE) file.
