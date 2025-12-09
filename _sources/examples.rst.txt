Examples
========

Example 1: Simple Function Approximation
-----------------------------------------

.. code-block:: python

   import nnapprox as nna
   import numpy as np

   # Generate data
   x = np.linspace(-5, 5, 200)
   y = x**3 - 2*x**2 + x + 5

   # Create approximator
   func = nna.create_approximator(
       input=['x'],
       output=['y'],
       backend='torch'
   )

   # Train
   func.fit({'x': x, 'y': y}, epochs=2000)

   # Predict
   x_test = np.array([0, 1, 2, 3])
   y_pred = func(x_test)
