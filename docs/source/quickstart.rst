Quick Start
===========

Basic Usage
-----------

.. code-block:: python

   import nnapprox as nna
   import numpy as np

   # Create sample data
   x1 = np.linspace(0, 10, 100)
   x2 = np.linspace(0, 5, 100)
   y = np.sin(x1) * np.cos(x2)

   data = {'x1': x1, 'x2': x2, 'y': y}

   # Create and train approximator
   func = nna.create_approximator(
       input=['x1', 'x2'],
       output=['y'],
       backend='pytorch',
       hidden_dims=[64, 64],
   )

   func.fit(data, epochs=1000, lr=1e-3)

   # Make predictions
   predictions = func.predict(x1_new, x2_new)

With Transforms
---------------

.. code-block:: python

   # Use predefined transforms
   func.set_transform('x1', transform_type='log')
   func.set_transform('y', transform_type='log')

   # Or custom transforms
   func.set_transform(
       'x2',
       forward=lambda x: x**2,
       inverse=lambda x: np.sqrt(x)
   )

Saving and Loading
------------------

.. code-block:: python

   # Save
   func.save('my_model.pt')

   # Load
   func2 = nna.load_approximator('my_model.pt', backend='pytorch')