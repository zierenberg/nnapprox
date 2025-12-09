Installation
============

Requirements
------------

* Python >= 3.8
* PyTorch >= 1.9
* NumPy
* pandas
* scikit-learn

Install from PyPI
-----------------

.. code-block:: bash

   pip install nnapprox[torch]

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/yourusername/nnapprox.git
   cd nnapprox
   pip install -e .

Optional Dependencies
---------------------

For custom transform serialization:

.. code-block:: bash

   pip install cloudpickle