Getting Started
===============

Installing QMix using pip
-------------------------

The easiest way to install QMix is using ``pip``:

.. code-block:: console

   python -m pip install QMix

Or you can install it via GitHub to get the latest version:

.. code-block:: console

   python -m pip install git+https://github.com/garrettj403/QMix.git

Downloading QMix from GitHub
----------------------------

If you want a local copy of the repository, you can download QMix from `GitHub`_: 

.. _GitHub: https://github.com/garrettj403/QMix/

.. code-block:: console
   
   # Download QMix
   git clone https://github.com/garrettj403/QMix.git QMix
   cd QMix/

   # Create a virtual environment using Anaconda
   conda env create -f environment.yml
   conda activate qmix

   # Install QMix
   python -m pip install -e .

   # Test your installation
   pytest --verbose --color=yes tests/

If you encounter any other issues, (1) make sure that you are using the virtual environment that I provided with the package, and then (2) let me know through the `issue tracking system`__ on GitHub. You can also test the package by running the workflow examples that are contained in the ``QMix/notebooks/`` directory.

.. __: https://github.com/garrettj403/QMix/issues/
