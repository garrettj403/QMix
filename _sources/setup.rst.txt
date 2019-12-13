Getting Started
===============

Installing QMix using pip
-------------------------

The easiest way to install QMix is using ``pip``. You can install it via the Python Package Index (PyPI):

.. code-block:: console

   python -m pip install QMix

Or you can install it via GitHub to get the latest version:

.. code-block:: console

   python -m pip install git+https://github.com/garrettj403/QMix.git

Downloading QMix from GitHub
----------------------------

If you want the example files, you can download QMix from `GitHub`_. 

.. _GitHub: https://github.com/garrettj403/QMix/

.. warning:: The instructions below are for macOS. Some of the commands will be slightly different for Windows or Linux.

To clone the repository, open a terminal window, navigate to the directory in which you would like to put QMix, and then run:

.. code-block:: console
   
   git clone https://github.com/garrettj403/QMix.git QMix

This will download QMix into a new directory called ``QMix/``. You then need to install QMix.

.. code-block:: console

   cd QMix/
   python -m pip install -e .

Configuring Python
------------------

QMix is written using Python version 3.7. If you don't want to change your Python installation or in case you are missing any packages, I have included a virtual environment file that contains all of the necessary packages. If you are using the Anaconda package manager, you can run:

.. code-block:: console

   conda env create -f environment.yaml

This may take a few minutes. You can then activate the virtual environment using:

.. code-block:: console

   conda activate qmix

Testing Your Installation
-------------------------

From the ``QMix/`` root directory, you can test your QMix installation by running:

.. code-block:: python

   pytest --verbose --color=yes tests/

All of the tests should pass; although, a few ``FutureWarnings`` may pop up. These are caused by the SciPy package, and they are normally suppressed when QMix is running. If you encounter any other issues, (1) make sure that you are using the virtual environment that I provided with the package, and then (2) let me know through the `issue tracking system`__ on GitHub. You can also test the package by running the workflow examples that are contained in the ``QMix/examples/`` directory.

.. __: https://github.com/garrettj403/QMix/issues/
