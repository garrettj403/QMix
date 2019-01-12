Getting Started
===============

Installing QMix using pip
-------------------------

The easiest way to install QMix is using ``pip``:

.. code-block:: bash

   pip install QMix

Installing QMix from GitHub
---------------------------

If you want the latest version, you can download QMix from `GitHub`_. In order to clone the repository, open a terminal window, navigate to the directory in which you would like to put QMix, and then run:

.. _GitHub: https://github.com/garrettj403/QMix/

.. code-block:: bash
   
   git clone https://github.com/garrettj403/QMix.git QMix

This will download QMix into a new directory called ``QMix/``. You then need to add the ``QMix/`` directory to your ``PYTHONPATH`` environment variable. If you are on a macOS system, open your bash profile (normally located in ``~/.bash_profile`` or ``~/.profile``) and add the following:

.. code-block:: bash

   export PYTHONPATH="<path-to-qmix>:$PYTHONPATH"

where ``<path-to-qmix>`` is the path to the ``QMix/`` directory on your computer. 

Configuring Python
------------------

QMix is intended for Python version 3.7. If you don't want to change your Python installation or in case you are missing any packages, I have included a virtual environment file that contains all of the neccesary packages. If you are using the Anaconda package manager, you can run:

.. code-block:: bash

   conda env create -f environment.yml

This may take a few minutes. You can then activate the virtual environment using:

.. code-block:: bash

   source activate qmix

Testing Your Installation
-------------------------

From the ``QMix/`` root directory, you can test your QMix installation by running:

.. code-block:: python

   pytest --verbose --color=yes tests/

All of the tests should pass; although, a few ``FutureWarnings`` may pop up. These are caused by the SciPy package, and they are normally suppressed when QMix is running. If you encounter any other issues, (1) make sure that you are using the virtual environment that I provided with the package, and then (2) let me know through the `issue tracking system`__ on GitHub. You can also test the package by running the workflow examples that are contained in the ``QMix/examples/`` directory.

.. __: https://github.com/garrettj403/QMix/issues/