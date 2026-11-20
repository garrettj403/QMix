"""Silence the OpenMP notice from Numba's threading layer.

Numba's OpenMP threading layer calls the deprecated ``omp_set_nested``, which
prints "OMP: Info #276" once per process. This turns off OpenMP's info
messages. It must run before the OpenMP runtime loads (i.e., before the first
parallel kernel), so it is imported first in ``qmix/__init__.py``. Setting
``KMP_WARNINGS`` in the environment overrides it.

"""

import os

os.environ.setdefault('KMP_WARNINGS', '0')
