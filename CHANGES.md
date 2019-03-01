v1.0.2 (unreleased)
===================

New Features
------------

- All plotting functions now accept and return Matplotlib axis objects.
- When importing experimental data, you can now skip any headers that may be present in the CSV files.

Deprecated Features
-------------------

- Dropped Travis-CI testing with Python 3.5-dev and 3.6-dev. QMix is now only tested with 3.5, 3.6 and 3.7-dev.
- The ``f_idc``, ``f_ikk``, ``f_didc``, and ``f_dikk`` attributes were all removed from the ``qmix.respfn.RespFn`` class. They are replaced by the ``idc``, ``ikk``, ``didc`` and ``dikk`` methods, respectively.

Bug Fixes
---------

- Fixed potential issue with file paths. Previously, file paths were built for macOS (e.g., ``"some-dir/this-figure.png"``). The package was updated to now use ``os.path.join("some-dir", "this-figure.png")``, which is machine independent.

Documentation
-------------

- Added package API to web page.
- Added information on getting support and contributing to QMix.
- Added descriptions of different response function types to docstring in ``qmix.respfn.py``.
- Added more comments on keyword arguments and attributes for classes in ``qmix.circuit.py`` and ``qmix.respfn.py``. This should help to explain some of the SIS-related jargon.
- Added use cases for many class methods.
- Fixed documentation for ``qmix.exp.parameters.py``.
- Updated workflow examples.
- Other (minor) changes to documentation.



v1.0.1 (5-Feb-2019)
===================

Bug Fixes
---------

- Fixed long description in setup.py. Previously, there were some non-ASCII characters, which were causing issues on Windows installations.



v1.0.1 (11-Jan-2019)
====================

Initial release.
