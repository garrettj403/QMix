v1.0.3 (20-Mar-2019)
====================

- This release was created after the JOSS review.
- There are minor changes to the documentation, but no changes to the code base.



v1.0.2 (11-Mar-2019)
====================

New Features
------------

- All plotting functions now accept and return Matplotlib axis objects.
- When analyzing experimental data, you can now pass either CSV files or Numpy arrays to ``RawData`` and ``RawData0``.

Changes
-------

- In the ``qmix.respfn.RespFn`` class, the ``f_idc``, ``f_ikk``, ``f_didc``, and ``f_dikk`` attributes were all made private by adding a leading underscore. They are replaced by the ``idc``, ``ikk``, ``didc`` and ``dikk`` methods, respectively.
- In the experimental data module, the parameter ``iv_multiplier`` was renamed ``i_multiplier``. ``RawData0`` and ``RawData`` use this parameter to correct imported current values. (All imported current data is multiplied by ``i_multiplier``.) I also added ``v_multiplier``, which does the same things, except for imported voltage data.

Testing
-------

- Dropped Travis-CI testing with Python 3.5-dev and 3.6-dev. QMix is now tested with 3.5, 3.6 and 3.7-dev.
- Added automatic coverage testing through coveralls.io.

Bug Fixes
---------

- Fixed potential issue with file paths. Previously, file paths were built assuming a mac operating system (e.g., ``"some-dir/this-figure.png"``). The package was updated to now use ``os.path.join("some-dir", "this-figure.png")``, which is machine independent.

Documentation
-------------

- Added package API to web page.
- Added more information about how experimental is stored and imported into QMix.
- Added information on getting support and contributing to QMix.
- Added descriptions of different response function types to docstring in ``qmix.respfn.py``.
- Added more comments on keyword arguments and attributes for classes in ``qmix.circuit.py`` and ``qmix.respfn.py``. (This should help to explain some of the SIS-related jargon.)
- Added use cases for many class methods.
- Fixed documentation for ``qmix.exp.parameters.py``.
- Updated workflow examples.
- Other (minor) changes.



v1.0.1 (5-Feb-2019)
===================

Bug Fixes
---------

- Fixed description in setup.py. Previously, there were some non-ASCII characters, which were causing issues on Windows installations.



v1.0.0 (11-Jan-2019)
====================

Initial release.
