v1.0.5 (30-Jul-2019)
====================

Experimental data
-----------------

- Process experimental data in a consistent order (i.e., I-V and IF data should be processed in the same way).
- Add warning if normal resistance is too low or too high. This can help to detect if you are using the wrong units for the current.
- Make parameters in ``qmix.exp.parameters`` more consistent. Changes include:
    - ``voffset_range``, which defines where the voltage offset is potentially found, is now defined as a list. E.g., if the experimental voltage offset could be found anywhere between -0.5mV and 0.1mV, you now use ``voffset_range=(-0.5e-3, 0.1e-3)``. This change is backwards compatible, so you can still define it as a float if you like. For example, if you set ``voffset_range=1e-3``, this is equivalent to ``voffset_range=(-1e-3, 1e-3)``.
    - ``rn_vmin`` and ``rn_vmax``, which previously defined where the normal resistance was calculated, are now combined into ``vrn``. Previously, this was defined using ``rn_vmin`` and ``rn_vmax``. Now, it is defined using ``vrn`` as a list. For example, if you previously used ``rn_vmin=4e-3`` and ``rn_vmax=5e-3``, you would now use ``vrn=(4e-3, 5e-3)``. This change is backwards compatible, so you can still use ``rn_vmin`` and ``rn_vmax``.
    - ``vshot``, which controls where the shot noise slope is calculated, is now a list of lists, so that multiple voltage ranges can be defined. For example, if the shot noise slope is smooth between from 4-5mV and 5.5-6mV, you can define ``vshot=((4e-3,5e-3),(5.5e-3,6e-3))``. This change is backwards compatible, so you can still use a normal list.
    - ``cut_low`` and ``cut_high``, which previously defined the region of the first photon step that would be used for impedance recovery, are now combined into ``fit_range``. For example, if you previously used ``cut_low=0.25`` and ``cut_high=0.2``, you should now use ``fit_interval=(0.25, 0.8)``. (NB: The way that the upper limit is defined has been changed!). In this example, the script will ignore the first 25% of the photon step and the last 20% during the impdance recovery process. This is backwards compatible, so you can still use ``cut_low`` and ``cut_high`` if you like.
    - ``vgap_guess`` and ``igap_guess`` have been removed. They actually weren't needed all along.
    - ``ifdata_vmax``, which previously defined the maximum IF voltage to import, has been removed. QMix now uses the value from ``vmax`` instead.
    - ``ifdata_sigma``, which defines the width of the filter for the IF data, is now defined in units [V]. Previously, it was defined by the number of data points. This is also backwards compatible (if ``ifdata_sigma`` is >0.5, it will assume that you are defining it by the number steps).

Optimization
------------

- Add basic timing scripts for ``qmix.harmonic_balance``. See ``speed/`` directory.
- Use a better initial guess for the junction voltage in the ``qmix.harmonic_balance.harmonic_balance`` function.

Testing
-------

- Improve code coverage. Now 99.0% covered!

Command line
------------

- Add command line script to plot IF response from experimental data (``bin/plot-if-response.py``).


v1.0.4 (15-Apr-2019)
====================

Optimization
------------

- Optimize ``qmix.qtcurrent`` using Numba and JIT. 4 tone simulations are now 1.5 times faster!
- Add basic timing scripts for ``qmix.qtcurrent``. See ``speed/`` directory.

Changes
-------

- Move the code that is used for analyzing the IF response into a new module: ``qmix.exp.if_response``.

New Features
------------

- Add ability to force embedding impedance in ``qmix.exp.exp_data.RawData``. The embedding voltage will then be calculated using this value.

Testing
-------

- Improve test coverage for ``qmix.circuit``, ``qmix.respfn``, and ``qmix.exp.exp_data``.
- Add tests for new module (``qmix.exp.if_response``).
- Add notebook to recreate plots from Kittara's thesis. (For the purpose of validation.)

Minor Changes
-------------

- Improved progress messages from ``qmix.harmonic_balance.harmonic_balance``.
- Add ability to plot response function (``qmix.respfn.RespFn``).
- Always use default parameters from ``qmix.exp.parameters`` (i.e., reduce the number of magic numbers).
- Add ability to call ``qmix.respfn.RespFn`` by adding ``__call__`` method. This will return the interpolated response function (a complex value).
- Fix error in ``qmix.harmonic_balance.harmonic_balance`` (when Thevenin voltage is checked).
- Other minor changes to documentation.



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
