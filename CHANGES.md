Unreleased
==========

Functionality Changes
---------------------

- Added superconducting transmission lines. The new module ``qmix.superconductor`` gives the complex conductivity of superconducting films from the Mattis-Bardeen theory in the local limit (``mattis_bardeen``), with the temperature dependence of the energy gap from the BCS gap equation (``energy_gap``), and the surface impedance of films of finite thickness and of stacks of films (``film_impedance``; classes ``SuperconductingFilm``, ``NormalFilm`` and ``Multilayer``). ``qmix.network.microstrip`` gives the characteristic impedance and propagation constant of a microstrip line with these films (Hammerstad & Jensen, Kirschning & Jansen and Yassin & Withington, as in SuperMix), including the kinetic inductance and the loss of the films, and ``qmix.network.microstrip_line`` the line as a ``Network``. The results agree with SuperMix (to round-off for normal metals; for superconductors to the accuracy of SuperMix's integration, ~2e-8, and of its table of the energy gap vs temperature, ~1e-4; see ``profile/supermix/compare_lines.py``).
- Added DC bias networks for several junctions, e.g., series arrays. ``qmix.circuit.MultiJunctionCircuit.set_dc_network`` sets linear DC equations between the DC voltages and DC currents of the junctions (``a @ v0 + b @ i0 = rhs``), and ``set_series_bias`` sets up one or several series strings across one bias voltage, optionally through a bias resistor. ``qmix.harmonic_balance.harmonic_balance`` then solves for the DC voltages too and returns them (``vj, vb = harmonic_balance(...)``; new ``vb_initial`` argument). ``check_hb_error`` checks the DC equations (new ``vb`` argument). Series strings typically need 15-30 iterations, so the default ``max_it`` is now 60 with a DC network (``max_it=None`` resolves to ``MAX_IT`` = 30 or ``MAX_IT_DC`` = 60). Circuits without a DC network are unchanged.
- ``qmix.superconductor`` now uses the ``mattis-bardeen`` package (https://github.com/garrettj403/Mattis-Bardeen) for the Mattis-Bardeen conductivity, the BCS energy gap, the surface impedance of films and stacks of films, and the penetration depth; QMix keeps the same functions and film classes as a thin layer on top. QMix now depends on ``mattis-bardeen>=1.0.0``. The conductivities agree with the previous implementation to ~1e-13 and are ~7x faster (a fixed tanh-sinh rule, vectorized over frequency, instead of adaptive quadrature at each frequency).
- Added ``qmix.network.transmission_line`` (a line from its characteristic impedance and propagation constant) and ``qmix.network.tee`` (an ideal junction of several ports).

Packaging
---------

- QMix now requires Python 3.10 or later (the classifiers listed 3.5-3.8). The ``tests`` package is no longer installed with QMix, and ``pytest-cov`` is now part of the ``test`` extra. Removed the obsolete ``python setup.py test`` command.

Documentation
-------------

- Added a Theory page on superconducting transmission lines (Mattis-Bardeen conductivity, surface impedance of films, microstrip model, comparison with SuperMix).
- Example #6 (multiple junctions) has a new section that designs the twin-junction tuning circuit from Nb microstrip, and shows how far the resonance moves if the kinetic inductance is ignored (from 230 to ~180 GHz).

v1.1.0 (20-Nov-2026)
====================

Functionality Changes
---------------------

- Removed the limit of 4 tones. ``qmix.qtcurrent.qtcurrent`` now uses one general current coefficient kernel for any number of tones (replacing the separate 1, 2, 3 and 4 tone functions), and ``qmix.circuit.EmbeddingCircuit`` accepts any ``num_f >= 1``. Results are unchanged (to ~1e-15) and the speed is the same.
- Added a Fourier-shift method for the summation over the phase factor coefficients (new ``method`` argument of ``qmix.qtcurrent.qtcurrent``, ``qmix.qtcurrent.interpolate_respfn`` and ``qmix.harmonic_balance.harmonic_balance``: ``'direct'``, ``'fourier'`` or ``'auto'``). With ``'fourier'``, the response function is represented by a windowed Fourier series so that the cost scales with the *sum* of ``2 * num_b + 1`` over the tones instead of the *product*, and the large interpolated response function matrix is not needed. This makes 5 or more tones (or large ``num_b``) practical. The accuracy is limited by the smoothness of the response function splines: ~1e-7 (relative to the direct method) with the default cubic splines and ~1e-9 or better with ``spline_order=5``. The default, ``'auto'``, uses the direct method unless the response function matrix would exceed ``qmix.qtcurrent.MAX_RESP_MATRIX_BYTES`` (2 GB). ``interpolate_respfn`` returns a ``qmix.qtcurrent.FourierResp`` object in this case, which can be passed to ``qtcurrent`` as ``resp_matrix``.
- Response functions (``qmix.respfn.RespFn`` and subclasses) are now exactly symmetric: the DC I-V spline is odd and the KK spline is even (to ~1e-14). Previously, the spline knots were not placed symmetrically about zero, so the two sides differed by up to ~5e-5 near the gap (within the interpolation error). The knots at positive voltages are now mirrored. Tunneling currents change by up to ~1e-5 for strongly pumped junctions.
- ``qmix.harmonic_balance.harmonic_balance`` now uses a backtracking line search at each bias voltage (new ``line_search`` argument, default ``True``): a Newton step is only accepted if it reduces the error enough, otherwise it is shortened. Bias voltages where no acceptable step exists even with a new Jacobian (i.e., the iteration is stuck near a local minimum of the error) are restarted from the solution at the nearest converged bias voltage. This fixes divergence when the junction loads the embedding circuit strongly (e.g., large embedding impedance near the gap): over a grid of 96 hard single-tone cases, the number of bias voltages that failed to converge in 50 iterations dropped from 52 to 0, and the solutions match SuperMix. For cases that already converged, the results, number of iterations and run time are unchanged (full steps cost nothing extra).
- The default maximum number of iterations in ``qmix.harmonic_balance.harmonic_balance`` (``max_it``) is now 30 instead of 10. Most cases still converge (and stop) in ~5 iterations, but in the hard cases above, some bias voltages need up to ~20; all 96 converge within 20.
- Added a ``warm_start`` option to ``qmix.harmonic_balance.harmonic_balance`` (default ``False``). For multi-tone simulations, it first solves the harmonic balance for the strongest tone on its own (cheap, since single-tone tunneling currents are much faster to calculate), and uses the result as the initial guess for that tone. When one tone (e.g., the LO) is much stronger than the others, this typically cuts the number of multi-tone iterations by 2-3x (e.g., 8 to 3 for the 4-tone benchmark, ~20% faster), but in some hard cases it is slower, so it is not the default.
- Added ``qmix.small_signal`` for the small-signal analysis of SIS mixers (Tucker theory). ``admittance_matrix`` calculates the conversion admittance matrix between the sidebands of the LO harmonics (m * f_lo + f_if) from a single-tone LO state (e.g., from ``harmonic_balance``), at any IF frequency. ``sideband_impedances``, ``sideband_voltages`` and ``solve_sidebands`` solve the sideband network, and ``conversion_gain`` gives the transducer gain between any two sidebands. This is much faster than a multi-tone simulation with a weak signal (e.g., 0.01 s vs 0.3 s for an LO + USB + LSB + IF mixer) and has no frequency rounding limits. The admittance matrix agrees with SuperMix to ~1e-13, and the gain agrees with QMix's multi-tone simulations to ~1e-6 in the limit of a weak signal. It also includes the noise analysis: ``noise_matrix`` calculates the junction's current noise correlation matrix (shot noise, including its thermal and quantum limits), ``embedding_noise`` the thermal noise of the embedding impedances (Callen-Welton, including zero-point fluctuations), and ``output_noise`` and ``noise_temperature`` the noise delivered to the IF load and the (SSB) noise temperature of the mixer. These agree with SuperMix's mixer analysis to ~1e-15 (~1e-9 including the embedding noise) when the same physical constants are used; SuperMix's CODATA 1986 constants differ from SciPy's by ~6e-6 in e / k.
- Added ``qmix.exp.PumpedData.mixer_noise`` (and ``plot_mixer_noise``) to compare measured hot/cold gain and noise temperature with a small-signal simulation of the mixer. It simulates the mixer with the recovered embedding circuit (DSB conversion gain and noise temperature), and fits the gain and noise temperature of the RF/optics stage in front of it, using the IF noise from the shot noise. For the example data, two fitted constants (G_RF = 0.51, T_RF = 23 K) reproduce the bias dependence of the measured gain across the photon steps, and the measured noise temperature to ~10% around the optimum bias; the simulated DSB mixer noise is ~3-4 K of the ~38 K receiver noise. The IF frequency, IF load impedance and physical temperature must be given (they are not in the data), but the results hardly depend on them. ``mixer_noise(..., smeared=True)`` uses the smeared response function (``DCData.resp_smear``), which halves the error in the shape of the simulated gain for the example data.
- Importing ``qmix`` now sets ``KMP_WARNINGS=0`` (unless it is already set) to silence the harmless ``OMP: Info #276: omp_set_nested routine deprecated`` notice that Numba's OpenMP threading layer prints once per process.
- Added ``qmix.network`` for linear networks described by S-parameters (e.g., HFSS simulations or VNA measurements): ``read_touchstone`` reads Touchstone v1 files (Hz-GHz; MA, DB, RI; the reference impedance from the option line, or the per-frequency ``! Port Impedance`` comments of HFSS files that are not renormalized), and ``Network`` can be interpolated, renormalized (power waves, complex references), connected to other networks and terminated, and gives the impedance seen at any port. ``embedding_impedance`` turns this into the embedding impedance of a junction as a function of frequency, with the junction capacitance in parallel. This is a first step towards S-parameter embedding circuits.
- The ``qmix.exp.PumpedData.mixer_noise`` docstring now explains the role of ``z_if``: the IF load sets the split of the measured gain into ``g_rf`` and ``g_mix`` (not the mixer noise), and the measured gain is referred to a reference plane at ``rn`` (the shot noise calibration ignores the IF mismatch), so the gain into an IF impedance ``z_if`` corresponds to a measured gain of ``g_mix / (1 - |G|^2)``, ``G = (z_if - rn) / (z_if + rn)``.
- ``qmix.exp.PumpedData.mixer_noise`` has a new ``z_emb`` argument: the embedding impedance of the RF circuit as a function of frequency (e.g., from ``qmix.network.embedding_impedance``), used for the sideband impedances instead of the recovered LO impedance (``anchor_lo=True`` shifts it to match the recovered impedance at the LO). The sideband impedances used are returned (``z_usb``, ``z_lsb`` in ``MixerNoise``).
- Added comparisons with the thesis data of the 230 GHz mixer (``profile/thesis/``; the data are not distributed): the HFSS embedding impedance vs the recovered impedances, two strong LOs (multi-tone harmonic balance vs measured I-V curves) and an LO power sweep (gain vs LO power). See ``profile/thesis/README.md``.
- ``qmix.mathfn.kktrans.kk_trans`` now handles I-V curves with an offset above the gap (e.g., excess current: ``i ~ v + b`` at large ``v``), which is common in experimental data. The FFT-based Hilbert transform assumes that ``i - v`` goes to zero at the ends of the voltage range; with an offset, the KK transform was off by ~3e-3 at ``|v| < 8`` for ``b = 0.2`` (and much more towards ``|v| = 35``). The offset is now subtracted as a smooth step whose transform is known analytically, and added back, which brings the error down to ~1e-6. For the example experimental data (``b ~ -0.07``), the ac tunneling currents change by ~1e-5 (relative) and the recovered embedding circuit is unchanged; I-V models without an offset (e.g., the polynomial model) are unaffected.
- Added ``qmix.network.thevenin`` and ``qmix.network.set_thevenin``: the Thevenin equivalent circuit seen by the junction when a source with a given available power drives another port of an S-parameter network (e.g., the LO in the waveguide), including the junction capacitance. ``set_thevenin`` puts it into an ``EmbeddingCircuit`` for any tone and harmonic, so S-parameter circuits can drive the harmonic balance directly. Example #4 uses it to find the LO power in the waveguide from a pumped I-V curve (47 nW at 220.1 GHz; the available power at the junction agrees with the recovered embedding circuit to 5%).
- Added support for several junctions in one embedding network. ``qmix.circuit.MultiJunctionCircuit`` holds the multi-port Thevenin equivalent at each tone and harmonic (open-circuit voltages ``vt[j, f, p]`` and impedance matrices ``zt[j, l, f, p]``; the junction axis is first and junctions are numbered from 0), the bias voltage of each junction (``vb[j, i]``) and their relative areas (``area``, which scale the tunneling currents). ``qmix.harmonic_balance.harmonic_balance`` solves all junctions together (``vj[j, f, p, i]``) with the same Newton method, line search, restarts and warm start; since each junction's current only depends on its own voltage, a new Jacobian costs the same number of current calculations per junction as for one junction. ``qmix.qtcurrent.qtcurrent``, ``interpolate_respfn``, ``suggest_num_b`` and ``qmix.harmonic_balance.check_hb_error`` accept the multi-junction circuit, and the response function can be different for each junction (a list). ``qmix.network.thevenin`` and ``set_thevenin`` accept a list of junction ports and give the multi-port Thevenin equivalent of an S-parameter network, with one junction capacitance per port. Single-junction results are unchanged. The small-signal analysis (``qmix.small_signal``) is still for one junction.
- The small-signal analysis (``qmix.small_signal``) now supports several junctions. ``admittance_matrix`` and ``noise_matrix`` accept a ``MultiJunctionCircuit`` and return one matrix per junction (``y[j, m, n, i]``, scaled by the relative areas). The junctions are coupled by the embedding network, described at each sideband by its impedance matrix over the junction ports and one external port (the RF input at the RF sidebands, the IF load at the IF; ``sideband_networks``), terminated by ``z_ext``. ``network_gain`` gives the transducer gain between the external ports, and ``network_output_noise`` and ``network_noise_temperature`` the noise, including the noise of the junctions, of the terminations, and of a lossy embedding network (Twiss: open-circuit noise voltages with correlation ``2 k T_cw (Z + Z^dagger)``). These also work for one junction behind a lossy network. Checked against the single-junction functions (one junction behind a lossless network), two junctions in parallel vs one junction of twice the area, thermal equilibrium (unpumped junctions and a lossy network at one temperature deliver ``k T_cw (1 - |Gamma|^2)`` to the IF load, including the quantum regime), and a multi-junction multi-tone harmonic balance with a weak RF signal (~1e-6). New ``qmix.network.impedance_matrix``: the impedance matrix of several ports of a network (other ports terminated), with capacitances in parallel, for these embedding networks. New example notebook (Example #6, ``multi-junction-mixer.ipynb``): a twin-junction mixer at 230 GHz.
- ``qmix.harmonic_balance.harmonic_balance`` now uses an analytic Jacobian by default (new ``jacobian`` argument: ``'analytic'``, or ``'numerical'`` for the finite differences used before). New ``qmix.qtcurrent.qtcurrent_derivatives`` gives the derivatives of the tunneling current with respect to the real and imaginary parts of the junction voltage. Perturbing harmonic q of a tone shifts its phase factor coefficients by +/- q (``dC_k / dRe V_q = (C_{k-q} - C_{k+q}) / 2 q w``, ``dC_k / dIm V_q = -j (C_{k-q} + C_{k+q}) / 2 q w``), so the derivatives are the same sums as the current, with one tone's pair products replaced by their derivatives, and no derivatives of the response function are needed. All of the derivatives are evaluated in one pass over the response function matrix per batch, and the sum over the innermost tone is shared by the derivatives with respect to the other tones (new kernel ``_deriv_coeff_batched``). The Fourier-shift method and several junctions are supported. The derivatives match central differences to ~1e-9 (the error of the differences; the forward differences used before had errors of ~4e-3 for a low-frequency IF tone). Harmonic balance is 1.2-1.5x faster for one junction (1-4 tones) and ~1.9x for two junctions, and converges in some hard cases even without the line search. ``qtcurrent`` itself is unchanged (the term evaluation was refactored, but the reference currents still agree to 14 decimals).
- **Bug fix:** ``qmix.network``: loads at ports with complex reference impedances (``Network.terminate``, ``Network.impedance`` with ``terminations``, and unlisted "matched" ports) used the reflection coefficient ``(Z - Z_ref^*) / (Z + Z_ref)``, which is the reflection seen looking into a port, not that of a load at a network port, ``(Z - Z_ref) / (Z + Z_ref^*)``. A matched port is therefore terminated by ``Z_ref``, not by its complex conjugate. Real reference impedances (the usual case, including the HFSS files in the examples, which change by < 0.01 ohms) are unaffected.
- ``qmix.exp``: ``vgap_threshold`` (removed when the gap voltage method was changed to the "corner" method: half the current where the tangent at the steepest point meets the normal-resistance line) now gives a warning instead of being silently ignored, and it was removed from the docstrings, notebooks and tests. Use ``vgap`` to set the gap voltage manually (now documented).
- ``qmix.exp`` (``DCData``, ``dciv_curve``): if there are no I-V data in the range used to find the normal resistance (``vrn``, above 3.5 mV by default), a ``ValueError`` now explains that the units are probably wrong (``v_fmt``/``i_fmt`` are 'mV'/'mA' by default; data in V need ``v_fmt='V'``), instead of a ``TypeError`` from ``np.polyfit``. **Bug fix:** with coarse I-V data (e.g., 20 uV steps) and ``filter_data=False``, the gap voltage was fitted to a single point (a window of +/- 10 uV around the steepest point); the window is now at least +/- 2.5 points wide.
- **Bug fix:** ``qmix.qtcurrent.qtcurrent`` now includes all combinations of tone indices (a_1, ..., a_F) that land on an output frequency (thesis Eqn. 5.36), not only those with ``|a_f| <= num_p``. With commensurate tones, e.g., an LO, USB, LSB and IF, combinations such as 2 * LO - LSB = USB contribute to the current, and some are first order in the signals. Dropping them caused errors of ~1e-2 in the currents (normalized) and, with both sidebands as tones and ``num_p = 1``, errors of up to ~50% in the simulated mixer response (found by comparing with SuperMix's small-signal theory). Combinations are included if an upper bound on their contribution exceeds ``mixing_tol`` times the size of that current (new argument; default ``qmix.qtcurrent.MIXING_TOL = 1e-6``, giving relative errors of ~1e-7 or better, also for weak signals). ``mixing_tol=np.inf`` gives the old behavior. Simulations with commensurate tones are slower (e.g., 4.0 s instead of 1.1 s for the 4-tone harmonic balance benchmark, but only ~1.8x for weak signals); simulations with unrelated tones are unchanged. The stored 3- and 4-tone benchmark currents (``profile/benchmark/data/``) changed by 3e-2 and were regenerated.
- ``qmix.qtcurrent.interpolate_respfn`` (and therefore ``qtcurrent`` and ``harmonic_balance``) now warns if the response function is needed beyond the voltage range over which it is defined (``|v| <= 35`` for the standard response functions; new ``v_max`` attribute of ``qmix.respfn.RespFn``), i.e., if the bias voltage plus ``num_b`` times the frequencies exceeds it. Beyond that, the response function is extrapolated, which is inaccurate (e.g., the KK transform of the polynomial model is off by ~7e-4 at v = 50).
- ``harmonic_balance(..., mode='x')`` now reports whether the error target was reached, instead of whether the maximum number of iterations was hit (which was wrong if the target was reached on the last iteration).

Optimization
------------

- ``qmix.qtcurrent.qtcurrent`` (direct method) now evaluates all of the current coefficients of all output frequencies together, in batches of up to 32 terms per pass over the interpolated response function matrix (``_current_coeff_batched``), instead of one pass per term. The kernel is limited by memory bandwidth, so this matters most with many terms, i.e., commensurate tones with the mixing products that are now included: e.g., ~2.4x faster for 4 strong commensurate tones and ~2x for 5 tones; the 4-tone harmonic balance benchmark takes 2.6 s instead of 4.0 s. 1 and 2 tone calculations are also ~1.3-1.9x faster. Results agree with the previous kernel to ~5e-15.
- ``qmix.harmonic_balance.harmonic_balance`` now reuses the Jacobian from the previous iteration while the error is still decreasing quickly (new ``reuse_jacobian`` argument, default ``True``). Combined with faster current coefficient calculations in ``qmix.qtcurrent`` (for 2-4 tones), harmonic balance is ~2x faster for 3 or 4 tones.
- The current coefficient calculations in ``qmix.qtcurrent`` now run in parallel (multi-threaded, via Numba) for 3 or 4 tones. Harmonic balance is ~2.5x faster for 4 tones.
- Added ``qmix.qtcurrent.suggest_num_b``, which suggests the summation limits (``num_b``) for each tone based on the junction voltage. Weakly driven tones need far fewer terms, so this can make 3 or 4 tone simulations several times faster (e.g., ~3x for the 4 tone benchmark).
- Response functions (``qmix.respfn.RespFn`` and subclasses) are now evaluated as one complex piecewise polynomial (built from the DC I-V and KK splines) in a parallel Numba kernel, instead of evaluating two splines. Results are the same to ~1e-14. Together with building the interpolation voltages in a single vectorized step, ``qmix.qtcurrent.interpolate_respfn`` is ~7x faster for 4 tones, and ``qtcurrent`` (without a pre-interpolated ``resp_matrix``) is ~5x faster for 4 tones.

Documentation
-------------

- Added two example notebooks (also on the documentation website): ``hfss-embedding-impedance.ipynb`` (the embedding impedance from HFSS S-parameters with ``qmix.network``, compared with the impedances recovered over an LO sweep, and used in the noise model) and ``lo-saturation.ipynb`` (power saturation: gain and noise vs LO power compared with hot/cold measurements, and gain compression by a strong RF signal compared with a measurement of the IF output). The HFSS notebook also simulates the broken photon step at 240 GHz (a spurious tone at 3/2 f_LO with the HFSS embedding impedance at f_LO / 2). Their data are in ``notebooks/example-data/hfss/``, ``lo-sweep/``, ``lo-saturation/`` and ``rf-saturation/``.
- Added a Theory section to the documentation (before the examples): multi-tone spectral domain analysis, harmonic balance, and the Fourier-shift method (which moved under it).
- Added the ``qmix.exp.if_response`` module to the API documentation (it was missing).
- Updated ``environment.yml``: added ``scienceplots`` (required by the notebooks) and removed unused or unmaintained packages.
- Added a page to the online documentation describing the Fourier-shift method (``docs/source/fourier-shift-method.rst``).
- Added a Theory page on the small-signal analysis (``docs/source/small-signal.rst``): the admittance and noise correlation matrices, gain and noise for one junction, and for several junctions coupled by a multi-port embedding network (Thevenin reduction of the junction ports, network noise from Twiss' theorem), with the limits of the analysis.

v1.0.6 (8-Jul-2020)
===================

Functionality Changes
---------------------

- Renamed ``cct.vph`` to ``cct.freq`` (in ``qmix.circuit.EmbeddingCircuit``). The normalized photon voltage is equivalent to the normalized frequency. I think frequency is easier to understand so I changed it.
- Deleted the command line scripts (``bin/``). There was only one script and it was pretty obscure so I descided to delete it.
- Moved ``qtcurrent_all_freq()`` function from ``qmix.qtcurrent`` to ``qmix.harmonic_balance``. Again, it's a pretty obscure function and it's only used in the ``harmonic_balance`` module, so I moved it.
- Deleted ``qtcurrent_std`` from ``qmix.qtcurrent``. This function calculated the "standard" output values, but this was pretty arbitrary. It also didn't really simplify the package at all so I deleted it.

Optimization
------------

- Added more detailed benchmarking and profiling scripts (found in ``QMix/profile/``).
- ``qmix.qtcurrent.qtcurrent`` is now ~20% faster for 1 tone.
- ``qmix.harmonic_balance.harmonic_balance`` is ~20% faster for 1 or 2 tones, and ~10% faster for 3 tones.

Documentation
-------------

- Travis CI now automatically rebuilds the online documentation on each new commit.
- Added information on how to contribute to QMix to README.
- Added docstrings to many private functions. This should allow new users to understand the inner workings a bit better.

Miscellaneous Minor Changes
---------------------------

- ``QMix/examples/`` renamed ``QMix/notebooks/``
- Removed ``codemeta.json`` and ``setup.cfg``. 
- Fixed several bugs with Travis CI.

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
