.. _small-signal:

Small-Signal Analysis
=====================

In an SIS mixer, a strong local oscillator (LO) pumps the junction, and the signals (the RF input and the IF output) are weak. Harmonic balance can simulate the LO and the signals together as tones, but this is slow, since every tone multiplies the cost of the tunneling current calculation. If the signals are weak enough, they don't change the LO state, and their response is linear. This is the small-signal analysis of Tucker (1979; Tucker and Feldman, 1985): from a single-tone harmonic balance of the LO alone, it gives the conversion gain and the noise of the mixer at any IF frequency. The functions are in :mod:`qmix.small_signal`.

As on the previous pages, all values are normalized to the gap voltage, gap current, normal-state resistance and gap frequency. Phasors are peak amplitudes.

Sidebands
---------

The LO at frequency :math:`\omega` (with harmonics :math:`p \omega`) mixes a weak signal at the IF frequency :math:`\omega_\mathrm{IF}` with the sidebands of every LO harmonic. QMix labels them by an integer :math:`m`, from :math:`-M` to :math:`M` (``num_sb``):

.. math::

   \omega_m = m \, \omega + \omega_\mathrm{IF}.

Sideband 0 is the IF, sideband :math:`p > 0` is the upper sideband (USB) of harmonic :math:`p`, and sideband :math:`-p` is its lower sideband (LSB). For :math:`m < 0` the frequency is negative, so the phasors at these sidebands are the complex conjugates of the phasors at the physical LSB frequency :math:`p \omega - \omega_\mathrm{IF}`, and so are the impedances. The helper functions (:func:`qmix.small_signal.sideband_impedances`, :func:`qmix.small_signal.sideband_voltages` and :func:`qmix.small_signal.sideband_networks`) take the physical values and do this conversion, and :func:`qmix.small_signal.lsb` converts results back. Arrays over sidebands are stored in natural order: sideband :math:`m` is at index ``m + num_sb``.

The admittance matrix
---------------------

The small signals at all sidebands are linearly related by the conversion admittance matrix:

.. math::

   I_m = \sum_{n=-M}^{M} Y_{mn} V_n,

where :math:`V_n` and :math:`I_n` are the small-signal voltage and current phasors of the junction at sideband :math:`n`. The matrix depends on the LO state through the phase factor coefficients :math:`C_k` of the LO (see :ref:`mtsda`), and on the junction through its response function :math:`\tilde{I}(V) = I_{kk}(V) + j I_{dc}(V)`. With :math:`T(V) = -j \tilde{I}(V) = I_{dc}(V) - j I_{kk}(V)`,

.. math::
   :label: ss-admittance

   Y_{mn} = \frac{1}{2 \omega_n} \sum_k C_k \, C^*_{k+m-n}
   \Big[ T(V_0 + k \omega) - T(V_0 + (k - n) \omega - \omega_\mathrm{IF})
   + T^*(V_0 + (k + m) \omega + \omega_\mathrm{IF}) - T^*(V_0 + (k + m - n) \omega) \Big],

at bias voltage :math:`V_0`. This is Tucker's admittance matrix, written as one expression for sidebands of either sign. It is equivalent to Eqn. 29 of Rice et al. (1999) and to SuperMix's ``small_signal``. QMix calculates it with :func:`qmix.small_signal.admittance_matrix`, from the junction voltage of a single-tone harmonic balance (the LO and its harmonics). The matrix is real only in the classical limit; the Kramers-Kronig part of the response function makes it complex. At :math:`\omega_\mathrm{IF} = 0`, the bracket in column :math:`n = 0` and the denominator :math:`2 \omega_0 = 2 \omega_\mathrm{IF}` both vanish, and the limit needs the derivatives of the response function:

.. math::
   :label: ss-admittance-zero-if

   Y_{m0} = \frac{1}{2} \sum_k C_k \, C^*_{k+m}
   \Big[ T'(V_0 + k \omega) + T'^*(V_0 + (k + m) \omega) \Big]
   \qquad (\omega_\mathrm{IF} = 0).

In particular, :math:`Y_{00} = \sum_k |C_k|^2 I_{dc}'(V_0 + k \omega)` is the slope of the pumped I-V curve. The other columns are given by Eqn. :eq:`ss-admittance` with :math:`\omega_\mathrm{IF} = 0`. This is the limit of a low IF: sidebands :math:`m` and :math:`-m` are still treated as separate signals, although their physical frequencies coincide (and :math:`Y_{-m,-n} = Y^*_{mn}`). :func:`qmix.small_signal.admittance_matrix` uses it for ``f_if=0``. It agrees with SuperMix's ``small_signal`` at :math:`f_\mathrm{IF} = 0` to round-off, once SuperMix's IF port, which is a plain DC value rather than an RMS phasor, is converted. The noise matrix :math:`\mathbf{H}` has no such singularity and is continuous at :math:`\omega_\mathrm{IF} = 0`.

The junction also generates shot noise. The noise currents at different sidebands are correlated, because they come from the same pumped tunneling current. Their correlation matrix (Tucker and Feldman, 1985) is

.. math::
   :label: ss-noise

   \langle \delta I_m \, \delta I_n^* \rangle = H_{mn} \, e \, I_\mathrm{gap} \, \Delta f,
   \qquad
   H_{mn} = \sum_k C_k \, C^*_{k+m-n}
   \Big[ N(V_0 + (k + m) \omega + \omega_\mathrm{IF}) + N(V_0 + (k - n) \omega - \omega_\mathrm{IF}) \Big],

where :math:`\Delta f` is the bandwidth, the noise currents are RMS values (so :math:`H` is a one-sided spectral density), and

.. math::

   N(V) = I_{dc}(V) \coth\left( \frac{V}{2 T} \right)

is the noise of the unpumped junction, with the normalized temperature :math:`T = k_B T_\mathrm{phys} / e V_\mathrm{gap}`. This includes the thermal and quantum limits of shot noise, but not the noise of the embedding circuit. QMix calculates :math:`H` with :func:`qmix.small_signal.noise_matrix`, which needs the physical temperature and the gap voltage.

One junction
------------

At each sideband, the embedding circuit seen by the junction is a Thevenin equivalent circuit: a source voltage :math:`V_{T,m}` in series with an impedance :math:`Z_m` (:func:`qmix.small_signal.sideband_impedances`). Usually, only the source at the input sideband is nonzero. The junction voltages satisfy :math:`V_m = V_{T,m} - Z_m I_m`, so

.. math::

   (\mathbf{1} + \mathbf{Z} \mathbf{Y}) \, \mathbf{V} = \mathbf{V}_T,

where :math:`\mathbf{Z}` is the diagonal matrix of the embedding impedances (:func:`qmix.small_signal.solve_sidebands`). The transducer gain from sideband :math:`m` to sideband :math:`m'` is the power delivered to the embedding impedance at :math:`m'`, divided by the power available from the source at :math:`m`:

.. math::

   G_{m' m} = 4 \, \mathrm{Re}\, Z_m \, \mathrm{Re}\, Z_{m'} \left| \frac{I_{m'}}{V_{T,m}} \right|^2

(:func:`qmix.small_signal.conversion_gain`). For a mixer, :math:`m = \pm 1` and :math:`m' = 0`.

The noise sources are the junction's noise currents :math:`\delta \mathbf{I}`, and the thermal noise voltages :math:`\mathbf{E}` of the embedding impedances, in series with the sources. Then :math:`V_m = E_m - Z_m I_m` and :math:`\mathbf{I} = \mathbf{Y} \mathbf{V} + \delta \mathbf{I}`, so

.. math::

   \mathbf{I} = \mathbf{A} \left( \delta \mathbf{I} + \mathbf{Y} \mathbf{E} \right),
   \qquad
   \mathbf{A} = (\mathbf{1} + \mathbf{Y} \mathbf{Z})^{-1},

and the correlation matrix of the noise currents is :math:`\mathbf{A} (\mathbf{H} + \mathbf{Y} \mathbf{S} \mathbf{Y}^\dagger) \mathbf{A}^\dagger`. :math:`\mathbf{S}` is the diagonal matrix of the spectral densities of :math:`\mathbf{E}`, given by the Callen-Welton (1951) formula:

.. math::

   S_m = 4 k_B T_{cw} \, \mathrm{Re}\, Z_m,
   \qquad
   T_{cw} = \frac{h f}{2 k_B} \coth\left( \frac{h f}{2 k_B T_\mathrm{phys}} \right),

at the physical frequency of the sideband, :math:`f = |\omega_m|` (:func:`qmix.small_signal.embedding_noise`). :math:`T_{cw}` includes the zero-point fluctuations: it tends to :math:`T_\mathrm{phys}` at low frequencies, and to :math:`h f / 2 k_B` at zero temperature. The noise power delivered to the load at the output sideband, per unit bandwidth and in units of :math:`k_B`, is

.. math::

   T_\mathrm{out} = \mathrm{Re}\, Z_{m'} \left[ \mathbf{A} (\mathbf{H} + \mathbf{Y} \mathbf{S} \mathbf{Y}^\dagger) \mathbf{A}^\dagger \right]_{m' m'} \frac{e V_\mathrm{gap}}{k_B},

without the noise of the load itself (:func:`qmix.small_signal.output_noise`). The single-sideband (SSB) noise temperature of the mixer is the output noise, without the noise of the source at the input sideband, referred to the input: :math:`T_\mathrm{SSB} = T_\mathrm{out} / G_{m' m}` (:func:`qmix.small_signal.noise_temperature`). For the double-sideband (DSB) noise temperature, the noise of both sidebands is left out, and the output noise is divided by the sum of the gains from the USB and the LSB. With the junction noise only (:math:`\mathbf{S} = 0`), the zero-point fluctuations of the embedding are missing, so the noise temperature can be below the quantum limit (:math:`h f / 2 k_B` for a DSB mixer).

Several junctions
-----------------

With several junctions (:class:`qmix.circuit.MultiJunctionCircuit`), each junction :math:`j` has its own admittance and noise matrices, :math:`\mathbf{Y}_j` and :math:`\mathbf{H}_j`, from its own LO state. They are scaled by the relative area of the junction (``cct.area[j]``), and the noise of different junctions is uncorrelated. Junctions with a different gap voltage (``cct.gap_voltage[j]``) use the scaled response function described in :ref:`harmonic-balance`. Since that function is in circuit units, the matrices need no other change (the temperature in :math:`N(V)` is normalized with the circuit's gap voltage, like :math:`V`). :func:`qmix.small_signal.admittance_matrix` and :func:`qmix.small_signal.noise_matrix` return them as ``y[j, m, n, i]`` and ``h[j, m, n, i]``.

The junctions are coupled by the embedding network. At each sideband, the network is described by its impedance matrix over the :math:`J` junction ports and one external port :math:`x`: the RF input (or the image termination) at the RF sidebands, and the IF load at the IF. With the currents :math:`I^\mathrm{net}` into the network and the open-circuit noise voltages :math:`\mathbf{e}` of a lossy network,

.. math::

   \begin{bmatrix} \mathbf{V}_J \\ V_x \end{bmatrix} =
   \begin{bmatrix} \mathbf{Z}_{JJ} & \mathbf{Z}_{Jx} \\ \mathbf{Z}_{xJ} & Z_{xx} \end{bmatrix}
   \begin{bmatrix} \mathbf{I}^\mathrm{net}_J \\ I_x \end{bmatrix}
   + \begin{bmatrix} \mathbf{e}_J \\ e_x \end{bmatrix},

with the junction capacitances in parallel with the junction ports. :func:`qmix.network.impedance_matrix` gives this matrix from S-parameters, and :func:`qmix.small_signal.sideband_networks` collects the matrices of all sidebands. The external port is terminated by an impedance :math:`z_x` (the source or load impedance, ``z_ext``), with a source voltage :math:`V_s` and a thermal noise voltage :math:`e_\mathrm{ext}` in series: :math:`V_x = V_s + e_\mathrm{ext} - z_x I_x`. The junction currents flow out of the network, :math:`\mathbf{I}^\mathrm{net}_J = -\mathbf{I}`, so eliminating the external port gives

.. math::

   I_x = \frac{V_s + e_\mathrm{ext} - e_x}{D} + \mathbf{w} \cdot \mathbf{I},
   \qquad
   \mathbf{V}_J = \mathbf{n} - \mathbf{Z}_T \, \mathbf{I},

with :math:`D = Z_{xx} + z_x`, :math:`\mathbf{w} = \mathbf{Z}_{xJ} / D` and :math:`\mathbf{k} = \mathbf{Z}_{Jx} / D`. The Thevenin impedance matrix of the junction ports is

.. math::

   \mathbf{Z}_T = \mathbf{Z}_{JJ} - \frac{\mathbf{Z}_{Jx} \mathbf{Z}_{xJ}}{D},

and the voltages that drive the junction ports are :math:`\mathbf{n} = \mathbf{e}_J + \mathbf{k} \, (V_s + e_\mathrm{ext} - e_x)`. Each junction obeys :math:`\mathbf{I}_j = \mathbf{Y}_j \mathbf{V}_j + \delta \mathbf{I}_j` over the sidebands. Together:

.. math::

   (\mathbf{1} + \mathbf{Y} \mathbf{Z}_T) \, \mathbf{I} = \delta \mathbf{I} + \mathbf{Y} \mathbf{n},

where the vectors run over all junctions and sidebands. :math:`\mathbf{Y}` is block diagonal over the junctions (it couples the sidebands of one junction), and :math:`\mathbf{Z}_T` is block diagonal over the sidebands (it couples the junctions at one sideband). For :math:`J` junctions and :math:`2M + 1` sidebands, this is a linear system of size :math:`J (2M + 1)` at each bias voltage.

The gain is the transducer gain between the external ports, from the source at sideband :math:`m` to the load at sideband :math:`m'`:

.. math::

   G_{m' m} = 4 \, \mathrm{Re}\, z_{x,m} \, \mathrm{Re}\, z_{x,m'} \left| \frac{I_{x,m'}}{V_{s,m}} \right|^2

(:func:`qmix.small_signal.network_gain`). It includes the loss of the network.

The noise in the load (:func:`qmix.small_signal.network_output_noise`) has three sources, which are uncorrelated between sidebands:

- the junctions, with the correlation matrices :math:`\mathbf{H}_j`;
- the embedding network, if it is lossy (temperature ``temp``). A passive network has open-circuit noise voltages with the correlation matrix (Twiss, 1955)

  .. math::

     \langle \mathbf{e} \, \mathbf{e}^\dagger \rangle = 2 k_B T_{cw} \left( \mathbf{Z} + \mathbf{Z}^\dagger \right) \Delta f,

  with :math:`\mathbf{Z}` the full :math:`(J + 1)`-port matrix at the sideband. A lossless network (:math:`\mathbf{Z} + \mathbf{Z}^\dagger = 0`) has no noise of its own;
- the terminations of the external port (temperature ``temp_ext``), with :math:`\langle |e_\mathrm{ext}|^2 \rangle = 4 k_B T_{cw} \, \mathrm{Re}\, z_x \, \Delta f`. The load at the output sideband is left out.

The output noise temperature is :math:`T_\mathrm{out} = \mathrm{Re}\, z_{x,m'} \langle |I_{x,m'}|^2 \rangle / k_B \Delta f`. :func:`qmix.small_signal.network_noise_temperature` divides it by the gain to give the SSB noise temperature, as for one junction. The admittance and noise matrices may also be given without a junction axis, for one junction behind a lossy network.

These equations are checked in the test suite in four ways. One junction behind a lossless network gives the same gain and noise as the single-junction functions. Two identical junctions in parallel give the same results as one junction of twice the area. A multi-junction, multi-tone harmonic balance with a weak RF signal gives the same gain to about :math:`10^{-6}`. In thermal equilibrium (unpumped junctions at zero bias, a lossy network and the terminations all at one temperature), the noise delivered to the IF load is :math:`k_B T_{cw} (1 - |\Gamma|^2)`, where :math:`\Gamma` is the reflection coefficient at the IF port. This holds in the quantum regime too, as the fluctuation-dissipation theorem requires.

Limits of the small-signal analysis
-----------------------------------

- **The signals must be weak.** The analysis is linear in the signals and doesn't include saturation. Use a multi-tone harmonic balance for strong signals, e.g., :doc:`lo-saturation`. At a low IF, "weak" is stricter than it seems: the IF photon voltage :math:`\omega_\mathrm{IF}` is small, so even a small IF voltage gives a large IF drive level :math:`|V_\mathrm{IF}| / \omega_\mathrm{IF}`. When comparing with a multi-tone simulation at a 5 GHz IF, an RF signal of 1 pW gave errors of 0.5%, and :math:`10^{-16}` W was needed to agree to :math:`10^{-6}`.
- **The number of sidebands.** ``num_sb`` sets how many LO harmonics have sidebands, :math:`M`. The default, the number of LO harmonics in the circuit (``cct.num_p``), is usually enough, since the junction capacitance shorts out the higher sidebands.
- **The LO state is fixed.** It comes from the single-tone harmonic balance, so the LO must be strong compared with all signals, including the noise.

:meth:`qmix.exp.exp_data.PumpedData.mixer_noise` uses the single-junction analysis to compare measured gain and noise with a simulation. The :doc:`multi-junction-mixer` example shows the multi-junction analysis for a twin-junction mixer, including a check against a multi-tone harmonic balance.
