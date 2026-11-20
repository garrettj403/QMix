.. _mtsda:

Multi-Tone Spectral Domain Analysis
===================================

This page describes how QMix calculates the quasiparticle tunneling current through an SIS junction when the voltage across the junction is known. The voltage can contain any number of tones (e.g., a local oscillator, an RF signal and the IF), each with any number of harmonics.

All quantities are normalized, as everywhere in QMix: voltages to the gap voltage :math:`V_\mathrm{gap}`, currents to the gap current :math:`I_\mathrm{gap} = V_\mathrm{gap} / R_n`, impedances to the normal-state resistance :math:`R_n`, and frequencies to the gap frequency :math:`f_\mathrm{gap} = e V_\mathrm{gap} / h`. In these units, a frequency is equal to the photon voltage :math:`h f / e` (also normalized to the gap voltage).

The response function
---------------------

In Tucker's theory of quasiparticle tunneling (Tucker and Feldman, 1985), everything about the junction itself is contained in its *response function*

.. math::

   \tilde{I}(V) = I_{kk}(V) + j \, I_{dc}(V),

where :math:`I_{dc}(V)` is the DC I-V curve of the unpumped junction and :math:`I_{kk}(V)` is its Kramers-Kronig transform. The DC I-V curve gives the in-phase (dissipative) part of the response, and the Kramers-Kronig transform gives the quadrature (reactive) part. The reactive part is what makes the junction impedance complex near the gap.

In QMix, the response function is built by the classes in :mod:`qmix.respfn`, either from an I-V curve model (e.g., :class:`qmix.respfn.RespFnPolynomial`) or from experimental data (:class:`qmix.respfn.RespFnFromIVData`). The Kramers-Kronig transform is calculated with an FFT-based Hilbert transform (:func:`qmix.mathfn.kktrans.kk_trans`), and both curves are interpolated with splines so that :math:`\tilde{I}` can be evaluated at any voltage.

One tone
--------

Consider a junction with a DC bias voltage :math:`V_0` and a sinusoidal voltage at frequency :math:`\omega`:

.. math::

   V(t) = V_0 + |V_\omega| \cos(\omega t + \phi).

The voltage modulates the phase of the quasiparticle wavefunctions on one side of the barrier by the *phase factor*

.. math::

   W(t) = \exp\left[ -j \int |V_\omega| \cos(\omega t' + \phi) \, \mathrm{d}t' \right]
        = \sum_{n=-\infty}^{\infty} J_n(\alpha) \, e^{-j n \phi} \, e^{-j n \omega t},

where the second form follows from the Jacobi-Anger expansion, :math:`J_n` is a Bessel function of the first kind, and

.. math::

   \alpha = \frac{|V_\omega|}{\omega}

is the *drive level* (in physical units, :math:`\alpha = e |V_\omega| / \hbar \omega`). Each term in the sum is a quasiparticle that has absorbed (:math:`n > 0`) or emitted (:math:`n < 0`) :math:`|n|` photons. The DC current is the classic photon-assisted tunneling result of Tien and Gordon:

.. math::

   I_{dc}^{\mathrm{pumped}}(V_0) = \sum_{n=-\infty}^{\infty} J_n^2(\alpha) \, I_{dc}(V_0 + n \omega).

This produces the photon steps in the pumped I-V curve, at :math:`V_0 = 1 - n \omega` below the gap.

One tone with harmonics
-----------------------

In general, the voltage at a tone also has harmonics, :math:`p = 1, 2, \ldots, P`:

.. math::

   V(t) = V_0 + \sum_{p=1}^{P} |V_{p}| \cos(p \omega t + \phi_p).

The phase factor is then a product of one Jacobi-Anger series per harmonic, each with its own drive level :math:`\alpha_p = |V_p| / (p \omega)` and coefficients

.. math::

   c^{(p)}_n = J_n(\alpha_p) \, e^{-j n \phi_p}

(Eqns. 5.5 and 5.7 in Kittara's thesis). Collecting the terms by their overall frequency :math:`k \omega` gives

.. math::

   W(t) = \sum_{k} C_k \, e^{-j k \omega t},
   \qquad
   C_k = \sum_{n_1 + 2 n_2 + \cdots + P n_P = k} \; \prod_{p=1}^{P} c^{(p)}_{n_p},

where :math:`C_k` are the *phase factor coefficients* of the tone (Eqn. 5.12). QMix calculates them in :func:`qmix.qtcurrent.calculate_phase_factor_coeff`, by convolving the coefficients of the harmonics one at a time. The sum over :math:`k` is truncated at :math:`|k| \le B`, where :math:`B` is the summation limit (``num_b``). Since :math:`J_n(\alpha)` falls off quickly once :math:`|n| > \alpha`, :math:`B` must be somewhat larger than the drive level; :func:`qmix.qtcurrent.suggest_num_b` estimates it from a solved junction voltage.

Multiple tones
--------------

Now consider :math:`F` tones with frequencies :math:`\omega_1, \ldots, \omega_F` that are not harmonically related (e.g., an LO and an RF signal), each with its own harmonics. The phase factor of the total voltage is the product of the phase factors of the individual tones, so it is described by the coefficients :math:`C^{(f)}_k` of each tone :math:`f`, calculated as above:

.. math::

   W(t) = \prod_{f=1}^{F} \sum_{k_f} C^{(f)}_{k_f} \, e^{-j k_f \omega_f t}.

This is the central idea of *multi-tone spectral domain analysis*: the phase factor is never formed in the time domain. Each tone is described by its own set of coefficients, and the tunneling current is assembled directly at the frequencies of interest.

The tunneling current
---------------------

The tunneling current at an output frequency :math:`\omega` (e.g., 0 for the DC current, the LO frequency, or the IF) comes from every combination of tone harmonics :math:`\mathbf{a} = (a_1, \ldots, a_F)` that lands on that frequency,

.. math::

   a_1 \omega_1 + a_2 \omega_2 + \cdots + a_F \omega_F = \omega.

For each combination, the contribution is given by a sum over the summation indices of all tones (Eqns. 5.25 and 5.35 in Kittara's thesis):

.. math::
   :label: mtsda-sum

   R_{\mathbf{a}} + j S_{\mathbf{a}} = \sum_{k_1 = -B_1}^{B_1} \cdots \sum_{k_F = -B_F}^{B_F}
   \left[ \prod_{f=1}^{F} C^{(f)}_{k_f} \, C^{(f)*}_{k_f + a_f} \right]
   \tilde{I}\!\left( V_0 + \sum_{f=1}^{F} k_f \omega_f \right).

The response function is evaluated at the bias voltage shifted by every combination of photon energies, and each value is weighted by products of the phase factor coefficients. The current phasor at :math:`\omega` is then (Eqn. 5.36)

.. math::

   I_{\omega} = \sum_{\mathbf{a}} \left[ \left( S_{\mathbf{a}} + S_{-\mathbf{a}} \right) - j \left( R_{\mathbf{a}} - R_{-\mathbf{a}} \right) \right],

and the DC current is :math:`I_{0} = S_{\mathbf{0}}`. For one tone and one harmonic, :math:`S_{\mathbf{0}}` reduces to the Tien-Gordon formula above.

Three practical points follow:

- **Which combinations contribute.** QMix includes all combinations with :math:`|a_f| \le P` (the number of harmonics). When the tones are commensurate, combinations with larger indices can land on the same output frequency, e.g., :math:`2 \, \omega_{LO} - \omega_{LSB} = \omega_{USB}` for a mixer with the LO, both sidebands and the IF as tones. Some of these are first order in the signals, so QMix also includes every combination whose contribution could exceed a small fraction (``MIXING_TOL``) of the current at that frequency. To decide which frequencies are equal, QMix rounds them to 4 decimal places (normalized), so tones closer together than about :math:`10^{-4} f_\mathrm{gap}` cannot be told apart.
- **Cost.** Equation :eq:`mtsda-sum` has :math:`\prod_f (2 B_f + 1)` terms for every bias voltage and every combination :math:`\mathbf{a}`. QMix evaluates the response function once at all of the shifted voltages (:func:`qmix.qtcurrent.interpolate_respfn`) and then evaluates all of the combinations together, in batches. The cost still grows *multiplicatively* with the number of tones. With five or more tones, or with large summation limits, the :ref:`fourier-shift-method` is more efficient.
- **The junction voltage must be known.** These equations give the current for a given voltage across the junction, :math:`V_{f,p}` for every tone :math:`f` and harmonic :math:`p`. In a real circuit, this voltage depends on the current that the junction draws. Harmonic balance (next page) solves this.

In QMix, the voltages are stored in an array ``vj[f, p, i]`` (tone, harmonic, bias voltage) with the tone and harmonic axes indexed from 1. The current is calculated with :func:`qmix.qtcurrent.qtcurrent`, which takes the junction voltage, the embedding circuit (for the frequencies), the response function, and the output frequencies.
