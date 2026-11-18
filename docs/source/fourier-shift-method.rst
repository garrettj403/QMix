.. _fourier-shift-method:

The Fourier-Shift Method
========================

QMix provides two ways to evaluate the sum over the phase factor coefficients that gives the quasiparticle tunneling current: the *direct* summation, which is the method described by Kittara (2002) and used by QMix since its first release, and the *Fourier-shift* method, which was added to make simulations with many tones (or large summation limits) practical. This page describes the Fourier-shift method, when to use it, and what accuracy to expect.

All quantities below are normalized in the usual way: voltages to the gap voltage, currents to the gap current, and frequencies to the gap frequency.

The sum that has to be evaluated
--------------------------------

In multi-tone spectral domain analysis, the junction is driven by :math:`F` non-harmonically related tones with frequencies :math:`\omega_1, \ldots, \omega_F`. For each tone :math:`f`, the phase factor coefficients :math:`C^{(f)}_k` (Eqn. 5.12 in Kittara's thesis) are known for :math:`k = -B_f, \ldots, B_f`, where :math:`B_f` is the summation limit (``num_b`` in QMix). The tunneling current at an output frequency :math:`\omega` is assembled from the harmonic combinations :math:`\mathbf{a} = (a_1, \ldots, a_F)` for which :math:`a_1 \omega_1 + \cdots + a_F \omega_F = \omega`. For each such combination, Eqn. 5.25 (and its generalization, Eqn. 5.35) gives

.. math::
   :label: direct-sum

   R_{\mathbf{a}} + j S_{\mathbf{a}} = \sum_{k_1 = -B_1}^{B_1} \cdots \sum_{k_F = -B_F}^{B_F}
   \left[ \prod_{f=1}^{F} P^{(f)}_{a_f}(k_f) \right]
   \tilde{I}\!\left( V_b + \sum_{f=1}^{F} k_f \omega_f \right),

where :math:`V_b` is the bias voltage, :math:`\tilde{I}(V) = I_{kk}(V) + j I_{dc}(V)` is the response function, and

.. math::

   P^{(f)}_{a}(k) = C^{(f)}_{k} \, C^{(f)*}_{k+a}

are the *pair products* of one tone. The current at :math:`\omega` then follows from Eqn. 5.36:

.. math::

   \tilde{I}_{\omega} = \sum_{\mathbf{a}} \left[ \left( S_{\mathbf{a}} + S_{-\mathbf{a}} \right) - j \left( R_{\mathbf{a}} - R_{-\mathbf{a}} \right) \right].

The weight in square brackets in :eq:`direct-sum` factorizes into one term per tone, but the response function does not: its argument contains all of the summation indices at once. The direct method therefore has to evaluate :math:`\tilde{I}` on the full grid of index combinations. For every bias point this is

.. math::

   N_{\mathrm{direct}} = \prod_{f=1}^{F} \left( 2 B_f + 1 \right)

values, which are stored in the response function matrix (``resp_matrix`` in QMix) and visited once per harmonic combination. Both the memory and the time per current calculation grow *multiplicatively* with the number of tones. For four tones with ``num_b = (15, 10, 10, 15)`` and 201 bias points, the matrix is 1.4 GB; for five tones with ``num_b = (15, 10, 10, 15, 15)`` it would be 42 GB.

Factorizing the shift
---------------------

The Fourier-shift method removes the coupling through the response function by writing :math:`\tilde{I}` as a Fourier series over a period :math:`L` that starts at a voltage :math:`v_0`:

.. math::
   :label: fourier-series

   \tilde{I}(V) \approx \sum_{n=-N}^{N} c_n \, e^{j \tau_n (V - v_0)},
   \qquad \tau_n = \frac{2 \pi n}{L}.

(How the series is constructed, and what "approximately" means, is the subject of the next section.) A shift of the argument is now a product of phase factors:

.. math::

   \tilde{I}\!\left( V_b + \sum_f k_f \omega_f \right)
   = \sum_{n} c_n \, e^{j \tau_n (V_b - v_0)} \prod_{f=1}^{F} e^{j \tau_n k_f \omega_f}.

Substituting this into :eq:`direct-sum` and exchanging the order of summation, the sum over each tone's index can be done on its own:

.. math::
   :label: fourier-sum

   R_{\mathbf{a}} + j S_{\mathbf{a}} = \sum_{n=-N}^{N} c_n \, e^{j \tau_n (V_b - v_0)}
   \prod_{f=1}^{F} M^{(f)}_{a_f}(\tau_n),
   \qquad
   M^{(f)}_{a}(\tau) = \sum_{k=-B_f}^{B_f} P^{(f)}_{a}(k) \, e^{j \tau k \omega_f}.

Each :math:`M^{(f)}_a` is a one-dimensional sum over :math:`2 B_f + 1` terms, evaluated at the :math:`2N + 1` values of :math:`\tau_n`. Over all bias points this is one matrix product between the per-tone *shift matrix* :math:`E^{(f)}_{nk} = e^{j \tau_n k \omega_f}`, which depends only on the frequencies and the summation limits, and the pair products :math:`P^{(f)}_a(k)`, which depend on the junction voltage. The product over tones and the final sum over :math:`n` cost :math:`F` multiplications per mode and bias point. For every bias point, the work is therefore

.. math::

   N_{\mathrm{fourier}} \approx (2N + 1) \left[ \sum_{f=1}^{F} n_{a,f} \left( 2 B_f + 1 \right) + F \, n_{\mathbf{a}} \right],

where :math:`n_{a,f}` is the number of distinct values of :math:`a_f` that occur (at most :math:`2 \, p_{\max} + 1`, with :math:`p_{\max}` the number of harmonics) and :math:`n_{\mathbf{a}}` is the number of harmonic combinations. The cost is *additive* in the tones, and no multi-dimensional response function matrix exists at all: the largest arrays are the :math:`M^{(f)}_a`, with one row per mode and one column per bias point. QMix processes the modes in blocks (``FOURIER_BLOCK``) so that even these stay small.

The structure of :eq:`fourier-sum` is worth a remark. The variable :math:`\tau` is conjugate to voltage, so it plays the role of time: the Fourier transform of the response function with respect to voltage is, up to constants, the time-domain response function :math:`\psi(t)` of Withington and Kollberg (1989). The sum over :math:`n` is therefore a time integral, and :math:`M^{(f)}_a(\tau)` is the time-domain autocorrelation of one tone's phase factor. Withington and Kollberg rejected a time-domain analysis because the response function has to be sampled over many gap-frequency cycles for every current evaluation, and because a multi-tone waveform has no short period over which the output frequency could be extracted. The Fourier-shift method keeps the spectral part of their argument: the output frequency is still selected analytically through the index tuples :math:`\mathbf{a}`, and only the shift of the response function is moved to the :math:`\tau` domain. The number of modes :math:`N` is the counterpart of their number of time samples, which is why, for one or two tones, the direct sum is still cheaper, exactly as they found. With many tones the multiplicative cost of the direct sum takes over and the additive cost of :eq:`fourier-sum` wins.

Constructing the Fourier series
-------------------------------

The response function is not periodic: :math:`I_{dc}(V)` grows linearly for large :math:`|V|`. It is therefore multiplied by a window :math:`w(V)` before the series is formed. The window is exactly one on the interval

.. math::

   \left[ v_{lo}, \, v_{hi} \right] = \left[ \min V_b - \sum_f B_f \omega_f, \;\; \max V_b + \sum_f B_f \omega_f \right],

which contains every voltage at which :eq:`direct-sum` evaluates :math:`\tilde{I}`, and it falls smoothly to zero over a taper of width :math:`T` (``FOURIER_TAPER``, 3 by default) on either side. The period is :math:`L = v_{hi} - v_{lo} + 2T` and it starts at :math:`v_0 = v_{lo} - T`. The taper is built from the infinitely differentiable step

.. math::

   h(x) = \frac{e^{-1/x}}{e^{-1/x} + e^{-1/(1-x)}}, \qquad 0 < x < 1,

with :math:`h = 0` for :math:`x \le 0` and :math:`h = 1` for :math:`x \ge 1`, so that

.. math::

   w(V) = h\!\left( \frac{V - v_0}{T} \right) h\!\left( \frac{v_0 + L - V}{T} \right).

The windowed function :math:`w \tilde{I}` vanishes with all of its derivatives at both ends of the period, so its periodic extension is as smooth as :math:`\tilde{I}` itself. Its Fourier coefficients are obtained from a single FFT of :math:`M = 2N + 1` equally spaced samples,

.. math::

   c_n = \frac{1}{M} \sum_{m=0}^{M-1} w(v_m) \, \tilde{I}(v_m) \, e^{-2 \pi j n m / M},
   \qquad v_m = v_0 + \frac{m}{M} L,

which is the trapezoidal rule and is spectrally accurate for a smooth periodic function.

Two points follow from this construction. First, inside :math:`[v_{lo}, v_{hi}]` the window is exactly one, so it introduces no approximation whatsoever; the only approximation in the whole method is the truncation of the series at :math:`|n| \le N` in :eq:`fourier-series`. Second, the series is built once per response function, bias voltage grid, set of frequencies and set of summation limits, and it does not depend on the junction voltage. QMix therefore builds it in :func:`qmix.qtcurrent.interpolate_respfn`, which returns a :class:`qmix.qtcurrent.FourierResp` object in place of the response function matrix, and reuses it across all of the current calculations in a harmonic balance run.

Accuracy
--------

How quickly the series converges depends on how smooth the response function is. In QMix the response function is a spline through the DC I-V curve and its Kramers-Kronig transform. A cubic spline has jumps in its third derivative at the knots, so its Fourier coefficients decay only like :math:`n^{-4}`; a quintic spline (``spline_order=5`` when constructing the :class:`qmix.respfn.RespFn`) has jumps in the fifth derivative and its coefficients decay like :math:`n^{-6}`. The table lists the largest error of the truncated series, relative to the spline it represents, over a window from -5 to 8 for the polynomial I-V model (``RespFnPolynomial(50)``). The error is always largest at the gap voltage, where the knots are densest.

===========  ===============  ===============
:math:`N`    cubic spline     quintic spline
===========  ===============  ===============
2000         3e-5             1e-6
4000         8e-6             1e-7
8000         8e-7             3e-9
16000        8e-8             5e-11
32000        1e-8             2e-12
===========  ===============  ===============

With the default of :math:`N = 8000` modes (``qmix.qtcurrent.NUM_MODES``), the tunneling currents from the Fourier-shift method differ from the direct method by roughly one part in :math:`10^6` with cubic splines and by less than one part in :math:`10^7` with quintic splines. For comparison, the splines themselves represent the underlying I-V data to a default tolerance of :math:`10^{-3}` (``max_interp_error``), so both methods are far more accurate than the response function is a representation of the junction. If the Fourier-shift method is going to be used, quintic splines are recommended: they cost nothing extra at run time and remove most of the difference between the two methods.

Two limitations follow from the dependence on smoothness. The method cannot be used with :class:`qmix.respfn.RespFnPerfect` without smearing, whose DC I-V curve is discontinuous at the gap: the series then converges too slowly to be useful (Gibbs oscillations). And the direct method remains the reference for the stored benchmark values, which are checked to 14 decimal places.

Cost
----

The two methods have opposite scaling, so neither is best everywhere. For a small number of tones with modest summation limits, the direct method is faster: for the four-tone benchmark (``num_b = (10, 5, 5, 5)``, 201 bias points) it takes about 40 ms where the Fourier-shift method takes about 0.6 s. The Fourier-shift method takes over when the product of the summation limits becomes large, which is what happens with five or more tones, or with large ``num_b`` on several tones (for example, a strongly driven IF tone in a saturation study). The following were measured on the same machine for five tones at 201 bias points.

=============================  ===========================  =============================
``num_b``                      direct                       Fourier-shift (:math:`N = 8000`)
=============================  ===========================  =============================
(6, 4, 4, 4, 4)                0.3 s, 270 MB matrix         1.2 s
(8, 5, 5, 5, 5)                0.9 s, 800 MB matrix         1.3 s
(15, 10, 10, 15, 15)           42 GB matrix (not feasible)  1.4 s
=============================  ===========================  =============================

The time of the Fourier-shift method is almost independent of ``num_b``, because the shift matrices are small and the work is dominated by the :math:`(2N+1) \times n_{pts}` products over the modes.

Using it
--------

The method is selected with the ``method`` argument of :func:`qmix.qtcurrent.qtcurrent`, :func:`qmix.qtcurrent.interpolate_respfn` and :func:`qmix.harmonic_balance.harmonic_balance`:

- ``'direct'``: the direct summation.
- ``'fourier'``: the Fourier-shift method.
- ``'auto'`` (the default): the direct method, unless the response function matrix would be larger than ``qmix.qtcurrent.MAX_RESP_MATRIX_BYTES`` (2 GB by default), in which case the Fourier-shift method is used and a notice is printed when ``verbose`` is true.

For example, a five-tone simulation with quintic splines:

.. code-block:: python

   import qmix

   resp = qmix.respfn.RespFnPolynomial(50, spline_order=5)

   cct = qmix.circuit.EmbeddingCircuit(5, 1)
   cct.freq[1:] = [0.30, 0.33, 0.27, 0.03, 0.07]
   # ... set cct.vt and cct.zt ...

   num_b = (15, 10, 10, 15, 15)
   vj = qmix.harmonic_balance.harmonic_balance(cct, resp, num_b, method='fourier')
   idc = qmix.qtcurrent.qtcurrent(vj, cct, resp, 0., num_b, method='fourier')

As with the direct method, the response function representation can be built once and reused when ``qtcurrent`` is called repeatedly with the same circuit:

.. code-block:: python

   fresp = qmix.qtcurrent.interpolate_respfn(cct, resp, num_b, method='fourier', num_modes=16000)
   idc = qmix.qtcurrent.qtcurrent(vj, cct, resp, 0., num_b, resp_matrix=fresp)

The number of modes is set by ``num_modes`` (default ``qmix.qtcurrent.NUM_MODES``, 8000). Doubling it roughly doubles the run time, and improves the accuracy by the factors in the table above. The :class:`qmix.qtcurrent.FourierResp` object has a ``resp`` method that evaluates the truncated series, which is a convenient way to check the accuracy of a given ``num_modes`` against the response function directly:

.. code-block:: python

   import numpy as np

   v = np.linspace(-1, 3, 1001)
   error = np.abs(fresp.resp(v) - resp(v)).max()
