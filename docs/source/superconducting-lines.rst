.. _superconducting-lines:

Superconducting Transmission Lines
==================================

The tuning circuits of SIS mixers are usually made from superconducting microstrip: e.g., a Nb strip on a thin SiO\ :sub:`2` dielectric above a Nb ground plane. Below the gap frequency of the films, the films have very little loss, but they have a large *kinetic inductance*: the supercurrent penetrates the films to a depth :math:`\lambda` (~80 nm for Nb films), and the kinetic energy of the Cooper pairs adds to the inductance of the line. When the dielectric is only a few penetration depths thick, the waves are slowed down by ~20%, which shifts the frequency at which a tuning circuit resonates by the same amount. Above the gap frequency (~700 GHz for Nb), photons break Cooper pairs and the films become lossy.

QMix models these lines in three steps: the complex conductivity of the films (Mattis-Bardeen theory), the surface impedance of a film of finite thickness, and the microstrip line with these surface impedances. The functions are in :mod:`qmix.superconductor` (films) and :mod:`qmix.network` (lines). The model is the same as in SuperMix (Ward et al., 1999), and the results agree with it (see the end of this page).

Unlike the rest of QMix, the values here are not normalized: frequencies are in Hz, temperatures in K, lengths in m and impedances in ohms. Complex values use the engineering convention :math:`e^{+j\omega t}`, as for the impedances in the rest of QMix: inductive reactances are positive.

The energy gap
--------------

The energy gap :math:`\Delta(T)` of the superconductor decreases with temperature and vanishes at the critical temperature :math:`T_c`. QMix takes the gap at zero temperature from the gap voltage of the film, :math:`V_\mathrm{gap} = 2 \Delta(0) / e`, and its temperature dependence from the weak-coupling BCS gap equation,

.. math::

   \ln \frac{\Delta(0)}{\Delta(T)} = 2 \int_0^\infty \frac{f(E)}{E} \, \mathrm{d}\xi,
   \qquad E = \sqrt{\xi^2 + \Delta(T)^2},
   \qquad \Delta(0) = \frac{\pi}{e^{\gamma_E}} \, k_B T_c \approx 1.764 \, k_B T_c,

where :math:`f(E) = 1 / (e^{E / k_B T} + 1)` is the Fermi function and :math:`\gamma_E` is Euler's constant. With :math:`\xi = \Delta \sinh u`, the integral becomes :math:`2 \int_0^\infty f(\Delta \cosh u) \, \mathrm{d}u`, which :func:`qmix.superconductor.energy_gap` solves for :math:`\Delta(T) / \Delta(0)`. Only this ratio comes from BCS theory, so :math:`V_\mathrm{gap}` and :math:`T_c` don't need to satisfy the BCS ratio (for Nb films, :math:`2 \Delta(0) / k_B T_c \approx 3.5-3.9`). Near :math:`T_c`, :math:`\Delta(T) / \Delta(0) \approx 1.74 \sqrt{1 - T / T_c}`.

The Mattis-Bardeen conductivity
-------------------------------

In the local (dirty) limit, i.e., when the mean free path of the electrons is much shorter than the penetration depth (true for most sputtered Nb and NbTiN films), the complex conductivity :math:`\sigma = \sigma_1 - j \sigma_2` of a superconductor follows from the Mattis-Bardeen theory (Mattis and Bardeen, 1958; here in the form of Kautz, 1978). With energies normalized to :math:`\Delta(T)`, the reduced frequency :math:`\Omega = h f / \Delta(T)` and the reduced temperature :math:`\tau = k_B T / \Delta(T)`,

.. math::

   \frac{\sigma_1}{\sigma_n} = \frac{2}{\Omega} \int_1^\infty \frac{[f(x) - f(x + \Omega)] \, g(x)}{\sqrt{x^2 - 1} \sqrt{(x + \Omega)^2 - 1}} \, \mathrm{d}x
   + \frac{1}{\Omega} \int_{1 - \Omega}^{-1} \frac{[1 - 2 f(x + \Omega)] \, |g(x)|}{\sqrt{x^2 - 1} \sqrt{(x + \Omega)^2 - 1}} \, \mathrm{d}x,

.. math::

   \frac{\sigma_2}{\sigma_n} = \frac{1}{\Omega} \int_{\max(1 - \Omega, -1)}^{1} \frac{[1 - 2 f(x + \Omega)] \, g(x)}{\sqrt{1 - x^2} \sqrt{(x + \Omega)^2 - 1}} \, \mathrm{d}x,

with :math:`g(x) = x^2 + \Omega x + 1`, the Fermi function :math:`f(x) = 1 / (e^{x / \tau} + 1)`, and the normal-state conductivity :math:`\sigma_n = 1 / \rho_n`. The first term of :math:`\sigma_1` is the absorption by thermally excited quasiparticles, and the second term (only for :math:`h f > 2 \Delta`) is pair breaking. :func:`qmix.superconductor.mattis_bardeen` evaluates the integrals with adaptive quadrature: the inverse square-root singularities at the ends of the intervals are handled with algebraic weights (or the substitution :math:`x = 1 + s^2`), so the integrals converge to ~1e-12, also at the gap edge. Checks:

* At :math:`T = 0`, the integrals have a closed form in terms of complete elliptic integrals (Mattis and Bardeen, 1958), which QMix matches to ~1e-12.
* At low frequencies, :math:`\sigma_2 / \sigma_n \to (\pi \Delta / h f) \tanh(\Delta / 2 k_B T)`, which gives the London penetration depth in the local limit, :math:`\lambda = \sqrt{\hbar \rho_n / (\pi \mu_0 \Delta \tanh(\Delta / 2 k_B T))}` (:meth:`qmix.superconductor.SuperconductingFilm.penetration_depth`).
* At high frequencies, :math:`\sigma \to \sigma_n`, and above :math:`T_c`, :math:`\sigma = \sigma_n`.

The integrals take ~0.2 ms per frequency, and :class:`qmix.superconductor.SuperconductingFilm` caches the values.

Surface impedance of a film
---------------------------

The surface impedance :math:`Z_s` of a film is the ratio of the tangential electric field to the total surface current, at the surface that faces the dielectric of the line. For a local conductor, a plane wave in the metal has the wavenumber and wave impedance

.. math::

   k_1 = k_0 \, n, \qquad Z_1 = \frac{Z_\mathrm{vac}}{n}, \qquad n = \sqrt{1 - j \frac{Z_\mathrm{vac} \, \sigma}{k_0}},

with :math:`k_0 = \omega / c` and the branch of the square root with :math:`\mathrm{Im}(n) \le 0`, so that the wave decays into the metal. The film is then a section of transmission line of length :math:`t` (its thickness), terminated on the far side by an impedance :math:`Z_b` (by default the impedance of free space :math:`Z_\mathrm{vac}`, i.e., radiation from the back of the film):

.. math::

   Z_s = Z_1 \frac{1 + \Gamma}{1 - \Gamma},
   \qquad \Gamma = \frac{Z_b - Z_1}{Z_b + Z_1} \, e^{-2 j k_1 t}.

This is :func:`qmix.superconductor.film_impedance`. For stacks of films (:class:`qmix.superconductor.Multilayer`, e.g., a thin Nb film on NbTiN), the formula is applied layer by layer, from the back: the surface impedance of the layers behind a film is its termination. Limits: for a thick normal metal, :math:`Z_s = (1 + j) \sqrt{\pi f \mu_0 \rho}` (skin effect); for a thick superconductor at low frequencies, :math:`Z_s \approx j \omega \mu_0 \lambda`; and for a film much thinner than the skin or penetration depth, :math:`Z_s \approx 1 / (\sigma t + 1 / Z_b)`. QMix uses SciPy's physical constants. The thickness also enters the geometry of the strip (below).

Microstrip lines
----------------

:func:`qmix.network.microstrip` gives the characteristic impedance :math:`Z_c` and the propagation constant :math:`\gamma = \alpha + j \beta` (waves vary as :math:`e^{-\gamma z}`) of a microstrip line: a strip of width :math:`w` and thickness :math:`t`, on a dielectric of thickness :math:`h` and relative permittivity :math:`\epsilon_r` (and loss tangent :math:`\tan\delta`) above a ground plane, with an optional superstrate above the strip (default: vacuum). It combines:

1. **The effective permittivity** :math:`\epsilon_\mathrm{eff}` of the lossless line: the quasi-static value of Hammerstad and Jensen (1980), including the strip thickness as a correction to its width, with the dispersion of Kirschning and Jansen (1982) (:math:`\epsilon_\mathrm{eff}` rises towards :math:`\epsilon_r` at high frequencies).
2. **Geometric factors** from the conformal mapping of Yassin and Withington (1995): :math:`g_1`, the external inductance per unit length divided by :math:`\mu_0`, which gives the characteristic impedance of the lossless line :math:`Z_0 = Z_\mathrm{vac} \, g_1 / \sqrt{\epsilon_\mathrm{eff}}`; and :math:`g_s` and :math:`g_g` (in 1/m), which weight the surface impedances of the strip and the ground plane by the current distributions on them (roughly, one over the effective width of each conductor).
3. **The surface impedances** of the strip and the ground plane, which add to the series impedance per unit length:

.. math::

   Z' = j k Z_0 + g_s Z_{s,\mathrm{strip}} + g_g Z_{s,\mathrm{ground}},
   \qquad
   Y' = \frac{k}{Z_0} \left( 2 \delta + j (1 - \delta^2) \right),
   \qquad k = \frac{\omega \sqrt{\epsilon_\mathrm{eff}}}{c},

.. math::

   Z_c = \sqrt{Z' / Y'}, \qquad \gamma = \sqrt{Z' \, Y'},

where :math:`\delta` is the effective dielectric loss (the loss tangents of the dielectric and the superstrate, weighted by the filling factor; it is half of the loss tangent of :math:`\gamma`). The real part of the surface impedance gives conductor loss, and its imaginary part (the kinetic inductance of a superconductor) adds to the inductance of the line: for a wide line (:math:`g_1 \to h / w`, :math:`g_s, g_g \to 1 / w`) with thick films at low frequencies, this gives Swihart's (1961) result :math:`\beta / \beta_0 = \sqrt{1 + (2 \lambda / h) \coth(t / \lambda)}`.

The strip and the ground plane can be any object with a surface impedance method ``zs(freq, temp)`` and a ``thickness``: :class:`qmix.superconductor.SuperconductingFilm`, :class:`qmix.superconductor.NormalFilm` or :class:`qmix.superconductor.Multilayer`; ``None`` means a perfect conductor. :func:`qmix.network.microstrip_line` returns the line as a 2-port :class:`qmix.network.Network`, which can be connected with other networks (e.g., :func:`qmix.network.tee`, :func:`qmix.network.transmission_line`) and used for the embedding circuit of the junctions (:func:`qmix.network.set_thevenin`, :func:`qmix.network.impedance_matrix`). Example #6 designs a twin-junction tuning circuit from Nb microstrip this way.

Comparison with SuperMix
------------------------

``profile/supermix/compare_lines.py`` runs SuperMix's ``super_film``, ``normal_film`` and ``microstrip`` classes (``smx_lines.cc``) on several lines (Nb and NbTiN, narrow and wide strips, thin films, lossy dielectrics and superstrates, 10 GHz to 3 THz), and ``tests/test_supermix_reference.py`` checks QMix against the stored results. With SuperMix's physical constants:

* For normal-metal films, the surface impedances and the line parameters agree to ~1e-15, so the microstrip model is the same.
* For superconductors below :math:`T / T_c = 0.04` (where both give :math:`\Delta(T) = \Delta(0)`), the conductivities agree to ~2e-8, and to 2e-6 right at the gap edge. This is the accuracy of SuperMix's integration: QMix agrees with the closed form at :math:`T = 0` to ~1e-12. The line parameters agree to ~4e-7.
* At higher temperatures, SuperMix interpolates Mühlschlegel's (1959) table of :math:`\Delta(T)` (4 digits), which differs from the BCS gap equation by up to ~1e-4, and the conductivities differ by up to ~1.5e-4 (at 4.2 K for Nb; most of it is the thermally excited quasiparticles, which depend on :math:`e^{-\Delta / k_B T}`).
