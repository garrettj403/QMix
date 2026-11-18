""" This module contains functions to calculate the quasiparticle tunneling
currents passing through an SIS junction.

Description:

    Given the voltages applied across an SIS junction, the quasiparticle
    tunneling currents can be calculated using multi-tone spectral domain
    analysis (MTSDA; see references in online docs).

    Two methods are available for the summation over the phase factor
    coefficients (Eqn. 5.25 in Kittara's thesis):

      - ``'direct'``: the response function is interpolated at every
        combination of the summation indices (``interpolate_respfn``), and
        the sum is evaluated directly. The size of this matrix, and the time
        per current calculation, scale with the product of (2 * num_b + 1)
        over all tones.

      - ``'fourier'``: the response function is represented by a (windowed)
        Fourier series, so that the shift in its argument factorizes into one
        term per tone. The cost then scales with the sum of (2 * num_b + 1)
        over all tones, and no large matrix is needed. This makes 5+ tones
        (or large ``num_b``) tractable. The accuracy is limited by the
        smoothness of the response function splines: ~1e-7 (relative to the
        direct method) with the default cubic splines, ~1e-9 or better with
        ``spline_order=5`` (see ``qmix.respfn.RespFn``). It is not suitable
        for ``RespFnPerfect`` without smearing.

    By default (``method='auto'``), the direct method is used unless the
    interpolated response function matrix would be larger than
    ``MAX_RESP_MATRIX_BYTES``.

    The current at an output frequency is the sum over all combinations of
    tone indices (a_1, ..., a_F) with a_1 * f_1 + ... + a_F * f_F equal to
    that frequency (Eqn. 5.36). All combinations with ``|a_f| <= num_p`` are
    included. When the tones are commensurate (e.g., LO, USB, LSB and IF),
    combinations with larger indices can also land on the output frequency,
    some of them first order in the signals (e.g., 2 * LO - LSB = USB). These
    are included if an upper bound on their contribution exceeds
    ``MIXING_TOL`` times the size of that current (see ``qtcurrent``).

Note:

    This code is largely based on P. Kittara's 2002 DPhil thesis (see
    references in online docs). I include some inline comments to refer to
    specific equations.

    Also, all of the values in this module are normalized, i.e., voltages are
    normalized to the gap voltage, frequencies are normalized to the gap
    frequency, etc.

"""

import functools
import itertools
import warnings
from timeit import default_timer as timer

import numba as nb
import numpy as np
from scipy.special import jv as bessel
from scipy.special import j0, j1


# round frequency values to this number of decimal places
# required when comparing frequency to frequency_list
ROUND_FREQ = 4

# With method='auto', switch from the direct summation to the Fourier-shift
# method when the interpolated response function matrix would be larger than
# this (in bytes)
MAX_RESP_MATRIX_BYTES = 2 * 1024 ** 3

# Default number of Fourier modes (on each side of zero) for the
# Fourier-shift method
NUM_MODES = 8000

# Width (in normalized voltage) of the window taper for the Fourier-shift
# method
FOURIER_TAPER = 3.

# Fourier modes are processed in blocks of this size (limits memory use)
FOURIER_BLOCK = 2048

METHODS = ('auto', 'direct', 'fourier')

# Include combinations of tone indices with |a_f| > num_p that land on an
# output frequency if the upper bound on their contribution (the product over
# tones of the summed magnitudes of the pair products C_k * conj(C_{k+a}))
# exceeds this. The bound is conservative: with 1e-6, the error in the
# currents is ~1e-8. np.inf gives the old behavior (only |a_f| <= num_p).
MIXING_TOL = 1e-6


# Determine the dc/ac tunneling currents -------------------------------------

def qtcurrent(vj, cct, resp, freq_list, num_b=15, verbose=True, resp_matrix=None, method='auto', num_modes=None,
              mixing_tol=None):
    """Calculate the quasiparticle tunneling current.

    This function uses multi-tone spectral domain analysis (MTSDA; see
    references in online docs). The current is calculated based on the
    voltage applied across the junction (vj).

    Note:

        This function will return the tunneling current for all of the
        frequencies listed in freq_list (normalized to the gap frequency).
        E.g., to solve for the dc tunneling current and the ac tunneling
        current at 230 GHz, the ``freq_list`` would  be ``[0, 230e9 / fgap]``
        where ``fgap`` is the gap frequency.

        Any number of non-harmonically related tones is supported. See the
        module docstring for the two summation methods (``'direct'`` and
        ``'fourier'``).

    Args:
        vj (ndarray): Voltage across the SIS junction
        cct (qmix.circuit.EmbeddingCircuit): Embedding circuit
        resp (qmix.respfn.RespFn): Response function
        freq_list: Calculate the tunneling currents for these frequencies
            (normalized to the gap frequency)
        num_b (float/tuple, optional): Summation limits for phase factor
            coefficients, default is 15
        verbose (bool, optional): Print info to the terminal if true, default
            is True
        resp_matrix (ndarray/FourierResp, optional): The interpolated
            response function matrix (or its Fourier representation),
            generated by interpolate_respfn(), default is None. If given, it
            determines the method (``method`` is ignored).
        method (str, optional): ``'direct'``, ``'fourier'`` or ``'auto'``
            (see module docstring), default is ``'auto'``
        num_modes (int, optional): Number of Fourier modes for the
            Fourier-shift method, default is ``NUM_MODES``
        mixing_tol (float, optional): include combinations of tone indices
            with ``|a_f| > num_p`` that land on an output frequency if the
            upper bound on their contribution exceeds this times the size of
            that current (see module docstring), default is ``MIXING_TOL``
            (1e-6, which gives relative errors of ~1e-7 or better). Larger
            values are faster for commensurate tones; ``np.inf`` only
            includes ``|a_f| <= num_p``. To change it for
            ``harmonic_balance``, set ``qmix.qtcurrent.MIXING_TOL``.

    Returns:
        ndarray: Quasiparticle tunneling current

    """

    # Load, prepare and check input data -------------------------------------

    num_f = cct.num_f   # number of frequencies
    num_p = cct.num_p   # number of harmonics
    npts = cct.vb_npts  # number of bias voltages

    assert cct.freq[1:].min() > 0., "All freq must be > 0!"

    # TODO: there must be a better way...
    try:
        freq_list = list(freq_list)
        freq_is_list = True
    except TypeError:
        freq_list = [float(freq_list)]
        freq_is_list = False

    # TODO: there must be a better way...
    for i, freq_val in enumerate(freq_list):
        freq_list[i] = round(freq_val, ROUND_FREQ)
    freq_npts = len(freq_list)

    freq = cct.freq

    nb_list = _unpack_num_b(num_b, num_f)

    if verbose:
        print("Calculating tunneling current...")
        print(" - {0} tone(s)".format(cct.num_f))
        print(" - {0} harmonic(s)".format(cct.num_p))
        start_time = timer()

    # Convolution coefficients ------------------------------------------------

    ccc = calculate_phase_factor_coeff(vj, freq, num_f, num_p, num_b)

    # Interpolate response function ------------------------------------------

    if resp_matrix is None:
        resp_matrix = interpolate_respfn(cct, resp, num_b, method=method, num_modes=num_modes, verbose=verbose)
    elif isinstance(resp_matrix, FourierResp):
        assert resp_matrix.vb.size == npts
    else:
        assert resp_matrix.ndim == num_f + 1
        assert resp_matrix.shape[-1] == npts

    if verbose:
        print(" - method: {0}".format('fourier' if isinstance(resp_matrix, FourierResp) else 'direct'))

    # Harmonic combinations (a, b, c, ...) that match each output frequency ---

    tuple_lists = [_matching_tuples(freq_out, freq, num_f, num_p) for freq_out in freq_list]

    # Combinations with |a_f| > num_p that also land on an output frequency
    if mixing_tol is None:
        mixing_tol = MIXING_TOL
    if np.isfinite(mixing_tol):
        tuple_lists = _add_mixing_products(tuple_lists, freq_list, freq, num_p, nb_list, ccc, mixing_tol)

    # Calculate the tunneling currents ----------------------------------------

    if isinstance(resp_matrix, FourierResp):
        current_out = _current_fourier(resp_matrix, ccc, tuple_lists, nb_list)
    else:
        current_out = np.zeros((freq_npts, npts), dtype=complex)
        for i, tuple_list in enumerate(tuple_lists):
            for a_tuple in tuple_list:
                current_out[i] += _current_coeff(a_tuple, ccc, resp_matrix, nb_list)

    # Done --------------------------------------------------------------------

    if verbose:
        print("Done.")
        print("Time: {0:.4f} s\n".format(timer() - start_time))

    if freq_is_list:
        return current_out
    else:
        if freq_list[0] == 0.:
            return current_out[0].real
        else:
            return current_out[0]


def _matching_tuples(freq_out, freq, num_f, num_p):
    """All harmonic combinations (a_1, ..., a_F), with |a_f| <= num_p, for
    which a_1 * freq[1] + ... + a_F * freq[F] matches the output frequency.

    Frequencies are compared after rounding to ROUND_FREQ decimal places.

    Args:
        freq_out (float): Output frequency
        freq (ndarray): Frequencies of the tones (1-indexed)
        num_f (int): Number of tones
        num_p (int): Number of harmonics

    Returns:
        list: Matching index tuples

    """

    freq_out = round(freq_out, ROUND_FREQ)
    harmonics = range(num_p, -(num_p + 1), -1)
    tuple_list = []
    for a_tuple in itertools.product(harmonics, repeat=num_f):
        freq_a = round(sum(a * freq[f + 1] for f, a in enumerate(a_tuple)), ROUND_FREQ)
        if freq_a == freq_out:
            tuple_list.append(a_tuple)

    return tuple_list


def _add_mixing_products(tuple_lists, freq_list, freq, num_p, nb_list, ccc, mixing_tol):
    """Add the combinations of tone indices with |a_f| > num_p that land on
    each output frequency and whose contribution may be significant.

    The contribution of (a_1, ..., a_F) to the current is bounded by the
    product over tones of sum_k |C_k| |C_{k+a_f}| (times the magnitude of
    the response function). A combination is included if this bound exceeds
    mixing_tol times the largest bound of the |a_f| <= num_p combinations at
    the same output frequency (i.e., relative to the size of that current).

    Args:
        tuple_lists (list): for each output frequency, the combinations with
            |a_f| <= num_p (from ``_matching_tuples``)
        freq_list (list): output frequencies
        freq (ndarray): frequencies of the tones (1-indexed)
        num_p (int): number of harmonics
        nb_list (tuple): summation limits, one per tone
        ccc (ndarray): convolution coefficients
        mixing_tol (float): tolerance for the bound

    Returns:
        list: for each output frequency, the list of combinations

    """

    num_f = len(nb_list)
    freq_key = tuple(float(freq[f]) for f in range(1, num_f + 1))
    candidate_lists = [_candidate_tuples(round(freq_out, ROUND_FREQ), freq_key, tuple(nb_list), num_p)
                       for freq_out in freq_list]
    if all(len(candidates) == 0 for candidates in candidate_lists):
        return tuple_lists
    bounds = _pair_product_bounds(ccc, nb_list)

    def weights(a_tuples):
        a_tuples = np.asarray(a_tuples, dtype=np.int64).reshape(-1, num_f)
        weight = np.ones(len(a_tuples))
        for f, num_b in enumerate(nb_list):
            in_range = np.abs(a_tuples[:, f]) <= 2 * num_b
            weight *= np.where(in_range, bounds[f][np.clip(a_tuples[:, f] + 2 * num_b, 0, 4 * num_b)], 0.)
        return weight

    tuple_lists_out = []
    for candidates, tuple_list in zip(candidate_lists, tuple_lists):
        if len(candidates) > 0:
            # Relative to the largest bound of the |a_f| <= num_p combinations,
            # so that e.g. first-order terms at a weak signal's frequency are
            # included regardless of the signal strength
            scale = weights(tuple_list).max() if len(tuple_list) > 0 else 0.
            if scale == 0:
                scale = 1.
            extra = [tuple(int(a) for a in a_tuple) for a_tuple in candidates[weights(candidates) > mixing_tol * scale]]
            tuple_list = list(tuple_list) + extra
        tuple_lists_out.append(tuple_list)

    return tuple_lists_out


def _pair_product_bounds(ccc, nb_list):
    """For each tone f and index a = -2 num_b_f ... 2 num_b_f (stored at
    a + 2 num_b_f), the maximum over bias voltages of sum_k |C_k| |C_{k+a}|,
    with k and k + a within [-num_b_f, num_b_f]."""

    bounds = []
    for f, num_b in enumerate(nb_list, 1):
        abs_c = np.abs(ccc[f][_natural_indices(num_b)])  # k = -num_b ... num_b
        dim = 2 * num_b + 1
        bound_f = np.zeros(4 * num_b + 1)
        for a in range(-2 * num_b, 2 * num_b + 1):
            lo, hi = max(0, -a), min(dim, dim - a)
            bound_f[a + 2 * num_b] = (abs_c[lo:hi] * abs_c[lo + a:hi + a]).sum(axis=0).max()
        bounds.append(bound_f)

    return bounds


def _natural_indices(num_b):
    """Storage indices of k = -num_b ... num_b (wraparound indexing)."""

    return np.r_[np.arange(-num_b, 0), np.arange(0, num_b + 1)]


@functools.lru_cache(maxsize=1024)
def _candidate_tuples(freq_out, freq, nb_list, num_p):
    """All combinations of tone indices with |a_f| <= 2 * num_b_f (beyond
    that, the pair products vanish) that land on the output frequency, except
    those with all |a_f| <= num_p (see ``_matching_tuples``).

    The tones are split into two groups, and the sums of the frequencies over
    each group are matched (meet in the middle), so the cost scales with the
    square root of the number of combinations.

    Returns:
        ndarray: combinations, shape (number of combinations, number of tones)

    """

    num_f = len(nb_list)
    half = num_f // 2

    def combinations(tones):
        if len(tones) == 0:
            return np.zeros((1, 0), dtype=np.int64), np.zeros(1)
        ranges = [np.arange(-2 * nb_list[f], 2 * nb_list[f] + 1) for f in tones]
        a = np.stack(np.meshgrid(*ranges, indexing='ij'), axis=-1).reshape(-1, len(tones))
        return a, a @ np.array([freq[f] for f in tones])

    a_1, sum_1 = combinations(range(half))
    a_2, sum_2 = combinations(range(half, num_f))

    # Pairs whose total frequency is within the rounding window, then the
    # exact check (the same as in _matching_tuples)
    order = np.argsort(sum_2)
    sum_2_sorted = sum_2[order]
    window = 10. ** -ROUND_FREQ
    lo = np.searchsorted(sum_2_sorted, freq_out - sum_1 - window)
    hi = np.searchsorted(sum_2_sorted, freq_out - sum_1 + window, side='right')
    result = []
    for i in np.flatnonzero(hi > lo):
        for j in order[lo[i]:hi[i]]:
            a_tuple = tuple(int(a) for a in np.r_[a_1[i], a_2[j]])
            if all(abs(a) <= num_p for a in a_tuple):
                continue
            if round(sum(a * freq[f] for f, a in enumerate(a_tuple)), ROUND_FREQ) == freq_out:
                result.append(a_tuple)

    result = np.array(sorted(result, reverse=True), dtype=np.int64).reshape(-1, num_f)
    result.flags.writeable = False

    return result


# Response function matrices --------------------------------------------------
# The response function (the dc I-V curve and it's KK transform) needs to be
# repeatedly interpolated in this module. The functions below do all of the
# necessary interpolations all at once to save time.
#
# Runs once per qtcurrent function call

def interpolate_respfn(cct, resp, num_b, method='auto', num_modes=None, verbose=False):
    """Interpolate the response function at all necessary voltages.

    I have included this as a stand-alone function because if you are going
    to be running ``qtcurrent`` over and over again with the same input
    signal frequencies, it can save time by pre-interpolating the response
    function.

    For the Fourier-shift method (see module docstring), this returns a
    ``FourierResp`` object instead of a matrix. Either can be passed to
    ``qtcurrent`` as ``resp_matrix``.

    Args:
        cct (qmix.circuit.EmbeddingCircuit): Embedding circuit
        resp (qmix.respfn.RespFn): Response function
        num_b (int/tuple): Summation limits for phase factor coefficients
        method (str, optional): ``'direct'``, ``'fourier'`` or ``'auto'``,
            default is ``'auto'``
        num_modes (int, optional): Number of Fourier modes for the
            Fourier-shift method, default is ``NUM_MODES``
        verbose (bool, optional): Print a notice if ``'auto'`` selects the
            Fourier-shift method, default is False

    Returns:
        ndarray/FourierResp: The interpolated response function as a matrix
        (direct method) or as a Fourier representation (Fourier-shift method)

    """

    nb_list = _unpack_num_b(num_b, cct.num_f)

    _check_respfn_range(resp, cct.vb, cct.freq, nb_list)

    if _select_method(method, nb_list, cct.vb_npts, verbose) == 'fourier':
        return FourierResp(resp, cct.vb, cct.freq, nb_list, num_modes)

    return resp(_interpolation_voltages(cct.vb, cct.freq, nb_list))


def _check_respfn_range(resp, vb, freq, nb_list):
    """Warn if the response function is needed beyond the voltage range over
    which it is defined (``resp.v_max``; e.g., |v| <= 35 for the standard
    response functions). Beyond that, it is extrapolated, which is
    inaccurate. Response functions without ``v_max`` are not checked."""

    v_max = getattr(resp, 'v_max', None)
    if v_max is None:
        return

    reach = np.abs(vb).max() + sum(num_b * freq[f] for f, num_b in enumerate(nb_list, 1))
    if reach > v_max:
        msg = ("The response function is only defined for |v| <= {:.1f}, but voltages up to {:.1f} are needed "
               "(bias voltage + num_b * freq). It will be extrapolated, which is inaccurate. Reduce num_b or the "
               "bias voltage range.")
        warnings.warn(msg.format(v_max, reach))


def _select_method(method, nb_list, npts, verbose=False):
    """Resolve method='auto' based on the size of the response function matrix.

    Args:
        method (str): 'auto', 'direct' or 'fourier'
        nb_list (tuple): Summation limits, one for each tone
        npts (int): Number of bias voltage points
        verbose (bool): Print a notice when 'auto' selects 'fourier'

    Returns:
        str: 'direct' or 'fourier'

    """

    assert method in METHODS, "method must be one of {0}".format(METHODS)

    if method != 'auto':
        return method

    nbytes = 16 * npts * resp_matrix_size(nb_list)
    if nbytes > MAX_RESP_MATRIX_BYTES:
        if verbose:
            msg = " - response function matrix would be {0:.1f} GB: using the Fourier-shift method"
            print(msg.format(nbytes / 1e9))
        return 'fourier'

    return 'direct'


def resp_matrix_size(nb_list):
    """Number of response function values per bias point for the direct
    method, i.e., the product of (2 * num_b + 1) over all tones.

    Args:
        nb_list (tuple): Summation limits, one for each tone

    Returns:
        int: Number of values

    """

    size = 1
    for num_b in nb_list:
        size *= 2 * num_b + 1

    return size


def _interpolation_voltages(vb, freq, nb_list):
    """Build the voltages required to interpolate the response function.

    I.e., vb + k * freq[1] + l * freq[2] + ... for all summation indices
    (k, l, ...). The summation indices use wraparound indexing (see
    ``calculate_phase_factor_coeff``).

    Args:
        vb (ndarray): Bias voltages, normalized
        freq (ndarray): Frequencies, normalized
        nb_list (tuple): Summation limits for phase factor coefficients, one
            for each tone

    Returns:
        ndarray: Voltages, with shape (2 * num_b1 + 1, ..., npts)

    """

    num_f = len(nb_list)
    shape = [2 * num_b + 1 for num_b in nb_list] + [len(vb)]

    voltage = np.zeros(shape)
    voltage += vb
    for f, num_b in enumerate(nb_list, 1):
        ind = _wraparound_indices(num_b)
        ind_shape = [1] * (num_f + 1)
        ind_shape[f - 1] = 2 * num_b + 1
        voltage += (ind * freq[f]).reshape(ind_shape)

    return voltage


def _wraparound_indices(num_b):
    """Summation indices in storage order: 0, 1, ..., num_b, -num_b, ..., -1.

    Args:
        num_b (int): Summation limit

    Returns:
        ndarray: Summation indices

    """

    return np.r_[np.arange(0, num_b + 1), np.arange(-num_b, 0)]


# Fourier representation of the response function ----------------------------

def _window_edge(x):
    """Smooth (infinitely differentiable) step from 0 at x <= 0 to 1 at x >= 1.

    Args:
        x (ndarray): Position

    Returns:
        ndarray: Step function

    """

    x = np.clip(x, 0., 1.)
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        e0 = np.where(x > 0., np.exp(-1. / np.maximum(x, 1e-300)), 0.)
        e1 = np.where(x < 1., np.exp(-1. / np.maximum(1. - x, 1e-300)), 0.)

    return e0 / (e0 + e1)


class FourierResp(object):
    """Fourier representation of the response function, for the Fourier-shift
    method (see module docstring).

    The response function is multiplied by a smooth window that is exactly
    one over every voltage that the direct method would interpolate (the bias
    voltages plus all combinations of the summation indices), tapered to zero
    over ``FOURIER_TAPER`` on either side, and then represented by a Fourier
    series over that period. Within the window, the only approximation is the
    truncation of the series to ``num_modes`` on each side of zero.

    Instances are returned by ``interpolate_respfn`` and are passed to
    ``qtcurrent`` as ``resp_matrix``. They depend on the response function,
    the bias voltages, the frequencies and ``num_b``, but not on the junction
    voltage, so they can be reused in the same way as the response function
    matrix.

    Args:
        resp (qmix.respfn.RespFn): Response function
        vb (ndarray): Bias voltages, normalized
        freq (ndarray): Frequencies, normalized (1-indexed)
        nb_list (tuple): Summation limits for phase factor coefficients, one
            for each tone
        num_modes (int, optional): Number of Fourier modes on each side of
            zero, default is ``NUM_MODES``

    Attributes:
        coeff (ndarray): Fourier coefficients c_n
        tau (ndarray): Angular "frequencies" (in 1/voltage) of the modes,
            tau_n = 2 pi n / period
        v0 (float): Start of the period
        period (float): Length of the period (in normalized voltage)
        vb (ndarray): Bias voltages
        shift (list): Per-tone shift matrices exp(j * tau_n * k * freq_f),
            with shape (num_coeff, 2 * num_b_f + 1), using wraparound indexing
            for k

    """

    def __init__(self, resp, vb, freq, nb_list, num_modes=None):

        if num_modes is None:
            num_modes = NUM_MODES

        vb = np.asarray(vb, dtype=float)
        nb_list = tuple(nb_list)

        # Window: covers vb + k * freq[1] + l * freq[2] + ... for all indices
        reach = sum(num_b * freq[f] for f, num_b in enumerate(nb_list, 1))
        v_lo = vb.min() - reach
        v_hi = vb.max() + reach
        self.v0 = v_lo - FOURIER_TAPER
        self.period = (v_hi - v_lo) + 2 * FOURIER_TAPER

        # Fourier coefficients of the windowed response function
        num_coeff = 2 * num_modes + 1
        voltage = self.v0 + self.period * np.arange(num_coeff) / num_coeff
        window = _window_edge((voltage - self.v0) / FOURIER_TAPER)
        window *= _window_edge((self.v0 + self.period - voltage) / FOURIER_TAPER)
        self.coeff = np.fft.fft(resp(voltage) * window) / num_coeff
        mode = np.rint(np.fft.fftfreq(num_coeff, 1. / num_coeff)).astype(int)
        self.tau = 2 * np.pi * mode / self.period

        # Per-tone shift matrices
        self.shift = []
        for f, num_b in enumerate(nb_list, 1):
            k = _wraparound_indices(num_b)
            self.shift.append(np.exp(1j * np.outer(self.tau, k * freq[f])))

        self.vb = vb
        self.freq = np.array(freq, dtype=float)
        self.nb_list = nb_list
        self.num_modes = num_modes

    def __str__(self):  # pragma: no cover

        return "Fourier representation of the response function ({0} modes)".format(self.num_modes)

    def __repr__(self):  # pragma: no cover

        return self.__str__()

    def bias_phase(self, modes=slice(None)):
        """Fourier coefficients times the phase factors of the bias voltages,
        i.e., c_n * exp(j * tau_n * (vb - v0)).

        Args:
            modes (slice, optional): Subset of the modes, default is all

        Returns:
            ndarray: With shape (number of modes, npts)

        """

        return self.coeff[modes, None] * np.exp(1j * np.outer(self.tau[modes], self.vb - self.v0))

    def resp(self, voltage):
        """Evaluate the Fourier series (i.e., the represented response
        function). Only accurate within the window.

        Args:
            voltage (ndarray): Voltage, normalized

        Returns:
            ndarray: Response function

        """

        voltage = np.asarray(voltage, dtype=float)
        flat = voltage.ravel()
        resp_out = np.empty(flat.size, dtype=complex)
        for i in range(0, flat.size, 256):
            phase = np.exp(1j * np.outer(flat[i:i + 256] - self.v0, self.tau))
            resp_out[i:i + 256] = phase @ self.coeff

        return resp_out.reshape(voltage.shape)


# Calculate the overall phase factor spectrum coefficients -------------------

def calculate_phase_factor_coeff(vj, freq, num_f, num_p, num_b):
    """Calculate the overall phase factor spectrum coefficients.

    Runs once per qtcurrent function call.

    Eqns. 5.7 and 5.12 in Kittara's thesis.

    Args:
        vj (ndarray): Voltage across the SIS junction
        freq (ndarray): Frequencies
        num_f (int): Number of non-harmonically related frequencies
        num_p (int): Number of harmonics
        num_b (int): Summation limits for phase factor coefficients

    Returns:
        ndarray: Phase factor spectrum coefficients (C_k(H) in Kittara)

    """

    # Number of bias voltage points
    npts = len(vj[0, 0, :])

    # Summation limits for phase factor coefficients
    if isinstance(num_b, int):
        num_b = tuple([num_b] * num_f)

    # Junction drive level:
    # alpha[f, p, i] in R^(num_f+1)(num_p+1)(npts)
    # Eqn. 5.5 in Kittara's thesis
    alpha = np.zeros_like(vj, dtype=float)
    for f in range(1, num_f + 1):
        for p in range(1, num_p + 1):
            alpha[f, p, :] = np.abs(vj[f, p, :]) / (p * freq[f])

    # Junction voltage phase:
    # phi[f, p, i] in R^(num_f+1)(num_p+1)(npts)
    phi = np.angle(vj)  # in radians

    # Complex coefficients from the Jacobi-Anger equality:
    # jac[f, p, n, i] in C^(num_f+1)(num_p+1)(num_b*2+1)(npts)
    # Equation 5.7 in Kittara's thesis
    # Note: This chunk of code dominates the computation time of this function
    # I tried using the recurrence relation, but ran into numerical errors
    jac = np.zeros((num_f + 1, num_p + 1, max(num_b) * 2 + 1, npts), dtype=complex)
    for f in range(1, num_f + 1):
        for p in range(1, num_p + 1):
            jac[f, p,  0] = j0(alpha[f, p])
            for n in range(1, num_b[f - 1] + 1):
                # using Bessel function identity
                if n == 1:
                    jn = j1(alpha[f, p])
                else:
                    jn = bessel(n, alpha[f, p])
                jac[f, p,  n] = jn * np.exp(-1j * n * phi[f, p])
                jac[f, p, -n] = (-1)**n * np.conj(jac[f, p,  n])

    # Overall phase factor coefficients:
    # ckh[f, k, i] in C^(num_f+1)(num_b*2+1)(npts)
    ckh = _convolve_coefficients(jac)

    return ckh


@nb.njit("c16[:,:,:](c16[:,:,:,:])", cache=True)
def _convolve_coefficients(jac):  # pragma: no cover
    """Convolve spectrum coefficients (recursively).

    See Withington and Kollberg, 1989.

    This function is only used if there are higher-order harmonics
    (num_p > 1).

    Calculation time is proportional to num_p.

    Eqn. 5.12 in Kittara's thesis.

    Args:
        jac (ndarray): Complex coefficients from the Jacobi-Anger equality
            (Eqn. 5.7 in Kittara's thesis)

    Returns:
        ndarray: Overall phase factor spectrum coefficients

    """

    _, num_p, num_b, _ = jac.shape
    num_p -= 1                # number of harmonics
    num_b = (num_b - 1) // 2  # number of bessel functions

    ckh_last = jac[:, 1, :, :]
    if num_p == 1:
        return ckh_last

    for p in range(2, num_p + 1):
        ckh_next = np.zeros_like(ckh_last)
        for k in range(-num_b, num_b + 1):
            # Don't exceed indices
            l_min = max(-num_b, int((k - num_b) / p))
            l_max = min( num_b, int((k + num_b) / p))
            for l in range(l_min, l_max + 1):
                ckh_next[1:, k] += ckh_last[1:, k - p * l] * jac[1:, p, l]
        ckh_last = ckh_next

    return ckh_last


def suggest_num_b(vj, cct, tol=1e-5, max_b=50):
    """Suggest summation limits for the phase factor coefficients (num_b).

    The computation time of ``qtcurrent`` (and therefore
    ``harmonic_balance``) scales with the product of (2 * num_b + 1) over all
    tones, but weakly driven tones only need a few terms. This function
    calculates the phase factor coefficients for each tone and returns the
    smallest summation limit beyond which all coefficients are below ``tol``
    (at every bias voltage).

    Note:

        This requires the actual junction voltage (e.g., the output of
        ``harmonic_balance``), not the Thevenin voltage. Some tones (e.g.,
        an IF tone) can have a large junction voltage even though they are
        not driven by the embedding circuit (i.e., ``cct.vt`` is zero),
        because they are generated by mixing. A typical workflow is to run
        ``harmonic_balance`` once with a conservative ``num_b``, call this
        function, and then use the result for subsequent simulations with
        similar drive levels (e.g., sweeps).

    Args:
        vj (ndarray): Voltage across the SIS junction
        cct (qmix.circuit.EmbeddingCircuit): Embedding circuit
        tol (float, optional): Neglect phase factor coefficients smaller than
            this value, default is 1e-5
        max_b (int, optional): Largest summation limit to consider, default
            is 50

    Returns:
        tuple: Suggested summation limits, one for each tone

    """

    ckh = calculate_phase_factor_coeff(vj, cct.freq, cct.num_f, cct.num_p, max_b)

    # Summation index (k >= 0), used with wraparound indexing for -k
    k = np.arange(max_b + 1)

    num_b = []
    for f in range(1, cct.num_f + 1):
        # Largest coefficient magnitude at +/-k (over all bias voltages)
        ckh_max = np.abs(ckh[f]).max(axis=-1)
        ckh_max = np.maximum(ckh_max[k], ckh_max[-k])
        k_above_tol = k[ckh_max >= tol]
        num_b_f = int(k_above_tol.max()) if len(k_above_tol) > 0 else 1
        if num_b_f == max_b:
            print("Warning: tone {0} may need num_b > max_b ({1})".format(f, max_b))
        num_b.append(max(1, num_b_f))

    return tuple(num_b)


@nb.njit("c16[:,:](c16[:,:], i4, i4)", cache=True)
def _pair_products(ccc_f, a, num_b):  # pragma: no cover
    """Pre-compute the phase factor pair products for one tone.

    Returns P[k, i] = ccc_f[k, i] * conj(ccc_f[k + a, i]), with zeros where
    k + a falls outside of [-num_b, num_b]. Uses the same wraparound indexing
    as ccc_f.

    Used in the current coefficient functions, where this product would
    otherwise be recomputed for every combination of the other summation
    indices.

    Args:
        ccc_f (ndarray): Phase factor coefficients for one tone
        a (int): Offset of the conjugated coefficient
        num_b (int): Summation limits for phase factor coefficients

    Returns:
        ndarray: Pair products

    """

    npts = ccc_f.shape[1]
    pair = np.zeros((2 * num_b + 1, npts), dtype=np.complex128)
    for k in range(-num_b, num_b + 1):
        if -num_b <= k + a <= num_b:
            for i in range(npts):
                pair[k, i] = ccc_f[k, i] * np.conj(ccc_f[k + a, i])

    return pair


# Tunneling current functions ------------------------------------------------
# These are the functions that actually calculate the tunneling current
# coefficients I(a, b, c, ...) for one harmonic combination (a, b, c, ...),
# Eqns. 5.25 and 5.36 in Kittara's thesis:
#
#   R + jS = sum_{k,l,...} P1[k] * P2[l] * ... * resp(vb + k*f1 + l*f2 + ...)
#
# where Pf[k] = Cf[k] * conj(Cf[k + a_f]) are the per-tone pair products. The
# weights factorize per tone but the response function does not, so the sum
# runs over every combination of the summation indices.

def _current_coeff(a_tuple, ccc, resp_matrix, nb_list):
    """Calculate the tunneling current coefficient (direct method).

    Calculate I(a, b, c, ...) for any number of tones.

    Equations 5.25 and 5.36 in Kittara's thesis.

    Args:
        a_tuple (tuple): Indices (a, b, c, ...) in Eqn. 5.25, one per tone
        ccc (ndarray): Convolution coefficients
        resp_matrix (ndarray): Response function matrix, generated by
            interpolate_respfn
        nb_list (tuple): Summation limits for phase factor coefficients, one
            per tone

    Returns:
        ndarray: Tunneling current coefficient

    """

    npts = ccc.shape[2]
    pair_p, pair_m, valid_p, valid_m = _pair_products_all(a_tuple, ccc, nb_list)
    dims = np.array([2 * num_b + 1 for num_b in nb_list], dtype=np.int64)
    resp_flat = np.ascontiguousarray(resp_matrix).reshape(-1, npts)
    num_outer = int(np.prod(dims[:-1]))
    num_chunks = max(1, min(num_outer, 64 * nb.get_num_threads()))

    rs = _current_coeff_n_tones(pair_p, pair_m, valid_p, valid_m, dims, resp_flat, num_chunks)

    return _combine_rs(a_tuple, rs[0], rs[1])


def _pair_products_all(a_tuple, ccc, nb_list):
    """Pair products for all tones, for the +a and -a terms of Eqn. 5.36.

    The arrays are padded to the largest num_b; each tone's products are
    stored in positions 0 to 2 * num_b_f (wraparound storage order). The
    validity masks mark where the conjugated coefficient exists (i.e., k + a
    is within [-num_b, num_b]), so that the kernel can skip empty terms.

    Args:
        a_tuple (tuple): Indices (a, b, c, ...), one per tone
        ccc (ndarray): Convolution coefficients
        nb_list (tuple): Summation limits, one per tone

    Returns:
        tuple: pair_p, pair_m (num_f, 2 * max_b + 1, npts) and valid_p,
        valid_m (num_f, 2 * max_b + 1)

    """

    num_f = len(nb_list)
    max_b = max(nb_list)
    npts = ccc.shape[2]

    pair_p = np.zeros((num_f, 2 * max_b + 1, npts), dtype=np.complex128)
    pair_m = np.zeros_like(pair_p)
    valid_p = np.zeros((num_f, 2 * max_b + 1), dtype=np.bool_)
    valid_m = np.zeros_like(valid_p)
    for f, (a, num_b) in enumerate(zip(a_tuple, nb_list)):
        dim = 2 * num_b + 1
        pair_p[f, :dim] = _pair_products(ccc[f + 1], a, num_b)
        pair_m[f, :dim] = _pair_products(ccc[f + 1], -a, num_b)
        k = _wraparound_indices(num_b)
        valid_p[f, :dim] = np.abs(k + a) <= num_b
        valid_m[f, :dim] = np.abs(k - a) <= num_b

    return pair_p, pair_m, valid_p, valid_m


def _combine_rs(a_tuple, rs_p, rs_m):
    """Combine the +a and -a sums into the current coefficient.

    Equation 5.26/5.36 in Kittara's thesis: I = (S_a + S_-a) - j (R_a - R_-a)

    Args:
        a_tuple (tuple): Indices (a, b, c, ...), one per tone
        rs_p (ndarray): R + jS for +a
        rs_m (ndarray): R + jS for -a

    Returns:
        ndarray: Tunneling current coefficient

    """

    if all(a == 0 for a in a_tuple):
        return rs_p.imag + 1j * 0.
    else:
        return (rs_p.imag + rs_m.imag) - 1j * (rs_p.real - rs_m.real)


@nb.njit("c16[:,:](c16[:,:,:], c16[:,:,:], b1[:,:], b1[:,:], i8[:], c16[:,:], i8)", parallel=True, cache=True)
def _current_coeff_n_tones(pair_p, pair_m, valid_p, valid_m, dims, resp_flat, num_chunks):  # pragma: no cover
    """Calculate R + jS (Eqn. 5.25) for any number of tones.

    The nested sum over the per-tone summation indices is one flattened loop
    over the rows of the response function matrix, reshaped to 2-D (all
    tones' indices in C order, bias points last). The outer tones' indices
    are decoded from the row number (mixed radix), the running products of
    their pair products are updated only from the first index that changed
    since the previous row (like an odometer), and the innermost tone and the
    bias points are looped over contiguous memory.

    Parallelized over chunks of rows: each chunk accumulates into its own
    row, which is summed at the end (avoids race conditions).

    Args:
        pair_p (ndarray): Pair products for +a, see _pair_products_all
        pair_m (ndarray): Pair products for -a
        valid_p (ndarray): Validity masks for +a
        valid_m (ndarray): Validity masks for -a
        dims (ndarray): 2 * num_b + 1 for each tone
        resp_flat (ndarray): Response function matrix reshaped to 2-D, with
            shape (product of dims, npts)
        num_chunks (int): Number of parallel chunks

    Returns:
        ndarray: R + jS for +a (row 0) and -a (row 1)

    """

    num_f = dims.shape[0]
    npts = resp_flat.shape[1]
    d_last = dims[num_f - 1]

    # Rows of the outer tones (all but the last)
    num_outer = 1
    for f in range(num_f - 1):
        num_outer *= dims[f]
    rows_per_chunk = (num_outer + num_chunks - 1) // num_chunks

    acc = np.zeros((2, num_chunks, npts), dtype=np.complex128)

    # Index of the last outer tone (-1 for one tone, where the running
    # product is just 1)
    last = num_f - 2
    num_prefix = max(num_f - 1, 1)

    for c in nb.prange(num_chunks):

        # Running products of the outer tones' pair products:
        # prefix[f] = pair[0, s_0] * ... * pair[f, s_f]
        prefix_p = np.ones((num_prefix, npts), dtype=np.complex128)
        prefix_m = np.ones((num_prefix, npts), dtype=np.complex128)
        s_arr = np.empty(num_f, dtype=np.int64)
        s_prev = np.full(num_f, -1, dtype=np.int64)

        row_lo = c * rows_per_chunk
        row_hi = min(num_outer, row_lo + rows_per_chunk)
        for row in range(row_lo, row_hi):

            # Decode row -> storage indices of the outer tones, and check
            # whether any term exists
            r = row
            ok_p = True
            ok_m = True
            for f in range(last, -1, -1):
                s = r % dims[f]
                r = r // dims[f]
                s_arr[f] = s
                ok_p = ok_p and valid_p[f, s]
                ok_m = ok_m and valid_m[f, s]
            if not (ok_p or ok_m):
                s_prev[0] = -1  # force a full update at the next valid row
                continue

            # Update the running products from the first changed index
            start = 0
            while start <= last and s_arr[start] == s_prev[start]:
                start += 1
            for f in range(start, last + 1):
                s = s_arr[f]
                s_prev[f] = s
                if f == 0:
                    for i in range(npts):
                        prefix_p[0, i] = pair_p[0, s, i]
                        prefix_m[0, i] = pair_m[0, s, i]
                else:
                    for i in range(npts):
                        prefix_p[f, i] = prefix_p[f - 1, i] * pair_p[f, s, i]
                        prefix_m[f, i] = prefix_m[f - 1, i] * pair_m[f, s, i]
            t_p = prefix_p[max(last, 0)]
            t_m = prefix_m[max(last, 0)]

            # Innermost tone and bias points (contiguous in resp_flat)
            base = row * d_last
            for s in range(d_last):
                if ok_p and valid_p[num_f - 1, s]:
                    for i in range(npts):
                        acc[0, c, i] += t_p[i] * pair_p[num_f - 1, s, i] * resp_flat[base + s, i]
                if ok_m and valid_m[num_f - 1, s]:
                    for i in range(npts):
                        acc[1, c, i] += t_m[i] * pair_m[num_f - 1, s, i] * resp_flat[base + s, i]

    return acc.sum(axis=1)


def _current_fourier(fresp, ccc, tuple_lists, nb_list):
    """Calculate the tunneling currents with the Fourier-shift method.

    With the response function as a Fourier series, resp(v) = sum_n c_n
    exp(j tau_n v), the shift of its argument in Eqn. 5.25 factorizes:

        R + jS = sum_n c_n exp(j tau_n vb) * prod_f M_f(a_f)[n]
        M_f(a)[n] = sum_k Pf[k] exp(j tau_n k freq_f)

    so each tone is summed independently (a matrix product per tone), and
    the F-dimensional sum over the response function matrix is avoided.

    The modes are processed in blocks to limit memory use.

    Args:
        fresp (FourierResp): Fourier representation of the response function
        ccc (ndarray): Convolution coefficients
        tuple_lists (list): For each output frequency, the list of matching
            index tuples (a, b, c, ...)
        nb_list (tuple): Summation limits, one per tone

    Returns:
        ndarray: Tunneling currents, with shape (number of output
        frequencies, npts)

    """

    npts = ccc.shape[2]
    num_coeff = fresp.coeff.size

    # All distinct index tuples, and the per-tone pair products they need
    all_tuples = sorted(set(itertools.chain.from_iterable(tuple_lists)), reverse=True)
    pairs = {}
    for a_tuple in all_tuples:
        for f, a in enumerate(a_tuple):
            for sign in (1, -1):
                if (f, sign * a) not in pairs:
                    pairs[(f, sign * a)] = _pair_products(ccc[f + 1], sign * a, nb_list[f])

    # R + jS for +a and -a, accumulated over blocks of modes
    rs_p = {a_tuple: np.zeros(npts, dtype=complex) for a_tuple in all_tuples}
    rs_m = {a_tuple: np.zeros(npts, dtype=complex) for a_tuple in all_tuples}
    for start in range(0, num_coeff, FOURIER_BLOCK):
        modes = slice(start, min(num_coeff, start + FOURIER_BLOCK))
        bias = fresp.bias_phase(modes)
        factor = {key: fresp.shift[key[0]][modes] @ pair for key, pair in pairs.items()}
        for a_tuple in all_tuples:
            prod_p = bias.copy()
            for f, a in enumerate(a_tuple):
                prod_p *= factor[(f, a)]
            rs_p[a_tuple] += prod_p.sum(axis=0)
            if any(a != 0 for a in a_tuple):
                prod_m = bias.copy()
                for f, a in enumerate(a_tuple):
                    prod_m *= factor[(f, -a)]
                rs_m[a_tuple] += prod_m.sum(axis=0)

    current_out = np.zeros((len(tuple_lists), npts), dtype=complex)
    for i, tuple_list in enumerate(tuple_lists):
        for a_tuple in tuple_list:
            current_out[i] += _combine_rs(a_tuple, rs_p[a_tuple], rs_m[a_tuple])

    return current_out


# Helper functions -----------------------------------------------------------

def _unpack_num_b(num_b, num_f):
    """Unpack num_b (summation limits for phase factor coefficients).

    Args:
        num_b: Summation limits for phase factor coefficients
        num_f: Number of frequencies

    Returns:
        tuple: Summation limits for phase factor coefficients in tuple form,
        with one value for each tone

    """

    # Note: num_b is 0-indexed if it is a tuple
    # I.e.: num_b[0] is for the fundamental frequency
    if isinstance(num_b, tuple):
        assert len(num_b) >= num_f, \
            "There must be one value of num_b for each fundamental frequency."
        num_b = tuple(num_b[:num_f])
        return num_b

    return tuple([num_b] * num_f)
