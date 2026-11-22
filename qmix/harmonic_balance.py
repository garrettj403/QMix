""" This module contains functions to perform harmonic balance of non-linear
SIS mixer circuits.

**Description**

    Each signal that is applied to an SIS junction can be represented by a
    Thevenin equivalent circuit (see qmix.circuit). This circuit will then
    induce a voltage across the SIS junction. The exact voltage that is
    induced depends on the impedance of the SIS junction. However, this
    impedance changes depending on the other signals that are applied to
    the junction.

    Harmonic balance is a procedure to solve for the voltage across the SIS
    junction for each signal that is applied to the junction. This techniques
    uses Newton's method to find the solution numerically.

    Several junctions in one embedding network
    (``qmix.circuit.MultiJunctionCircuit``) are solved together: the
    unknowns are the voltages of all junctions, and the Jacobian couples
    them through the off-diagonal Thevenin impedances. Since the current of
    each junction only depends on its own voltage, the derivatives of all
    junctions' currents are found from the same perturbed voltages, so a new
    Jacobian costs 2 * num_n current calculations per junction (as for one
    junction).

"""

# import qmix
import copy

import numpy as np
from timeit import default_timer as timer
from qmix.circuit import EmbeddingCircuit, MultiJunctionCircuit
from qmix.misc.progbar import progress_bar
from qmix.qtcurrent import (FourierResp, interpolate_respfn, junction_resp, qtcurrent, qtcurrent_derivatives,
                            _check_respfn_range, _interpolation_voltages, _is_multi, _select_method, _unpack_num_b)

# default maximum number of iterations: most cases converge in ~5, but strong
# pumping into a large embedding impedance can take ~20, and series arrays (DC
# bias network) typically 15-30
MAX_IT = 30
MAX_IT_DC = 60
# minimum Thevenin voltage (required to avoid div by 0 errors)
MIN_VT = 1e-10
# voltage step for numerical derivatives
DV = 1e-3
# round frequency values to this number of decimal places when comparing
ROUND_FREQ = 4
# reuse the Jacobian from the previous iteration as long as each iteration
# reduces the maximum relative error by at least this factor
JACOBIAN_REUSE_RATIO = 0.5
# line search: a step must reduce the squared error norm |e|^2 at a bias point
# to at most (1 - 2 * LINE_SEARCH_ALPHA * step) times its previous value
# (sufficient decrease condition); otherwise the step is shortened, up to
# LINE_SEARCH_MAX_BACKTRACK times
LINE_SEARCH_ALPHA = 1e-4
LINE_SEARCH_MAX_BACKTRACK = 8
# if the line search fails even with a new Jacobian (i.e., the iteration is
# stuck near a local minimum of the error, not a root), restart that bias
# voltage from the solution at the nearest converged bias voltage, up to this
# many times per bias voltage
MAX_RESTARTS = 3
# DC bias network: the error of the DC equations is checked relative to the
# largest DC voltage at each bias point, but at least this voltage
DC_VREF = 0.01
# DC bias network: a bias point that hasn't reduced its squared error norm by
# STALL_RATIO within STALL_ITERATIONS iterations is restarted from the nearest
# converged bias point (as if the line search had failed)
STALL_RATIO = 0.5
STALL_ITERATIONS = 5
# DC bias network: step for the central differences with respect to the DC
# voltages
DV_DC = 1e-5


# Harmonic Balance ----------------------------------------------------------

# This is the main harmonic balance function. It uses the Newton-Raphson
# method to find the solution to the highly non-linear equation.


def harmonic_balance(cct, resp, num_b=15, max_it=None, stop_rerror=0.001, vj_initial=None, damp_coeff=1., mode="o",
                     verbose=True, zj_guess=0.67, reuse_jacobian=True, method='auto', line_search=True,
                     warm_start=False, jacobian='analytic', vb_initial=None):
    """Perform harmonic balance.

    Determine the harmonic balance of the junction + embedding circuit system.
    Uses Newton's method to find the solution. For more information, see
    Garrett (2018); Kittara (2002); Kittara, Winthington & Yassin (2007); or
    Withington, Kittara & Yassin (2003). [Full references in online docs.]

    For several junctions, pass a ``qmix.circuit.MultiJunctionCircuit``.
    The junction voltage then has the junction axis first,
    ``vj[j, f, p, i]``, and ``resp`` can be a list (one response function
    per junction).

    With a DC bias network (``qmix.circuit.MultiJunctionCircuit.set_dc_network``
    or ``set_series_bias``, e.g., for a series array), the DC voltages of
    the junctions are unknowns too: they are solved together with the
    junction voltage and returned with it (``vj, vb``; ``cct.vb`` is only
    the initial guess).

    Args:
        cct (qmix.circuit.EmbeddingCircuit): Embedding circuit (or
            ``qmix.circuit.MultiJunctionCircuit``)
        resp (qmix.respfn.RespFn): Response function (or a list, one per
            junction)
        num_b (int/tuple, optional): Summation limits for phase factor
            coefficients, default is 15
        method (str, optional): Method used by ``qmix.qtcurrent.qtcurrent``
            to sum over the phase factor coefficients: ``'direct'``,
            ``'fourier'`` or ``'auto'``, default is ``'auto'`` (see
            ``qmix.qtcurrent`` for details)
        max_it (int, optional): Maximum number of iterations, default is 30
            (``MAX_IT``), or 60 with a DC bias network (``MAX_IT_DC``).
            Most cases converge in ~5 iterations (and stop there), but strong
            pumping into a large embedding impedance can take ~20.
        stop_rerror (float, optional): Maximum acceptable relative error,
            default is 0.001
        vj_initial (ndarray, optional): Initial guess of junction voltage
            (vj), default is None
        damp_coeff (float, optional): Dampening coefficient for correction
            factor (0-1), default is 1. With ``line_search=True``, this is
            the initial step length of the line search.
        mode (string, optional): output vj ('o'), print ('p'), output extra
            data ('x'), default is "o"
        verbose (bool, optional): print info to terminal if true, default is
            True
        zj_guess (float, optional): guess of the junction impedance, used
            to generate the initial guess of vj, default is 0.67
        reuse_jacobian (bool, optional): reuse the Jacobian from the
            previous iteration while the error is still decreasing quickly
            (each new Jacobian requires 2 * num_n calls to the tunneling
            current function), default is True
        line_search (bool, optional): backtracking line search, default is
            True. At each bias voltage, a Newton step is only accepted if it
            reduces the error enough; otherwise it is shortened. This
            prevents divergence when the junction loads the embedding circuit
            strongly (e.g., large embedding impedance near the gap). Each
            shortened step costs one call to the tunneling current function,
            but full steps cost nothing extra.
        warm_start (bool, optional): for multi-tone simulations, first
            solve the harmonic balance for the strongest tone (largest
            Thevenin voltage) on its own, and use the result as the initial
            guess for that tone, default is False. This is cheap (single-tone
            tunneling currents are much faster to calculate) and typically
            cuts the number of multi-tone iterations by 2-3x when one tone
            (e.g., the LO) is much stronger than the others, but it is not
            always faster. Ignored if ``vj_initial`` is given or there is
            only one tone.
        jacobian (str, optional): how the Jacobian is calculated:
            ``'analytic'`` (default) or ``'numerical'``. The analytic
            Jacobian (``qmix.qtcurrent.qtcurrent_derivatives``) is exact and
            evaluates all of the derivatives together, which is usually
            faster than ``'numerical'`` (finite differences, 2 * num_n calls
            to the tunneling current function, with a forward step of
            ``DV``). With a DC bias network, the derivatives with respect to
            the DC voltages are always forward differences.
        vb_initial (ndarray, optional): With a DC bias network: initial
            guess of the DC voltages, ``vb[j, i]``, default is ``cct.vb``

    Returns:
        ndarray: junction voltage that satisfies the circuit (mode 'o');
        also the number of iterations and whether the solution converged
        (mode 'x'), or a mask of the converged bias voltages (mode 'm').
        With a DC bias network, the DC voltages ``vb[j, i]`` follow the
        junction voltage (e.g., ``vj, vb`` for mode 'o', or ``vj, vb,
        iterations, converged`` for mode 'x').

    """

    if verbose:
        print("Running harmonic balance:")

    start_time = timer()

    assert jacobian in ('analytic', 'numerical'), "jacobian must be 'analytic' or 'numerical'"
    analytic = jacobian == 'analytic'

    # Check input data ------------------------------------------------------

    num_f = cct.num_f
    num_p = cct.num_p
    num_n = cct.num_n
    npts = cct.vb_npts

    # Several junctions: num_j * num_n unknowns per bias voltage
    multi = _is_multi(cct)
    num_j = cct.num_j if multi else 1
    num_k = num_j * num_n

    # DC bias network: the DC voltages (one per junction) are unknowns too
    dc = multi and cct.has_dc_network
    if dc:
        assert cct.dc_rhs.shape == (num_j, npts), "dc_rhs must have shape (num_j, vb_npts)"
    num_u = num_k + num_j if dc else num_k
    if max_it is None:
        max_it = MAX_IT_DC if dc else MAX_IT

    vt = cct.vt
    zt = cct.zt

    # Harmonic balance isn't required if all zt are equal to zero
    if (zt == 0).all() and not dc:
        if verbose:
            print("Done.\n")
        return vt[..., None] * np.ones(npts, dtype=complex)

    # Check whether any Thevenin voltages are zero (prevents div0 errors) ----

    vt[np.abs(vt) < MIN_VT] = MIN_VT
    vt[..., 0, :] = 0  # TODO: is this needed?
    vt[..., :, 0] = 0

    # Initial guess of the junction voltage (vj) -----------------------------

    if vj_initial is None:
        if multi:
            vj_initial = _initial_guess_multi(vt, zt, zj_guess, npts)
        else:
            vj_initial = vt[:, :, None] * \
                         zj_guess / (zj_guess + zt[:, :, None]) * \
                         np.ones(npts, dtype=complex)

        # Warm start: solve the strongest tone on its own
        if warm_start and num_f > 1:
            vt_fp = np.abs(cct.vt[:, 1:, 1:]).max(axis=0) if multi else np.abs(cct.vt[1:, 1:])
            f_strong = 1 + int(np.argmax(vt_fp.max(axis=1)))
            if verbose:
                print(" - warm start: solving tone {} on its own".format(f_strong))
            vj_f = _single_tone_guess(cct, resp, f_strong, _unpack_num_b(num_b, num_f)[f_strong - 1], max_it,
                                      stop_rerror, damp_coeff, zj_guess, reuse_jacobian, method, line_search,
                                      jacobian)
            if dc:
                vj_f, vb_f = vj_f
                if vb_initial is None:
                    vb_initial = vb_f
            if multi:
                vj_initial[:, f_strong] = vj_f
            else:
                vj_initial[f_strong] = vj_f
    elif multi:
        assert vj_initial.shape == (num_j, num_f + 1, num_p + 1, npts), "vj_initial must have shape (j, f, p, i)"

    # Prepare input data ----------------------------------------------------

    # Note on matrix shapes: Most of the circuit parameters are stored in a
    # [f, p, vb] format. I.e., the first index is for the tone, the second is
    # for the harmonic and the third is for the bias voltage. In the harmonic
    # balance function, I change these to [k, vb] to make the matrix
    # manipulation easier to understand. Therefore, a 3 tone, 2 harmonic
    # variable (with shape (3+1) x (2+1) x npts, where npts = length of bias
    # voltage) will be transformed into a matrix with shape 6 x npts.

    # For several junctions, the junction is the outer index:
    # k = j * num_n + (index of [f, p])

    if verbose:
        if multi:
            print(" - {0} junctions".format(num_j))
        print((" - {0} tone(s) and {1} harmonic(s)".format(num_f, num_p)))
        if analytic:
            print(" - analytic Jacobian")
        else:
            msg = " - {0} calls to the quasiparticle tunneling current (qtc) function per iteration"
            if multi:
                msg += " (for each junction)"
            print((msg.format(num_n * 2 + 1)))
        if dc:
            print(" - DC bias network (DC voltages solved)")
        print(" - max. iterations: {}".format(max_it))

    # Interpolate response function for all required voltages (a list, one
    # per junction, for several junctions)
    # (with a DC bias network, this is done for every error vector instead)
    if not dc:
        respfn_interp = interpolate_respfn(cct, resp, num_b, method=method, verbose=verbose)

    # Change format of data from [f, p] to [k] (reduce by one dimension).
    # The error vector and the inverse Jacobian are calculated by these
    # functions of vj_2d.
    if dc:
        vb_initial = cct.vb if vb_initial is None else np.asarray(vb_initial, dtype=float)
        assert vb_initial.shape == (num_j, npts), "vb_initial must have shape (num_j, vb_npts)"
        vj_2d = np.vstack((vj_initial[:, 1:, 1:, :].reshape((num_k, npts)), vb_initial.astype(complex)))
        system = _DCBiasSystem(cct, resp, num_b, method, verbose=verbose)
        error_vector = system.error_vector
        inv_jacobian = system.inv_jacobian_analytic if analytic else system.inv_jacobian
    elif multi:
        vj_2d = vj_initial[:, 1:, 1:, :].reshape((num_k, npts))
        system = _MultiJunctionSystem(cct, resp, num_b, respfn_interp)
        error_vector = system.error_vector
        inv_jacobian = system.inv_jacobian_analytic if analytic else system.inv_jacobian
    else:
        vj_2d = vj_initial[1:, 1:, :].reshape((num_n, npts))
        vt_2d = vt[1:, 1:].reshape(num_n)
        zt_2d = zt[1:, 1:].reshape(num_n)

        def error_vector(vj_2d):
            return _error_vector(vj_2d, vt_2d, zt_2d, cct, resp, num_b, respfn_interp)

        def inv_jacobian(err_all, vj_2d, verbose):
            if analytic:
                return _inv_jacobian_analytic(vj_2d, zt_2d, cct, resp, num_b, respfn_interp)
            return _inv_jacobian(err_all, vj_2d, vt_2d, zt_2d, cct, resp, num_b, resp_matrix=respfn_interp,
                                 verbose=verbose)

    # Perform harmonic balance -----------------------------------------------

    # Error vector at the initial guess
    time_current = timer()
    err_all = error_vector(vj_2d)
    if verbose:
        print("Estimated time:")
        call_time = timer() - time_current
        msg = ' - time per qtc call:  {:7.2f} s / {:6.2f} min / {:5.2f} hrs'
        print((msg.format(call_time, call_time / 60., call_time / 3600.)))
        it_time = call_time * (num_n * 2 + 1)  # finite differences (an upper bound for the analytic Jacobian)
        msg = ' - time per iteration: {:7.2f} s / {:6.2f} min / {:5.2f} hrs'
        print((msg.format(it_time, it_time / 60., it_time / 3600.)))
        max_time = it_time * max_it
        msg = ' - max sim time:       {:7.2f} s / {:6.2f} min / {:5.2f} hrs'
        print((msg.format(max_time, max_time / 60., max_time / 3600.)))

    iteration = 0
    inv_j = None
    last_max_rel_error = np.inf
    force_new_jacobian = False
    num_restarts = np.zeros(npts, dtype=int)
    # DC bias network: last iteration at which each bias point made progress
    # (see STALL_ITERATIONS)
    best_merit = np.full(npts, np.inf)
    it_best = np.zeros(npts, dtype=int)
    for iteration in range(max_it + 1):

        # Check error at each tone and harmonic
        if verbose:
            print(("Error after {0} iteration(s):".format(iteration)))
        max_rel_error = 0.  # initialize maximum relative error of all signals
        msg = "\tf:{:d}, p:{:d},   med. rel. error: {:9.3f},   max. rel. error: {:9.3f},   {:5.1f} % complete"
        if multi:
            msg = "\tj:{:d}, " + msg[1:]
        finished_points = np.ones(npts, dtype=bool)
        if dc:
            # DC equations: relative to the largest DC voltage (at least DC_VREF)
            v_ref_dc = np.maximum(np.abs(vj_2d[num_k:].real).max(axis=0), DC_VREF)
        for k in range(num_u):
            with np.errstate(divide='ignore'):
                # Check relative error at k
                abs_error = np.abs(err_all[k, :])
                v_ref = v_ref_dc if k >= num_k else np.abs(vj_2d[k, :])
                rel_error = abs_error / v_ref
                max_rel_error = max(np.max(rel_error), max_rel_error)
                med_rel_error = np.median(rel_error)
                good_points = rel_error < stop_rerror
                finished_points = finished_points & good_points
                complete = float(np.sum(good_points)) / npts

            # Print to terminal
            if verbose and k >= num_k:
                msg_dc = ("\tj:{:d}, DC,        med. rel. error: {:9.3f},   max. rel. error: {:9.3f},   "
                          "{:5.1f} % complete")
                print(msg_dc.format(k - num_k, min(med_rel_error, 99999.999), min(np.max(rel_error), 99999.999),
                                    complete * 100))
            elif verbose:
                f, p = _k_to_fp(k % num_n, num_p)

                if med_rel_error > 99999.999:
                    _med_rel_error = 99999.999
                else:
                    _med_rel_error = med_rel_error

                if np.max(rel_error) > 99999.999:
                    _rel_error = 99999.999
                else:
                    _rel_error = np.max(rel_error)

                values = (int(f), int(p), _med_rel_error, _rel_error, complete * 100)
                print((msg.format(*((k // num_n,) + values if multi else values))))

        # Exit if the error is good enough
        if max_rel_error <= stop_rerror:
            if verbose:
                print("Done: Minimum error target was achieved.")
            break

        # Exit if this is the last iteration
        if max_rel_error > stop_rerror and iteration == max_it:
            print("*** DID NOT ACHIEVE TARGET ERROR VALUE ***\n")
            break

        # Update junction voltages (i.e., the business end of this function)
        # Only recalculate the Jacobian if the last one has stopped working well
        slow_convergence = max_rel_error > JACOBIAN_REUSE_RATIO * last_max_rel_error
        new_jacobian = inv_j is None or not reuse_jacobian or slow_convergence or force_new_jacobian
        if new_jacobian:
            inv_j = inv_jacobian(err_all, vj_2d, verbose)
        elif verbose:
            print("Reusing Jacobian from previous iteration")
        last_max_rel_error = max_rel_error
        if verbose:
            print("Applying correction")
        corr = np.zeros((num_u, npts), dtype=complex)
        for p in range(num_u):
            for q in range(num_u):
                corr[p, :] -= (inv_j[p * 2, q * 2] * np.real(err_all[q]) +
                               inv_j[p * 2, q * 2 + 1] * np.imag(err_all[q]))
                corr[p, :] -= 1j * (inv_j[p * 2 + 1, q * 2] * np.real(err_all[q]) +
                                    inv_j[p * 2 + 1, q * 2 + 1] * np.imag(err_all[q]))

        if line_search:
            # Points that have already converged take the full step
            vj_2d, err_all, num_backtrack, stuck = _line_search(vj_2d, corr, err_all, ~finished_points, damp_coeff,
                                                                new_jacobian, error_vector)
            # A step that had to be shortened (or failed) with an old Jacobian
            # suggests that the Jacobian is out of date
            force_new_jacobian = not new_jacobian and (num_backtrack > 0 or stuck.any())
            if verbose and (num_backtrack > 0 or stuck.any()):
                msg = "Line search: {} step reduction(s); no acceptable step at {} bias point(s)"
                print(msg.format(num_backtrack, np.sum(stuck)))

            # Stuck despite a new Jacobian: restart from a converged neighbor
            restart = stuck & new_jacobian
            if dc:
                # Also restart bias points that have stopped making progress
                # (a non-root local minimum of the error, which is common with
                # series arrays: the line search keeps accepting short steps)
                merit = np.sum(np.abs(err_all) ** 2, axis=0)
                progress = merit < STALL_RATIO * best_merit
                best_merit[progress] = merit[progress]
                it_best[progress] = iteration
                restart |= ~finished_points & (iteration - it_best >= STALL_ITERATIONS)
            restart &= num_restarts < MAX_RESTARTS
            if restart.any() and finished_points.any():
                idx_done = np.flatnonzero(finished_points)
                for i in np.flatnonzero(restart):
                    vj_2d[:, i] = vj_2d[:, idx_done[np.abs(idx_done - i).argmin()]]
                num_restarts[restart] += 1
                best_merit[restart], it_best[restart] = np.inf, iteration
                err_all = error_vector(vj_2d)
                # The Jacobian at the restarted points is out of date
                force_new_jacobian = True
                if verbose:
                    print("Restarted {} bias point(s) from a converged neighbor".format(np.sum(restart)))
        else:
            vj_2d = vj_2d + corr * damp_coeff
            err_all = error_vector(vj_2d)

    # Format results (from [k,vb] to [f,p,vb], or [j,f,p,vb])
    if multi:
        vj_out = np.zeros((num_j, num_f + 1, num_p + 1, npts), dtype=complex)
        vj_out[:, 1:, 1:, :] = vj_2d[:num_k].reshape((num_j, num_f, num_p, npts))
    else:
        vj_out = np.zeros((num_f + 1, num_p + 1, npts), dtype=complex)
        vj_out[1:, 1:, :] = vj_2d.reshape((num_f, num_p, npts))

    time = timer() - start_time
    if verbose:
        print((" - sim time:\t\t{:7.2f} s / {:6.2f} min / {:5.2f} hrs".format(time, time / 60., time / 3600.)))
        if iteration >= 1:
            tit = time / iteration  # time per iteration
            msg = " - {} iterations required"
            print((msg.format(iteration)))
            print((" - time per iteration:\t{:7.2f} s / {:6.2f} min / {:5.2f} hrs".format(tit, tit / 60., tit / 3600.)))

    # With a DC bias network, the DC voltages follow vj
    out = (vj_out, vj_2d[num_k:].real.copy()) if dc else (vj_out,)

    if mode == "o":  # return vj
        return out if dc else vj_out
    elif mode == "x":  # return vj, number of iterations, and if converged
        converged = max_rel_error <= stop_rerror
        return out + (iteration, converged)
    elif mode == "m":  # return vj and mask
        return out + (finished_points,)


# Check results -------------------------------------------------------------

# Double check the error from the harmonic balance function. This is mainly
# for debugging purposes to ensure that everything was done properly.


def check_hb_error(vj_check, cct, resp, num_b=15, stop_rerror=0.001, vb=None):
    """Check the results from the `harmonic_balance` function.

    Just to double check. Mostly for debugging purposes.

    Args:
        vj_check (ndarray): The voltage across junction to check
        cct (qmix.circuit.EmbeddingCircuit): Embedding circuit (or
            ``qmix.circuit.MultiJunctionCircuit``)
        resp (qmix.respfn.RespFn): Response function (or a list, one per
            junction)
        num_b (optional): Summation limits for phase factor coefficients,
            default is 15
        stop_rerror (float, optional): Maximum acceptable relative error,
            default is 0.001
        vb (ndarray, optional): DC voltages from ``harmonic_balance``, for a
            circuit with a DC bias network (the DC equations are checked
            too), default is ``cct.vb``

    Raises:
        AssertionError: If the stop error is not met

    """

    # Note: speed is not important here...

    print("Double-checking harmonic balance error:")

    if vb is not None:
        cct = copy.copy(cct)
        cct.vb = vb

    # Tunnelling current for all tones/harmonics (see below)
    ij_all = _qtcurrent_all_freq(vj_check, cct, resp, num_b)

    # Several junctions: vt - sum_l zt[j, l] ij[l] - vj for each junction
    multi = _is_multi(cct)
    if multi:
        error_all = cct.vt[..., None] - np.einsum('jlfp,lfpi->jfpi', cct.zt, ij_all) - vj_check
    else:
        error_all = cct.vt[:, :, None] - cct.zt[:, :, None] * ij_all - vj_check

    for j in range(cct.num_j if multi else 1):
        for f in range(1, cct.num_f + 1):
            for p in range(1, cct.num_p + 1):
                if multi:
                    error_fp, vj_fp = error_all[j, f, p], vj_check[j, f, p]
                else:
                    error_fp, vj_fp = error_all[f, p], vj_check[f, p]
                max_rel_error = np.max(np.abs(error_fp) / np.abs(vj_fp))
                good_error = max_rel_error < stop_rerror
                if good_error:
                    err_str = 'Yes'
                else:
                    err_str = 'No'
                msg = "\tf:{0},\tp:{1},\tmax rel. error: {2:.2E},\tPass? {3}"
                if multi:
                    msg = "\tj:{4}," + msg
                print((msg.format(f, p, max_rel_error, err_str, j)))
                assert good_error

    # DC bias network
    if multi and cct.has_dc_network:
        i0 = qtcurrent(vj_check, cct, resp, [0.], num_b, verbose=False)[:, 0].real
        error_dc = cct.dc_rhs - cct.dc_a @ cct.vb - cct.dc_b @ i0
        max_rel_error = np.max(np.abs(error_dc) / np.maximum(np.abs(cct.vb).max(axis=0), DC_VREF))
        good_error = max_rel_error < stop_rerror
        print("\tDC network,\tmax rel. error: {0:.2E},\tPass? {1}".format(max_rel_error, 'Yes' if good_error else 'No'))
        assert good_error
    print("")


def _qtcurrent_all_freq(vj, cct, resp, num_b=15):
    """Calculate the AC tunneling current for all tones and all harmonics.

    This function will return the tunneling current in a 3-D array:
    (num_f+1) x (num_p+1) x (npts).

    This is used in the harmonic balance procedure.

    Args:
        vj (ndarray): Voltage across the junction
        cct (qmix.circuit.EmbeddingCircuit): Embedding circuit class
        resp (qmix.respfn.RespFn): Response function
        num_b (int/tuple, optional): Summation limits for phase factor
            coefficients, default is 15

    Returns:
        ndarray: Quasiparticle tunneling current

    """

    num_f = cct.num_f
    num_p = cct.num_p
    npts = cct.vb_npts

    # Get frequency list with all tones/harmonics represented
    freq_list = (cct.freq[1:, None] * np.arange(1, num_p + 1)).flatten()

    current = qtcurrent(vj, cct, resp, freq_list, num_b, verbose=False)

    # Several junctions: 4D matrix (j x f x p x vb)
    if _is_multi(cct):
        current_out = np.zeros((cct.num_j, num_f + 1, num_p + 1, npts), dtype=complex)
        current_out[:, 1:, 1:] = current.reshape((cct.num_j, num_f, num_p, npts))
        return current_out

    # Arrange back into a 3D matrix (f x p x vb)
    current_out = np.zeros((num_f + 1, num_p + 1, npts), dtype=complex)
    current_out[1:, 1:] = current.reshape((num_f, num_p, npts))

    return current_out


# Calculate Jacobian matrix and its inverse ----------------------------------

# Calculate the Jacobian matrix and invert it. This is the function that
# takes the bulk of the computing time. This requires num_n*2+1 calls to the
# non-linear tunneling current calculations.


def _inv_jacobian(error_all, vj_2d, vt_2d, zt_2d, cct, resp, num_b, resp_matrix=None, verbose=True):
    """ Find the inverse Jacobian matrix. Used to update vj."""

    num_n = cct.num_n
    npts = cct.vb_npts

    jacobian = np.zeros((num_n * 2, num_n * 2, npts), dtype=float)
    for q in range(num_n):

        if verbose:
            progress_bar(q, num_n, prefix="Calculating inverse Jacobian")

        dvj = np.zeros((num_n, npts), dtype=float)
        dvj[q, :] = DV

        # The current at vj+dv (vj for signal 'q' increased by dv)
        ij_drev = _qt_current_for_hb(vj_2d + dvj, cct, resp, num_b, resp_matrix=resp_matrix)
        error_drev = vt_2d[:, None] - zt_2d[:, None] * ij_drev - (vj_2d + dvj)

        # The current at vj+dv (vj for signal 'q' increased by 1j*dv)
        ij_dimv = _qt_current_for_hb(vj_2d + 1j * dvj, cct, resp, num_b, resp_matrix=resp_matrix)
        error_dimv = vt_2d[:, None] - zt_2d[:, None] * ij_dimv - (vj_2d + 1j * dvj)

        # Numerical derivatives (hoisted out of the loop below)
        derror_drev = (error_drev - error_all) / DV
        derror_dimv = (error_dimv - error_all) / DV

        # Calculate the 2x2 Jacobian block
        block = np.zeros((2, 2, npts), dtype=float)
        for p in range(num_n):
            block[0, 0, :] = np.real(derror_drev[p])
            block[0, 1, :] = np.real(derror_dimv[p])
            block[1, 0, :] = np.imag(derror_drev[p])
            block[1, 1, :] = np.imag(derror_dimv[p])
            jacobian[p * 2:p * 2 + 2, q * 2:q * 2 + 2, :] = block

    if verbose:
        progress_bar(num_n, num_n, prefix="Calculating inverse Jacobian")

    # Invert the Jacobian matrix
    # See: https://stackoverflow.com/a/49582794
    inv_jacobian = np.linalg.inv(jacobian.transpose(2, 0, 1)).transpose(1, 2, 0)

    return inv_jacobian


def _hb_freq_list(cct):
    """Frequencies of all tones and harmonics, in [f, p] order (rounded)."""

    return [round(cct.freq[ft] * pt, ROUND_FREQ) for ft in range(1, cct.num_f + 1) for pt in range(1, cct.num_p + 1)]


def _current_derivatives_2d(vj_2d, cct, resp, num_b, resp_matrix):
    """Analytic derivatives of the currents at all tones and harmonics,
    d[n, q, c, i] = d ij[n] / d (Re, Im)[c] vj[q], from
    qmix.qtcurrent.qtcurrent_derivatives (n, q = index of [f, p])."""

    num_n, npts = cct.num_n, cct.vb_npts
    vj = np.zeros((cct.num_f + 1, cct.num_p + 1, npts), dtype=complex)
    vj[1:, 1:, :] = vj_2d.reshape((cct.num_f, cct.num_p, npts))
    d = qtcurrent_derivatives(vj, cct, resp, _hb_freq_list(cct), num_b, resp_matrix=resp_matrix)

    return d[:, 1:, 1:].reshape((num_n, num_n, 2, npts))


def _inv_jacobian_analytic(vj_2d, zt_2d, cct, resp, num_b, resp_matrix):
    """Inverse Jacobian from the analytic derivatives of the current (same
    layout as _inv_jacobian): de[n] / dvj[q] = -zt[n] dij[n] / dvj[q] -
    delta_nq."""

    num_n, npts = cct.num_n, cct.vb_npts
    derr = -zt_2d[:, None, None, None] * _current_derivatives_2d(vj_2d, cct, resp, num_b, resp_matrix)
    idx = np.arange(num_n)
    derr[idx, idx, 0] -= 1.
    derr[idx, idx, 1] -= 1j

    # [n, re/im of the error, q, re/im of the perturbation, i]
    jacobian = np.empty((num_n, 2, num_n, 2, npts))
    jacobian[:, 0] = derr.real
    jacobian[:, 1] = derr.imag
    jacobian = jacobian.reshape((2 * num_n, 2 * num_n, npts))

    return np.linalg.inv(jacobian.transpose(2, 0, 1)).transpose(1, 2, 0)


# Determine mixer current (2D) ------------------------------------------------

def _qt_current_for_hb(vj_2d, cct, resp, num_b, resp_matrix=None):
    """Calculate the DC and AC tunneling currents required for the harmonic
     balance function.

    This function will return the tunneling current for all of the tones and
    all of the harmonics in a 2-D array. This function is used by the harmonic
    balance module.

    Args:
        vj_2d (ndarray): Junction voltage in a 2-D matrix
        cct (class): Embedding circuit
        resp (class): Response function
        num_b (int/tuple): Summation limits for phase factor coefficients
        resp_matrix (ndarray): The interpolate response function matrix

    Returns:
        ndarray: AC tunneling currents in a matrix

    """

    # 2d [k, i] -> 3d [f, p, i]
    vj = np.zeros((cct.num_f + 1, cct.num_p + 1, cct.vb_npts), dtype=complex)
    vj[1:, 1:, :] = vj_2d.reshape((cct.num_f, cct.num_p, cct.vb_npts))

    current_out = qtcurrent(vj, cct, resp, _hb_freq_list(cct), num_b, verbose=False, resp_matrix=resp_matrix)

    return current_out


def _single_tone_guess(cct, resp, f, num_b, max_it, stop_rerror, damp_coeff, zj_guess, reuse_jacobian, method,
                       line_search, jacobian='analytic'):
    """Solve the harmonic balance for tone f on its own (all harmonics).

    Used as the initial guess for that tone in a multi-tone harmonic balance
    (``warm_start``).

    Returns:
        ndarray: junction voltage for tone f, shape (num_p + 1, npts), or
        (num_j, num_p + 1, npts) for several junctions

    """

    if _is_multi(cct):
        cct_1 = MultiJunctionCircuit(cct.num_j, 1, cct.num_p, vb_npts=cct.vb_npts)
        cct_1.vb = cct.vb.copy()
        cct_1.area = cct.area.copy()
        cct_1.gap_voltage = cct.gap_voltage.copy()
        cct_1.freq[1] = cct.freq[f]
        cct_1.vt[:, 1] = cct.vt[:, f]
        cct_1.zt[:, :, 1] = cct.zt[:, :, f]
        if cct.has_dc_network:
            cct_1.dc_a, cct_1.dc_b, cct_1.dc_rhs = cct.dc_a, cct.dc_b, cct.dc_rhs
            vj_1, vb_1 = harmonic_balance(cct_1, resp, num_b=num_b, max_it=max_it, stop_rerror=stop_rerror,
                                          damp_coeff=damp_coeff, verbose=False, zj_guess=zj_guess,
                                          reuse_jacobian=reuse_jacobian, method=method, line_search=line_search,
                                          jacobian=jacobian)
            return vj_1[:, 1], vb_1
        vj_1 = harmonic_balance(cct_1, resp, num_b=num_b, max_it=max_it, stop_rerror=stop_rerror,
                                damp_coeff=damp_coeff, verbose=False, zj_guess=zj_guess,
                                reuse_jacobian=reuse_jacobian, method=method, line_search=line_search,
                                jacobian=jacobian)
        return vj_1[:, 1]

    cct_1 = EmbeddingCircuit(1, cct.num_p, vb_npts=cct.vb_npts)
    cct_1.vb = cct.vb.copy()
    cct_1.freq[1] = cct.freq[f]
    cct_1.vt[1] = cct.vt[f]
    cct_1.zt[1] = cct.zt[f]

    vj_1 = harmonic_balance(cct_1, resp, num_b=num_b, max_it=max_it, stop_rerror=stop_rerror, damp_coeff=damp_coeff,
                            verbose=False, zj_guess=zj_guess, reuse_jacobian=reuse_jacobian, method=method,
                            line_search=line_search, jacobian=jacobian)

    return vj_1[1]


def _error_vector(vj_2d, vt_2d, zt_2d, cct, resp, num_b, resp_matrix):
    """Harmonic balance error vector, vt - zt * ij - vj, for each signal and
    bias voltage (2-D, [k, i])."""

    ij_2d = _qt_current_for_hb(vj_2d, cct, resp, num_b, resp_matrix=resp_matrix)

    return vt_2d[:, None] - zt_2d[:, None] * ij_2d - vj_2d


# Several junctions -----------------------------------------------------------

def _initial_guess_multi(vt, zt, zj_guess, npts):
    """Initial guess of the junction voltages, vj[j, f, p, i], with each
    junction replaced by the impedance zj_guess: (1 + zt / zj) vj = vt."""

    num_j = vt.shape[0]
    a = np.eye(num_j)[:, :, None, None] + zt / zj_guess        # [j, l, f, p]
    v = np.linalg.solve(a.transpose(2, 3, 0, 1), vt.transpose(1, 2, 0)[..., None])[..., 0]  # [f, p, j]

    return v.transpose(2, 0, 1)[..., None] * np.ones(npts, dtype=complex)


class _MultiJunctionSystem(object):
    """Error vector and inverse Jacobian for several junctions.

    The unknowns are vj_2d[k, i] with k = j * num_n + n (junction j, signal
    n = index of [f, p]). The error is

        e[j, n] = vt[j, n] - sum_l zt[j, l, n] ij[l, n] - vj[j, n]

    The current of junction l only depends on its own voltage, so the
    derivatives with respect to signal q of all junctions are found by
    perturbing signal q of all junctions at once:
    de[j, n] / dvj[l, q] = -zt[j, l, n] dij[l, n] / dvj[l, q] - delta_jl delta_nq.

    """

    def __init__(self, cct, resp, num_b, resp_matrix):

        self.num_j, self.num_n, self.npts = cct.num_j, cct.num_n, cct.vb_npts
        self.num_b = num_b
        self.ccts = [cct.junction(j) for j in range(self.num_j)]
        self.resp = junction_resp(resp, self.num_j, cct.gap_voltage)
        self.resp_matrix = resp_matrix
        self.area = cct.area.copy()
        self.vt_2d = cct.vt[:, 1:, 1:].reshape(self.num_j * self.num_n)
        # [n, j, l]
        self.zt_2d = cct.zt[:, :, 1:, 1:].reshape(self.num_j, self.num_j, self.num_n).transpose(2, 0, 1)

    def currents(self, vj_2d):
        """Tunneling currents, ij[j, n, i]."""
        vj_3d = vj_2d.reshape((self.num_j, self.num_n, self.npts))
        return np.array([self.area[j] * _qt_current_for_hb(vj_3d[j], self.ccts[j], self.resp[j], self.num_b,
                                                           resp_matrix=self.resp_matrix[j])
                         for j in range(self.num_j)])

    def error_vector(self, vj_2d):
        """Harmonic balance error vector, [k, i]."""
        zi = np.einsum('njl,lni->jni', self.zt_2d, self.currents(vj_2d))
        return self.vt_2d[:, None] - zi.reshape(vj_2d.shape) - vj_2d

    def inv_jacobian_analytic(self, err_all, vj_2d, verbose=True):
        """Inverse Jacobian from the analytic derivatives of the currents
        (see inv_jacobian)."""

        num_j, num_n, npts = self.num_j, self.num_n, self.npts
        vj_3d = vj_2d.reshape((num_j, num_n, npts))
        # dij[l, n, q, c, i]
        dij = np.array([self.area[j] * _current_derivatives_2d(vj_3d[j], self.ccts[j], self.resp[j], self.num_b,
                                                               self.resp_matrix[j])
                        for j in range(num_j)])
        # [j, n, l, q, c, i]
        derr = -np.einsum('njl,lnqci->jnlqci', self.zt_2d, dij)
        for j in range(num_j):
            for q in range(num_n):
                derr[j, q, j, q, 0] -= 1.
                derr[j, q, j, q, 1] -= 1j

        # [j, n, re/im of the error, l, q, re/im of the perturbation, i]
        jacobian = np.empty((num_j, num_n, 2, num_j, num_n, 2, npts))
        jacobian[:, :, 0] = derr.real
        jacobian[:, :, 1] = derr.imag
        num_k = num_j * num_n
        jacobian = jacobian.reshape((2 * num_k, 2 * num_k, npts))

        return np.linalg.inv(jacobian.transpose(2, 0, 1)).transpose(1, 2, 0)

    def inv_jacobian(self, err_all, vj_2d, verbose=True):
        """Inverse Jacobian, [2k, 2k, i] (real and imaginary parts, as in
        _inv_jacobian)."""

        num_j, num_n, npts = self.num_j, self.num_n, self.npts
        ij_0 = self.currents(vj_2d)
        # [j, n, re/im of the error, l, q, re/im of the perturbation, i]
        jacobian = np.zeros((num_j, num_n, 2, num_j, num_n, 2, npts), dtype=float)
        for q in range(num_n):

            if verbose:
                progress_bar(q, num_n, prefix="Calculating inverse Jacobian")

            dvj = np.zeros((num_j, num_n, npts), dtype=float)
            dvj[:, q, :] = DV
            dvj = dvj.reshape(vj_2d.shape)

            for c, unit in enumerate((1., 1j)):
                # dij[l, n] / dvj[l, q]
                dij = (self.currents(vj_2d + unit * dvj) - ij_0) / DV
                derr = -np.einsum('njl,lni->jnli', self.zt_2d, dij)
                for j in range(num_j):
                    derr[j, q, j] -= unit
                jacobian[:, :, 0, :, q, c] = derr.real
                jacobian[:, :, 1, :, q, c] = derr.imag

        if verbose:
            progress_bar(num_n, num_n, prefix="Calculating inverse Jacobian")

        num_k = num_j * num_n
        jacobian = jacobian.reshape((2 * num_k, 2 * num_k, npts))

        return np.linalg.inv(jacobian.transpose(2, 0, 1)).transpose(1, 2, 0)


class _DCBiasSystem(object):
    """Error vector and inverse Jacobian for several junctions with a DC bias
    network (``MultiJunctionCircuit.set_dc_network``).

    The unknowns are vj_2d[k, i]: k = j * num_n + n for the signals (as in
    _MultiJunctionSystem), followed by the DC voltages v0[j] at
    k = num_j * num_n + j (stored as complex numbers; their imaginary parts
    are pinned to zero). The errors are

        e[j, n] = vt[j, n] - sum_l zt[j, l, n] ij[l, n] - vj[j, n]
        e[r]    = rhs[r] - sum_l a[r, l] v0[l] - sum_l b[r, l] i0[l] - j Im(v0[r])

    The response function matrix depends on the DC voltage, so it is
    interpolated again for every error vector. The derivatives with respect
    to the signals are the same as in _MultiJunctionSystem (plus those of
    the DC currents); the derivatives with respect to v0 are central
    differences (step DV_DC), with v0 of all junctions perturbed at once
    (the currents of junction l only depend on v0[l]), i.e., two more
    current calculations per Jacobian. (A forward step of DV is too crude
    at the sharp gap knee of the DC I-V curve.)

    """

    def __init__(self, cct, resp, num_b, method, verbose=False):

        self.num_j, self.num_n, self.npts = cct.num_j, cct.num_n, cct.vb_npts
        self.num_k = self.num_j * self.num_n
        self.num_b = num_b
        self.nb_list = _unpack_num_b(num_b, cct.num_f)
        self.freq = cct.freq.copy()
        self.ccts = [cct.junction(j) for j in range(self.num_j)]
        self.resp = junction_resp(resp, self.num_j, cct.gap_voltage)
        self.area = cct.area.copy()
        self.dc_a, self.dc_b, self.dc_rhs = cct.dc_a, cct.dc_b, cct.dc_rhs
        self.vt_2d = cct.vt[:, 1:, 1:].reshape(self.num_k)
        # [n, j, l]
        self.zt_2d = cct.zt[:, :, 1:, 1:].reshape(self.num_j, self.num_j, self.num_n).transpose(2, 0, 1)
        # Output frequencies: DC, then all tones and harmonics
        self.freq_list = [0.] + _hb_freq_list(cct)

        # The DC voltages are unknown: interpolation range from the bias network
        v_max = max(np.abs(self.dc_rhs).max(), np.abs(cct.vb).max()) * 1.2 + 0.1
        self.fourier = _select_method(method, self.nb_list, self.npts, verbose) == 'fourier'
        for resp_j in set(self.resp):
            _check_respfn_range(resp_j, np.array([-v_max, v_max]), self.freq, self.nb_list)
        if self.fourier:
            # One Fourier series per response function, covering all DC voltages
            fresp = {}
            for resp_j in self.resp:
                if resp_j not in fresp:
                    fresp[resp_j] = FourierResp(resp_j, np.array([-v_max, v_max]), self.freq, self.nb_list)
            self.fresp = [fresp[resp_j] for resp_j in self.resp]

    def resp_matrix(self, j, v0):
        """Interpolated response function of junction j at DC voltages v0."""
        if self.fourier:
            fresp = copy.copy(self.fresp[j])
            fresp.vb = v0
            return fresp
        return self.resp[j](_interpolation_voltages(v0, self.freq, self.nb_list))

    def _split(self, vj_2d):
        """Signals [j, n, i] and DC voltages [j, i]."""
        return vj_2d[:self.num_k].reshape((self.num_j, self.num_n, self.npts)), vj_2d[self.num_k:].real

    def _vj(self, vj_n):
        """[n, i] -> [f, p, i] for one junction."""
        cct = self.ccts[0]
        vj = np.zeros((cct.num_f + 1, cct.num_p + 1, self.npts), dtype=complex)
        vj[1:, 1:, :] = vj_n.reshape((cct.num_f, cct.num_p, self.npts))
        return vj

    def currents(self, vj_2d, dv0=0.):
        """DC currents i0[j, i] and AC currents ij[j, n, i] (incl. area), with
        all DC voltages shifted by dv0."""
        vj_3d, v0 = self._split(vj_2d)
        out = np.array([self.area[j] * qtcurrent(self._vj(vj_3d[j]), self.ccts[j], self.resp[j], self.freq_list,
                                                 self.num_b, verbose=False,
                                                 resp_matrix=self.resp_matrix(j, v0[j] + dv0))
                        for j in range(self.num_j)])
        return out[:, 0].real, out[:, 1:]

    def _errors(self, vj_2d, i0, ij):
        vj_3d, v0 = self._split(vj_2d)
        err_ac = self.vt_2d[:, None] - np.einsum('njl,lni->jni', self.zt_2d, ij).reshape(self.num_k, -1) - \
            vj_2d[:self.num_k]
        err_dc = self.dc_rhs - self.dc_a @ v0 - self.dc_b @ i0 - 1j * vj_2d[self.num_k:].imag
        return np.vstack((err_ac, err_dc))

    def error_vector(self, vj_2d):
        """Harmonic balance error vector, [k, i]."""
        return self._errors(vj_2d, *self.currents(vj_2d))

    def _dc_derivatives(self, vj_2d):
        """Central differences with respect to v0 of all junctions:
        di0[l, i] and dij[l, n, i] (d / d v0[l])."""
        i0_p, ij_p = self.currents(vj_2d, DV_DC)
        i0_m, ij_m = self.currents(vj_2d, -DV_DC)
        return (i0_p - i0_m) / (2 * DV_DC), (ij_p - ij_m) / (2 * DV_DC)

    def _assemble(self, dij, di0, dij_v0, di0_v0):
        """Inverse Jacobian from the derivatives of the currents.

        Args:
            dij: d ij[l, n] / d (re, im)[c] vj[l, q], [l, n, q, c, i]
            di0: d i0[l] / d (re, im)[c] vj[l, q], [l, q, c, i] (real)
            dij_v0: d ij[l, n] / d v0[l], [l, n, i]
            di0_v0: d i0[l] / d v0[l], [l, i]

        """

        num_j, num_n, num_k, npts = self.num_j, self.num_n, self.num_k, self.npts
        num_u = num_k + num_j
        # Complex derivative of each error with respect to each real
        # parameter: [error, unknown, re/im of the unknown, i]
        derr = np.zeros((num_u, num_u, 2, npts), dtype=complex)

        # Signal errors with respect to the signals
        d_ac = -np.einsum('njl,lnqci->jnlqci', self.zt_2d, dij)
        for j in range(num_j):
            for q in range(num_n):
                d_ac[j, q, j, q, 0] -= 1.
                d_ac[j, q, j, q, 1] -= 1j
        derr[:num_k, :num_k] = d_ac.reshape((num_k, num_k, 2, npts))

        # Signal errors with respect to v0 (real part only)
        derr[:num_k, num_k:, 0] = -np.einsum('njl,lni->jnli', self.zt_2d, dij_v0).reshape((num_k, num_j, npts))

        # DC errors with respect to the signals: -b[r, l] di0[l] / dvj[l, q]
        d_dc = -self.dc_b[:, :, None, None, None] * di0[None]
        derr[num_k:, :num_k] = d_dc.reshape((num_j, num_k, 2, npts))

        # DC errors with respect to v0: real part -a - b di0/dv0, imaginary part -j (pins Im v0)
        derr[num_k:, num_k:, 0] = -self.dc_a[:, :, None] - self.dc_b[:, :, None] * di0_v0[None]
        idx = np.arange(num_j)
        derr[num_k + idx, num_k + idx, 1] -= 1j

        # [error, re/im of the error, unknown, re/im of the unknown, i]
        jacobian = np.empty((num_u, 2, num_u, 2, npts))
        jacobian[:, 0] = derr.real
        jacobian[:, 1] = derr.imag
        jacobian = jacobian.reshape((2 * num_u, 2 * num_u, npts))

        return np.linalg.inv(jacobian.transpose(2, 0, 1)).transpose(1, 2, 0)

    def inv_jacobian_analytic(self, err_all, vj_2d, verbose=True):
        """Inverse Jacobian, analytic derivatives with respect to the signals."""

        vj_3d, v0 = self._split(vj_2d)
        d = np.array([self.area[j] * qtcurrent_derivatives(self._vj(vj_3d[j]), self.ccts[j], self.resp[j],
                                                           self.freq_list, self.num_b,
                                                           resp_matrix=self.resp_matrix(j, v0[j]))
                      for j in range(self.num_j)])  # [l, o, f, p, c, i]
        d = d[:, :, 1:, 1:].reshape((self.num_j, self.num_n + 1, self.num_n, 2, self.npts))
        di0_v0, dij_v0 = self._dc_derivatives(vj_2d)

        return self._assemble(d[:, 1:], d[:, 0].real, dij_v0, di0_v0)

    def inv_jacobian(self, err_all, vj_2d, verbose=True):
        """Inverse Jacobian, forward differences."""

        num_j, num_n, npts = self.num_j, self.num_n, self.npts
        i0, ij = self.currents(vj_2d)
        dij = np.zeros((num_j, num_n, num_n, 2, npts), dtype=complex)
        di0 = np.zeros((num_j, num_n, 2, npts))
        for q in range(num_n):

            if verbose:
                progress_bar(q, num_n, prefix="Calculating inverse Jacobian")

            dvj = np.zeros_like(vj_2d, dtype=float)
            dvj[:self.num_k].reshape((num_j, num_n, npts))[:, q, :] = DV
            for c, unit in enumerate((1., 1j)):
                i0_d, ij_d = self.currents(vj_2d + unit * dvj)
                dij[:, :, q, c] = (ij_d - ij) / DV
                di0[:, q, c] = (i0_d - i0) / DV

        if verbose:
            progress_bar(num_n, num_n, prefix="Calculating inverse Jacobian")

        di0_v0, dij_v0 = self._dc_derivatives(vj_2d)

        return self._assemble(dij, di0, dij_v0, di0_v0)


# Line search -----------------------------------------------------------------

# Backtracking line search along the Newton step, separately at each bias
# voltage (Numerical Recipes, Section 9.7). The merit function is the squared
# error norm f = |e|^2. For an exact Newton step, its initial slope along the
# step is -2 f, so the sufficient decrease condition for step length lam is
# f(lam) <= (1 - 2 * alpha * lam) * f(0). Rejected steps are shortened to the
# minimum of a quadratic fit (limited to 0.1 - 0.5 of the previous length).

def _line_search(vj_2d, corr, err_all, active, lam0, fresh_jacobian, error_vector):
    """Apply the Newton correction with a backtracking line search.

    Args:
        vj_2d (ndarray): junction voltage, [k, i]
        corr (ndarray): Newton correction, [k, i]
        err_all (ndarray): error vector at vj_2d, [k, i]
        active (ndarray): bias voltages that need the line search (others
            take the full step)
        lam0 (float): initial step length (damping coefficient)
        fresh_jacobian (bool): was the Jacobian calculated at vj_2d? If not,
            bias voltages without an acceptable step keep their old voltage
            (so that the Jacobian can be recalculated); otherwise they take
            the shortest step tried.
        error_vector (function): error vector as a function of the junction
            voltage, [k, i]

    Returns:
        tuple: new junction voltage, its error vector, the number of step
        reductions, and a mask of the bias voltages without an acceptable step

    """

    merit0 = np.sum(np.abs(err_all) ** 2, axis=0)
    lam = np.full(vj_2d.shape[1], float(lam0))

    vj_new = vj_2d + lam * corr
    err_new = error_vector(vj_new)
    merit = np.sum(np.abs(err_new) ** 2, axis=0)
    pending = active & ~(merit <= (1 - 2 * LINE_SEARCH_ALPHA * lam) * merit0)

    num_backtrack = 0
    while pending.any() and num_backtrack < LINE_SEARCH_MAX_BACKTRACK:
        num_backtrack += 1

        # Minimum of the quadratic through f(0), f'(0) = -2 f(0) and f(lam)
        lam_old, f0, f1 = lam[pending], merit0[pending], merit[pending]
        denom = f1 - f0 + 2 * f0 * lam_old
        with np.errstate(divide='ignore', invalid='ignore'):
            lam_quad = np.where(denom > 0, f0 * lam_old ** 2 / denom, 0.5 * lam_old)
        lam[pending] = np.clip(np.nan_to_num(lam_quad, nan=0.5 * lam_old), 0.1 * lam_old, 0.5 * lam_old)

        # Evaluating all bias voltages costs about the same as a subset
        vj_try = vj_2d + lam * corr
        err_try = error_vector(vj_try)
        vj_new[:, pending] = vj_try[:, pending]
        err_new[:, pending] = err_try[:, pending]
        merit[pending] = np.sum(np.abs(err_try[:, pending]) ** 2, axis=0)
        pending &= ~(merit <= (1 - 2 * LINE_SEARCH_ALPHA * lam) * merit0)

    stuck = pending
    if stuck.any() and not fresh_jacobian:
        vj_new[:, stuck] = vj_2d[:, stuck]
        err_new[:, stuck] = err_all[:, stuck]

    return vj_new, err_new, num_backtrack, stuck


# General helper functions --------------------------------------------------

def _k_to_fp(k, num_p):
    """ Given the index 'k' (i.e., the 1-d index), find the equivalent index
    in '[f,p]' (i.e., the 2-d index). """

    p = (k % num_p) + 1
    f = (k + num_p - p + 1) / num_p

    return f, p
