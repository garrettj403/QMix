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

"""

import qmix
import numpy as np
from timeit import default_timer as timer
from qmix.misc.progbar import progress_bar
from qmix.qtcurrent import qtcurrent_all_freq, interpolate_respfn


# minimum Thevenin voltage (required to avoid div by 0 errors)
MIN_VT = 1e-10
# voltage step for numerical derivatives
DV = 1e-3
# round vph values to this number of decimal places when comparing
ROUND_VPH = 4


# Harmonic Balance ----------------------------------------------------------

# This is the main harmonic balance function. It uses the Newton-Raphson
# method to find the solution to the highly non-linear equation.


def harmonic_balance(cct, resp, num_b=15, max_it=10, stop_rerror=0.001, vj_initial=None, damp_coeff=1., mode="o", verbose=True, zj_guess=0.67,):
    """Perform harmonic balance.

    Determine the harmonic balance of the junction + embedding circuit system. 
    Uses Newton's method to find the solution. For more information, see 
    Garrett (2018); Kittara (2002); Kittara, Winthington & Yassin (2007); or 
    Withington, Kittara & Yassin (2003). [Full references in online docs.]

    Args:
        cct (qmix.circuit.EmbeddingCircuit): Embedding circuit
        resp (qmix.respfn.RespFn): Response function

    Keyword arguments:
        num_b (int_or_tuple): Number of Bessel functions to include
        max_it (int): Maximum number of iterations
        stop_rerror (float): Maximum acceptable relative error
        vj_initial (ndarray): Initial guess of junction voltage (vj)
        damp_coeff (float): Dampening coefficient for correction factor (0-1)
        mode (string): output vj ('o'), print ('p'), output extra data ('x')
        verbose (bool): print info to terminal if true

    Returns:
        ndarray: junction voltage that satisfies the circuit

    """

    if verbose:
        print("Running harmonic balance:")

    start_time = timer()

    # Check input data ------------------------------------------------------

    num_f = cct.num_f
    num_p = cct.num_p
    num_n = cct.num_n
    npts = cct.vb_npts

    vt = cct.vt
    zt = cct.zt

    # Harmonic balance isn't required if all zt are equal to zero
    if (zt == 0).all():
        if verbose:
            print("Done.\n")
        return vt[:, :, None] * np.ones(npts, dtype=complex)

    # Check whether any Thevenin voltages are zero (prevents div0 errors) ----

    vt[np.abs(vt) < MIN_VT] = MIN_VT
    vt[0, :] = 0  # TODO: is this needed?
    vt[:, 0] = 0

    # Initial guess of the junction voltage (vj) -----------------------------

    if vj_initial is None:

        vj_initial = vt[:, :, None] * \
                     zj_guess / (zj_guess + zt[:, :, None]) * \
                     np.ones(npts, dtype=complex)

    # Prepare input data ----------------------------------------------------

    # Note on matrix shapes: Most of the circuit parameters are stored in a
    # [f, p, vb] format. I.e., the first index is for the tone, the second is
    # for the harmonic and the third is for the bias voltage. In the harmonic
    # balance function, I change these to [k, vb] to make the matrix
    # manipulation easier to understand. Therefore, a 3 tone, 2 harmonic
    # variable (with shape (3+1) x (2+1) x npts, where npts = length of bias
    # voltage) will be transformed into a matrix with shape 6 x npts.

    # Change format of data from [f, p] to [p] (reduce by one dimension)
    vj_2d = vj_initial[1:, 1:, :].reshape((num_n, npts))
    vt_2d = vt[1:, 1:].reshape(num_n)
    zt_2d = zt[1:, 1:].reshape(num_n)

    if verbose:
        print((" - {0} tone(s) and {1} harmonic(s)".format(num_f, num_p)))
        msg = " - {0} calls to the quasiparticle tunneling current (qtc) function per iteration"
        print((msg.format(num_n * 2 + 1)))
        print(" - max. iterations: {}".format(max_it))

    # Interpolate response function for all required voltages
    respfn_interp = interpolate_respfn(cct, resp, num_b)

    # Perform harmonic balance -----------------------------------------------

    iteration = 0
    for iteration in range(max_it + 1):

        # Current vector (junction current)
        time_current = timer()
        ij_2d = _qt_current_for_hb(vj_2d, cct, resp, num_b, resp_matrix=respfn_interp)
        if iteration == 0 and verbose:
            print("Estimated time:")
            call_time = timer() - time_current
            print((' - time per qtc call:  {:7.2f} s / {:6.2f} min / {:5.2f} hrs'.format(call_time, call_time/60., call_time/3600.)))
            it_time = call_time * (num_n * 2 + 1)
            print((' - time per iteration: {:7.2f} s / {:6.2f} min / {:5.2f} hrs'.format(it_time, it_time/60., it_time/3600.)))
            max_time = it_time * max_it
            print((' - max sim time:       {:7.2f} s / {:6.2f} min / {:5.2f} hrs'.format(max_time, max_time/60., max_time/3600.)))

        # Error vector
        err_all = vt_2d[:, None] - zt_2d[:, None] * ij_2d - vj_2d

        # Check error at each tone and harmonic
        if verbose:
            print(("Error after {0} iteration(s):".format(iteration)))
        max_rel_error = 0.  # initialize maximum relative error of all signals
        msg = "\tf:{:d}, p:{:d},   med. rel. error: {:9.3f},   max. rel. error: {:9.3f},   {:5.1f} % complete"
        finished_points = np.ones(npts, dtype=bool)
        for k in range(num_n):
            with np.errstate(divide='ignore'):
                # Check relative error at k
                abs_error = np.abs(err_all[k, :])
                rel_error = abs_error / np.abs(vj_2d[k, :])
                max_rel_error = max(np.max(rel_error), max_rel_error)
                med_rel_error = np.median(abs_error / np.abs(vj_2d[k, :]))
                good_points = rel_error < stop_rerror
                finished_points = finished_points & good_points
                complete = float(np.sum(good_points)) / npts

            # Print to terminal
            if verbose:
                f, p = _k_to_fp(k, num_p)

                if med_rel_error > 99999.999:
                    _med_rel_error = 99999.999
                else:
                    _med_rel_error = med_rel_error

                if np.max(rel_error) > 99999.999:
                    _rel_error = 99999.999
                else:
                    _rel_error = np.max(rel_error)
                    
                print((msg.format(int(f), int(p), _med_rel_error, _rel_error, complete * 100)))

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
        inv_j = _inv_jacobian(err_all, vj_2d, vt_2d, zt_2d, cct, resp, num_b, resp_matrix=respfn_interp, verbose=verbose)
        if verbose:
            print("Applying correction")
        corr = np.zeros((num_n, npts), dtype=complex)
        for p in range(num_n):
            for q in range(num_n):
                corr[p, :] -=      (inv_j[p * 2,     q * 2    ] * np.real(err_all[q]) +
                                    inv_j[p * 2,     q * 2 + 1] * np.imag(err_all[q]))
                corr[p, :] -= 1j * (inv_j[p * 2 + 1, q * 2    ] * np.real(err_all[q]) +
                                    inv_j[p * 2 + 1, q * 2 + 1] * np.imag(err_all[q]))
        vj_2d += corr * damp_coeff

    # Format results (from [k,vb] to [f,p,vb])
    vj_out = np.zeros((num_f + 1, num_p + 1, npts), dtype=complex)
    vj_out[1:, 1:, :] = vj_2d.reshape((num_f, num_p, npts))

    time = timer() - start_time
    if verbose:
        print((" - sim time:\t\t{:7.2f} s / {:6.2f} min / {:5.2f} hrs".format(time, time/60., time/3600.)))
        if iteration >= 1:
            tit = time / iteration  # time per iteration
            msg = " - {} iterations required"
            print((msg.format(iteration)))
            print((" - time per iteration:\t{:7.2f} s / {:6.2f} min / {:5.2f} hrs".format(tit, tit/60., tit/3600.)))

    if mode == "o":  # return vj
        return vj_out
    elif mode == "x":  # return vj, number of iterations, and if converged
        did_not_hit_max_it = iteration != max_it
        return vj_out, iteration, did_not_hit_max_it
    elif mode == "m":  # return vj and mask
        return vj_out, finished_points


# Check results -------------------------------------------------------------

# Double check the error from the harmonic balance function. This is mainly
# for debugging purposes to ensure that everything was done properly.


def check_hb_error(vj_check, cct, resp, num_b=15, stop_rerror=0.001):
    """Check the results from the `harmonic_balance` function.

    Just to double check. Mostly for debugging purposes.

    Args:
        vj_check (ndarray): The voltage across junction to check
        cct (qmix.circuit.EmbeddingCircuit): Embedding circuit
        resp (qmix.respfn.RespFn): Response function

    Keyword arguments:
        num_b: Number of Bessel functions to include
        stop_rerror (float): Maximum acceptable relative error

    Raises:
        AssertionError: If the stop error is not met

    """

    # Note: speed is not important here...

    print("Double-checking harmonic balance error:")

    # Tunnelling current for all tones/harmonics
    ij_all = qtcurrent_all_freq(vj_check, cct, resp, num_b)

    for f in range(1, cct.num_f + 1):
        for p in range(1, cct.num_p + 1):
            error_fp = cct.vt[f, p] - \
                cct.zt[f, p] * ij_all[f, p, :] - \
                vj_check[f, p, :]
            max_rel_error = np.max(np.abs(error_fp) /
                                   np.abs(vj_check[f, p, :]))
            good_error = max_rel_error < stop_rerror
            if good_error:
                err_str = 'Yes'
            else:
                err_str = 'No'
            msg = "\tf:{0},\tp:{1},\tmax rel. error: {2:.2E},\tPass? {3}"
            print((msg.format(f, p, max_rel_error, err_str)))
            assert good_error
    print("")


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

        # Calculate the 2x2 Jacobian block
        block = np.zeros((2, 2, npts), dtype=float)
        for p in range(num_n):
            block[0, 0, :] = np.real((error_drev - error_all) / DV)[p]
            block[0, 1, :] = np.real((error_dimv - error_all) / DV)[p]
            block[1, 0, :] = np.imag((error_drev - error_all) / DV)[p]
            block[1, 1, :] = np.imag((error_dimv - error_all) / DV)[p]
            jacobian[p * 2:p * 2 + 2, q * 2:q * 2 + 2, :] = block

    if verbose:
        progress_bar(num_n, num_n, prefix="Calculating inverse Jacobian")

    # Invert the Jacobian matrix
    inv_jacobian = np.zeros((num_n * 2, num_n * 2, npts), dtype=float)
    for i in range(npts):
        inv_jacobian[:, :, i] = np.linalg.inv(jacobian[:, :, i])

    return inv_jacobian


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
        num_b (int/tuple): Number of Bessel functions to include
        resp_matrix (ndarray): The interpolate response function matrix

    Returns:
        ndarray: AC tunneling currents in a matrix

    """

    # 2d [k, i] -> 3d [f, p, i]
    vj = np.zeros((cct.num_f + 1, cct.num_p + 1, cct.vb_npts), dtype=complex)
    vj[1:, 1:, :] = vj_2d.reshape((cct.num_f, cct.num_p, cct.vb_npts))

    vph_list = []
    for ft in range(1, cct.num_f + 1):
        for pt in range(1, cct.num_p + 1):
            vph_list.append(round(cct.vph[ft] * pt, ROUND_VPH))

    current_out = qmix.qtcurrent.qtcurrent(vj, cct, resp, vph_list, num_b, verbose=False, resp_matrix=resp_matrix)

    return current_out


# General helper functions --------------------------------------------------

def _k_to_fp(k, num_p):
    """ Given the index 'k' (i.e., the 1-d index), find the equivalent index
    in '[f,p]' (i.e., the 2-d index). """

    p = (k % num_p) + 1
    f = (k + num_p - p + 1) / num_p

    return f, p
