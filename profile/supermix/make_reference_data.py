"""Generate SuperMix reference data for tests/test_supermix_reference.py.

Runs the tight-tolerance SuperMix driver (Bessel truncation 1e-14) on the
same cases as compare.py and saves the inputs and SuperMix's results to
``tests/supermix-data/supermix_reference.npz``. The test then checks QMix
against these values without needing SuperMix.

The I-V curve is a table (QMix's ``RespFnPolynomial(50)`` sampled every
``V_STEP`` for v >= 0, mirrored for v < 0). Every bias voltage and LO
frequency below is a multiple of ``V_STEP``, so every voltage vb + k * flo
that either code evaluates is a table point: SuperMix's spline returns the
table values exactly, and the test can look them up directly. The reference
data therefore checks ``qtcurrent`` and ``harmonic_balance`` only, and does
not depend on how QMix builds its response function.

Usage (from this directory, after ``./build.sh --tight smx_driver.cc``)::

    python make_reference_data.py

"""

import os

import numpy as np

import qmix
from qmix.qtcurrent import suggest_num_b

import compare

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, '..', '..', 'tests', 'supermix-data', 'supermix_reference.npz')
DRIVER = os.path.join(HERE, 'bin', 'smx_driver_tight')

V_STEP = 0.005
V_MAX = 25.


class TableResp(object):
    """Response function looked up from a table (v >= 0, mirrored)."""

    def __init__(self, step, table):
        self.step, self.table = step, table

    def __call__(self, vbias):
        vbias = np.asarray(vbias, dtype=float)
        x = np.abs(vbias) / self.step
        idx = np.rint(x).astype(int)
        assert np.abs(x - idx).max() < 1e-6, "Voltage is not on the table grid"
        r = self.table[idx]
        return r.real + 1j * np.sign(vbias) * r.imag

    resp = __call__


def export_table(table):
    os.makedirs(compare.WORK, exist_ok=True)
    v = np.arange(len(table)) * V_STEP
    files = []
    for label, values in (('idc', table.imag), ('ikk', table.real)):
        path = os.path.join(compare.WORK, 'table_{}.dat'.format(label))
        np.savetxt(path, np.c_[v, values], fmt='%.17g')
        files.append(path)
    return files


def main():

    compare.DRIVER = DRIVER
    if not os.path.isfile(DRIVER):
        raise SystemExit("Tight driver not found; run ./build.sh --tight smx_driver.cc first")

    v_table = np.arange(0, V_MAX + V_STEP / 2, V_STEP)
    table = qmix.respfn.RespFnPolynomial(50, verbose=False)(v_table)
    resp = TableResp(V_STEP, table)
    files = export_table(table)

    data = {'v_step': V_STEP, 'resp_table': table}
    names = []

    # Tunneling current with a given junction voltage
    current_cases = [(0.3, [0.5 * 0.3]), (0.3, [1.0 * 0.3]), (0.3, [3.0 * 0.3]),
                     (0.33, [0.35, 0.08 * np.exp(0.7j), 0.03 * np.exp(-1.2j)])]
    for flo, vph in current_cases:
        vb = np.round(np.linspace(0, 2, 401), 12)
        num_p = len(vph)
        vj = np.zeros((num_p + 1, len(vb)), dtype=complex)
        vj[0] = vb
        vj[1:] = np.array(vph)[:, None]
        cct = compare.make_circuit(flo, num_p, vb)
        vj3 = np.zeros((2, num_p + 1, len(vb)), dtype=complex)
        vj3[1, 1:] = vj[1:]
        num_b = suggest_num_b(vj3, cct, tol=1e-14, max_b=60)[0]

        smx = compare.supermix_current(files, flo, vb, vj)
        ij_q = compare.qmix_current(resp, cct, vj, num_b)

        key = 'current{}'.format(len(names))
        names.append(key)
        data.update({key + '_flo': flo, key + '_vb': vb, key + '_vj': vj, key + '_num_b': num_b,
                     key + '_ij': smx['ij']})
        print('{}: flo {}, {} harmonic(s), num_b {}, max |QMix - SuperMix| = {:.1e}'.format(
              key, flo, num_p, num_b, np.abs(ij_q - smx['ij']).max()))

    # Harmonic balance
    balance_cases = [(0.3, [0.5], [0.3 - 0.3j]),
                     (0.3, [0.5, 0, 0], [0.3 - 0.3j, 0.1 - 0.2j, 0.05 - 0.1j])]
    for flo, vt_list, zt_list in balance_cases:
        vb = np.round(np.linspace(0.02, 2, 199), 12)
        vt = np.r_[0, vt_list].astype(complex)
        zt = np.r_[0, zt_list].astype(complex)
        smx = compare.supermix_balance(files, flo, vb, vt, zt)
        assert (smx['status'] == 0).all(), "SuperMix balance failed"

        cct = compare.make_circuit(flo, len(vt_list), vb)
        vj3 = np.zeros((2, len(vt), len(vb)), dtype=complex)
        vj3[1, 1:] = smx['vj'][1:]
        num_b = suggest_num_b(vj3, cct, tol=1e-14, max_b=60)[0]

        key = 'balance{}'.format(len(names))
        names.append(key)
        data.update({key + '_flo': flo, key + '_vb': vb, key + '_vt': vt, key + '_zt': zt, key + '_num_b': num_b,
                     key + '_vj': smx['vj'], key + '_ij': smx['ij']})
        print('{}: flo {}, {} harmonic(s), num_b {}, SuperMix iterations <= {}'.format(
              key, flo, len(vt_list), num_b, smx['iterations'].max()))

    data['names'] = np.array(names)
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    np.savez_compressed(OUTPUT, **data)
    print('Saved {} ({:.0f} kB)'.format(os.path.relpath(OUTPUT, HERE), os.path.getsize(OUTPUT) / 1e3))


if __name__ == '__main__':
    main()
