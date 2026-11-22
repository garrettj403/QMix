"""Compare QMix's passive circuit elements (qmix.network) with SuperMix's.

Runs SuperMix's elements (``smx_elements.cc``) for the cases below and
compares their S-parameters with QMix's versions (``qmix_element`` in
``tests/test_supermix_reference.py``, which is imported here, so that the
script and the test use the same mapping).

Usage (from this directory, after ``./build.sh smx_elements.cc``)::

    python compare_elements.py
    python compare_elements.py --save   # also write tests/supermix-data/supermix_elements.npz

QMix is run with SuperMix's mu_0 = 4 pi 1e-7 (the waveguide, the via and the
films of the radial stub depend on it).

"""

import argparse
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import scipy.constants as sc

import mattisbardeen
import qmix.network
import qmix.superconductor

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.join(HERE, '..', '..', 'tests')
sys.path.insert(0, TESTS)
from test_supermix_reference import qmix_element  # noqa: E402

PROGRAM = os.environ.get('SMX_ELEMENTS', os.path.join(HERE, 'bin', 'smx_elements'))
OUTPUT = os.path.join(TESTS, 'supermix-data', 'supermix_elements.npz')

SUPERMIX_CONSTANTS = SimpleNamespace(e=1.60217733e-19, k=1.380658e-23, h=6.6260755e-34,
                                     hbar=6.6260755e-34 / (2 * np.pi), mu_0=4e-7 * np.pi, c=sc.c)

F_WIDE = np.geomspace(1e9, 1e12, 31)
F_WG = np.linspace(100e9, 400e9, 31) + 1e6    # TE10 cutoff of a = 1 mm: 149.9 GHz
F_STUB = np.linspace(100e9, 500e9, 41)

# element, z0, parameters (SI units), frequencies
CASES = [
    ('resistor_series', 50., [30.], F_WIDE),
    ('resistor_shunt', 50., [30.], F_WIDE),
    ('resistor_series', 75., [0.], F_WIDE),
    ('capacitor_series', 50., [100e-15], F_WIDE),
    ('capacitor_shunt', 50., [100e-15], F_WIDE),
    ('inductor_series', 50., [50e-12], F_WIDE),
    ('inductor_shunt', 20., [50e-12], F_WIDE),
    ('series_rlc_series', 50., [5., 40e-12, 300e-15], F_WIDE),
    ('series_rlc_shunt', 20., [5., 40e-12, 300e-15], F_WIDE),
    ('series_rlc_series', 50., [5., 40e-12, 0.], F_WIDE),          # C = 0: open circuit
    ('parallel_rlc_series', 50., [200., 40e-12, 300e-15], F_WIDE),
    ('parallel_rlc_shunt', 50., [200., 40e-12, 300e-15], F_WIDE),
    ('series_tee', 50., [], F_WIDE),
    ('branch', 50., [4], F_WIDE),
    ('zterm', 50., [20., -30.], F_WIDE),
    ('zterm', 50., [0., 0.], F_WIDE),
    ('yterm', 50., [0.01, 0.02], F_WIDE),
    ('yterm', 50., [0., 0.], F_WIDE),
    ('attenuator', 50., [6.], F_WIDE),
    ('time_delay', 50., [3e-12], F_WIDE),
    ('transformer', 50., [20., 5., 80., -10.], F_WIDE),
    ('transformer', 10., [3., 0., 12., 0.], F_WIDE),
    ('circulator', 50., [], F_WIDE),
    ('hybrid90', 50., [], F_WIDE),
    ('hybrid180', 50., [], F_WIDE),
    ('power_divider', 50., [], F_WIDE),
    ('slot_antenna', 50., [35., 20.], F_WIDE),
    ('twin_slot_1', 50., [35., 20.], F_WIDE),
    ('twin_slot_2', 50., [30., 10e-12, 1e-12], F_WIDE),
    ('trline', 50., [40., -1., 1.3, 230e9, 0.05], F_WIDE),
    ('deembed', 50., [60., 1.1, 230e9, 30e-15, 10., 20e-12, 500e-15], F_WIDE),
    ('r_waveguide', 50., [1.0e-3, 0.5e-3, 2e-3, 1., 0., 0., 0.], F_WG),
    ('r_waveguide', 50., [1.0e-3, 0.5e-3, 2e-3, 2.1, 1e-3, 1.7e-8, 2e-6], F_WG),
    ('cpw', 50., [10e-6, 5e-6, 200e-6, 11.9, 0, 1e-3], F_WIDE),
    ('cpw', 50., [10e-6, 5e-6, 200e-6, 11.9, 1, 1e-3], F_WIDE),
    ('cpw', 50., [30e-6, 3e-6, 50e-6, 4.5, 1, 1e-3], F_WIDE),
    ('radial_stub', 50., [60e-6, 70 * np.pi / 180, 10e-6, 3e-6, 0.25e-6, 3.8, 5e-9, 300e-9, 300e-9], F_STUB),
    ('via', 50., [200e-6, 300e-6], F_WIDE[:1]),
]


def supermix(element, z0, params, freq):
    """Run smx_elements: S[k, i, j] (or the inductance for 'via')."""

    lines = ['{} {:.17g} {}'.format(element, z0, len(freq))] + ['{:.17g}'.format(f) for f in freq]
    lines.append(' '.join('{:.17g}'.format(p) for p in params))
    out = subprocess.run([PROGRAM], input='\n'.join(lines) + '\n', capture_output=True, text=True, check=True)
    if element == 'via':
        return float(out.stdout.split()[0])
    rows = np.array([[float(x) for x in line.split()] for line in out.stdout.splitlines()])
    vals = rows[:, 1::2] + 1j * rows[:, 2::2]
    n = int(round(np.sqrt(vals.shape[1])))
    return vals.reshape(len(freq), n, n)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true', help="save SuperMix's results for the tests")
    args = parser.parse_args()

    for module in (qmix.network, qmix.superconductor, mattisbardeen):
        module.sc = SUPERMIX_CONSTANTS

    data = {'names': np.array(['case{}'.format(k) for k in range(len(CASES))])}
    worst = 0.
    for k, (element, z0, params, freq) in enumerate(CASES):
        ref = supermix(element, z0, params, freq)
        if element == 'via':
            err = abs(qmix_element(element, z0, params, freq) - ref) / ref
        else:
            net = qmix_element(element, z0, params, freq)
            assert np.allclose(net.z0, z0), "QMix's reference impedances differ"
            err = np.abs(net.s - ref).max()
        worst = max(worst, err)
        print('case{:<3d} {:<20s} z0 {:>4.0f}  max |QMix - SuperMix| = {:.1e}'.format(k, element, z0, err))
        data.update({'case{}_element'.format(k): element, 'case{}_z0'.format(k): z0,
                     'case{}_params'.format(k): np.array(params, dtype=float), 'case{}_freq'.format(k): freq,
                     'case{}_s'.format(k): ref})
    print('Worst: {:.1e}'.format(worst))

    if args.save:
        np.savez_compressed(OUTPUT, **data)
        print('Saved {}'.format(os.path.relpath(OUTPUT)))


if __name__ == '__main__':
    main()
