"""Compare QMix's superconducting films and microstrip lines with SuperMix.

Runs SuperMix's ``super_film`` / ``normal_film`` surface impedances and its
``microstrip`` class (``smx_lines.cc``) for several lines and compares the
conductivity, surface impedances, characteristic impedance and propagation
constant with ``qmix.superconductor`` and ``qmix.network.microstrip``.

Usage (from this directory, after ``./build.sh smx_lines.cc``)::

    python compare_lines.py
    python compare_lines.py --save   # also write tests/supermix-data/supermix_lines.npz

SuperMix uses the CODATA 1986 values of e, k and h, and mu_0 = 4 pi 1e-7, so
QMix is run with the same constants (``--scipy`` uses SciPy's instead). The
temperature dependence of the gap is the remaining difference: SuperMix
interpolates Muhlschlegel's 1959 table (4 digits), QMix solves the BCS gap
equation. Below T / Tc = 0.04 both give exactly 1, so the cases at 0.3 K
isolate everything else.

"""

import argparse
import os
import subprocess
from types import SimpleNamespace

import numpy as np
import scipy.constants as sc

import qmix.network
import qmix.superconductor
from qmix.network import microstrip
from qmix.superconductor import NormalFilm, SuperconductingFilm

HERE = os.path.dirname(os.path.abspath(__file__))
PROGRAM = os.environ.get('SMX_LINES', os.path.join(HERE, 'bin', 'smx_lines'))
OUTPUT = os.path.join(HERE, '..', '..', 'tests', 'supermix-data', 'supermix_lines.npz')

SUPERMIX_CONSTANTS = SimpleNamespace(e=1.60217733e-19, k=1.380658e-23, h=6.6260755e-34,
                                     hbar=6.6260755e-34 / (2 * np.pi), mu_0=4e-7 * np.pi, c=sc.c)

NB = dict(vgap=2.8e-3, tc=9.2, rho=5e-8)       # Nb, 5 uOhm cm
NBTIN = dict(vgap=4.9e-3, tc=14.4, rho=1e-6)   # NbTiN, 100 uOhm cm

# name, film type, film parameters, t_strip, t_ground, width, height, eps, tand, eps_up, tand_up, temp
CASES = [
    ('Nb/SiO2/Nb, 0.3 K', 'super', NB, 300e-9, 300e-9, 5e-6, 0.25e-6, 3.8, 0., 1., 0., 0.3),
    ('Nb/SiO2/Nb, 4.2 K', 'super', NB, 300e-9, 300e-9, 5e-6, 0.25e-6, 3.8, 0., 1., 0., 4.2),
    ('Nb/SiO/Nb narrow (u < 2), 0.3 K', 'super', NB, 400e-9, 200e-9, 0.6e-6, 0.4e-6, 5.6, 1e-3, 1., 0., 0.3),
    ('Nb/SiO/Nb wide (u > 5), 2 K', 'super', NB, 200e-9, 300e-9, 3e-6, 0.2e-6, 5.6, 0., 1., 0., 2.),
    ('NbTiN/SiO2/NbTiN, lossy, superstrate, 4.2 K', 'super', NBTIN, 300e-9, 300e-9, 2e-6, 0.3e-6, 3.8, 1e-3,
     2., 1e-4, 4.2),
    ('Thin Nb (50 nm), 0.3 K', 'super', NB, 50e-9, 50e-9, 4e-6, 0.3e-6, 3.8, 0., 1., 0., 0.3),
    ('Al, normal, 4.2 K', 'normal', dict(vgap=0., tc=0., rho=5e-9), 300e-9, 300e-9, 5e-6, 0.25e-6, 3.8, 0., 1., 0.,
     4.2),
]

FREQ = np.unique(np.r_[np.geomspace(10e9, 3e12, 61), [670e9, 676e9, 677e9, 678e9, 690e9]])


def supermix(ftype, film, t1, t2, w, h, eps, tand, eps_up, tand_up, temp, freq):
    """Run smx_lines; returns sigma, zs_strip, zs_ground, zchar, kprop."""

    lines = ['{} {:.17g} {:.17g} {:.17g}'.format(ftype, film['vgap'], film['tc'], film['rho']),
             ' '.join('{:.17g}'.format(x) for x in (t1, t2, w, h, eps, tand, eps_up, tand_up, temp)),
             str(len(freq))] + ['{:.17g}'.format(f) for f in freq]
    out = subprocess.run([PROGRAM], input='\n'.join(lines) + '\n', capture_output=True, text=True, check=True)
    rows = np.array([[float(x) for x in line.split()] for line in out.stdout.splitlines()])
    vals = rows[:, 1::2] + 1j * rows[:, 2::2]

    return vals.T


def qmix_lines(ftype, film, t1, t2, w, h, eps, tand, eps_up, tand_up, temp, freq):
    """The same with QMix; returns sigma, zs_strip, zs_ground, zchar, kprop."""

    if ftype == 'super':
        strip = SuperconductingFilm(film['vgap'], film['tc'], film['rho'], t1)
        ground = SuperconductingFilm(film['vgap'], film['tc'], film['rho'], t2)
    else:
        strip = NormalFilm(film['rho'], t1)
        ground = NormalFilm(film['rho'], t2)
    zc, gamma = microstrip(freq, w, h, eps, strip=strip, ground=ground, temp=temp, tan_d=tand, eps_up=eps_up,
                           tan_up=tand_up)

    return strip.sigma(freq, temp), strip.zs(freq, temp), ground.zs(freq, temp), zc, gamma


def rel(a, b):
    return np.max(np.abs(a - b) / np.abs(b))


def main():

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--scipy', action='store_true', help="use SciPy's constants in QMix")
    parser.add_argument('--save', action='store_true', help="save SuperMix's results for the tests")
    args = parser.parse_args()
    if not args.scipy:
        qmix.superconductor.sc = SUPERMIX_CONSTANTS
        qmix.network.sc = SUPERMIX_CONSTANTS

    print("Max relative differences, QMix vs SuperMix, {} frequencies from {:.0f} GHz to {:.0f} THz{}".format(
          FREQ.size, FREQ[0] / 1e9, FREQ[-1] / 1e12, " (SciPy constants)" if args.scipy else ""))
    print("{:46s} {:>9s} {:>9s} {:>9s} {:>9s} {:>9s}".format('case', 'sigma', 'Zs strip', 'Zs gnd', 'Zchar',
                                                             'gamma'))
    data = {'freq': FREQ, 'names': np.array([c[0] for c in CASES])}
    for k, (name, ftype, film, *rest) in enumerate(CASES):
        case = (ftype, film, *rest)
        smx = supermix(*case, FREQ)
        qmx = qmix_lines(*case, FREQ)
        diffs = [rel(q, s) for q, s in zip(qmx, smx)]
        print("{:46s} ".format(name) + " ".join("{:9.1e}".format(d) for d in diffs))
        data['case{}_inputs'.format(k)] = np.array([film['vgap'], film['tc'], film['rho'], *rest])
        data['case{}_type'.format(k)] = ftype
        for key, val in zip(('sigma', 'zs_strip', 'zs_ground', 'zchar', 'gamma'), smx):
            data['case{}_{}'.format(k, key)] = val

    if args.save:
        np.savez_compressed(OUTPUT, **data)
        print('Saved {} ({:.0f} kB)'.format(os.path.relpath(OUTPUT, HERE), os.path.getsize(OUTPUT) / 1e3))


if __name__ == '__main__':
    main()
