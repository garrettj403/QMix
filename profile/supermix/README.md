# QMix vs SuperMix

Cross-check of QMix against [SuperMix](http://www.submm.caltech.edu/supermix/)
v1.6, Caltech's C++ library for SIS mixers (Ward, Rice, Chattopadhyay &
Zmuidzinas 1999). SuperMix implements the single-tone spectral-domain method of
Withington & Kollberg (1989) with an analytic-Jacobian Newton harmonic balance
(Rice et al. 1999), independently of QMix's MTSDA code, so agreement between
the two is a strong check of `qtcurrent` and `harmonic_balance`.

The SuperMix source is not part of this repository. These scripts expect a copy
at `reference/supermix` (git-ignored), or set `SUPERMIXDIR`.

## Files

- `build.sh`: builds `libMiX.a` into `$SUPERMIXDIR/obj.<arch>` with modern
  clang/libc++ (macOS) and compiles programs into `bin/`.
- `compat/`: shims needed because SuperMix was written for g++ 2.x-5: a forced
  include (`prefix.h`) that loads the std headers before renaming SuperMix's
  `complex` alias and `ratio` array (both clash with libc++ under
  `using namespace std`), and stand-ins for two libstdc++-internal headers. The
  SuperMix source itself is unmodified.
- `smx_driver.cc`: runs SuperMix on a job from stdin, in normalized QMix units
  (peak phasors). Mode `current` computes the tunneling currents for a given
  junction voltage; mode `balance` runs `mixer::balance()` with a Thevenin
  source per harmonic (a small custom `nport`) and an ideal dc bias.
- `tight_ckdata.py`: generates a copy of SuperMix's `ckdata.cc` with the Bessel
  tolerance at 1e-14 instead of 1e-6, for the `--tight` build.
- `compare.py`: exports a QMix response function as SuperMix I-V files, runs
  both codes on the same inputs and prints the differences.

## Usage

```bash
cd profile/supermix
./build.sh --tight smx_driver.cc                     # library + both drivers
python compare.py                                    # stock SuperMix
SMX_DRIVER=bin/smx_driver_tight python compare.py    # tight Bessel tolerance
python compare.py --plot                             # also save plots to work/
```

SuperMix's own tests can be run against their stored outputs (e.g.,
`./build.sh $SUPERMIXDIR/test/src/test_balance.cc`, then run `bin/test_balance`
from `$SUPERMIXDIR/test/datafiles` and diff against
`$SUPERMIXDIR/test/compare.i386-intel-linux/test_balance.base`; arguments for
the others are in `$SUPERMIXDIR/test/cfast.all`).

## Results (Nov 2026)

Maximum |QMix - SuperMix| in normalized units over each bias sweep (polynomial
I-V curve, order 50):

| Case | stock SuperMix, QMix resp | stock, symmetrized resp | tight SuperMix, symmetrized resp |
|---|---|---|---|
| current, 1 harmonic, alpha 0.5 | 9.3e-09 | 3.0e-09 | 2.3e-15 |
| current, 1 harmonic, alpha 1.0 | 1.2e-07 | 2.1e-09 | 1.4e-15 |
| current, 1 harmonic, alpha 3.0 | 5.5e-06 | 2.2e-10 | 2.2e-15 |
| current, 3 harmonics with phases | 1.1e-06 | 2.2e-08 | 1.8e-15 |
| balance, 1 harmonic | 4.5e-07 | 1.0e-08 | 1.0e-12 |
| balance, 3 harmonics | 1.0e-06 | 3.7e-07 | 2.0e-10 |

- With the same I-V curve and Bessel truncation, the tunneling currents agree
  to machine precision, and the harmonic balance to QMix's stopping tolerance.
  The two codes also use the same phasor and phase conventions.
- The remaining differences in the stock comparison come from two sources:
  - QMix's interpolated response function is only symmetric (Idc odd, Ikk even)
    to within its interpolation error (up to 5e-5 in Ikk near the gap), because
    its spline knots are not symmetric about v = 0. SuperMix mirrors the v >= 0
    data. This dominates the ac current differences, which grow with pump
    level since larger alpha samples more negative voltages.
  - SuperMix truncates the Bessel functions and phase factors at 1e-6.
- The table step divides every voltage `vb + k * flo` evaluated here, so
  SuperMix's re-interpolation of the exported table does not contribute.
- Speed: SuperMix balances a 199-point sweep in a few ms (4-7 Newton iterations
  per point, about 15 us each), versus a few tenths of a second for QMix (two
  harmonic-balance passes; vectorized over bias, numerical Jacobian).
