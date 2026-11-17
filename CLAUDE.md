# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What QMix is

QMix simulates quasiparticle tunneling currents in SIS (Superconductor/Insulator/Superconductor) junctions using multi-tone spectral domain analysis (MTSDA). The math follows P. Kittara's 2002 DPhil thesis; inline comments reference specific equations from it (e.g., "Eqn. 5.25").

## Commands

```bash
pip install -e .                          # install for development (CI uses `pip install -e .[test]`)
pytest --verbose --color=yes tests/       # run test suite
pytest tests/test_qtcurrent.py::test_compare_qtcurrent_to_tucker_theory  # single test
pytest --cov=qmix/ --cov-report=html tests/   # coverage
flake8 qmix/                              # lint (what CI runs; config in .flake8)
sphinx-apidoc -o docs/source/ qmix/       # regenerate API docs after adding modules
```

Benchmark/profiling scripts live in `profile/benchmark/` and `profile/profile/`. Update `CHANGES.md` (unreleased section at top) for user-visible changes.

## Architecture

A simulation is a pipeline through four core modules:

1. **`qmix/respfn.py`** — `RespFn` classes wrap a DC I-V curve and its Kramers-Kronig transform (the "response function") behind cubic-spline interpolation. Subclasses (`RespFnPolynomial`, `RespFnExponential`, `RespFnPerfect`, `RespFnFromIVData`) differ only in the I-V model used (models in `qmix/mathfn/ivcurve_models.py`, KK transform in `qmix/mathfn/kktrans.py`).
2. **`qmix/circuit.py`** — `EmbeddingCircuit` holds the Thevenin equivalent circuit: frequencies (`freq`), voltages (`vt`), impedances (`zt`) for up to 4 tones (`num_f`) and their harmonics (`num_p`).
3. **`qmix/harmonic_balance.py`** — `harmonic_balance()` solves for the junction voltage `vj` self-consistently via Newton's method: each iteration builds a numerical Jacobian (2·num_n + 1 calls to `qtcurrent`) and applies the correction.
4. **`qmix/qtcurrent.py`** — `qtcurrent()` computes the tunneling current for given `vj` at requested output frequencies. This is the compute core.

`qmix/exp/` analyzes experimental data (I-V and IF spectra), fitting it into the same pipeline via `RespFnFromIVData`; `exp_data.py` is its main entry point.

Worked end-to-end examples (single-tone and multi-tone simulations, experimental-data analysis) are in `notebooks/`; they mirror the examples on the documentation website.

### Conventions that span the codebase

- **Everything is normalized**: voltages to the gap voltage `Vgap`, currents to `Igap = Vgap/Rn`, frequencies to the gap frequency. Un-normalized values appear only at the `exp/` and `RespFn` construction boundaries.
- **Array index order** (per CODE_GUIDELINES.md): frequency/tone `f`, harmonic `p`, summation index `k`, bias-voltage point `i` — e.g. `vt[f, p]`, `ckh[f, p, k, i]`. The `f` and `p` axes are **1-indexed**; row/column 0 is unused padding.
- **Wraparound indexing for summation indices**: arrays over the Bessel summation index `k` (response-function matrices, phase-factor coefficients) store k = 0..num_b in positions 0..num_b and k = -num_b..-1 at the end, so code indexes them directly with negative `k`. Both NumPy and Numba handle this via negative-index wraparound. Keep this convention when touching `qtcurrent.py`.
- **Frequency matching by rounding**: output frequencies are matched to harmonic combinations `a·f1 + b·f2 + ...` after rounding to `ROUND_FREQ` (4) decimal places (defined in both `qtcurrent.py` and `harmonic_balance.py`). Tones or IFs closer than ~1e-4 (normalized) will alias.
- **Per-tone-count specialization**: the current functions in `qtcurrent.py` have separate hand-written implementations for 1, 2, 3, and 4 tones (`_current_coeff_1_tone` ... `_current_coeff_4_tones`). A change to one usually needs mirroring in the others. (The response-function interpolation is general: `_interpolation_voltages` builds the voltage grid for any number of tones.) Maximum 4 non-harmonically related tones.
- **Numba kernels**: the hot kernels use `@nb.njit` with explicit signatures (eager compilation) and `cache=True`. They carry `# pragma: no cover` because coverage can't trace compiled code — the surrounding pure-Python wrappers are what tests exercise directly. Kernels avoid temporary array allocations (explicit loops over bias points, pair products pre-computed with `_pair_products`, invariant products hoisted per loop level); preserve that style when editing. The 3- and 4-tone current coefficient kernels use `parallel=True` with `prange` over flattened (k, l) pairs, each accumulating into its own row (no shared writes); keep the bias-point loop innermost so memory access stays contiguous. `RespFn.resp()` evaluates a complex piecewise polynomial (`_setup_resp_poly` / `_eval_resp_poly` in `respfn.py`) rather than the splines; `RespFnPerfect` without smearing has no polynomial and falls back to its analytic functions. With `cache=True`, never compile the same Python function twice with different flags (e.g., serial and `parallel=True`): the cache keys on function and signature, so one silently loads the other.

## Code style

- Google-style docstrings for public modules/functions/classes; private (`_`-prefixed) functions may use inline comments instead.
- Dependencies restricted to the standard library, NumPy, SciPy, Numba, and Matplotlib.
- ASCII only (no unicode in source); American spelling.
- flake8 line limit is 120; ignores include alignment-related codes (E221, E222, E251), so vertically aligned assignments are fine and already common in the code.
