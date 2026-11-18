# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What QMix is

QMix simulates quasiparticle tunneling currents in SIS (Superconductor/Insulator/Superconductor) junctions using multi-tone spectral domain analysis (MTSDA). The math follows P. Kittara's 2002 DPhil thesis; inline comments reference specific equations from it (e.g., "Eqn. 5.25"). The source papers are in `reference/`: the PDF with the hashed name is Chapter 5 of the thesis (pp. 113-156), the others are Withington & Kollberg 1989 (single-tone spectral method, harmonic balance) and Kittara, Withington & Yassin 2007 (multitone theory applied to high-harmonic mixers). Text extraction drops the thesis equations; render the pages to images with Ghostscript (`gs -sDEVICE=pnggray`) to read them.

## Commands

```bash
pip install -e .                          # install for development (CI uses `pip install -e .[test]`)
pytest --verbose --color=yes tests/       # run test suite
pytest tests/test_qtcurrent.py::test_compare_qtcurrent_to_tucker_theory  # single test
pytest --cov=qmix/ --cov-report=html tests/   # coverage
flake8 qmix/                              # lint (what CI runs; config in .flake8)
sphinx-apidoc -o docs/source/ qmix/       # regenerate API docs after adding modules
```

Update `CHANGES.md` (unreleased section at top) for user-visible changes.

Benchmarks must be run from `profile/benchmark/` (they use relative paths). `benchmark-qtcurrent.py` also checks the 1-4 tone currents against stored reference values in `data/` to 14 decimals, so run it after touching any `qtcurrent` kernel. `-s` appends the timings to `results/*.txt`; leave that to the user unless asked. Profiling notebooks are in `profile/profile/`.

```bash
cd profile/benchmark && python benchmark-qtcurrent.py && python benchmark-harmonic-balance.py
```

Docs are Sphinx (`docs/source/*.rst`, built with `make html` in `docs/`, output goes to `../../QMix-docs/html/`); the simulation pages come from `notebooks/` via nbsphinx, and `fourier-shift-method.rst` is a hand-written theory page using `.. math::` (mathjax). Sphinx is not installed in the usual dev environment; to check a page, build it in a throwaway venv with a minimal `conf.py` (`extensions = ['sphinx.ext.mathjax']`) and `sphinx-build -W`.

## Architecture

A simulation is a pipeline through four core modules:

1. **`qmix/respfn.py`** — `RespFn` classes wrap a DC I-V curve and its Kramers-Kronig transform (the "response function") behind spline interpolation. For speed, `resp()` evaluates both splines as one complex piecewise polynomial (`_setup_resp_poly` / `_eval_resp_poly`); `RespFnPerfect` without smearing has no polynomial and uses its analytic functions instead. Subclasses (`RespFnPolynomial`, `RespFnExponential`, `RespFnPerfect`, `RespFnFromIVData`) differ only in the I-V model used (models in `qmix/mathfn/ivcurve_models.py`, KK transform in `qmix/mathfn/kktrans.py`).
2. **`qmix/circuit.py`** — `EmbeddingCircuit` holds the Thevenin equivalent circuit: frequencies (`freq`), voltages (`vt`), impedances (`zt`) for any number of tones (`num_f`) and their harmonics (`num_p`).
3. **`qmix/harmonic_balance.py`** — `harmonic_balance()` solves for the junction voltage `vj` self-consistently via Newton's method. A numerical Jacobian costs 2·num_n calls to `qtcurrent` (plus 1 for the error vector), so it is reused across iterations while each iteration still cuts the max relative error by `JACOBIAN_REUSE_RATIO` (`reuse_jacobian=False` gives plain Newton).
4. **`qmix/qtcurrent.py`** — `qtcurrent()` computes the tunneling current for given `vj` at requested output frequencies. This is the compute core: for 3+ tones nearly all simulation time is in `_current_coeff_n_tones`, whose cost (and the size of `resp_matrix`) scales with the product of (2·num_b + 1) over tones. `suggest_num_b(vj, cct)` picks per-tone limits from a solved `vj` (not `cct.vt`: IF tones can be driven by mixing even when `vt` is 0). When that product gets large (`method='auto'` switches above `MAX_RESP_MATRIX_BYTES`, or pass `method='fourier'`), the Fourier-shift path (`FourierResp` / `_current_fourier`) sums each tone separately at a cost that is additive in the tones; its accuracy is set by the spline smoothness of the response function (~1e-7 with cubic splines, ~1e-9 with `spline_order=5`), so it is not used for the 14-decimal benchmark checks. The theory is written up in `docs/source/fourier-shift-method.rst`. `interpolate_respfn` returns a `FourierResp` instead of a matrix in that case; `qtcurrent` accepts either as `resp_matrix`.

`qmix/exp/` analyzes experimental data (I-V and IF spectra), fitting it into the same pipeline via `RespFnFromIVData`; `exp_data.py` is its main entry point.

Worked end-to-end examples (single-tone and multi-tone simulations, experimental-data analysis) are in `notebooks/`; they mirror the examples on the documentation website.

### Conventions that span the codebase

- **Everything is normalized**: voltages to the gap voltage `Vgap`, currents to `Igap = Vgap/Rn`, frequencies to the gap frequency. Un-normalized values appear only at the `exp/` and `RespFn` construction boundaries.
- **Array index order** (per CODE_GUIDELINES.md): frequency/tone `f`, harmonic `p`, summation index `k`, bias-voltage point `i` — e.g. `vt[f, p]`, `ckh[f, p, k, i]`. The `f` and `p` axes are **1-indexed**; row/column 0 is unused padding.
- **Wraparound indexing for summation indices**: arrays over the Bessel summation index `k` (response-function matrices, phase-factor coefficients) store k = 0..num_b in positions 0..num_b and k = -num_b..-1 at the end, so code indexes them directly with negative `k`. Both NumPy and Numba handle this via negative-index wraparound. Keep this convention when touching `qtcurrent.py`.
- **Frequency matching by rounding**: output frequencies are matched to harmonic combinations `a·f1 + b·f2 + ...` after rounding to `ROUND_FREQ` (4) decimal places (defined in both `qtcurrent.py` and `harmonic_balance.py`). Tones or IFs closer than ~1e-4 (normalized) will alias.
- **One kernel for any number of tones**: `_current_coeff_n_tones` sums over `resp_matrix` reshaped to 2-D (all tone indices flattened in C order, bias points last). The outer tones' indices are decoded from the row number (mixed radix), their running pair products are updated odometer-style from the first index that changed since the previous row, and the last tone plus the bias points run over contiguous memory. Per-tone validity masks (`_pair_products_all`) replace the old "k + a in range" checks. Output frequencies are matched to index tuples by `_matching_tuples` (itertools product over |a_f| <= num_p; higher-order combinations that land exactly on an output frequency are deliberately not included, which matches the old behavior but not thesis Eqn. 5.36). Because the summation order depends on the number of tones, results for different tone counts agree only to ~1e-13: tests compare them with `assert_allclose`, not `assert_equal`.
- **Numba kernels**: the hot kernels use `@nb.njit` with explicit signatures (eager compilation) and `cache=True`. They carry `# pragma: no cover` because coverage can't trace compiled code — the surrounding pure-Python wrappers are what tests exercise directly. Kernels avoid temporary array allocations (explicit loops over bias points, pair products pre-computed with `_pair_products`, invariant products hoisted per loop level); preserve that style when editing.
- **Parallel kernels**: `_current_coeff_n_tones` and `_eval_resp_poly` use `parallel=True` with `prange`. The current kernel parallelizes over chunks of rows, each accumulating into its own slot (no shared writes); keep the bias-point loop innermost so memory access stays contiguous (parallelizing over bias points instead was ~3x slower). With `cache=True`, never compile the same Python function twice with different flags (e.g., serial and parallel): the cache keys on function and signature, so one silently loads the other. With Anaconda's Numba (OpenMP threading layer), a harmless `OMP: Info #276 ... omp_set_nested` notice prints once per process.

## Code style

- Google-style docstrings for public modules/functions/classes; private (`_`-prefixed) functions may use inline comments instead.
- Dependencies restricted to the standard library, NumPy, SciPy, Numba, and Matplotlib.
- ASCII only (no unicode in source); American spelling.
- flake8 line limit is 120; ignores include alignment-related codes (E221, E222, E251), so vertically aligned assignments are fine and already common in the code.
