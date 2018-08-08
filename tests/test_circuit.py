"""This module really doesn't need any tests..."""

import pytest
import numpy as np
from qmix.circuit import *
import tempfile


def test_matrix_sizes():

    Nf = 3
    Np = 2
    npts = 101

    cct = EmbeddingCircuit(Nf, Np, vb_npts=npts)

    results = cct.vph
    msize = np.shape(results)
    ndims = results.ndim
    assert msize[0] == Nf+1
    assert ndims == 1

    results = cct.vt
    msize = np.shape(results)
    ndims = results.ndim
    assert msize[0] == Nf+1
    assert msize[1] == Np+1
    assert ndims == 2

    results = cct.vt
    msize = np.shape(results)
    ndims = results.ndim
    assert msize[0] == Nf+1
    assert msize[1] == Np+1
    assert ndims == 2

    results = cct.initialize_vj()
    msize = np.shape(results)
    ndims = results.ndim
    assert msize[0] == Nf+1
    assert msize[1] == Np+1
    assert msize[2] == npts
    assert ndims == 3


def test_printing():

    Nf = 3
    Np = 2
    npts = 101

    cct = EmbeddingCircuit(Nf, Np, fgap=600e9, vgap=2.7e-3, rn=15)
    cct.vph[1] = 0.3
    cct.vph[2] = 0.4
    cct.vph[3] = 0.5
    cct.vt[1:, 1:] = np.ones((3, 2)) * 0.5
    cct.zt[1:, 1:] = np.ones((3, 2)) * (0.3 - 1j * 0.3)
    cct.print_info()

    _, path = tempfile.mkstemp()
    cct.save_info(path)


def test_input_values():

    with pytest.raises(AssertionError):
        cct = EmbeddingCircuit(-1, -1)
    with pytest.raises(AssertionError):
        cct = EmbeddingCircuit(-1, 1)
    with pytest.raises(AssertionError):
        cct = EmbeddingCircuit(1, -1)
    with pytest.raises(AssertionError):
        cct = EmbeddingCircuit(0, 0)
