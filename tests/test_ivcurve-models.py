"""Test the I-V curve models (qmix.mathfn.ivcurve_models).

"""

import numpy as np
import pytest

import qmix.mathfn.ivcurve_models as iv
from qmix.mathfn.misc import slope

# Model parameters
vgap  = 3e-3
rn    = 14.
rsg   = 5e2
agap  = 5e4
a0    = 1e4
ileak = 5e-6
vnot  = 2.85e-3
inot  = 1e-5
anot  = 2e4
ioff  = 1.2e-5


def test_perfect_iv_model():
    """Test the `perfect' I-V curve model."""

    # Test I-V curve model
    x = np.array([-2., -1., 0., 1., 2.])
    y = iv.perfect(x)
    np.testing.assert_equal(y, [-2., -0.5, 0., 0.5, 2.])

    # Test KK transform
    x = np.array([-1e10, -1, 1, 1e10])
    y = iv.perfect_kk(x)
    assert np.abs(y[0]) <  1e-7
    assert np.abs(y[1]) ==  100.
    assert np.abs(y[2]) ==  100.
    assert np.abs(y[3]) <  1e-7


class TestExponentialModel:
    """Test the exponential I-V curve model that is used by the Chalmers goup.

    For example, see:

        H. Rashid, S. Krause, D. Meledin, V. Desmaris, A. Pavolotsky, and 
        V. Belitsky, "Frequency Multiplier Based on Distributed 
        Superconducting Tunnel Junctions: Theory, Design, and 
        Characterization," IEEE Trans. Terahertz Sci. Technol., pp. 1-13, 
        2016.

    """

    def test_corrected_exponential_model(self):
        """Test corrected model."""

        # Generate I-V curve
        x = np.linspace(-3, 3, 201)
        y = iv.exponential(x, vgap=vgap, rn=rn, rsg=rsg, agap=agap)
        x *= vgap
        y *= (vgap / rn)

        # Calculate resistance
        dxdy = slope(y, x)

        # Check subgap resistance
        idx = np.abs(x).argmin()
        assert round(dxdy[idx], 10) == rsg

        # Check normal resistance
        idx = np.abs(x - 7.5e-3).argmin()
        assert round(dxdy[idx], 10) == rn

    def test_original_exponential_model(self):
        """Test original model."""

        # Generate I-V curve
        x = np.linspace(-3, 3, 201)
        y = iv.exponential(x, vgap=vgap, rn=rn, rsg=rsg, agap=agap, model='original')
        x *= vgap
        y *= (vgap / rn)

        # Calculate resistance
        dxdy = slope(y, x)

        # Check subgap resistance
        idx = np.abs(x).argmin()
        assert rsg / 2 - 1 < round(dxdy[idx], 10) < rsg / 2 + 1

        # Check normal resistance
        idx = np.abs(x - 7.5e-3).argmin()
        assert rn - 1 < round(dxdy[idx], 10) < rn

    def test_other_models(self):
        """Try using other model type."""

        # Generate I-V curve
        x = np.linspace(-3, 3, 201)
        with pytest.raises(ValueError):
            y = iv.exponential(x, vgap=vgap, rn=rn, rsg=rsg, agap=agap, model='other')


def test_expanded_model():
    """Test the expanded I-V curve model."""

    # Generate I-V curve
    x = np.linspace(-2, 2, 201)
    y = iv.expanded(x, vgap=vgap, rn=rn, rsg=rsg, agap=agap, a0=a0,
                    ileak=ileak, vnot=vnot, inot=inot, anot=anot, ioff=ioff)
    x *= vgap
    y *= (vgap / rn)

    # Calculate resistance
    dxdy = slope(y, x)

    # Calculate sub-gap resistance
    idx = np.abs(x - 2e-3).argmin()
    assert round(dxdy[idx], 2) == rsg

    # Calculate normal resistance
    idx = np.abs(x - 5e-3).argmin()
    assert round(dxdy[idx], 2) == rn
