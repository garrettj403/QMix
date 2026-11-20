""" Test the S-parameter networks (qmix.network) against analytic circuits."""

import numpy as np
import pytest

from qmix.network import Network, embedding_impedance, read_touchstone

FREQ = np.linspace(1e9, 300e9, 300)


def _line(zc, length, freq=FREQ, eps=1., loss=0.):
    """Transmission line (characteristic impedance zc, length in m), with
    reference impedance zc at both ports."""
    beta = 2 * np.pi * freq * np.sqrt(eps) / 299792458. * (1 - 1j * loss)
    t = np.exp(-1j * beta * length)
    s = np.zeros((freq.size, 2, 2), dtype=complex)
    s[:, 0, 1] = s[:, 1, 0] = t
    return Network(freq, s, zc), beta


def _zin_line(zc, beta, length, z_load):
    tan = np.tan(beta * length)
    return zc * (z_load + 1j * zc * tan) / (zc + 1j * z_load * tan)


def _random_network(n, seed=1, nf=5):
    rng = np.random.default_rng(seed)
    s = (rng.normal(size=(nf, n, n)) + 1j * rng.normal(size=(nf, n, n))) * 0.3
    return Network(np.linspace(1e9, 5e9, nf), s, 50.)


def _z_matrix(ntwk):
    """Impedance matrix (power waves)."""
    out = []
    for s, g in zip(ntwk.s, ntwk.z0):
        f = np.diag(1 / (2 * np.sqrt(g.real)))
        sp = np.linalg.inv(f) @ s @ f
        out.append(np.linalg.solve(np.eye(len(g)) - sp, np.diag(np.conj(g)) + sp @ np.diag(g)))
    return np.array(out)


def test_line_impedance():
    """Terminated line, renormalized to 50 ohms, vs the textbook formula."""

    zc, length, z_load = 30., 2.3e-3, 12. - 7j
    line, beta = _line(zc, length, loss=0.01)
    line50 = line.renormalize(50.)
    assert np.allclose(line50.z0, 50.)
    z_in = line50.terminate(2, z_load).impedance(1)
    np.testing.assert_allclose(z_in, _zin_line(zc, beta, length, z_load), rtol=1e-12)
    np.testing.assert_allclose(line50.impedance(1, {2: z_load}), z_in, rtol=1e-12)

    # Open circuit and short circuit
    np.testing.assert_allclose(line50.impedance(1, {2: 0.}), _zin_line(zc, beta, length, 0.), rtol=1e-10)
    np.testing.assert_allclose(line50.impedance(1, {2: np.inf}), -1j * zc / np.tan(beta * length), rtol=1e-9)

    # Unlisted ports are matched (here: to the reference impedance of the line itself)
    np.testing.assert_allclose(line.impedance(1), zc, rtol=1e-12)


def test_renormalize():
    """Renormalizing preserves the impedance matrix and is reversible,
    including complex reference impedances."""

    ntwk = _random_network(3)
    z0_new = np.array([25., 75. + 10j, 13.3 - 2j])
    new = ntwk.renormalize(z0_new)
    np.testing.assert_allclose(_z_matrix(new), _z_matrix(ntwk), rtol=1e-10)
    np.testing.assert_allclose(new.renormalize(50.).s, ntwk.s, atol=1e-12)

    # Per-frequency reference impedances
    z0_f = np.linspace(20, 80, 5)[:, None] * np.ones(3)
    np.testing.assert_allclose(_z_matrix(ntwk.renormalize(z0_f)), _z_matrix(ntwk), rtol=1e-10)


def test_connect():
    """Two lines connected = one long line; mismatched references are
    handled; port order of the result."""

    zc, l1, l2 = 30., 1.1e-3, 0.7e-3
    line1, beta = _line(zc, l1)
    line2, _ = _line(zc, l2)
    both = line1.renormalize(50.).connect(2, line2.renormalize([75., 20.]), 1)
    z_in = both.impedance(1, {2: 10.})
    np.testing.assert_allclose(z_in, _zin_line(zc, beta, l1 + l2, 10.), rtol=1e-10)
    np.testing.assert_allclose(both.z0[:, 1], 20.)

    # Three-port tee from a random network: connecting a load via a line = terminating
    # with the transformed load
    tee = _random_network(3, nf=FREQ.size)
    tee.freq = FREQ
    z_load = 40. + 5j
    via_line = tee.connect(2, line1.renormalize(50.), 1).terminate(3, z_load)  # ports: 1, 3(tee)
    direct = tee.terminate(2, _zin_line(zc, beta, l1, z_load))
    np.testing.assert_allclose(via_line.s, direct.s, atol=1e-12)

    with pytest.raises(AssertionError):
        line1.connect(2, _random_network(2), 1)


def test_connect_below_cutoff():
    """Ports with imaginary reference impedances (waveguide below cutoff) are
    connected directly, and the other ports are not renormalized."""

    a, b = _random_network(2, seed=3), _random_network(2, seed=4)
    a.z0[:, 1] = b.z0[:, 0] = 0.2j
    a.z0[:, 0] = 30j
    both = a.connect(2, b, 1)
    np.testing.assert_allclose(both.z0[:, 0], 30j)
    # Direct connection: same as with real references of any (equal) value
    a.z0[:, 1] = b.z0[:, 0] = 50.
    a.z0[:, 0] = 50.
    np.testing.assert_allclose(both.s, a.connect(2, b, 1).s, atol=1e-12)
    assert np.isfinite(both.s).all()


def test_embedding_impedance():
    """Junction capacitance in parallel; interpolation between frequencies;
    range check."""

    zc, length = 20., 3e-3
    line, beta = _line(zc, length)
    cap = 100e-15
    z_emb = embedding_impedance(line.renormalize(50.), 1, cap=cap, terminations={2: 50.})
    f = np.array([100e9, 222.2e9])
    z_line = _zin_line(zc, 2 * np.pi * f / 299792458., length, 50.)
    expected = 1 / (1 / z_line + 2j * np.pi * f * cap)
    np.testing.assert_allclose(z_emb(f), expected, rtol=2e-4)
    np.testing.assert_allclose(z_emb(FREQ[10]), 1 / (1 / _zin_line(zc, beta[10], length, 50.)
                                                     + 2j * np.pi * FREQ[10] * cap), rtol=1e-12)
    with pytest.raises(ValueError):
        z_emb(400e9)


def test_interpolate():

    line, _ = _line(30., 1e-3)
    new = line.interpolate([10e9, 20.5e9])
    assert new.s.shape == (2, 2, 2)
    np.testing.assert_allclose(new.s[0], line.s[np.argmin(abs(FREQ - 10e9))], atol=1e-12)
    with pytest.raises(ValueError):
        line.interpolate(0.5e9)


@pytest.mark.parametrize("fmt", ['MA', 'DB', 'RI'])
def test_read_touchstone(tmp_path, fmt):
    """Write networks in each format and read them back (2-port column order,
    multi-line 3-port records, option line)."""

    for n in (2, 3):
        ntwk = _random_network(n, seed=n)
        lines = ['! test file', '# MHz S {} R 75'.format(fmt)]
        for f, s in zip(ntwk.freq, ntwk.s):
            vals = s.T.flatten() if n == 2 else s.flatten()
            if fmt == 'MA':
                pairs = [(abs(v), np.angle(v, deg=True)) for v in vals]
            elif fmt == 'DB':
                pairs = [(20 * np.log10(abs(v)), np.angle(v, deg=True)) for v in vals]
            else:
                pairs = [(v.real, v.imag) for v in vals]
            nums = ['{:.15e} {:.15e}'.format(*p) for p in pairs]
            if n == 2:
                lines.append('{:.6f} '.format(f / 1e6) + ' '.join(nums) + ' ! comment')
            else:
                for row in range(n):
                    prefix = '{:.6f} '.format(f / 1e6) if row == 0 else '   '
                    lines.append(prefix + ' '.join(nums[row * n:(row + 1) * n]))
        path = tmp_path / 'test.s{}p'.format(n)
        path.write_text('\n'.join(lines) + '\n')
        new = read_touchstone(str(path))
        np.testing.assert_allclose(new.freq, ntwk.freq)
        np.testing.assert_allclose(new.s, ntwk.s, atol=1e-12)
        np.testing.assert_allclose(new.z0, 75.)


def test_read_touchstone_hfss_port_impedance(tmp_path):
    """HFSS files that are not renormalized: per-frequency port impedances in
    comments, no R in the option line."""

    text = """! Exported from HFSS
# GHZ S MA
! Modal data exported
1.0 0.5 10 0.1 20 0.1 20 0.4 30
! Gamma  !      1 0 2 0
! Port Impedance  288.4 0 60.7 -0.003
2.0 0.6 11 0.2 21 0.2 21 0.3 31
! Gamma  !      1 0 2 0
! Port Impedance  280.0 0 60.0 -0.002
"""
    path = tmp_path / 'hfss.s2p'
    path.write_text(text)
    ntwk = read_touchstone(str(path))
    np.testing.assert_allclose(ntwk.z0, [[288.4, 60.7 - 0.003j], [280., 60. - 0.002j]])
    np.testing.assert_allclose(ntwk.s[1, 0, 1], 0.2 * np.exp(1j * np.deg2rad(21)))

    # Renormalized files keep the R of the option line
    path.write_text(text.replace('# GHZ S MA', '# GHZ S MA R 50'))
    np.testing.assert_allclose(read_touchstone(str(path)).z0, 50.)

    with pytest.raises(ValueError):
        path.write_text(text.replace('# GHZ S MA', '# GHZ Y MA'))
        read_touchstone(str(path))
    path.write_text(text)
    with pytest.raises(AssertionError):
        read_touchstone(str(path), num_ports=3)
