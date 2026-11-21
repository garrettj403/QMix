""" Test the S-parameter networks (qmix.network) against analytic circuits."""

import numpy as np
import pytest

from qmix.circuit import EmbeddingCircuit, MultiJunctionCircuit
from qmix.network import Network, embedding_impedance, impedance_matrix, read_touchstone, set_thevenin, thevenin

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


def test_thevenin_matched_line():
    """Source through a matched line: Z_T = Z_c and |V_T|^2 / (8 Z_c) = P,
    with the phase delay of the line."""

    zc, length, power = 50., 3.1e-3, 40e-9
    line, beta = _line(zc, length)
    f = FREQ[10:20]
    vt, zt = thevenin(line, f, port=2, source_port=1, power=power, phase=0.3)
    np.testing.assert_allclose(zt, zc, rtol=1e-10)
    v_expected = 2 * np.sqrt(zc) * np.sqrt(2 * power) * np.exp(0.3j) * np.exp(-1j * beta[10:20] * length)
    np.testing.assert_allclose(vt, v_expected, rtol=1e-9)
    np.testing.assert_allclose(np.abs(vt) ** 2 / (8 * zc), power, rtol=1e-9)

    # No source
    vt, zt = thevenin(line, f, port=2)
    assert np.all(vt == 0)


@pytest.mark.parametrize("port,source,other", [(1, 3, 2), (3, 1, 2), (2, 3, 1)])
def test_thevenin_vs_circuit(port, source, other):
    """Random 3-port with complex reference impedances: the Thevenin
    equivalent vs solving the circuit from the impedance matrix (generator
    with internal impedance Z_ref at the source port, a load at the third
    port, the junction port open)."""

    ntwk = _random_network(3, seed=4).renormalize(np.array([25. + 3j, 75. - 10j, 13.3]))
    z_load, power = 7. - 4j, 1e-6
    vt, zt = thevenin(ntwk, ntwk.freq, port, source, power=power, terminations={other: z_load})
    np.testing.assert_allclose(zt, ntwk.impedance(port, {other: z_load}), rtol=1e-10)
    z_mat = _z_matrix(ntwk)
    j, s_, o = port - 1, source - 1, other - 1
    for k, z in enumerate(z_mat):  # Z_T: source replaced by its internal impedance
        zo = z[[s_, o]][:, [s_, o]] + np.diag([ntwk.z0[k, s_], z_load])
        x = np.linalg.solve(zo, -z[[s_, o], j])
        np.testing.assert_allclose(zt[k], z[j, j] + z[j, [s_, o]] @ x, rtol=1e-9)

    for k, z in enumerate(z_mat):
        zg = ntwk.z0[k, s_]
        vg = np.sqrt(8 * power * zg.real)
        # Unknown currents I_s, I_o (I_j = 0): V_s = vg - zg I_s, V_o = -z_load I_o
        a = np.array([[z[s_, s_] + zg, z[s_, o]], [z[o, s_], z[o, o] + z_load]])
        i_s, i_o = np.linalg.solve(a, [vg, 0.])
        v_open = z[j, s_] * i_s + z[j, o] * i_o
        np.testing.assert_allclose(vt[k], v_open, rtol=1e-9)


def test_thevenin_capacitance_and_set_thevenin():
    """Junction capacitance in parallel; set_thevenin normalizes and keeps the
    available power."""

    zc, length, power, cap = 20., 1.7e-3, 25e-9, 100e-15
    line, beta = _line(zc, length)
    f = 230e9
    vt0, zt0 = thevenin(line, f, 2, 1, power=power)
    vt, zt = thevenin(line, f, 2, 1, power=power, cap=cap)
    np.testing.assert_allclose(zt, 1 / (1 / zt0 + 2j * np.pi * f * cap), rtol=1e-12)
    np.testing.assert_allclose(vt, vt0 * zt / zt0, rtol=1e-12)
    np.testing.assert_allclose(zt, embedding_impedance(line, 2, cap=cap)(f), rtol=1e-12)

    cct = EmbeddingCircuit(1, 2, vgap=2.8e-3, rn=14.)
    cct.freq[1] = f / 2 / cct.fgap  # p = 2 is at 230 GHz
    vt_n, zt_n = set_thevenin(cct, line, f=1, p=2, port=2, source_port=1, power=power)
    np.testing.assert_allclose(vt_n * cct.vgap, vt0, rtol=1e-12)
    np.testing.assert_allclose(zt_n * cct.rn, zt0, rtol=1e-12)
    np.testing.assert_allclose(cct.available_power(1, 2), power, rtol=1e-9)

    with pytest.raises(AssertionError):
        thevenin(line, f, 2, 2, power=power)          # source at the junction port
    with pytest.raises(AssertionError):
        thevenin(line, f, 2, power=power)             # power without a source
    with pytest.raises(AssertionError):
        set_thevenin(EmbeddingCircuit(1, 1), line, source_port=1)  # no vgap / rn
    with pytest.raises(ValueError):
        thevenin(line, 500e9, 2, 1, power=power)      # outside the frequency range


def test_terminations_complex_reference():
    """Loads at ports with complex reference impedances vs the impedance
    matrix (this used to use the conjugate reference impedance)."""

    ntwk = _random_network(3, seed=4).renormalize(np.array([25. + 3j, 75. - 10j, 13.3 - 6j]))
    zl2, zl3 = 7. - 4j, 30. + 12j
    zin = ntwk.impedance(1, {2: zl2, 3: zl3})
    zin_t = ntwk.terminate(3, zl3).terminate(2, zl2).impedance(1)
    for k, z in enumerate(_z_matrix(ntwk)):
        x = np.linalg.solve(z[1:, 1:] + np.diag([zl2, zl3]), -z[1:, 0])
        np.testing.assert_allclose(zin[k], z[0, 0] + z[0, 1:] @ x, rtol=1e-10)
        np.testing.assert_allclose(zin_t[k], zin[k], rtol=1e-10)

    # Matched (unlisted) ports are terminated by their reference impedance
    np.testing.assert_allclose(ntwk.impedance(1), ntwk.impedance(1, {2: 75. - 10j, 3: 13.3 - 6j}), rtol=1e-12)


def test_thevenin_source_below_cutoff():
    """No source power below the cutoff of a waveguide port (imaginary
    reference impedance); above cutoff it works."""

    line, _ = _line(50., 2e-3)
    z0 = np.ones((FREQ.size, 2), dtype=complex) * 50.
    z0[FREQ < 100e9, 0] = 300j  # port 1 below cutoff
    ntwk = Network(line.freq, line.s, z0)
    with pytest.raises(ValueError):
        thevenin(ntwk, 50e9, port=2, source_port=1, power=1e-9)
    vt, zt = thevenin(ntwk, 200e9, port=2, source_port=1, power=1e-9)
    np.testing.assert_allclose(np.abs(vt) ** 2 / (8 * zt.real), 1e-9, rtol=1e-6)


def test_thevenin_multiple_ports_vs_circuit():
    """Several junction ports (random 4-port with complex reference
    impedances): the multi-port Thevenin equivalent vs solving the circuit
    from the impedance matrix (junction ports open, generator with internal
    impedance Z_ref at the source port, a load at the other port)."""

    ntwk = _random_network(4, seed=7).renormalize(np.array([25. + 3j, 75. - 10j, 13.3, 40. + 20j]))
    ports, source, other, z_load, power, phase = [3, 1], 4, 2, 7. - 4j, 1e-6, 0.4
    vt, zt = thevenin(ntwk, ntwk.freq, ports, source, power=power, phase=phase, terminations={other: z_load})
    assert vt.shape == (ntwk.freq.size, 2) and zt.shape == (ntwk.freq.size, 2, 2)
    jj, oo = [p - 1 for p in ports], [source - 1, other - 1]
    for k, z in enumerate(_z_matrix(ntwk)):
        zg = ntwk.z0[k, source - 1]
        z_oo = z[oo][:, oo] + np.diag([zg, z_load])
        z_th = z[jj][:, jj] - z[jj][:, oo] @ np.linalg.solve(z_oo, z[oo][:, jj])
        np.testing.assert_allclose(zt[k], z_th, rtol=1e-9)
        vg = np.sqrt(8 * power * zg.real) * np.exp(1j * phase)
        i_o = np.linalg.solve(z_oo, [vg, 0.])
        np.testing.assert_allclose(vt[k], z[jj][:, oo] @ i_o, rtol=1e-9)

    # One port in a list: the same as the single-port function
    for port in (1, 3):
        vt_1, zt_1 = thevenin(ntwk, 2.2e9, [port], source, power=power, cap=3e-12, terminations={other: z_load})
        vt_s, zt_s = thevenin(ntwk, 2.2e9, port, source, power=power, cap=3e-12, terminations={other: z_load})
        np.testing.assert_allclose(vt_1, [vt_s], rtol=1e-12)
        np.testing.assert_allclose(zt_1, [[zt_s]], rtol=1e-12)

    # Junction capacitances in parallel with the junction ports
    cap, f = np.array([2e-12, 5e-12]), 3.1e9
    vt_0, zt_0 = thevenin(ntwk, f, ports, source, power=power)
    vt_c, zt_c = thevenin(ntwk, f, ports, source, power=power, cap=cap)
    y = np.linalg.inv(zt_0) + np.diag(2j * np.pi * f * cap)
    np.testing.assert_allclose(zt_c, np.linalg.inv(y), rtol=1e-10)
    np.testing.assert_allclose(vt_c, np.linalg.inv(y) @ np.linalg.solve(zt_0, vt_0), rtol=1e-10)

    # No source
    vt_n, zt_n = thevenin(ntwk, f, ports)
    np.testing.assert_equal(vt_n, 0.)
    np.testing.assert_allclose(zt_n, thevenin(ntwk, f, ports, source, power=power)[1], rtol=1e-12)

    with pytest.raises(AssertionError):
        thevenin(ntwk, f, [1, 1])                       # the same port twice
    with pytest.raises(AssertionError):
        thevenin(ntwk, f, [1, 2], 2, power=power)       # source at a junction port
    with pytest.raises(ValueError):
        thevenin(ntwk.renormalize(np.array([1j, 50., 50., 50.])), f, [1, 2])  # invalid junction reference


def test_set_thevenin_multiple_junctions():
    """Two identical junctions in parallel, fed through an ideal tee from a
    source: the harmonic balance gives the same voltage as one junction with
    twice the area."""

    from qmix.harmonic_balance import harmonic_balance
    from qmix.respfn import RespFnPolynomial

    freq = np.linspace(200e9, 260e9, 4)
    s_tee = np.full((4, 3, 3), 2 / 3, dtype=complex)
    s_tee[:, [0, 1, 2], [0, 1, 2]] = -1 / 3
    tee = Network(freq, s_tee, 10.)
    power, cap = 30e-9, 50e-15

    cct = MultiJunctionCircuit(2, 1, 1, vb_npts=41, vgap=2.8e-3, rn=14.)
    cct.freq[1] = 230e9 / cct.fgap
    vt, zt = set_thevenin(cct, tee, port=[1, 2], source_port=3, power=power, cap=[cap, cap])
    assert vt.shape == (2,) and zt.shape == (2, 2)
    np.testing.assert_allclose(cct.vt[:, 1, 1], vt)
    single = EmbeddingCircuit(1, 1, vb_npts=41, vgap=2.8e-3, rn=7.)  # two junctions in parallel: Rn / 2
    single.freq[1] = cct.freq[1]
    set_thevenin(single, tee.terminate(2, np.inf), port=1, source_port=2, power=power, cap=2 * cap)
    np.testing.assert_allclose(single.available_power(), power, rtol=1e-9)

    resp = RespFnPolynomial(50, verbose=False)
    vj = harmonic_balance(cct, resp, 15, stop_rerror=1e-6, verbose=False)
    vj_1 = harmonic_balance(single, resp, 15, stop_rerror=1e-6, verbose=False)
    np.testing.assert_allclose(vj[0], vj_1, rtol=0, atol=1e-6)
    np.testing.assert_allclose(vj[1], vj_1, rtol=0, atol=1e-6)

    with pytest.raises(AssertionError):
        set_thevenin(cct, tee, port=1, source_port=3, power=power)  # need a port per junction


def test_impedance_matrix():
    """Impedance matrix of several ports vs the Z-matrix (other ports
    terminated), with capacitances in parallel; only frequencies where all
    ports have valid references are used."""

    ntwk = _random_network(4, seed=11).renormalize(np.array([25. + 3j, 75. - 10j, 13.3, 40. + 20j]))
    ports, z_load = [4, 1, 3], 9. + 2j
    z = impedance_matrix(ntwk, ntwk.freq, ports, terminations={2: z_load})
    kk = [p - 1 for p in ports]
    for k, zf in enumerate(_z_matrix(ntwk)):
        expected = zf[kk][:, kk] - np.outer(zf[kk, 1], zf[1, kk]) / (zf[1, 1] + z_load)
        np.testing.assert_allclose(z[k], expected, rtol=1e-9)

    f, cap = 2.5e9, np.array([1e-12, 0., 4e-12])
    z0 = impedance_matrix(ntwk, f, ports)
    zc = impedance_matrix(ntwk, f, ports, cap=cap)
    np.testing.assert_allclose(zc, np.linalg.inv(np.linalg.inv(z0) + np.diag(2j * np.pi * f * cap)), rtol=1e-10)
    np.testing.assert_allclose(impedance_matrix(ntwk, f, [1], cap=2e-12)[0, 0],
                               embedding_impedance(ntwk, 1, cap=2e-12)(f), rtol=1e-10)

    # Port 2 below cutoff at the lowest frequency: only the others are used
    z0_ref = ntwk.z0.copy()
    z0_ref[0, 1] = 30j
    cut = Network(ntwk.freq, ntwk.s, z0_ref)
    impedance_matrix(cut, 3e9, [1, 2])
    with pytest.raises(ValueError):
        impedance_matrix(cut, 1.2e9, [1, 2])
    with pytest.raises(AssertionError):
        impedance_matrix(ntwk, f, [1, 1])
