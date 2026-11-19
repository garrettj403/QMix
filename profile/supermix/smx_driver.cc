// smx_driver.cc
//
// Run SuperMix's SIS junction model on inputs written by compare.py, so the
// results can be compared with QMix. Reads a job from stdin and writes one
// line per bias point to stdout.
//
// All values in and out are normalized the QMix way: voltages to Vgap (Vn),
// currents to Vgap/Rn, impedances to Rn, frequencies to the gap frequency.
// AC phasors are PEAK amplitudes (SuperMix works with RMS internally; the
// conversion is done here). The DC values are plain.
//
// Input (whitespace separated):
//
//   mode                 "current", "balance", "smallsignal" or "mixernoise"
//   vn_mV rn_ohm         normalizing voltage and resistance
//   idc_file ikk_file    I-V curve and KK transform (normalized)
//   flo num_p npts       LO frequency, number of harmonics, bias points
//   tol                  balance/smallsignal: Newton tolerance (normalized
//                        voltage)
//   fif                  smallsignal/mixernoise: IF frequency (normalized)
//   temp_K z0_ohm        mixernoise only: physical temperature and the
//                        normalizing impedance of the mixer ports
//   num_p x (vt_re vt_im zt_re zt_im)   balance/smallsignal: Thevenin
//                                       circuit at each LO harmonic
//   npts x (vb [num_p x (vj_re vj_im)]) bias voltage; current mode also
//                                       gives the junction voltage phasors
//
// Output: npts lines of
//
//   vb iterations status  then for p = 0..num_p: vj_re vj_im ij_re ij_im
//
// and in smallsignal mode, then the junction's small-signal admittance matrix
// (Tucker theory, after balancing the LO) Y[m][n] for m, n = -num_p..num_p
// (row by row), as re im pairs, normalized to 1/Rn. Sideband m is at
// frequency m * flo + fif; for m < 0, this is negative, so the phasor is the
// complex conjugate of the one at the positive frequency |m| * flo - fif.
//
// In mixernoise mode, the line continues with the junction's current noise
// correlation matrix H[m][n] (one-sided spectral density of the RMS noise
// currents, normalized to e * Vn / Rn), then the mixer's (small-signal
// analysis with all sideband ports open, i.e., terminated in Z0) scattering
// matrix S and noise wave correlation matrix C (in K) between the sideband
// ports m, n = -num_p..num_p, each as re im pairs, row by row. The last
// column is the IF noise temperature (in K) with the RF port terminated by the
// Thevenin terminators, which have thermal noise (Callen-Welton) at temp_K.
//
// where status is 0 for success (current mode always 0) and iterations is
// the number of Newton iterations (0 in current mode). The final line is
// "# time_s <seconds>" with the wall time spent in SuperMix calls.

#include "supermix.h"
#include <ctime>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

// A 1-port Thevenin source whose impedance and voltage depend on which LO
// harmonic is being evaluated (n = round(f / LO)). QMix specifies the
// embedding circuit this way, one (vt, zt) pair per harmonic.
class harmonic_thevenin : public nport
{
public:
  double LO;                // LO frequency (Hz)
  std::vector<Complex> Z;   // impedance at harmonic n (Ohm), n = 0..num_p
  std::vector<Complex> V;   // RMS voltage at harmonic n (Volt)

  harmonic_thevenin() : nport(1), LO(0.0)
  { info.source = true; info.noise = true; info.active = false; }

  int size() { return 1; }

private:
  void recalc_S()
  {
    double z0 = Z0;
    data.set_znorm(z0);
    int n = int(floor(f / LO + 0.5));
    Complex z(0.0), v(0.0);
    if (n >= 0 && n < int(Z.size())) { z = Z[n]; v = V[n]; }
    data.S[1][1] = (z - z0) / (z + z0);
    data.B[1] = sqrt(z0) * v / (z + z0);  // same convention as voltage_source
  }

  // Thermal noise of the impedance at the global temperature device::T
  // (Callen-Welton, including zero-point fluctuations), like any passive
  // SuperMix element. It only matters for mixer::get_term_data().
  void recalc() { recalc_S(); compute_passive_noise(data, f, T); }
};


static double now()
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}


int main()
{
  string mode, idc_file, ikk_file;
  double vn_mV, rn_ohm, flo_norm;
  int num_p, npts;

  cin >> mode >> vn_mV >> rn_ohm >> idc_file >> ikk_file >> flo_norm >> num_p >> npts;
  if (!cin || (mode != "current" && mode != "balance" && mode != "smallsignal" && mode != "mixernoise")) {
    cerr << "smx_driver: bad input header" << endl;
    return 1;
  }

  const double Vn = vn_mV * mVolt;
  const double Rn = rn_ohm * Ohm;
  const double In = Vn / Rn;
  const double Fgap = Vn * VoltToFreq;
  const double fLO = flo_norm * Fgap;

  ivcurve iv(idc_file.c_str(), ikk_file.c_str());

  sis_device sis;
  sis.set_iv(iv);
  sis.Vn = Vn;
  sis.Rn = Rn;
  sis.Cap = 0.0;  // any capacitance belongs in the QMix embedding impedance

  cout << setprecision(17);
  double t_total = 0.0;

  if (mode == "current") {
    // Tunneling current only: no circuit, the junction voltage is given.
    Vector V(num_p + 1, Index_C);
    for (int i = 0; i < npts; ++i) {
      double vb;
      cin >> vb;
      V[0] = vb * Vn;
      for (int p = 1; p <= num_p; ++p) {
        double re, im;
        cin >> re >> im;
        V[p] = Complex(re, im) * Vn / RmsToPeak;
      }
      double t0 = now();
      const Vector & I = sis.large_signal(V, fLO, num_p);
      t_total += now() - t0;

      cout << vb << " 0 0";
      for (int p = 0; p <= num_p; ++p) {
        double scale = (p == 0) ? 1.0 : RmsToPeak;
        Complex v = V.read(p) * scale / Vn, c = I.read(p) * scale / In;
        cout << " " << v.real << " " << v.imaginary << " " << c.real << " " << c.imaginary;
      }
      cout << "\n";
    }
  }

  else {  // balance, smallsignal or mixernoise
    double tol, fif_norm = 0.0, temp_K = 0.0, z0_ohm = 50.0;
    cin >> tol;
    if (mode == "smallsignal" || mode == "mixernoise") cin >> fif_norm;
    if (mode == "mixernoise") {
      cin >> temp_K >> z0_ohm;
      device::T = temp_K * Kelvin;
      device::Z0 = z0_ohm * Ohm;
    }
    const double fIF = fif_norm * Fgap;

    harmonic_thevenin lo;
    lo.LO = fLO;
    lo.Z.assign(num_p + 1, Complex(0.0));
    lo.V.assign(num_p + 1, Complex(0.0));
    for (int p = 1; p <= num_p; ++p) {
      double vr, vi, zr, zi;
      cin >> vr >> vi >> zr >> zi;
      lo.V[p] = Complex(vr, vi) * Vn / RmsToPeak;
      lo.Z[p] = Complex(zr, zi) * Rn;
    }

    // DC bias: ideal voltage source, so the junction DC voltage is exactly vb
    // (QMix also treats the bias voltage as the junction DC voltage).
    voltage_source bias;
    bias.R = 0.0;
    bias.source_f = 0.0;

    // RF and IF circuits: straight-through 2-ports. Port 1 goes to the
    // junction; the LO Thevenin source terminates RF port 2 during balance.
    branch rf(2), ifc(2);

    mixer m;
    m.add_junction(sis);
    m.set_LO(fLO).harmonics(num_p);
    m.set_bias(bias).set_if(ifc).set_rf(rf).set_balance_terminator(lo, 2);
    m.balance_parameters(200, tol * Vn, 1.0e-12, tol * Vn, 1.0e-4);

    for (int i = 0; i < npts; ++i) {
      double vb;
      cin >> vb;
      bias.source_voltage = vb * Vn;

      double t0 = now();
      if (i == 0) m.initialize_operating_state();
      int status = m.balance();  // starts from the previous point's state
      t_total += now() - t0;

      const Vector & V = sis.V();
      const Vector & I = sis.I();
      cout << vb << " " << m.balance_iterations() << " " << status;
      for (int p = 0; p <= num_p; ++p) {
        double scale = (p == 0) ? 1.0 : RmsToPeak;
        Complex v = V.read(p) * scale / Vn, c = I.read(p) * scale / In;
        cout << " " << v.real << " " << v.imaginary << " " << c.real << " " << c.imaginary;
      }
      if (mode == "smallsignal" || mode == "mixernoise") {
        double t1 = now();
        const Matrix & Y = sis.small_signal(fIF, num_p);
        t_total += now() - t1;
        for (int mm = -num_p; mm <= num_p; ++mm)
          for (int nn = -num_p; nn <= num_p; ++nn) {
            Complex y = Y.read(mm, nn) * Rn;
            cout << " " << y.real << " " << y.imaginary;
          }
      }
      if (mode == "mixernoise") {
        double t1 = now();
        const Matrix & H = sis.noise(fIF, device::T, num_p);
        const double h_norm = eCharge * Vn / Rn;
        for (int mm = -num_p; mm <= num_p; ++mm)
          for (int nn = -num_p; nn <= num_p; ++nn) {
            Complex h = H.read(mm, nn) / h_norm;
            cout << " " << h.real << " " << h.imaginary;
          }
        // Full small-signal analysis of the mixer at the IF frequency
        device::f = fIF;
        const sdata & sd = m.get_data();
        t_total += now() - t1;
        for (int mm = -num_p; mm <= num_p; ++mm)
          for (int nn = -num_p; nn <= num_p; ++nn) {
            Complex s = sd.S.read(m.port(2, mm), m.port(2, nn));
            cout << " " << s.real << " " << s.imaginary;
          }
        for (int mm = -num_p; mm <= num_p; ++mm)
          for (int nn = -num_p; nn <= num_p; ++nn) {
            Complex c = sd.C.read(m.port(2, mm), m.port(2, nn)) / Kelvin;
            cout << " " << c.real << " " << c.imaginary;
          }
        // RF ports terminated by the (noisy) Thevenin terminators: noise
        // temperature at the IF port, from the junction and the terminations
        const sdata & st = m.get_term_data();
        cout << " " << st.C.read(1, 1).real / Kelvin;
      }
      cout << "\n";
    }
  }

  cout << "# time_s " << t_total << endl;
  return 0;
}
