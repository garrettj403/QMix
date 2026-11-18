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
//   mode                 "current" or "balance"
//   vn_mV rn_ohm         normalizing voltage and resistance
//   idc_file ikk_file    I-V curve and KK transform (normalized)
//   flo num_p npts       LO frequency, number of harmonics, bias points
//   tol                  balance only: Newton tolerance (normalized voltage)
//   num_p x (vt_re vt_im zt_re zt_im)   balance only: Thevenin circuit
//   npts x (vb [num_p x (vj_re vj_im)]) bias voltage; current mode also
//                                       gives the junction voltage phasors
//
// Output: npts lines of
//
//   vb iterations status  then for p = 0..num_p: vj_re vj_im ij_re ij_im
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
  { info.source = true; info.noise = false; info.active = false; }

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

  void recalc() { recalc_S(); data.C[1][1] = 0.0; }
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
  if (!cin || (mode != "current" && mode != "balance")) {
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

  else {  // balance
    double tol;
    cin >> tol;

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
      cout << "\n";
    }
  }

  cout << "# time_s " << t_total << endl;
  return 0;
}
