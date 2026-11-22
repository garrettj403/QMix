// smx_elements.cc
//
// Evaluate SuperMix's passive circuit elements, so their S-parameters can be
// compared with qmix.network (compare_elements.py). Reads one job from stdin
// and writes one line per frequency to stdout. All values are in SI units
// (Hz, m, ohm, F, H, s, radians).
//
// Input (whitespace separated):
//
//   element z0_ohm nfreq   element name (see below), normalizing impedance
//                          (device::Z0) and number of frequencies
//   nfreq x freq_Hz
//   parameters             depending on the element:
//
//     resistor_series r | resistor_shunt r
//     capacitor_series c | capacitor_shunt c
//     inductor_series l | inductor_shunt l
//     series_rlc_series r l c | series_rlc_shunt r l c
//     parallel_rlc_series r l c | parallel_rlc_shunt r l c
//     series_tee | circulator | hybrid90 | hybrid180 | power_divider
//     branch n
//     zterm z_re z_im | yterm y_re y_im
//     attenuator db | time_delay t | transformer z1_re z1_im z2_re z2_im
//     slot_antenna z_re z_im
//     twin_slot_1 z_re z_im            antenna response: zterm(z)
//     twin_slot_2 r l c                antenna response: shunt series_RLC
//     trline zc_re zc_im theta f_ref loss
//     deembed zc theta f_ref c_out r l c
//         input: trline(zc, theta at f_ref), output: shunt capacitor c_out,
//         device: series series_RLC(r, l, c); "measured" is the circuit
//         input -- device -- output, and the output is SuperMix's deembed
//     r_waveguide a b length eps tand rho thick
//         filled with const_diel(eps, tand) unless eps == 1 and tand == 0;
//         walls normal_film(rho, thick) unless rho == 0 (perfect)
//     cpw width space height eps backed length
//         backed != 0: conducting plane below the substrate
//     radial_stub radius angle length width height eps rho t_strip t_ground
//         normal_film strip and ground plane, vacuum superstrate
//     via diameter height                  (prints L in H, one line)
//
// Output: nfreq lines of "freq S11_re S11_im S12_re S12_im ..." (row by row,
// normalized to z0).

#include "supermix.h"
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

static void print_s(nport & n, double f_hz)
{
  device::f = f_hz * Hertz;
  const Matrix & S = n.get_data_S().S;
  const int size = n.size();
  cout << f_hz;
  for (int i = 1; i <= size; ++i)
    for (int j = 1; j <= size; ++j) {
      Complex s = S.read(i, j);
      cout << " " << s.real << " " << s.imaginary;
    }
  cout << "\n";
}

static double get() { double x; cin >> x; return x; }

int main()
{
  string el;
  double z0;
  int nfreq;
  cin >> el >> z0 >> nfreq;
  vector<double> freq(nfreq);
  for (int i = 0; i < nfreq; ++i) cin >> freq[i];
  if (!cin) { cerr << "smx_elements: bad input" << endl; return 1; }
  device::Z0 = z0 * Ohm;
  device::T = 4.2 * Kelvin;
  cout << setprecision(17);

  // Elements are created here and evaluated in run()
  nport *n = 0;
  resistor r; capacitor c; inductor l; series_RLC srlc; parallel_RLC prlc;
  series_tee st; branch br; zterm zt; yterm yt; attenuator att; time_delay td;
  transformer tr; circulator circ; hybrid90 h90; hybrid180 h180; power_divider pd;
  slot_antenna sa; twin_slot_antenna tsa; trline tl;
  r_waveguide wg; cpw cp; radial_stub rs;

  if (el == "resistor_series" || el == "resistor_shunt") {
    r.R = get() * Ohm;
    if (el == "resistor_shunt") r.parallel();
    n = &r;
  }
  else if (el == "capacitor_series" || el == "capacitor_shunt") {
    c.C = get() * Farad;
    if (el == "capacitor_shunt") c.parallel();
    n = &c;
  }
  else if (el == "inductor_series" || el == "inductor_shunt") {
    l.L = get() * Henry;
    if (el == "inductor_shunt") l.parallel();
    n = &l;
  }
  else if (el == "series_rlc_series" || el == "series_rlc_shunt") {
    srlc.R = get() * Ohm; srlc.L = get() * Henry; srlc.C = get() * Farad;
    if (el == "series_rlc_shunt") srlc.parallel();
    n = &srlc;
  }
  else if (el == "parallel_rlc_series" || el == "parallel_rlc_shunt") {
    prlc.R = get() * Ohm; prlc.L = get() * Henry; prlc.C = get() * Farad;
    if (el == "parallel_rlc_shunt") prlc.parallel();
    n = &prlc;
  }
  else if (el == "series_tee") n = &st;
  else if (el == "branch") { br.set_branches(int(get())); n = &br; }
  else if (el == "zterm") { double re = get(), im = get(); zt.Z = Complex(re, im) * Ohm; n = &zt; }
  else if (el == "yterm") { double re = get(), im = get(); yt.Y = Complex(re, im) / Ohm; n = &yt; }
  else if (el == "attenuator") { att.dB = get(); n = &att; }
  else if (el == "time_delay") { td.time = get() * Second; n = &td; }
  else if (el == "transformer") {
    double a = get(), b = get(), cc = get(), d = get();
    tr.Z1 = Complex(a, b) * Ohm; tr.Z2 = Complex(cc, d) * Ohm;
    n = &tr;
  }
  else if (el == "circulator") n = &circ;
  else if (el == "hybrid90") n = &h90;
  else if (el == "hybrid180") n = &h180;
  else if (el == "power_divider") n = &pd;
  else if (el == "slot_antenna") {
    double re = get(), im = get();
    sa.Z = Complex(re, im) * Ohm;
    n = &sa;
  }
  else if (el == "twin_slot_1") {
    double re = get(), im = get();
    zt.Z = Complex(re, im) * Ohm;
    tsa.set(zt);
    n = &tsa;
  }
  else if (el == "twin_slot_2") {
    srlc.R = get() * Ohm; srlc.L = get() * Henry; srlc.C = get() * Farad;
    srlc.parallel();
    tsa.set(srlc);
    n = &tsa;
  }
  else if (el == "trline") {
    double re = get(), im = get();
    tl.zchar = Complex(re, im) * Ohm;
    tl.theta = get(); tl.freq = get() * Hertz; tl.loss = get();
    n = &tl;
  }
  else if (el == "deembed") {
    static trline in;
    static capacitor out;
    static series_RLC dut;
    static circuit meas;
    in.zchar = get() * Ohm; in.theta = get(); in.freq = get() * Hertz;
    out.C = get() * Farad; out.parallel();
    dut.R = get() * Ohm; dut.L = get() * Henry; dut.C = get() * Farad;
    meas.connect(in, 2, dut, 1);
    meas.connect(dut, 2, out, 1);
    meas.add_port(in, 1);
    meas.add_port(out, 2);
    static deembed de(in, out, meas);
    n = &de;
  }
  else if (el == "r_waveguide") {
    static const_diel fill;
    static normal_film wall;
    wg.a = get() * Meter; wg.b = get() * Meter; wg.length = get() * Meter;
    double eps = get(), tand = get(), rho = get(), thick = get();
    if (eps != 1.0 || tand != 0.0) { fill.eps = eps; fill.tand = tand; wg.fill(fill); }
    if (rho != 0.0) { wall.rho = rho * Ohm * Meter; wall.Thick = thick * Meter; wg.wall(wall); }
    n = &wg;
  }
  else if (el == "cpw") {
    static const_diel sub;
    static normal_film metal;  // required by SuperMix, but not used (lossless)
    metal.rho = 1e-8 * Ohm * Meter; metal.Thick = 1e-7 * Meter;
    cp.width = get() * Meter; cp.space = get() * Meter; cp.sub_thick = get() * Meter;
    sub.eps = get();
    double backed = get();
    cp.length = get() * Meter;
    cp.substrate(sub).top_strip(metal);
    if (backed != 0.0) cp.bottom_plane(metal);
    n = &cp;
  }
  else if (el == "radial_stub") {
    static const_diel sub, vac(1.0, 0.0);
    static normal_film strip, ground;
    rs.radius = get() * Meter; rs.angle = get(); rs.length = get() * Meter; rs.width = get() * Meter;
    rs.sub_thick = get() * Meter;
    sub.eps = get();
    double rho = get();
    strip.rho = rho * Ohm * Meter; ground.rho = rho * Ohm * Meter;
    strip.Thick = get() * Meter; ground.Thick = get() * Meter;
    rs.substrate(sub).superstrate(vac).top_strip(strip).ground_plane(ground);
    n = &rs;
  }
  else if (el == "via") {
    L_via v;
    v.D = get() * Meter; v.H = get() * Meter;
    cout << v.get() / Henry << endl;
    return 0;
  }
  else { cerr << "smx_elements: unknown element " << el << endl; return 1; }

  if (!cin) { cerr << "smx_elements: bad parameters" << endl; return 1; }
  for (int i = 0; i < nfreq; ++i) print_s(*n, freq[i]);
  return 0;
}
