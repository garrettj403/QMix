// smx_lines.cc
//
// Evaluate SuperMix's superconducting film and microstrip models, so the
// results can be compared with qmix.superconductor and qmix.network
// (compare_lines.py). Reads one job from stdin, writes one line per
// frequency to stdout. All values are in SI units (Hz, K, V, m, ohm m).
//
// Input (whitespace separated):
//
//   film_type            "super" (Mattis-Bardeen) or "normal"
//   vgap_V tc_K rho      super: gap voltage, Tc, normal-state resistivity;
//                        normal: rho only (vgap and tc are read and ignored)
//   t_strip t_ground     film thicknesses
//   width height         microstrip width and dielectric thickness
//   eps tand eps_up tand_up   substrate and superstrate dielectrics
//   temp_K nfreq         temperature and number of frequencies
//   nfreq x freq_Hz
//
// Output: nfreq lines of
//
//   freq  sigma_re sigma_im  zs_strip_re zs_strip_im  zs_ground_re zs_ground_im
//         zchar_re zchar_im  kprop_re kprop_im
//
// sigma is the strip film's conductivity in S/m (exp(+j w t): sigma1 - j
// sigma2), zs are surface impedances in ohms, zchar in ohms and kprop
// (waves vary as exp(-kprop z)) in 1/m. The super_film objects are used
// without interpolation (no_interpolate), so sigma is from the integrals.

#include "supermix.h"
#include <iomanip>
#include <iostream>
#include <string>

using namespace std;

int main()
{
  string film_type;
  double vgap, tc, rho, t_strip, t_ground, width, height, eps, tand, eps_up, tand_up, temp;
  int nfreq;

  cin >> film_type >> vgap >> tc >> rho >> t_strip >> t_ground >> width >> height
      >> eps >> tand >> eps_up >> tand_up >> temp >> nfreq;
  if (!cin || (film_type != "super" && film_type != "normal")) {
    cerr << "smx_lines: bad input" << endl;
    return 1;
  }

  super_film s_strip, s_ground;
  normal_film n_strip, n_ground;
  surfimp *strip, *ground;
  if (film_type == "super") {
    s_strip.Vgap = vgap * Volt;   s_ground.Vgap = vgap * Volt;
    s_strip.Tc = tc * Kelvin;     s_ground.Tc = tc * Kelvin;
    s_strip.rho_normal = rho * Ohm * Meter;
    s_ground.rho_normal = rho * Ohm * Meter;
    s_strip.Thick = t_strip * Meter;
    s_ground.Thick = t_ground * Meter;
    s_strip.no_interpolate();
    s_ground.no_interpolate();
    strip = &s_strip; ground = &s_ground;
  }
  else {
    n_strip.rho = rho * Ohm * Meter;  n_ground.rho = rho * Ohm * Meter;
    n_strip.Thick = t_strip * Meter;  n_ground.Thick = t_ground * Meter;
    strip = &n_strip; ground = &n_ground;
  }

  const_diel sub(eps, tand), sup(eps_up, tand_up);
  microstrip line;
  line.substrate(sub).superstrate(sup).top_strip(*strip).ground_plane(*ground);
  line.width = width * Meter;
  line.sub_thick = height * Meter;
  line.length = 1.0 * Micron;  // not used for zchar, kprop
  line.Temp = temp * Kelvin;

  cout << setprecision(17);
  for (int i = 0; i < nfreq; ++i) {
    double f;
    cin >> f;
    const double F = f * Hertz, T = temp * Kelvin;
    Complex sig = (film_type == "super") ? s_strip.sigma(F, T) : n_strip.sigma(F, T);
    sig *= Ohm * Meter;  // to S/m
    Complex zs1 = strip->Zsurf(F, T) / Ohm;
    Complex zs2 = ground->Zsurf(F, T) / Ohm;
    Complex zc = line.Zchar(F, T) / Ohm;
    Complex k = line.Kprop(F, T) * Meter;  // to 1/m
    cout << f << " " << sig.real << " " << sig.imaginary << " "
         << zs1.real << " " << zs1.imaginary << " " << zs2.real << " " << zs2.imaginary << " "
         << zc.real << " " << zc.imaginary << " " << k.real << " " << k.imaginary << endl;
  }
  return 0;
}
