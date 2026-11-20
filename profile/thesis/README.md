# Comparisons with the thesis data

These scripts compare QMix with measurements and HFSS simulations of the 230 GHz SIS mixer from the author's DPhil thesis. The data are expected in `thesis/` at the root of the repository; it is git-ignored and not distributed. Each script runs from any directory and prints its results; it takes a few minutes.

| Script | Data | What it checks |
|---|---|---|
| `hfss_embedding.py` | HFSS touchstones (`Supermix_with_Recovered_Optic_Loss/.../SParam_files_Paris`), device #5.6 LO sweep (2016-03-17), example data | `qmix.network` embedding impedance vs the impedances recovered by `qmix.exp`; junction capacitance fit; `mixer_noise` with HFSS sideband impedances (`z_emb`) |
| `two_lo.py` | Device #5.7, two strong LOs (2017-04-10, `test2_*`) | Multi-tone harmonic balance vs the measured two-LO pumped I-V curves, with each LO recovered from its own single-LO curve (no free parameters) |
| `lo_saturation.py` | Device #5.6, 255 GHz LO power sweep (2017-08-10, `LO_saturation`) | Gain and noise vs LO power from `mixer_noise`, with a fixed embedding impedance |

## Results (Nov 2026)

**HFSS embedding impedance.** The tuner `.s3p` (port 1: junction, 2: finline side, 3: IF) is connected to the finline `.s2p` (port 1: waveguide). The waveguide must be terminated in its own mode impedance (~288 ohm at 230 GHz). With the renormalized `*_norm` files, leaving it at 50 ohm gives spurious ~25 GHz ripples. With the raw `*_nonorm` files, which keep the HFSS port impedances, it is matched by default. The two give the same result to 2 milliohm. The nominal junction capacitance (80 fF/um^2 x 1.5 um^2 = 120 fF) gives an rms difference of 0.22 Rn from the recovered impedances (214-233 GHz, good fits only). A fitted 144 fF gives 0.12 Rn, and it also matches the good fits at 252-255 GHz. With the measured Rn (14.2 ohm, i.e., ~1.4 um^2 at RnA = 20), this corresponds to ~100 fF/um^2. Above ~235 GHz the impedance recovery itself is unreliable.

**Noise with HFSS sideband impedances.** For the example data, using the HFSS impedances at the USB/LSB instead of the recovered impedance changes `g_rf` by up to 25% but `t_rf` (18-21 K) and `t_mix` (3.5-4.1 K) hardly at all. So equal sideband impedances do not explain the ~8 K difference from the thesis noise breakdown. The HFSS IF impedance at the junction (~6-16j ohm at 5 GHz, mostly tuner capacitance, with the IF transformer into 50 ohm) is not consistent with the data: it requires `g_rf` > 1.

**Two LOs.** With the two LOs alone (2 tones), the simulation keeps sharp photon steps, while the measured curves are smooth. The rms error is 15-40% of the pump-induced change in current, 2-3x worse than the single-LO fits. Adding the beat at f2 - f1 = 4.3 GHz as a third tone (vt = 0, into the IF load) gives an IF voltage of ~0.18 Vgap, which smears the steps. The error then drops to the level of the single-LO fits (e.g., 0.022-0.033 with 25 ohm, vs 0.047-0.066 for 2 tones). The IF load is not recorded; lower values fit slightly better. The IF tone needs num_b ~ 45 (V_if / f_if is large).

**LO saturation.** Recovering zt separately at each power level fails at 255 GHz. With zt fixed from the two lowest-error levels and only vt fitted per level, the I-V fits are good (rms < 0.012). The gain at the best bias is then predicted to 0-7% over vt = 0.19-1.36 with one fixed RF stage. The exceptions are the most strongly pumped point at the first step (level 4, 23% low) and the nearly unpumped level 8. Tn is underpredicted at high LO power (e.g., 87 vs 111 K at level 4).
