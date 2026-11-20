"""Shared loading code for the thesis-data comparisons.

The data are in ``thesis/`` at the root of the repository (the 230 GHz SIS
mixer data from the author's DPhil thesis; git-ignored, not distributed).

"""

import contextlib
import io
import os
import warnings

import numpy as np

import qmix

warnings.filterwarnings('ignore')

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
THESIS = os.path.join(ROOT, 'thesis')
HFSS = os.path.join(THESIS, 'Supermix_with_Recovered_Optic_Loss', 'Supermix_clean_Paris_RF', 'SParam_files_Paris')
TWO_TONES = os.path.join(THESIS, '230 GHz SIS Device', 'two-strong-tones', 'data')
COOLDOWNS = os.path.join(THESIS, 'Cooldown Data')

# Parameters for the raw lab files (no header, mV / mA)
PARAMS = dict(delimiter=',', usecols=(0, 1), skip_header=0, v_fmt='mV', i_fmt='mA', area=1.5,
              filter_data=True, filter_nwind=21, verbose=False)

assert os.path.isdir(THESIS), "Thesis data not found: {}".format(THESIS)


def load(path):
    return np.genfromtxt(path, delimiter=',')


def quiet():
    """Silence the impedance-recovery printout of qmix.exp."""
    return contextlib.redirect_stdout(io.StringIO())


def dc_data(iv_file, if_file, **kw):
    with quiet():
        return qmix.exp.DCData(load(iv_file), load(if_file), **{**PARAMS, **kw})


def pumped_data(dciv, iv_file, freq, hot_file=None, cold_file=None, **kw):
    hot = None if hot_file is None else load(hot_file)
    cold = None if cold_file is None else load(cold_file)
    with quiet():
        return qmix.exp.PumpedData(load(iv_file), dciv, hot, cold, freq=freq, **{**PARAMS, **kw})


def rf_network(norm=False):
    """RF circuit of the 230 GHz mixer (HFSS): tuner (junction at port 1, IF at
    port 3) connected to the finline (waveguide at port 1).

    Returns a 3-port network with ports: 1 junction, 2 IF, 3 waveguide.
    Without ``norm``, the files keep the HFSS port impedances, so the
    waveguide is matched when it is left unterminated.

    """

    from qmix.network import read_touchstone
    sfx = 'norm' if norm else 'nonorm'
    tuner = read_touchstone(os.path.join(HFSS, 'Quartz_direct_tuner_wide_Paris_rho_01_FINAL_RnPort_supermix_'
                                         + sfx + '.s3p'))
    finline = read_touchstone(os.path.join(HFSS, '230GHz_Quartz_finline_2-5um_07_Paris_rho_nonotch_FINAL_supermix_'
                                           + sfx + '.s2p'))
    return tuner.connect(2, finline, 2)
