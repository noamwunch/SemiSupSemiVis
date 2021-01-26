from pathlib import Path
import sys

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from tensorflow import keras

from UTILS.utils import evs_txt2jets_df as load_data
from semisup import combine_SB, determine_feats
from UTILS.lstm_classifier import preproc_for_lstm

plot_path = "RESULTS/mjj_new"
Path(plot_path).mkdir(parents=True, exist_ok=True)

# settings_path =
# parse args =

mask = -10.0
n_constits = 80
feats, n_cols = determine_feats(with_displacement='True',
                                with_deltar='True',
                                with_pid='False')
met_cut = 5
nn_cut = 0.5

B_path = "/gpfs0/kats/users/wunch/semisup_evs/bkg_GenPtGt70_PtGt100/test"
S_path = "/gpfs0/kats/users/wunch/semisup_evs/sig_rinv_0.25_mjj_500_GenPtGt40_PtGt100/test"
sig_frac = 0.2
model1_path = "RESULTS/final_grid/rinv0.25sf0.20_PtGt100/j1_0"
model2_path = "RESULTS/final_grid/rinv0.25sf0.20_PtGt100/j2_0"
Ntest = 20000

#### S/B comparison plots ####
hist_dict = {'density': True, 'histtype': 'step', 'bins': 100}

print('loading data for S/B comparison')
bkg, _, _ = combine_SB(B_path, S_path, int(Ntest/2), 0)
print('loaded signal')
sig, _, _ = combine_SB(B_path, S_path, int(Ntest/2), 1)
print('loaded background')
print(f'loaded data for S/B comparison: (bkg_evs, sig_evs) = {(len(bkg), len(sig))}\n')

met_bkg, met_sig = bkg.MET, sig.MET
mjj_bkg, mjj_sig = bkg.Mjj, sig.Mjj

print('Plotting MET and Mjj')
# MET
plt.figure()
plt.hist([met_bkg, met_sig], label=['bkg', 'sig'], **hist_dict)
plt.savefig(plot_path + '/met')

# MJJ
plt.figure()
plt.hist([mjj_bkg, mjj_sig], label=['bkg', 'sig'], **hist_dict)
plt.yscale('log')
plt.savefig(plot_path + '/mjj')
print('Finished plotting MET and MJJ\n')

#### Bump hunt ####
print('Beginning bump hunt...')
j1, j2, label = combine_SB(B_path, S_path, Ntest, sig_frac)
mjj = j1.Mjj
met = j1.MET

# Before cuts
plt.figure()
plt.hist(mjj, label=f'signal fraction: {sig_frac}', **hist_dict)
plt.yscale('log')
plt.savefig(plot_path + f'/mjj_sf{sig_frac}')

# After met cut
valid = mjj>met_cut
plt.figure()
plt.hist(mjj.loc[valid], label=f'signal fraction: {sig_frac}', **hist_dict)
plt.yscale('log')
plt.savefig(plot_path + f'/mjj_sf{sig_frac}_metcut{met_cut}')

print('Inferring jets...')
# After nn cut
model1 = keras.models.load_model(model1_path)
model2 = keras.models.load_model(model2_path)

inp1 = preproc_for_lstm(j1.copy(deep=True), feats, mask, n_constits)
inp2 = preproc_for_lstm(j2.copy(deep=True), feats, mask, n_constits)

pred1 = model1.predict(inp1, batch_size=512).flatten()
pred2 = model2.predict(inp2, batch_size=512).flatten()

valid = (pred1+pred2)/2>nn_cut
print('Inferred jets\n')

plt.hist(mjj.loc[valid], label=f'signal fraction: {sig_frac}', **hist_dict)
plt.yscale('log')
plt.savefig(plot_path + f'/mjj_sf{sig_frac}_nncut{nn_cut}')
print('Done!')
