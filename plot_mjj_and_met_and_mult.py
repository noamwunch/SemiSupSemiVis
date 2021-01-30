from pathlib import Path
import sys

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from tensorflow import keras

from UTILS.utils import evs_txt2jets_df as load_data
from semisup import combine_SB, determine_feats
from UTILS.lstm_classifier import preproc_for_lstm
from UTILS.plots_and_logs import plot_mult

plot_path = "RESULTS/mjj_30_01_21_new"
Path(plot_path).mkdir(parents=True, exist_ok=True)

B_path = "/gpfs0/kats/users/wunch/semisup_evs/bkg_bb_GenMjjGt150_GenPtGt40_GenEtaSt3_MjjGt200_PtGt50_EtaSt2.5/test"
S_path = "/gpfs0/kats/users/wunch/semisup_evs/sig_rinv0.25_mzp1000_GenMjjGt150_GenPtGt40_GenEtaSt3_MjjGt200_PtGt50_EtaSt2.5/test"
Ntest = 30000

#### S/B comparison plots ####
hist_dict = {'histtype': 'step', }

print('loading data for S/B comparison')
bkg1, bkg2, _ = combine_SB(B_path, S_path, Ntest, 0)
print('loaded signal')
sig1, sig2, _ = combine_SB(B_path, S_path, Ntest, 1)
print('loaded background')
print(f'loaded data for S/B comparison: (bkg_evs, sig_evs) = {(len(bkg1), len(sig1))}\n')

met_bkg, met_sig = bkg1.MET, sig1.MET
mjj_bkg, mjj_sig = bkg1.Mjj, sig1.Mjj
mult_bkg1, mult_sig1 = bkg1.mult, sig1.mult
mult_bkg2, mult_sig2 = bkg2.mult, sig2.mult
drpart_sig1, drpart_sig2 = sig1.dR_closest_parton, sig2.dR_closest_parton

print('Plotting MET and Mjj and Mult')
# MET
plt.figure()
plt.hist([met_bkg, met_sig], label=['bkg', 'sig'], bins=np.arange(0, 400, 5), **hist_dict)
plt.xlim([0, 400])
plt.ylim([0, None])
plt.xlabel('MET/GeV')
plt.ylabel('events/(5 GeV)')
plt.legend()
plt.savefig(plot_path + '/met')

# MJJ
plt.figure()
plt.hist([mjj_bkg, mjj_sig], label=['bkg', 'sig'], bins=np.arange(50, 1500, 25), **hist_dict)
plt.yscale('log')
plt.xlim([500, 1500])
plt.ylim([0, None])
plt.xlabel('$M_{jj}/GeV$')
plt.ylabel('events/(25 GeV)')
plt.legend()
plt.savefig(plot_path + '/mjj')

# Mult
plot_mult(mult_bkg1, mult_sig1, mult_bkg2, mult_sig2, save_path=plot_path+'/mult')
print('Finished plotting MET, MJJ, and Mult\n')

# Distance to closest partons
dark_frac1 = sum(drpart_sig1<0.3)/len(drpart_sig1)
dark_frac2 = sum(drpart_sig2<0.3)/len(drpart_sig2)
print(f'Percentage of leading jets in dark events originating from dark parton:'
      f'\n leading jet: {dark_frac1*100:.2f}%'
      f'\n next-to-leading jet: {dark_frac2*100:.2f}%')

# #### Bump hunt ####
mask = -10.0
n_constits = 80
feats, n_cols = determine_feats(with_displacement='True',
                                with_deltar='True',
                                with_pid='False')
met_cut = 5
nn_cut = 0.5

sig_frac = 0.2
model1_path = "RESULTS/final_grid/rinv0.25sf0.20_PtGt100/j1_0"
model2_path = "RESULTS/final_grid/rinv0.25sf0.20_PtGt100/j2_0"

# hist_dict = {'histtype': 'step', 'bins': np.arange(50, 2000, 25)}
#
# print('Beginning bump hunt...')
# j1, j2, label = combine_SB(B_path, S_path, Ntest, sig_frac)
# mjj = np.sqrt(j1.Mjj)
# met = j1.MET
#
# # Before cuts
# plt.figure()
# plt.hist(mjj, label=f'signal fraction: {sig_frac}', **hist_dict)
# plt.yscale('log')
# plt.xlabel('$M_{jj}$')
# plt.ylabel('events/(25 GeV)')
# plt.legend()
# plt.savefig(plot_path + f'/mjj_sf{sig_frac}.png')
#
# # After met cut
# valid = mjj>met_cut
# plt.figure()
# plt.hist(mjj.loc[valid], label=f'signal fraction: {sig_frac}', **hist_dict)
# plt.yscale('log')
# plt.xlabel('$M_{jj}/GeV$')
# plt.ylabel('events/(25 GeV)')
# plt.legend()
# plt.savefig(plot_path + f'/mjj_sf{sig_frac}_metcut{met_cut}.png')
#
# print('Inferring jets...')
# # After nn cut
# model1 = keras.models.load_model(model1_path)
# model2 = keras.models.load_model(model2_path)
#
# inp1 = preproc_for_lstm(j1.copy(deep=True), feats, mask, n_constits)
# inp2 = preproc_for_lstm(j2.copy(deep=True), feats, mask, n_constits)
#
# pred1 = model1.predict(inp1, batch_size=512).flatten()
# pred2 = model2.predict(inp2, batch_size=512).flatten()
#
# valid = (pred1+pred2)/2>nn_cut
# print('Inferred jets\n')
#
# plt.figure()
# plt.hist(mjj.loc[valid], label=f'signal fraction: {sig_frac}', **hist_dict)
# plt.yscale('log')
# plt.xlabel('$M_{jj}$')
# plt.ylabel('events/(25 GeV)')
# plt.legend()
# plt.savefig(plot_path + f'/mjj_sf{sig_frac}_nncut{nn_cut}.png')

print('Done!')
