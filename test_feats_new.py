from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras

from semisup import combine_SB
from semisup import determine_feats
from UTILS.lstm_classifier import preproc_for_lstm, create_lstm_classifier, train_classifier
from UTILS.plots_and_logs import plot_rocs, plot_mult, plot_learn_curve

output_path = "./RESULTS/fullsup/rinv0.25sf0.20/"
B_path_test = "/gpfs0/kats/users/wunch/semisup_evs/bkg/test"
S_path_test = "/gpfs0/kats/users/wunch/semisup_evs/sig_rinv_0.25_mjj_500/test"
B_path = "/gpfs0/kats/users/wunch/semisup_evs/bkg/train"
S_path = "/gpfs0/kats/users/wunch/semisup_evs/sig_rinv_0.25_mjj_500/train"
Path(output_path).mkdir(parents=True, exist_ok=True)

N = 100000
sig_frac = 0.5

# TO ERASE
N = 30000
sig_frac = 0.1
pt_min = 100
# TO ERASE

print('Loading data...')
j1_dat, j2_dat, label = combine_SB(B_path, S_path, N, sig_frac)
print(f'Loaded  data: {len(label)} training examples \n')

# TO ERASE
print(f'Cutting on jet PT (both jet pt > {pt_min})..')
valid = (j1_dat.jet_PT>pt_min) & (j2_dat.jet_PT>pt_min)
j1_dat, j2_dat = j1_dat.loc[valid], j2_dat.loc[valid]
label = label[valid]
print(f'Cut on jet PT left with {np.sum(valid)} events')
print(f'{sum(label)} signal events and {sum(~label.astype(bool))} background events')
print(f'In total: {np.sum(valid)/len(valid):.2f} of background has both jetpt > {pt_min}')
# TO ERASE

# # Before preprocessing
# bkg1, bkg2 = j1_dat.iloc[~label.astype(bool)], j2_dat.iloc[~label.astype(bool)]
# sig1, sig2 = j1_dat.iloc[label.astype(bool)], j2_dat.iloc[label.astype(bool)]
#
# # Jet features
# hist_dict = {'density': True, 'histtype': 'step', 'bins': 100}
# label = ['bkg1', 'sig1', 'bkg2', 'sig2']
# # multiplicity
# plot_mult(bkg1.mult, sig1.mult, bkg2.mult, sig2.mult, save_path=output_path+'mult.png')
#
# # jet pt
# feat = 'jet_PT'
# plt.figure()
# plt.hist([bkg1[feat], sig1[feat], bkg2[feat], sig2[feat]],
#          label=label, range=[0, 400], **hist_dict)
# plt.legend()
# plt.savefig(output_path+feat)
#
# # jet eta
# feat = 'jet_Eta'
# plt.figure()
# plt.hist([bkg1[feat], sig1[feat], bkg2[feat], sig2[feat]],
#          label=label, **hist_dict)
# plt.legend()
# plt.savefig(output_path+feat)
#
# # jet phi
# feat = 'jet_Phi'
# plt.figure()
# plt.hist([bkg1[feat], sig1[feat], bkg2[feat], sig2[feat]],
#          label=label, **hist_dict)
# plt.legend()
# plt.savefig(output_path+feat)

# Track features
# track pt
# track eta
# track phi
# track deltar

# After preprocessing

# print('Preprocessing data...')
# j1_inp = preproc_for_lstm(j1_dat.copy(deep=True), feats, mask, n_constits)
# j2_inp = preproc_for_lstm(j2_dat.copy(deep=True), feats, mask, n_constits)
# print(f'Preprocessed data: shape={j1_inp.shape} \n')




