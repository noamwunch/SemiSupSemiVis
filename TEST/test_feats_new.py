from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from semisup import combine_SB
from semisup import determine_feats
from UTILS.lstm_classifier import preproc_for_lstm
from UTILS.plots_and_logs import plot_mult

output_path = "../RESULTS/fullsup/rinv0.25sf0.20/"
B_path = "/gpfs0/kats/users/wunch/semisup_evs/bkg/train"
S_path = "/gpfs0/kats/users/wunch/semisup_evs/sig_rinv_0.25_mjj_500/train"
Path(output_path).mkdir(parents=True, exist_ok=True)

N = 100000
sig_frac = 0.5

print('Loading data...')
j1_dat, j2_dat, label = combine_SB(B_path, S_path, N, sig_frac)
print(f'Loaded  data: {len(label)} training examples \n')

# Jet features
bkg1, bkg2 = j1_dat.iloc[~label.astype(bool)], j2_dat.iloc[~label.astype(bool)]
sig1, sig2 = j1_dat.iloc[label.astype(bool)], j2_dat.iloc[label.astype(bool)]

hist_dict = {'density': True, 'histtype': 'step', 'bins': 100}
label = ['bkg1', 'sig1', 'bkg2', 'sig2']
dR = 0.3
pt_min = 100

# multiplicity
# plot_mult(bkg1.mult, sig1.mult, bkg2.mult, sig2.mult, save_path=output_path+'mult.png')

# jet pt
# feat = 'jet_PT'
# plt.figure()
# plt.hist([bkg1[feat], sig1[feat], bkg2[feat], sig2[feat]],
#          label=label, range=[0, 400], **hist_dict)
# plt.legend()
# plt.savefig(output_path+feat)

# jet eta
# feat = 'jet_Eta'
# plt.figure()
# plt.hist([bkg1[feat], sig1[feat], bkg2[feat], sig2[feat]],
#          label=label, **hist_dict)
# plt.legend()
# plt.savefig(output_path+feat)

# jet phi
# feat = 'jet_Phi'
# plt.figure()
# plt.hist([bkg1[feat], sig1[feat], bkg2[feat], sig2[feat]],
#          label=label, **hist_dict)
# plt.legend()
# plt.savefig(output_path+feat)

# dR closest parton
feat = 'dR_closest_parton'
plt.figure()
plt.hist([sig1[feat], sig2[feat]],
         label=['signal - jet2', 'signal - jet1'], **hist_dict)
plt.legend()
plt.savefig(output_path+feat)

print('For all PT:')
print(f'frac of j1 from dark parton: {np.sum(sig1[feat]<dR)/len(sig1[feat]):.2f}')
print(f'frac of j2 from dark parton: {np.sum(sig2[feat]<dR)/len(sig2[feat]):.2f}')
print('')

print(f'Cutting on jet PT (both jet pt > {pt_min})..')
valid = (sig1.jet_PT>pt_min) & (sig2.jet_PT>pt_min)
sig1, sig2 = sig1.loc[valid], sig2.loc[valid]
print(f'Cut on jet PT left with {np.sum(valid)} events')
print('')

print(f'For PT > {pt_min}:')
print(f'frac of j1 from dark parton: {np.sum(sig1[feat]<dR)/len(sig1[feat]):.2f}')
print(f'frac of j2 from dark parton: {np.sum(sig2[feat]<dR)/len(sig2[feat]):.2f}')

# Track features
# track pt
# track eta
# track phi
# track deltar
#
# After preprocessing
#
# print('Preprocessing data...')
# j1_inp = preproc_for_lstm(j1_dat.copy(deep=True), feats, mask, n_constits)
# j2_inp = preproc_for_lstm(j2_dat.copy(deep=True), feats, mask, n_constits)
# print(f'Preprocessed data: shape={j1_inp.shape} \n')




