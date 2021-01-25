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

print('Loading data...')
j1_dat, j2_dat, label = combine_SB(B_path, S_path, N, sig_frac)
print(f'Loaded  data: {len(label)} training examples \n')

# Before preprocessing
bkg1, bkg2 = j1_dat.iloc[~label.astype(bool)], j2_dat.iloc[~label.astype(bool)]
sig1, sig2 = j1_dat.iloc[label.astype(bool)], j2_dat.iloc[label.astype(bool)]

# Jet features
hist_dict = {'density': True, 'histtype': 'step', 'bins': 100}
label = ['bkg1', 'sig1', 'bkg2', 'sig2']
# multiplicity
plot_mult(bkg1.mult, sig1.mult, bkg2.mult, sig2.mult, save_path=output_path+'mult.png')

# jet pt
feat = 'jet_PT'
plt.figure()
plt.hist([bkg1[feat], sig1[feat], bkg2[feat], sig2[feat]],
         label=label, range=[0, 400], **hist_dict)
plt.legend()
plt.savefig(output_path+feat)

# jet eta
feat = 'jet_Eta'
plt.figure()
plt.hist([bkg1[feat], sig1[feat], bkg2[feat], sig2[feat]],
         label=label, **hist_dict)
plt.legend()
plt.savefig(output_path+feat)

# jet phi
feat = 'jet_Phi'
plt.figure()
plt.hist([bkg1[feat], sig1[feat], bkg2[feat], sig2[feat]],
         label=label, **hist_dict)
plt.legend()
plt.savefig(output_path+feat)





