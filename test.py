from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from tensorflow import keras

from UTILS.lstm_classifier import preproc_for_lstm
from semisup import combine_SB_old, combine_SB
from UTILS.plots_and_logs import plot_rocs

plot_path = "RESULTS/test/"
Path(plot_path).mkdir(parents=True, exist_ok=True)

# B_path = "/gpfs0/kats/users/wunch/semisup_data/bkg"
# S_path = "/gpfs0/kats/users/wunch/semisup_data/sig"

B_path = "/gpfs0/kats/users/wunch/semisup_evs/bkg"
S_path = "/gpfs0/kats/users/wunch/semisup_evs/sig_rinv_0.25_mjj_500"

# B_path = "/gpfs0/kats/users/wunch/semisup_evs/bkg"
# S_path = "/gpfs0/kats/users/wunch/semisup_evs/sig_rinv_0.00_mjj_500"

# B_path = "/gpfs0/kats/users/wunch/semisup_evs/bkg"
# S_path = "/gpfs0/kats/users/wunch/semisup_evs/sig_rinv_0.50_mjj_500"

# B_path = "/gpfs0/kats/users/wunch/semisup_evs/bkg"
# S_path = "/gpfs0/kats/users/wunch/semisup_evs/sig_rinv_0.50_mjj_500_gen_ptcut"

# B_path = "/gpfs0/kats/users/wunch/semisup_evs/bkg"
# S_path = "/gpfs0/kats/users/wunch/semisup_evs/sig_rinv_0.50_mjj_500_gen_ptcut_rem_mjjcut"

old = False
pt_cut = True
exp_dir_path = "RESULTS/example_grid/iter_1/"
j1_model_save_path = exp_dir_path + f'j1_0/'
j2_model_save_path = exp_dir_path + f'j2_0/'

sig_frac = 0.5
N = 100000
mask = -10.0
n_constits = 80
feats = ["constit_relPT", "constit_relEta", "constit_relPhi",
         "constit_relDZ", "constit_D0",
         "constit_deltaR"]
if old:
    combine_SB = combine_SB_old

print('loading events...')
j1_df, j2_df, event_label = combine_SB(B_path, S_path, N, sig_frac)
print(f'finished loading events: N={len(event_label)}')

if pt_cut:
    PT_min = 100
    PT_max = 200
    valid_idx = (j1_df.jet_PT > PT_min) & (j2_df.jet_PT > PT_min) & (
                 j1_df.jet_PT < PT_max) & (j2_df.jet_PT < PT_max)
    j1_df = j1_df[valid_idx]
    j2_df = j2_df[valid_idx]
    event_label = event_label[valid_idx]
    print(f'number of signal events that passed PT cut = {sum(event_label)}'
          f' out of {int(sig_frac*len(event_label))}')
    print(f'number of background events that passed PT cut = {len(event_label) - sum(event_label)} '
          f'out of {int((1-sig_frac)*len(event_label))}')

print(j1_df.jet_PT.head())

j1_inp = preproc_for_lstm(j1_df, feats, mask, n_constits)
j2_inp = preproc_for_lstm(j2_df, feats, mask, n_constits)

model1 = keras.models.load_model(j1_model_save_path)
model2 = keras.models.load_model(j2_model_save_path)
print(f'Finished Loading models')

print(f'Predicting jet1')
preds1 = model1.predict(j1_inp, verbose=1, batch_size=512).flatten()
print(f'Finished prediciting jet1')
print(f'Predicting jet2')
preds2 = model2.predict(j2_inp, verbose=1, batch_size=512).flatten()
print(f'Finished prediciting jet2')
preds_comb = (preds1 + preds2) * 0.5

mult1 = j1_df.mult
mult2 = j2_df.mult
mult_comb = (mult1 + mult2) * 0.5

classifier_dicts = {'semisup event classifier': {'probS': preds_comb, 'plot_dict': {'linestyle': '-'}},
                    'semisup classifier on j1': {'probS': preds1, 'plot_dict': {'linestyle': '-'}},
                    'semisup classifier on j2': {'probS': preds2, 'plot_dict': {'linestyle': '-'}},
                    'unsup event classifier': {'probS': mult_comb, 'plot_dict': {'linestyle': '--'}},
                    'unsup classifier on j1': {'probS': mult1, 'plot_dict': {'linestyle': '--'}},
                    'unsup classifier on j2': {'probS': mult2, 'plot_dict': {'linestyle': '--'}}}

print(f'Plotting rocs')
plot_rocs(classifier_dicts=classifier_dicts, true_lab=event_label,
          save_path=plot_path+ 'ROC_new_same_ptcut_rinv0.25.png')



