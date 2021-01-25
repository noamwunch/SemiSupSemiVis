from pathlib import Path

import numpy as np

from semisup import combine_SB
from semisup import determine_feats
from UTILS.lstm_classifier import preproc_for_lstm, create_lstm_classifier, train_classifier
from UTILS.plots_and_logs import plot_rocs, plot_mult

output_path = "./RESULTS/fullsup/rinv0.25sf0.20/"
B_path_test = "/gpfs0/kats/users/wunch/semisup_evs/bkg/test"
S_path_test = "/gpfs0/kats/users/wunch/semisup_evs/sig_rinv_0.25_mjj_500/test"
B_path = "/gpfs0/kats/users/wunch/semisup_evs/bkg/train"
S_path = "/gpfs0/kats/users/wunch/semisup_evs/sig_rinv_0.25_mjj_500/train"
Path(output_path).mkdir(parents=True, exist_ok=True)

Ntrain = 100000
Ntest = 20000
sig_frac = 0.5
epochs = 15
reg_dict = {'dropout': 0.1, 'recurrent_dropout': 0.2}
mask = -10.0
n_constits = 80
feats, n_cols = determine_feats(with_displacement='True',
                                with_deltar='True',
                                with_pid='False')

# Train
print('Loading training data...')
j1_dat, j2_dat, label = combine_SB(B_path, S_path, Ntrain, sig_frac)
print(f'Loaded training data: {len(label)} training examples \n')

print('Preprocessing training data...')
j1_inp = preproc_for_lstm(j1_dat.copy(deep=True), feats, mask, n_constits)
j2_inp = preproc_for_lstm(j2_dat.copy(deep=True), feats, mask, n_constits)
print(f'Preprocessed training data: shape={j1_inp.shape} \n')

print('Creating models...')
model1, _ = create_lstm_classifier(n_constits, n_cols, reg_dict, mask)
model2, _ = create_lstm_classifier(n_constits, n_cols, reg_dict, mask)
print('Created models \n')

print('Training classifiers...')
hist1, _ = train_classifier(j1_inp, label, model1,
                            model_save_path=output_path + "j1/",
                            epochs=epochs)
hist2, _ = train_classifier(j2_inp, label, model2,
                            model_save_path=output_path + "j2/",
                            epochs=epochs)
print('Trained classifiers \n')

# Test
print('Loading testing data...')
j1_dat_test, j2_dat_test, label_test = combine_SB(B_path_test, S_path_test, Ntest, sig_frac)
print(f'Loaded testing data: {len(label_test)} test examples \n')

print('Preprocessing testing data...')
j1_inp_test = preproc_for_lstm(j1_dat_test.copy(deep=True), feats, mask, n_constits)
j2_inp_test = preproc_for_lstm(j2_dat_test.copy(deep=True), feats, mask, n_constits)
print(f'Preprocessed testing data: shape={j1_inp_test.shape} \n')

print('Predicting testing data...')
preds1 = model1.predict(j1_inp_test, batch_size=512).flatten()
preds2 = model2.predict(j2_inp_test, batch_size=512).flatten()
print('Predicted testing data \n')

print('Plotting rocs and multiplicity...')
mult1 = j1_dat_test.mult
mult2 = j2_dat_test.mult

classifier_dicts = {'fullysup average': {'probS': (preds1+preds2)/2, 'plot_dict': {'linestyle': '-'}},
                    'fullysup classifier on j1': {'probS': preds1, 'plot_dict': {'linestyle': '-'}},
                    'fully classifier on j2': {'probS': preds2, 'plot_dict': {'linestyle': '-'}},
                    'mult average': {'probS': (mult1+mult2)/2, 'plot_dict': {'linestyle': '--'}},
                    'mult j1': {'probS': mult1, 'plot_dict': {'linestyle': '--'}},
                    'mult j2': {'probS': mult2, 'plot_dict': {'linestyle': '--'}}}

with np.errstate(divide='ignore'):
    plot_rocs(classifier_dicts=classifier_dicts, true_lab=label_test,
              save_path=output_path+'log_ROC.png')

plot_mult(j1_dat_test.mult[label_test.astype(bool)],
          j2_dat_test.mult[label_test.astype(bool)],
          j1_dat_test.mult[~label_test.astype(bool)],
          j2_dat_test.mult[~label_test.astype(bool)],
          save_path=output_path+'mult.png')
print('Plotted rocs and multiplicity')
print('Done!')
