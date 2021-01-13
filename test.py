from pathlib import Path

from tensorflow import keras

from UTILS.lstm_classifier import preproc_for_lstm
from semisup import combine_SB_old
from UTILS.plots_and_logs import plot_rocs

plot_path = "RESULTS/test"
Path(plot_path).mkdir(parents=True, exist_ok=True)

B_path = "/gpfs0/kats/users/wunch/semisup_data/bkg"
S_path = "/gpfs0/kats/users/wunch/semisup_data/sig"
exp_dir_path = "RESULTS/example_grid/iter_1/"
j1_model_save_path = exp_dir_path+f'j1_0/'
j2_model_save_path = exp_dir_path+f'j2_0/'

sig_frac = 0.2
N = 4000
mask = -10.0
n_constits = 80
feats = ["constit_relPT", "constit_relEta", "constit_relPhi",
         "constit_relDZ", "constit_D0",
         "constit_deltaR"]

j1_df, j2_df, event_label = combine_SB_old(B_path, S_path, N, sig_frac)

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
          save_path=exp_dir_path+'log_ROC_new.pdf')





