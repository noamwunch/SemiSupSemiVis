import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt

from semisup import combine_SB, determine_feats
from UTILS.plots_and_logs import plot_rocs

from UTILS.lstm_classifier_modular import preproc_for_lstm, create_lstm_classifier, train_classifier
from UTILS.dense_classifier import preproc_for_dense, create_dense_classifier

exp_dir_path = "/gpfs0/kats/users/wunch/SemiSupSemiVis/test_fullsup_highlevel/"
model1_save_path = "/gpfs0/kats/users/wunch/SemiSupSemiVis/test_fullsup_highlevel/j1/"
model2_save_path = "/gpfs0/kats/users/wunch/SemiSupSemiVis/test_fullsup_highlevel/j2/"

B_path = "/gpfs0/kats/users/wunch/semisup_dataset/bkg_bb_GenMjjGt800_GenPtGt40_GenEtaSt3_MjjGt1000_PtGt50_EtaSt2.5_y*lt1/train"
S_path = "/gpfs0/kats/users/wunch/semisup_dataset/sig_dl0.5_rinv0.00_mZp1500_lambda20_GenMjjGt800_GenPtGt40_GenEtaSt3_MjjGt1000_PtGt50_EtaSt2.5_y*lt1/train"
Btest_path = "/gpfs0/kats/users/wunch/semisup_dataset/bkg_bb_GenMjjGt800_GenPtGt40_GenEtaSt3_MjjGt1000_PtGt50_EtaSt2.5_y*lt1/test"
Stest_path = "/gpfs0/kats/users/wunch/semisup_dataset/sig_dl0.5_rinv0.00_mZp1500_lambda20_GenMjjGt800_GenPtGt40_GenEtaSt3_MjjGt1000_PtGt50_EtaSt2.5_y*lt1/test"

Ntrain = 50000
Ntest = 10000
epochs = 20

print('Loading train data...')
j1_df, j2_df, event_labels = combine_SB(B_path, S_path, Ntrain, 0.5)
print('Training data loaded')

print('Train NN for jet 1')
preproc_create_train(j1_df, model1_save_path, epochs)
print('Finished training NN for jet 1')

print('Train NN for jet 2')
preproc_create_train(j2_df, model2_save_path, epochs)
print('Finished training NN for jet 2')

print('Loading test data...')
j1test_df, j2test_df, event_labels_test = combine_SB(Btest_path, Stest_path, Ntest, 0.5)
print('Test data loaded')

print('Infer jet 1 of test set')
j1_lstmpreds = preproc_load_infer(j1test_df, model1_save_path)
j1_multpreds = j1test_df.mult
print('Finished inferring jet 1 of test set')

print('Infer jet 2 of test set')
j2_lstmpreds = preproc_load_infer(j2test_df, model2_save_path)
j2_multpreds = j2test_df.mult
print('Finished inferring jet 1 of test set')

print("ploting ROCs")
lstmpreds = j1_lstmpreds + j2_lstmpreds
multpreds = j1_multpreds + j2_multpreds

plt.figure()
lstm_hist_preds = [lstmpreds[event_labels_test], lstmpreds[~event_labels_test]]
hist_dict = dict(histtype='step', density=True)
labels = ["S - LSTM", "B - LSTM"]
plt.hist(lstm_hist_preds, label=labels, **hist_dict)
plt.xlabel('LSTM output')
plt.legend()
plt.savefig(exp_dir_path+'LSTM_out_hist.png')

classifier_dicts = {'LSTM event classifier': {'probS': lstmpreds, 'plot_dict': {'linestyle': '-'}},
                    'LSTM classifier on j1': {'probS': j1_lstmpreds, 'plot_dict': {'linestyle': '-'}},
                    'LSTM classifier on j2': {'probS': j2_lstmpreds, 'plot_dict': {'linestyle': '-'}},
                    'Multiplicity classifier': {'probS': multpreds, 'plot_dict': {'linestyle': '--'}},
                    'Multiplicity classifier on j1': {'probS': j1_multpreds, 'plot_dict': {'linestyle': '--'}},
                    'Multiplicity classifier on j2': {'probS': j2_multpreds, 'plot_dict': {'linestyle': '--'}}}
with np.errstate(divide='ignore'):
    plot_rocs(classifier_dicts=classifier_dicts, true_lab=event_labels_test,
              save_path=exp_dir_path+'log_ROC.png')

