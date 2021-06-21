import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt

from semisup import combine_SB, determine_feats
from UTILS.plots_and_logs import plot_rocs
from UTILS.lstm_classifier_modular import train_classifier

from UTILS.lstm_classifier_modular import preproc_for_lstm, create_lstm_classifier, train_classifier
from UTILS.dense_classifier import preproc_for_dense, create_dense_classifier

exp_dir_path = "/gpfs0/kats/users/wunch/SemiSupSemiVis/test_fullsup_highlevel/"
model1_save_path = "/gpfs0/kats/users/wunch/SemiSupSemiVis/test_fullsup_highlevel/j1/"
model2_save_path = "/gpfs0/kats/users/wunch/SemiSupSemiVis/test_fullsup_highlevel/j2/"

B_path = "/gpfs0/kats/users/wunch/semisup_dataset/bkg_bb_GenMjjGt800_GenPtGt40_GenEtaSt3_MjjGt1000_PtGt50_EtaSt2.5_y*lt1/train"
S_path = "/gpfs0/kats/users/wunch/semisup_dataset/sig_dl0.5_rinv0.00_mZp1500_lambda20_GenMjjGt800_GenPtGt40_GenEtaSt3_MjjGt1000_PtGt50_EtaSt2.5_y*lt1/train"
Btest_path = "/gpfs0/kats/users/wunch/semisup_dataset/bkg_bb_GenMjjGt800_GenPtGt40_GenEtaSt3_MjjGt1000_PtGt50_EtaSt2.5_y*lt1/test"
Stest_path = "/gpfs0/kats/users/wunch/semisup_dataset/sig_dl0.5_rinv0.00_mZp1500_lambda20_GenMjjGt800_GenPtGt40_GenEtaSt3_MjjGt1000_PtGt50_EtaSt2.5_y*lt1/test"

Ntrain = int(1e5)
Ntest = int(2e4)
epochs = 40

def preproc_create_train_dense(j_df):
    feats = ['constit_mult', 'vert_count', 'ptwmean_dR', 'ptwmean_absD0', 'ptwmean_absDZ', 'c1b', 'photonE_over_jetpt']
    inp = preproc_for_dense(j_df, feats=feats)
    model = create_dense_classifier()
    train_classifier_dense(X, y, model, model_save_path, epochs, batch_size, log=''):
    pass

def preproc_infer_dense():
    pass

def preproc_create_train_lstm():
    pass

def preproc_infer_lstm():
    pass

def preproc_infer_verts():
    pass

print('Loading train data...')
j1_df, j2_df, event_labels = combine_SB(B_path, S_path, Ntrain, 0.5)
print('Training data loaded')

print('Train NN for jet 1')
model1_dense = preproc_create_train_dense()
model1_lstm = preproc_create_train_dense()
print('Finished training NN for jet 1')

print('Train NN for jet 2')
model2_dense = preproc_create_train_dense()
model2_lstm = preproc_create_train_dense()
print('Finished training NN for jet 2')

print('Loading test data...')
j1test_df, j2test_df, event_labels_test = combine_SB(Btest_path, Stest_path, Ntest, 0.5)
print('Test data loaded')

print('Infer jet 1 of test set')
j1_densepreds = preproc_infer_dense(j1test_df, model1_dense)
j1_lstmpreds = preproc_infer_lstm(j1test_df, model1_lstm)
j1_vertpreds = preproc_infer_verts(j1test_df)
print('Finished inferring jet 1 of test set')

print('Infer jet 2 of test set')
j2_densepreds = preproc_infer_dense(j2test_df, model2_dense)
j2_lstmpreds = preproc_infer_lstm(j2test_df, model2_lstm)
j2_vertpreds = preproc_infer_verts(j2test_df)
print('Finished inferring jet 2 of test set')

print("ploting ROCs")
densepreds = j1_densepreds*j2_densepreds
lstmpreds = j1_lstmpreds*j2_lstmpreds
vertpreds = j1_vertpreds + j2_vertpreds

classifier_dicts = {'LSTM event classifier': {'probS': lstmpreds, 'plot_dict': {'linestyle': '-'}},
                    'Dense event classifier': {'probS': densepreds, 'plot_dict': {'linestyle': '-'}},
                    'Vertex count classifier': {'probS': vertpreds, 'plot_dict': {'linestyle': '--'}},
                    }
with np.errstate(divide='ignore'):
    plot_rocs(classifier_dicts=classifier_dicts, true_lab=event_labels_test,
              save_path=exp_dir_path+'log_ROC.png')

