import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt

from semisup import combine_SB, determine_feats
from UTILS.plots_and_logs import plot_rocs
from UTILS.lstm_classifier_modular import train_classifier

from UTILS.lstm_classifier_modular import preproc_for_lstm, create_lstm_classifier, train_classifier
from UTILS.dense_classifier import preproc_for_dense, create_dense_classifier

exp_dir_path = "/gpfs0/kats/users/wunch/SemiSupSemiVis/test_fullsup_highlevel/"

B_path = "/gpfs0/kats/users/wunch/semisup_dataset/bkg_bb_GenMjjGt800_GenPtGt40_GenEtaSt3_MjjGt1000_PtGt50_EtaSt2.5_y*lt1/train"
S_path = "/gpfs0/kats/users/wunch/semisup_dataset/sig_dl0.5_rinv0.00_mZp1500_lambda20_GenMjjGt800_GenPtGt40_GenEtaSt3_MjjGt1000_PtGt50_EtaSt2.5_y*lt1/train"
Btest_path = "/gpfs0/kats/users/wunch/semisup_dataset/bkg_bb_GenMjjGt800_GenPtGt40_GenEtaSt3_MjjGt1000_PtGt50_EtaSt2.5_y*lt1/test"
Stest_path = "/gpfs0/kats/users/wunch/semisup_dataset/sig_dl0.5_rinv0.00_mZp1500_lambda20_GenMjjGt800_GenPtGt40_GenEtaSt3_MjjGt1000_PtGt50_EtaSt2.5_y*lt1/test"

Ntrain = int(1e5)
Ntest = int(2e4)

# Preproc create
def preproc_create_train_dense(j_df, event_labs, model_save_path, preprocparams):
    epochs = 40
    batch_size = 1024

    inp = preproc_for_dense(j_df, **preprocparams)
    model = create_dense_classifier()
    hist, log = train_classifier(inp, event_labs, model, epochs, batch_size, log='', model_save_path=model_save_path)
    return hist

def preproc_create_train_lstm(j_df, event_labs, model_save_path, preprocparams):
    epochs = 40
    batch_size = 1024

    inp = preproc_for_dense(j_df, **preprocparams)
    model = create_dense_classifier()
    hist, log = train_classifier(inp, event_labs, model, epochs, batch_size, log='', model_save_path=model_save_path)
    return hist

# Preproc infer
def preproc_infer_dense(j_df, model_save_path, preprocparams):
    batch_size = 1024

    inp = preproc_for_dense(j_df, **preprocparams)
    model = keras.models.load_model(model_save_path)
    preds = np.array(model.predict(inp, batch_size=batch_size)).flatten()
    return preds

def preproc_infer_lstm(j_df, model_save_path, preprocparams):
    batch_size = 1024

    inp = preproc_for_dense(j_df, **preprocparams)
    model = keras.models.load_model(model_save_path)
    preds = np.array(model.predict(inp, batch_size=batch_size)).flatten()
    return preds

def preproc_infer_verts():
    pass

def preproc_infer_mult():
    pass

# Preproc params
## Dense
model1_savepath_dense = exp_dir_path + "dense/j1/"
model2_savepath_dense = exp_dir_path + "dense/j2/"
preprocparams_dense = dict(feats='all')
modelparams_dense = dict()
## LSTM
n_constits = 100
with_displacement = True
with_deltar = True
with_pid = True
mask = -10.0
model1_savepath_lstm = exp_dir_path + "lstm/j1/"
model2_savepath_lstm = exp_dir_path + "lstm/j2/"
feats, n_cols = determine_feats(with_displacement,
                                with_deltar,
                                with_pid)
preprocparams_lstm = dict(feats=feats, n_constits=n_constits, mask=mask)

print('Loading train data...')
j1_df, j2_df, event_labels = combine_SB(B_path, S_path, Ntrain, 0.5)
print('Training data loaded')

print('Train NN for jet 1')
model1_dense = preproc_create_train_dense(j1_df, event_labels, model1_savepath_dense, preprocparams_dense)
model1_lstm = preproc_create_train_dense(j1_df, event_labels, model1_savepath_dense, preprocparams_dense)
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

