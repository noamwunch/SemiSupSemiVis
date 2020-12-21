# System
import sys
from pathlib import Path
from timeit import default_timer as timer
from datetime import timedelta

# Standard
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# ML
import tensorflow as tf
from tensorflow import keras

# Custom
from UTILS.utils import evs_txt2jets_df as load_data
from UTILS.utils import create_one_hot_encoder, nominal2onehot
from UTILS.lstm_classifier import preproc_for_lstm, create_lstm_classifier, train_classifier
from UTILS.plots_and_logs import log_args, log_events_info, log_semisup_labels_info, log_nn_inp_info
from UTILS.plots_and_logs import plot_event_histograms, plot_nn_inp_histograms, plot_learn_curve, plot_rocs, plot_nn_hists

def determine_feats(with_displacement, with_deltar, with_pid):
    feats = ["constit_relPT", "constit_relEta", "constit_relPhi"]
    n_cols = 3
    if with_displacement == 'True':
        feats.extend(["constit_relDZ", "constit_D0"])
        n_cols += 2
    if with_deltar == "True":
        feats.append("constit_deltaR")
        n_cols += 1
    if with_pid == "True":
        feats.append("constit_PID")
        n_cols += 8
    return feats, n_cols

def parse_args(argv):
    ## Hard coded params
    mask = -10.0
    n_constits = 80
    pid = [mask, -2212, -321, -211, -13, -11, 0, 1, 11, 13, 211, 321, 2212]
    classification = ['masked', 'h-', 'h-', 'h-', 'mu-', 'e-', 'photon', 'h0', 'e+', 'mu+', 'h+', 'h+', 'h+']
    class_dict = dict(zip(pid, classification))

    ## Data prep params
    B_path, S_path, exp_dir_path, N, sig_frac =  argv[1], argv[2], argv[3], int(argv[4]), float(argv[5])

    ## Determine features and nn columns
    with_displacement, with_deltar, with_pid = argv[6], argv[7], argv[8]
    feats, n_cols = determine_feats(with_displacement, with_deltar, with_pid)
    if with_pid:
        enc = create_one_hot_encoder(class_dict)
    else:
        enc = None

    ## unsup classifier params
    unsup_dict = {'unsup_type': argv[9]}

    ## semisup classifier params
    # General
    semisup_dict = {'feats': feats, 'with_pid': with_pid, 'n_cols': n_cols, 'enc': enc, 'class_dict': class_dict,
                    'epochs': int(argv[10]), 'reg_dict': {}, 'n_constits': n_constits,  'mask': mask}
    # Regularization
    # Weight regularization
    weight_reg_params = ["kernel_regularizer", "recurrent_regularizer", "bias_regularizer"]
    weight_reg_dict = {param: None if arg=="None" else
                       keras.regularizers.l2(float(arg)) for param, arg in zip(weight_reg_params, argv[11:14])}
    # Dropout
    drop_params = ["dropout", "recurrent_dropout"]
    drop_dict = {param: float(arg) for param, arg in zip(drop_params, argv[14:16])}

    semisup_dict['reg_dict'] = {**weight_reg_dict, **drop_dict}

    log_path = exp_dir_path + 'log.txt'

    return B_path, S_path, exp_dir_path, N, sig_frac, unsup_dict, semisup_dict, log_path

def combine_SB(B_path, S_path, N, sig_frac):
    n_B, n_S = int(N*(1 - sig_frac)), int(N * sig_frac)
    (B_j1_df, B_j2_df), (S_j1_df, S_j2_df) = load_data(B_path, n_ev=n_B), load_data(S_path, n_ev=n_S)
    j1_df = pd.concat([B_j1_df, S_j1_df]).reset_index(drop=True)
    j2_df = pd.concat([B_j2_df, S_j2_df]).reset_index(drop=True)
    event_label = np.array([0]*n_B + [1]*n_S)
    return j1_df, j2_df, event_label

def infer_unsup(j_df, unsup_dict):
    if unsup_dict['unsup_type'] == 'constituent_mult':
        j_unsup_probS = j_df.mult
    else:
        raise ValueError(f'Unsuported unsup method: {unsup_dict["unsup_type"]}')
    return j_unsup_probS

def train_infer_semisup(j_df, j_semisup_lab, model_save_path, param_dict):
    Path(model_save_path).mkdir(parents=True, exist_ok=True)
    log = ''
    # Create model
    model, log = create_lstm_classifier(param_dict['n_constits'], param_dict['n_cols'], param_dict['reg_dict'],
                                        param_dict['mask'], log=log)
    # Preprocessing
    j_semisup_inp = preproc_for_lstm(j_df, param_dict['feats'], param_dict['mask'], param_dict['n_constits'])
    if param_dict['with_pid'] == "True":
        j_semisup_inp = nominal2onehot(j_semisup_inp, param_dict['class_dict'], param_dict['enc'])

    plt.figure()
    plt.hist(j_semisup_inp[:, 0, 0], label='track 1', bins=100, histtype='step', range=[0, 10])
    plt.hist(j_semisup_inp[:, 1, 0], label='track 2', bins=100, histtype='step', range=[0, 10])
    plt.hist(j_semisup_inp[:, 4, 0], label='track 5', bins=100, histtype='step', range=[0, 10])
    plt.hist(j_semisup_inp[:, 9, 0], label='track 10', bins=100, histtype='step', range=[0, 10])
    plt.legend(loc='best')
    plt.savefig(model_save_path + 'PT')

    # Train model
    hist, log = train_classifier(j_semisup_inp, j_semisup_lab, model=model, model_save_path=model_save_path,
                                 epochs=param_dict['epochs'], log=log)

    # Load and infer model
    model = keras.models.load_model(model_save_path)
    j_semisup_probS = model.predict(j_semisup_inp).flatten()

    # nn_input hisogram for debug (NOT IMPLEMENTED)
    plot_nn_inp_histograms(j_semisup_inp)

    return j_semisup_probS, hist, log

def main_semisup(argv):
    B_path, S_path, exp_dir_path, N, sig_frac, unsup_dict, semisup_dict, log_path = parse_args(argv)
    Path(exp_dir_path).mkdir(parents=True, exist_ok=True)

    ## Data prep
    j1_df, j2_df, event_label = combine_SB(B_path, S_path, N, sig_frac)

    ## First (unsupervised) classifier
    j1_unsup_probS, j2_unsup_probS = infer_unsup(j1_df, unsup_dict), infer_unsup(j2_df, unsup_dict)

    ## Second (semisupervised) classifiers
    # semisup labels for one jet are unsup predictions for the other jet.
    j1_thresh = np.median(j1_unsup_probS)
    j2_thresh = np.median(j2_unsup_probS)
    j1_semisup_lab = j2_unsup_probS > j2_thresh
    j2_semisup_lab = j1_unsup_probS > j1_thresh
    # create model, preprocess, train, and infer
    j1_semisup_probS, hist1, log1 = train_infer_semisup(j1_df, j1_semisup_lab,
                                                        model_save_path=exp_dir_path + 'j1/',
                                                        param_dict=semisup_dict)
    j2_semisup_probS, hist2, log2 = train_infer_semisup(j2_df, j2_semisup_lab,
                                                        model_save_path=exp_dir_path + 'j2/',
                                                        param_dict=semisup_dict)

    ## Average of both jet classifiers serves as a final event prediction.
    # semisupervised prediction
    event_semisup_probS = (j1_semisup_probS + j2_semisup_probS)/2
    # unsupervised prediction for benchmark
    event_unsup_probS = (j1_unsup_probS + j2_unsup_probS)/2

    ## Logs and plots
    # Logs
    log_args(log_path, B_path, S_path, exp_dir_path, unsup_dict, semisup_dict)
    log_events_info(log_path, event_label)
    log_semisup_labels_info(log_path, j1_semisup_lab, j2_semisup_lab, j1_thresh, j2_thresh, event_label)
    log_nn_inp_info(log_path, log1, log2)

    # Plots
    plot_event_histograms(exp_dir_path, j1_df, j2_df, event_label)
    plot_learn_curve(hist1, save_path=exp_dir_path+'nn1_learn_curve.pdf')
    plot_learn_curve(hist2, save_path=exp_dir_path+'nn2_learn_curve.pdf')

    # rocs and nn histograms
    probS_dict = {'semisup event classifier': event_semisup_probS, 'unsup event classifier': event_unsup_probS,
                  'semisup classifier on j1': j1_semisup_probS, 'unsup classifier on j1': j1_unsup_probS,
                  'semisup classifier on j2': j2_semisup_probS, 'unsup classifier on j2': j2_unsup_probS}

    plot_rocs(probS_dict=probS_dict, true_lab=event_label, save_path=exp_dir_path + 'log_ROC.pdf')
    plot_nn_hists(probS_dict=probS_dict, true_lab=event_label, save_path=exp_dir_path + 'nn_hist.pdf')

if __name__ == '__main__':
    start = timer()
    main_semisup(sys.argv)
    end = timer()
    print('elapsed time = {}'.format(timedelta(seconds=(end - start))))

'''
##################################
#elif unsup_dict['unsup_type'] == 'autencoder':
        # ae_classifier = load()
        # j1_ae_inp, j2_ae_inp = preprocess_for_ae(j1_df, j2_df, unsup_dict)
        # j1_unsup_probS, j2_unsup_probS = (infer_classfier(ae_classifier, j1_ae_inp, unsup_dict),
        #                                   infer_classifier(ae_classifier, j2_ae_inp, unsup_dict))
def preprocess_for_ae(unsup_dict):
    j1_ae_inp = 1
    j2_ae_inp = 2
    return j1_ae_inp, j2_ae_inp
def ae_classifier(a):
    return ae_classifier
##################################
'''