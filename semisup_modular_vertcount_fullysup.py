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
from tensorflow import keras

# Custom
## Load
from UTILS.utils import evs_txt2jets_df as load_data_old
from UTILS.utils import evs_txt2jets_df_with_verts_withparton as load_data

## General
from UTILS.plots_and_logs import log_args, log_events_info, log_semisup_labels_info, log_nn_inp_info
from UTILS.plots_and_logs import log_semisup_labels_info_new
from UTILS.plots_and_logs import plot_learn_curve, plot_rocs, plot_rocs_significance, plot_nn_hists
from UTILS.plots_and_logs import plot_mult

## LSTM
from UTILS.lstm_classifier_modular import preproc_for_lstm, create_lstm_classifier, train_classifier
from UTILS.lstm_classifier_modular import plot_nn_inp_histograms_lstm, plot_event_histograms_lstm

## Dense
from UTILS.dense_classifier import preproc_for_dense, create_dense_classifier
from UTILS.dense_classifier import plot_nn_inp_histograms_dense, plot_event_histograms_dense

global preproc_handle, create_model_handle, plot_event_histograms_handle, plot_nn_inp_histograms_handle

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

def combine_SB_old(B_path, S_path, N, sig_frac):
    n_B, n_S = int(N*(1 - sig_frac)), int(N * sig_frac)

    idxs = np.arange(n_B+n_S)
    np.random.shuffle(idxs)

    event_label = np.array([0]*n_B + [1]*n_S)[idxs]

    (B_j1_df, B_j2_df), (S_j1_df, S_j2_df) = load_data_old(B_path, n_ev=n_B), load_data_old(S_path, n_ev=n_S)
    j1_df = pd.concat([B_j1_df, S_j1_df]).iloc[idxs].reset_index(drop=True)
    j2_df = pd.concat([B_j2_df, S_j2_df]).iloc[idxs].reset_index(drop=True)
    return j1_df, j2_df, event_label

def combine_SB(B_path, S_path, N, sig_frac):
    mjj_range = (1200, 1500)
    n_B, n_S = int(N*(1 - sig_frac)), int(N * sig_frac)

    idxs = np.arange(n_B+n_S)
    np.random.shuffle(idxs)

    event_label = np.array([0]*n_B + [1]*n_S)[idxs]

    S_j1_df, S_j2_df = load_data(S_path, n_ev=n_S, mjj_range=mjj_range)
    B_j1_df, B_j2_df = load_data(B_path, n_ev=n_B, mjj_range=mjj_range)

    if n_S==0:
        j1_df = B_j1_df.iloc[idxs].reset_index(drop=True)
        j2_df = B_j2_df.iloc[idxs].reset_index(drop=True)
    elif n_B==0:
        j1_df = S_j1_df.iloc[idxs].reset_index(drop=True)
        j2_df = S_j2_df.iloc[idxs].reset_index(drop=True)
    else:
        j1_df = pd.concat([B_j1_df, S_j1_df]).iloc[idxs].reset_index(drop=True)
        j2_df = pd.concat([B_j2_df, S_j2_df]).iloc[idxs].reset_index(drop=True)
    return j1_df, j2_df, event_label

class jet_mult_classifier:
    def predict(self, jet_df, **kwargs):
        return jet_df.mult

class jet_vert_classifier:
    def predict(self, jet_df, **kwargs):
        return jet_df.n_verts

def filter_quantile(train_set, preds, bkg_quant, sig_quant):
    assert (bkg_quant+sig_quant)<=1, 'The sum of signal and background quantiles should be smaller than 1'
    bkg_thresh, sig_thresh = np.quantile(preds, [bkg_quant, 1-sig_quant])
    valid_idx = (preds>=sig_thresh) | (preds<=bkg_thresh)
    return valid_idx, sig_thresh

def preproc_train_fullysup(j_data, labels, param_dict,
                             model_save_path=None,
                             preproc_handle=None, create_model_handle=None,
                             preproc_args=None, create_model_args=None):
    global plot_nn_inp_histograms_handle
    batch_size = 8192

    ### Train NN on j1 using weak labels ###############################################################################
    Path(model_save_path).mkdir(parents=True, exist_ok=True)
    log = ''

    # Preprocessing of j1 for training
    j_inp = j_data.copy(deep=True)
    j_inp = preproc_handle(j_inp, **preproc_args)

    # Create model
    N_train = len(j_inp)
    model, log = create_model_handle(**create_model_args, N_train=N_train, batch_size=batch_size, log=log)

    # Train model
    if param_dict.get('train_nn', "True")=="True":
        hist, log = train_classifier(j_inp, labels, model=model, model_save_path=model_save_path,
                                     batch_size=batch_size, epochs=param_dict['epochs'], log=log)
    else:
        model = keras.models.load_model(model_save_path)
        hist = False

    return hist

def preproc_infer_fullysup(j_data,
                           model_save_path=None,
                           preproc_handle=None,
                           preproc_args=None):
    global plot_nn_inp_histograms_handle
    batch_size = 512

    ### Train NN on j1 using weak labels ###############################################################################
    Path(model_save_path).mkdir(parents=True, exist_ok=True)
    log = ''

    # Preprocessing of j1 for training
    j_inp = j_data.copy(deep=True)
    j_inp = preproc_handle(j_inp, **preproc_args)

    model = keras.models.load_model(model_save_path)

    return np.array(model.predict(j_inp, batch_size=batch_size)).flatten()

def main_fullysup(B_path, S_path, Btest_path, Stest_path, exp_dir_path, Ntrain=int(1e5), Ntest=int(1e4), sig_frac=0.2,
                 unsup_type='constituent_mult', unsup_dict=None, semisup_dict=None, n_iter=2, split_data="False"):
    """Runs semisupervised classification scheme on simulated event collection.

        Args:
            B_path: Path to directory containing background events
            S_path: Path to directory containing signal events
            exp_dir_path: Path to directory the results will be stored in. Directory will be created if it does
                not exits.
            N: Number of events in simulated collection.
            sig_frac: Fraction of signal events out of N total events.
            unsup_type: type of unsupervised method currently supports "constituent_mult"
            unsup_dict: to be passed to unsupervised method.
            semisup_dict: Dictionary holding parameters for unsupervised classification step.
                {'epochs': number of epochs for training,
                 'reg_dict': dictionary for regularization following keras LSTM signature}
                 'with_displacement': whether to use track displacement ("True" or "False" (str)).
                 'with_deltar': whether to use constituent deltaR. ("True" or "False" (str))
                 'with_pid': whether to use particle id. ("True" or "False" (str))
                 }
            n_iter: Number of semisupervised iterations.
            split_data: Whether to train on a different set each iteration ("True") or not ("False").
        Outputs saved to exp_dir_path:
            j1, j2: Tensorflow models trained on jet1 and jet2 respectively.
            log.txt: Log of data information.
    """
    global preproc_handle, create_model_handle, plot_event_histograms_handle, plot_nn_inp_histograms_handle
    Path(exp_dir_path).mkdir(parents=True, exist_ok=True)

    print(sig_frac)

########################################################################################################################
########################################################################################################################
    classifier_type = 'dense'

    ## Initialize classifier handles and arguments
    if classifier_type == 'lstm':
        mask = -10.0
        n_constits = 100
        preproc_handle = preproc_for_lstm
        create_model_handle = create_lstm_classifier
        plot_event_histograms_handle = plot_event_histograms_lstm
        plot_nn_inp_histograms_handle = plot_nn_inp_histograms_lstm
        # Determine features and nn columns
        feats, n_cols = determine_feats(semisup_dict['with_displacement'],
                                        semisup_dict['with_deltar'],
                                        semisup_dict['with_pid'])
        preproc_args = dict(feats=feats, n_constits=n_constits, mask=mask)
        create_model_args = dict(n_constits=n_constits, n_cols=n_cols, reg_dict=semisup_dict['reg_dict'], mask_val=mask)
    elif classifier_type == 'dense':
        all_feats = ['constit_mult', 'vert_count', 'ptwmean_dR', 'ptwmean_absD0', 'ptwmean_absDZ', 'c1b', 'photonE_over_jetpt']
        feats = ['constit_mult', 'vert_count', 'ptwmean_dR', 'ptwmean_absD0', 'ptwmean_absDZ', 'photonE_over_jetpt']
        preproc_handle = preproc_for_dense
        create_model_handle = create_dense_classifier
        plot_event_histograms_handle = plot_event_histograms_dense
        plot_nn_inp_histograms_handle = plot_nn_inp_histograms_dense
        preproc_args = dict(feats=feats)
        create_model_args = dict(nfeats=len(feats))

    ## Data prep
    print('Loading train data...')
    j1_df, j2_df, event_label = combine_SB(B_path, S_path, Ntrain, sig_frac)
    print('Training data loaded')

    print('Training model on jet1...')
    hist1 = preproc_train_fullysup(j1_df,
                                   event_label,
                                   semisup_dict,
                                   exp_dir_path + classifier_type + '/j1/',
                                   preproc_handle=preproc_handle,
                                   create_model_handle=create_model_handle,
                                   preproc_args=preproc_args,
                                   create_model_args=create_model_args)
    print('Finished training model on jet1')
    print('Training model on jet2...')
    hist2 = preproc_train_fullysup(j2_df,
                                   event_label,
                                   semisup_dict,
                                   exp_dir_path + classifier_type + '/j2/',
                                   preproc_handle=preproc_handle,
                                   create_model_handle=create_model_handle,
                                   preproc_args=preproc_args,
                                   create_model_args=create_model_args)
    print('Finished training model on jet2')

########################################################################################################################
########################################################################################################################
    classifier_type = 'lstm'

    ## Initialize classifier handles and arguments
    if classifier_type == 'lstm':
        mask = -10.0
        n_constits = 100
        preproc_handle = preproc_for_lstm
        create_model_handle = create_lstm_classifier
        plot_event_histograms_handle = plot_event_histograms_lstm
        plot_nn_inp_histograms_handle = plot_nn_inp_histograms_lstm
        # Determine features and nn columns
        feats, n_cols = determine_feats(semisup_dict['with_displacement'],
                                        semisup_dict['with_deltar'],
                                        semisup_dict['with_pid'])
        preproc_args = dict(feats=feats, n_constits=n_constits, mask=mask)
        create_model_args = dict(n_constits=n_constits, n_cols=n_cols, reg_dict=semisup_dict['reg_dict'], mask_val=mask)
    elif classifier_type == 'dense':
        all_feats = ['constit_mult', 'vert_count', 'ptwmean_dR', 'ptwmean_absD0', 'ptwmean_absDZ', 'c1b', 'photonE_over_jetpt']
        feats = ['constit_mult', 'vert_count', 'ptwmean_dR', 'ptwmean_absD0', 'ptwmean_absDZ', 'photonE_over_jetpt']
        preproc_handle = preproc_for_dense
        create_model_handle = create_dense_classifier
        plot_event_histograms_handle = plot_event_histograms_dense
        plot_nn_inp_histograms_handle = plot_nn_inp_histograms_dense
        preproc_args = dict(feats=feats)
        create_model_args = dict(nfeats=len(feats))

    ## Data prep
    print('Loading train data...')
    j1_df, j2_df, event_label = combine_SB(B_path, S_path, Ntrain, sig_frac)
    print('Training data loaded')

    print('Training model on jet1...')
    hist1 = preproc_train_fullysup(j1_df,
                                   event_label,
                                   semisup_dict,
                                   exp_dir_path + classifier_type + '/j1/',
                                   preproc_handle=preproc_handle,
                                   create_model_handle=create_model_handle,
                                   preproc_args=preproc_args,
                                   create_model_args=create_model_args)
    print('Finished training model on jet1')
    print('Training model on jet2...')
    hist2 = preproc_train_fullysup(j2_df,
                                   event_label,
                                   semisup_dict,
                                   exp_dir_path + classifier_type + '/j2/',
                                   preproc_handle=preproc_handle,
                                   create_model_handle=create_model_handle,
                                   preproc_args=preproc_args,
                                   create_model_args=create_model_args)
    print('Finished training model on jet2')
########################################################################################################################
########################################################################################################################

    print('Loading test data')
    j1_test_df, j2_test_df, event_label_test = combine_SB(Btest_path, Stest_path, Ntest, 0.5)
    print('Test data loaded')

    print('Testing on test data...')
    ## Average of both jet classifiers serves as a final event prediction.
    print('Infering jet 1...')
    classifier_type = 'dense'
    j1_preds_dense = preproc_infer_fullysup(j1_test_df,
                                            exp_dir_path + classifier_type + '/j1/',
                                            preproc_handle=preproc_handle,
                                            preproc_args=preproc_args)
    classifier_type = 'lstm'
    j1_preds_lstm = preproc_infer_fullysup(j1_test_df,
                                           exp_dir_path + classifier_type + '/j1/',
                                           preproc_handle=preproc_handle,
                                           preproc_args=preproc_args)
    print('Finished infering jet 1')
    print('Infering jet 2...')
    classifier_type = 'dense'
    j2_preds_dense = preproc_infer_fullysup(j2_test_df,
                                            exp_dir_path + classifier_type + '/j2/',
                                            preproc_handle=preproc_handle,
                                            preproc_args=preproc_args)
    classifier_type = 'lstm'
    j2_preds_lstm = preproc_infer_fullysup(j1_test_df,
                                           exp_dir_path + classifier_type + '/j2/',
                                           preproc_handle=preproc_handle,
                                           preproc_args=preproc_args)
    print('Finished infering jet 2')
    event_preds_dense = j1_preds_dense * j2_preds_dense
    event_preds_lstm = j1_preds_lstm * j2_preds_lstm

    ## Logs and plots
    print('Creating plots and logs...')
    # Learning curve ###################################################################################################
    if hist1 and hist2:
        plot_learn_curve(hist1, save_path=exp_dir_path+'nn1_learn_curve.pdf')
        plot_learn_curve(hist2, save_path=exp_dir_path+'nn2_learn_curve.pdf')

    # NN output histograms #############################################################################################
    classifier_dicts =  {'event dense': {'probS': event_preds_dense, 'plot_dict': {'linestyle': '-'}},
                         'j1 dense': {'probS': j1_preds_dense, 'plot_dict': {'linestyle': '-'}},
                         'j2 dense': {'probS': j2_preds_dense, 'plot_dict': {'linestyle': '-'}}}
    plot_nn_hists(classifier_dicts=classifier_dicts, true_lab=event_label_test,
                  save_dir=exp_dir_path+'nn_out_hists/')

    # ROCS #############################################################################################################
    j1_verts = j1_test_df.n_verts
    j2_verts = j2_test_df.n_verts
    ev_verts = j1_verts + j2_verts
    j1_mult = j1_test_df.mult
    j2_mult = j2_test_df.mult
    ev_mult = j1_mult + j2_mult
    classifier_dicts = {'event dense': {'probS': event_preds_dense, 'plot_dict': {'linestyle': '-', 'color': 'red'}},
                        'event LSTM': {'probS': event_preds_lstm, 'plot_dict': {'linestyle': '-', 'color': 'blue'}},
                        'event vertex count': {'probS': ev_verts, 'plot_dict': {'linestyle': '--', 'color': 'green'}},
                        'event constituent count': {'probS': ev_mult, 'plot_dict': {'linestyle': '--', 'color': 'black'}}}

    classifier_dicts_mult = {'event constituent count': {'probS': ev_mult, 'plot_dict': {'linestyle': '--', 'color': 'black'}},
                             'j1 constituent count': {'probS': j1_mult, 'plot_dict': {'linestyle': '--', 'color': 'blue'}},
                             'j2 constituent count': {'probS': j2_mult, 'plot_dict': {'linestyle': '--', 'color': 'green'}}}

    classifier_dicts_vert = {'event vertex count': {'probS': ev_verts, 'plot_dict': {'linestyle': '--', 'color': 'black'}},
                             'j1 vertex count': {'probS': j1_verts, 'plot_dict': {'linestyle': '--', 'color': 'blue'}},
                             'j2 vertex count': {'probS': j2_verts, 'plot_dict': {'linestyle': '--', 'color': 'green'}}}

    classifier_dicts_dense = {'event dense': {'probS': event_preds_dense, 'plot_dict': {'linestyle': '-', 'color': 'black'}},
                              'j1 dense': {'probS': j1_preds_dense, 'plot_dict': {'linestyle': '-', 'color': 'blue'}},
                              'j2 dense': {'probS': j2_preds_dense, 'plot_dict': {'linestyle': '-', 'color': 'green'}}}

    classifier_dicts_lstm = {'event LSTM': {'probS': event_preds_lstm, 'plot_dict': {'linestyle': '-', 'color': 'black'}},
                             'j1 LSTM': {'probS': j1_preds_lstm, 'plot_dict': {'linestyle': '-', 'color': 'blue'}},
                             'j2 LSTM': {'probS': j2_preds_lstm, 'plot_dict': {'linestyle': '-', 'color': 'green'}}}

    with np.errstate(divide='ignore'):
        plot_rocs(classifier_dicts=classifier_dicts, true_lab=event_label_test,
                  save_path=exp_dir_path+'log_ROC.pdf')
        plot_rocs(classifier_dicts=classifier_dicts_dense, true_lab=event_label_test,
                  save_path=exp_dir_path+'log_ROC_dense.pdf')
        plot_rocs(classifier_dicts=classifier_dicts_lstm, true_lab=event_label_test,
                  save_path=exp_dir_path+'log_ROC_lstm.pdf')
        plot_rocs(classifier_dicts=classifier_dicts_vert, true_lab=event_label_test,
                  save_path=exp_dir_path+'log_ROC_vert.pdf')
        plot_rocs(classifier_dicts=classifier_dicts_mult, true_lab=event_label_test,
                  save_path=exp_dir_path+'log_ROC_mult.pdf')
    # Feature histograms ###############################################################################################
    # with np.errstate(divide='ignore'):
    #     plot_event_histograms_handle(j1_df, j2_df, event_label, pdf_path=exp_dir_path+'feature_hists.pdf')

    # save classifier outputs ##########################################################################################
    classifier_preds_save_dir = exp_dir_path + 'classifier_preds/'
    Path(classifier_preds_save_dir).mkdir(parents=True, exist_ok=True)
    for classifier_name, classifier_dict in zip(classifier_dicts.keys(), classifier_dicts.values()):
        probS = classifier_dict['probS']
        np.save(classifier_preds_save_dir+classifier_name+'.npy', probS)
    np.save(classifier_preds_save_dir+'event_labels.npy', event_label_test)

    # Significance #####################################################################################################
    # print('Evaluating significance')
    # Btest2_path = B_path
    # Stest2_path = S_path
    # Ntest2 = Ntrain
    # fig_path = exp_dir_path + 'significance.pdf'
    # eval_significance(model_j1, model_j2, Btest2_path, Stest2_path, Ntest2, sig_frac, preproc_args, create_model_args, semisup_dict, fig_path)
    #
    # print('Finished creating plots and logs')

# def eval_significance(model_j1, model_j2, B_path, S_path, N, sig_frac, preproc_args, create_model_args, semisup_dict, fig_path):
#     j1_df, j2_df, event_labels = combine_SB(B_path, S_path, N, sig_frac)
#     event_labels = event_labels.astype(bool)
#
#     j1_preds = train_infer_semisup(j1_df, model_j1, semisup_dict, infer_only=True,
#                                     preproc_handle=preproc_handle,
#                                     create_model_handle=create_model_handle,
#                                     preproc_args=preproc_args,
#                                     create_model_args=create_model_args)
#
#     j2_preds = train_infer_semisup(j2_df, model_j2, semisup_dict, infer_only=True,
#                                    preproc_handle=preproc_handle,
#                                    create_model_handle=create_model_handle,
#                                    preproc_args=preproc_args,
#                                    create_model_args=create_model_args)
#     semisup_preds = j1_preds+j2_preds
#     mult_preds = np.array(j1_df.mult + j2_df.mult)
#
#     plot_significance(semisup_preds, mult_preds, event_labels, fig_path)
#
# def plot_significance(semisup_preds, mult_preds, event_labels, fig_path):
#     data_eff = np.logspace(-3, 0, 19)
#     semisup_significance = calc_significance(semisup_preds, event_labels, data_eff)
#     mult_significance = calc_significance(mult_preds, event_labels, data_eff)

    # semisup_nans = np.isnan(semisup_significance)
    # mult_nans = np.isnan(mult_significance)
    # if any(semisup_nans):
    #     semisup_start_idx = np.argwhere(semisup_nans)[-1]
    # else:
    #     semisup_start_idx = 0
    # if any(mult_nans):
    #     mult_start_idx = np.argwhere(mult_nans)[-1]
    # else:
    #     mult_start_idx = 0
    # start_idx = np.max([semisup_start_idx, mult_start_idx])
    # data_eff = data_eff[start_idx:]
    # semisup_significance = semisup_significance[start_idx:]
    # mult_significance = mult_significance[start_idx:]

    # fig, ax = plt.subplots()
    # plt.plot(data_eff, semisup_significance, label='NN cut', linestyle='-', marker='x', color='black')
    # plt.plot(data_eff, mult_significance, label='Multiplicity cut', linestyle='--', marker='o', color='grey')
    # plt.legend(loc='best')
    # plt.xscale('log')
    # plt.xlabel('$\\epsilon$')
    # plt.ylabel('$\\sigma = N_{S}/\\sqrt{N_{B}}$')
    # plt.xlim(right=1)
    # fig.savefig(fig_path)

def calc_significance(preds, ev_lab, data_effs):
    sorted_idxs = np.argsort(-preds)
    preds_sorted = preds[sorted_idxs]
    ev_lab_sorted = ev_lab[sorted_idxs]

    Sn = np.cumsum(preds_sorted)
    Pn = (Sn-0.5*preds_sorted)/Sn[-1]
    cutoff_idxs = np.searchsorted(Pn, data_effs)
    cutoff_idxs[cutoff_idxs==len(Pn)] = len(Pn)-1

    cumsum_ev_lab_sorted = np.cumsum(ev_lab_sorted)
    Ns = cumsum_ev_lab_sorted[cutoff_idxs]
    Nb = cutoff_idxs - Ns
    significance = Ns/np.sqrt(Nb)

    return significance

def parse_args(argv):
    ## Data prep params
    B_path, S_path, Btest_path, Stest_path, exp_dir_path =  argv[1], argv[2], argv[3], argv[4], argv[5]
    Ntrain, Ntest, sig_frac = int(float(argv[6])), int(float(argv[7])), float(argv[8])

    ## unsup classifier params
    unsup_type = 'constituent_mult'
    unsup_dict = {}

    ## semisup classifier params
    # General
    bkg_quant = float(argv[9])
    sig_quant = float(argv[10])
    with_displacement, with_deltar, with_pid = argv[11], argv[12], argv[13]
    epochs = int(argv[14])
    semisup_dict = {'bkg_quant': bkg_quant,
                    'sig_quant': sig_quant,
                    'epochs': epochs,
                    'reg_dict': {},
                    'with_displacement': with_displacement,
                    'with_deltar': with_deltar,
                    'with_pid': with_pid}

    # Regularization
    # Weight regularization
    weight_reg_params = ["kernel_regularizer", "recurrent_regularizer", "bias_regularizer"]
    weight_reg_dict = {param: None if arg=="None" else
                       keras.regularizers.l2(float(arg)) for param, arg in zip(weight_reg_params, argv[15:18])}
    # Dropout
    drop_params = ["dropout", "recurrent_dropout"]
    drop_dict = {param: float(arg) for param, arg in zip(drop_params, argv[18:20])}

    semisup_dict['reg_dict'] = {**weight_reg_dict, **drop_dict}

    n_iter = int(argv[20])
    split_data = argv[21]

    if len(argv)>22:
        semisup_dict['train_nn'] = argv[22]

    return (B_path, S_path, Btest_path, Stest_path, exp_dir_path, Ntrain, Ntest, sig_frac,
            unsup_type, unsup_dict, semisup_dict, n_iter, split_data)

if __name__ == '__main__':
    #set_tensorflow_threads(n_threads=30)
    start = timer()
    print('Starting excecution')
    main_fullysup(*parse_args(sys.argv))
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