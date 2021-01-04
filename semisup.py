# System
import sys
from pathlib import Path
from timeit import default_timer as timer
from datetime import timedelta

# Standard
import numpy as np
import pandas as pd

# ML
from tensorflow import keras

# Custom
from UTILS.utils import evs_txt2jets_df as load_data
from UTILS.utils import create_one_hot_encoder, nominal2onehot, set_tensorflow_threads
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

def combine_SB(B_path, S_path, N, sig_frac):
    n_B, n_S = int(N*(1 - sig_frac)), int(N * sig_frac)
    (B_j1_df, B_j2_df), (S_j1_df, S_j2_df) = load_data(B_path, n_ev=n_B), load_data(S_path, n_ev=n_S)
    j1_df = pd.concat([B_j1_df, S_j1_df]).reset_index(drop=True)
    j2_df = pd.concat([B_j2_df, S_j2_df]).reset_index(drop=True)
    event_label = np.array([0]*n_B + [1]*n_S)
    return j1_df, j2_df, event_label

def infer_unsup(j_df, unsup_type, unsup_dict):
    if unsup_type == 'constituent_mult':
        j_unsup_probS = j_df.mult
    else:
        raise ValueError(f'Unsuported unsup method: {unsup_type}')
    return j_unsup_probS

def train_infer_semisup(train_set, weak_labels, infer_set, model_save_path, param_dict):
    Path(model_save_path).mkdir(parents=True, exist_ok=True)
    log = ''
    ## Hard coded params
    mask = -10.0
    n_constits = 80
    pid = [mask, -2212, -321, -211, -13, -11, 0, 1, 11, 13, 211, 321, 2212]
    classification = ['masked', 'h-', 'h-', 'h-', 'mu-', 'e-', 'photon', 'h0', 'e+', 'mu+', 'h+', 'h+', 'h+']
    class_dict = dict(zip(pid, classification))

    ## Default args
    if param_dict is None:
        param_dict = {'with_displacement': "True", 'with_deltar': "False", 'with_pid': "False",
                      'reg_dict': {}, 'epochs': 20}

    # Determine features and nn columns
    feats, n_cols = determine_feats(param_dict['with_displacement'],
                                    param_dict['with_deltar'],
                                    param_dict['with_pid'])
    # Create model
    model, log = create_lstm_classifier(n_constits, n_cols, param_dict['reg_dict'], mask, log=log)

    # Preprocessing
    j_inp = preproc_for_lstm(train_set, feats, mask, n_constits)
    if param_dict['with_pid'] == "True":
        enc = create_one_hot_encoder(class_dict)
        j_inp = nominal2onehot(j_inp, class_dict, enc)

    # Train model
    hist, log = train_classifier(j_inp, weak_labels, model=model, model_save_path=model_save_path,
                                 epochs=param_dict['epochs'], log=log)

    # Load and infer model
    model = keras.models.load_model(model_save_path)
    # Preprocessing
    j_inf = preproc_for_lstm(infer_set, feats, mask, n_constits)
    if param_dict['with_pid'] == "True":
        enc = create_one_hot_encoder(class_dict)
        j_inf = nominal2onehot(j_inf, class_dict, enc)
    j_probS = model.predict(j_inf).flatten()

    # nn_input hisograms
    plot_nn_inp_histograms(j_inp, plot_save_dir=model_save_path)

    return j_probS, hist, log


def main_semisup(B_path, S_path, exp_dir_path, N=int(1e5), sig_frac=0.2, unsup_type='constituent_mult',
                 unsup_dict=None, semisup_dict=None, n_iter=2):
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

        Outputs saved to exp_dir_path:
            j1, j2: Tensorflow models trained on jet1 and jet2 respectively.
            log.txt: Log of data information.
    """

    Path(exp_dir_path).mkdir(parents=True, exist_ok=True)
    log_path = exp_dir_path + 'log.txt'

    ## Data prep
    j1_df, j2_df, event_label = combine_SB(B_path, S_path, N, sig_frac)

    ## Iteration split
    split_size = int(len(event_label)/(n_iter+1))
    D1 = tuple((j1_df[iteration*split_size:(iteration+1)*split_size] for iteration in range(n_iter+1)))
    D2 = tuple((j2_df[iteration*split_size:(iteration+1)*split_size] for iteration in range(n_iter+1)))

    ## First (unsupervised) classifier
    j1_unsup_probS, j2_unsup_probS = infer_unsup(D1[0], unsup_type, unsup_dict), infer_unsup(D2[0], unsup_type, unsup_dict)

    ## Second (semisupervised) classifiers
    j1_curr_probS = j1_unsup_probS
    j2_curr_probS = j2_unsup_probS
    for iteration in range(n_iter):
        # semisup labels for one jet are unsup predictions for the other jet.
        j1_thresh = np.median(j1_curr_probS)
        j2_thresh = np.median(j2_curr_probS)
        j1_semisup_lab = j2_curr_probS > j2_thresh
        j2_semisup_lab = j1_curr_probS > j1_thresh
        # create model, preprocess, train, and infer
        j1_curr_probS, hist1, log1 = train_infer_semisup(train_set=D1[iteration], infer_set=D1[iteration+1],
                                                         weak_labels=j1_semisup_lab,
                                                         model_save_path=exp_dir_path+f'j1_{iteration}/',
                                                         param_dict=semisup_dict)
        j2_curr_probS, hist2, log2 = train_infer_semisup(train_set=D2[iteration], infer_set=D2[iteration+1],
                                                         weak_labels=j2_semisup_lab,
                                                         model_save_path=exp_dir_path+f'j2_{iteration}/',
                                                         param_dict=semisup_dict)
    j1_semisup_probS, j2_semisup_probS = j1_curr_probS, j2_curr_probS

    ## Average of both jet classifiers serves as a final event prediction.
    # semisupervised prediction
    event_semisup_probS = (j1_semisup_probS + j2_semisup_probS)/2
    # unsupervised prediction for benchmark
    event_unsup_probS = (j1_unsup_probS + j2_unsup_probS)/2

    ## Logs and plots
    # Logs
    log_args(log_path, B_path, S_path, exp_dir_path, unsup_dict, semisup_dict, n_iter)
    log_events_info(log_path, event_label)
    log_semisup_labels_info(log_path, j1_semisup_lab, j2_semisup_lab, j1_thresh, j2_thresh, event_label)
    log_nn_inp_info(log_path, log1, log2)
    with open(log_path, 'a') as f:
        f.write('Classifiers correlation\n')
        f.write(f'Unsup classifiers correlation: {np.corrcoef(j1_unsup_probS, j2_unsup_probS)[0, 1]:.3f}\n')
        f.write(f'Semisup classifiers correlation: {np.corrcoef(j1_semisup_probS, j2_semisup_probS)[0, 1]:.3f}\n')
        f.write('----------\n')
        f.write('\n')

    # Plots
    plot_event_histograms(j1_df, j2_df, event_label, save_dir=exp_dir_path+'event_hists/')
    plot_learn_curve(hist1, save_path=exp_dir_path+'nn1_learn_curve.pdf')
    plot_learn_curve(hist2, save_path=exp_dir_path+'nn2_learn_curve.pdf')

    # rocs and nn histograms
    probS_dict = {'semisup classifier on j1': j1_semisup_probS,
                  'semisup classifier on j2': j2_semisup_probS,
                  'semisup event classifier': event_semisup_probS,
                  'unsup classifier on j1': j1_unsup_probS,
                  'unsup classifier on j2': j2_unsup_probS,
                  'unsup event classifier': event_unsup_probS}
    plot_nn_hists(probS_dict=probS_dict, true_lab=event_label, semisup_labs=(j1_semisup_lab, j2_semisup_lab),
                  save_dir=exp_dir_path+'nn_out_hists/')
    roc_dict = plot_rocs(probS_dict=probS_dict, true_lab=event_label, save_path=exp_dir_path+'log_ROC.pdf')

    # save rocs
    roc_save_dir = exp_dir_path + 'roc_arrays/'
    Path(roc_save_dir).mkdir(parents=True, exist_ok=True)
    for roc_name in roc_dict:
        roc = roc_dict[roc_name]
        np.save(roc_save_dir+roc_name+'.npy', roc)

def parse_args(argv):
    ## Data prep params
    B_path, S_path, exp_dir_path, N, sig_frac =  argv[1], argv[2], argv[3], int(argv[4]), float(argv[5])

    ## unsup classifier params
    unsup_type = 'constituent_mult'
    unsup_dict = {}

    ## semisup classifier params
    # General
    with_displacement, with_deltar, with_pid = argv[6], argv[7], argv[8]
    semisup_dict = {'epochs': int(argv[9]),
                    'reg_dict': {},
                    'with_displacement': with_displacement,
                    'with_deltar': with_deltar,
                    'with_pid': with_pid}
    # Regularization
    # Weight regularization
    weight_reg_params = ["kernel_regularizer", "recurrent_regularizer", "bias_regularizer"]
    weight_reg_dict = {param: None if arg=="None" else
                       keras.regularizers.l2(float(arg)) for param, arg in zip(weight_reg_params, argv[10:13])}
    # Dropout
    drop_params = ["dropout", "recurrent_dropout"]
    drop_dict = {param: float(arg) for param, arg in zip(drop_params, argv[13:15])}

    semisup_dict['reg_dict'] = {**weight_reg_dict, **drop_dict}

    n_iter = int(argv[15])

    return B_path, S_path, exp_dir_path, N, sig_frac, unsup_type, unsup_dict, semisup_dict, n_iter

if __name__ == '__main__':
    #set_tensorflow_threads(n_threads=30)
    start = timer()
    main_semisup(*parse_args(sys.argv))
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