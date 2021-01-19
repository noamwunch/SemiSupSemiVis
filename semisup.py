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
from UTILS.utils import evs_txt2jets_df_old as load_data_old
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
    n_B, n_S = int(N*(1 - sig_frac)), int(N * sig_frac)

    idxs = np.arange(n_B+n_S)
    np.random.shuffle(idxs)

    event_label = np.array([0]*n_B + [1]*n_S)[idxs]

    (B_j1_df, B_j2_df), (S_j1_df, S_j2_df) = load_data(B_path, n_ev=n_B), load_data(S_path, n_ev=n_S)
    j1_df = pd.concat([B_j1_df, S_j1_df]).iloc[idxs].reset_index(drop=True)
    j2_df = pd.concat([B_j2_df, S_j2_df]).iloc[idxs].reset_index(drop=True)
    return j1_df, j2_df, event_label

def infer_unsup(j_df, unsup_type, unsup_dict):
    if unsup_type == 'constituent_mult':
        j_unsup_probS = j_df.mult
    else:
        raise ValueError(f'Unsuported unsup method: {unsup_type}')
    return j_unsup_probS

class jet_mult_classifier:
    def predict(self, jet_df):
        return jet_df.mult

def filter_quantile(train_set, preds, bkg_quant, sig_quant):
    assert (bkg_quant+sig_quant)<=1, 'The sum of signal and background quantiles should be smaller than 1'
    bkg_thresh, sig_thresh = np.quantile(preds, [bkg_quant, 1-sig_quant])
    valid_idx = (preds>sig_thresh) | (preds<bkg_thresh)
    train_set = train_set[valid_idx]
    labels = preds[valid_idx]>sig_thresh
    return labels, train_set, sig_thresh

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
    j_inp = preproc_for_lstm(train_set.copy(deep=True), feats, mask, n_constits)
    if param_dict['with_pid'] == "True":
        enc = create_one_hot_encoder(class_dict)
        j_inp = nominal2onehot(j_inp, class_dict, enc)

    # Train model
    if param_dict.get('train_nn', "True")=="True":
        hist, log = train_classifier(j_inp, weak_labels, model=model, model_save_path=model_save_path,
                                                 epochs=param_dict['epochs'], log=log)
    else:
        model = keras.models.load_model(model_save_path)
        hist = False

    # Preprocessing for inference
    j_inf = preproc_for_lstm(infer_set.copy(deep=True), feats, mask, n_constits)
    if param_dict['with_pid'] == "True":
        enc = create_one_hot_encoder(class_dict)
        j_inf = nominal2onehot(j_inf, class_dict, enc)
    # Infer jets using model
    j_probS = model.predict(j_inf).flatten()

    # nn_input hisograms
    plot_nn_inp_histograms(j_inp, plot_save_dir=model_save_path)

    return j_probS, hist, log

def train_infer_semisup_new(j2_data, weak_model_j2,
                            j1_data=None, model_save_path=None, param_dict=None,
                            infer_only=False):
    Path(model_save_path).mkdir(parents=True, exist_ok=True)
    log = ''
    ## Hard coded params
    mask = -10.0
    n_constits = 80
    pid = [mask, -2212, -321, -211, -13, -11, 0, 1, 11, 13, 211, 321, 2212]
    classification = ['masked', 'h-', 'h-', 'h-', 'mu-', 'e-', 'photon', 'h0', 'e+', 'mu+', 'h+', 'h+', 'h+']
    class_dict = dict(zip(pid, classification))
    bkg_quant = 0.5
    sig_quant = 0.5

    # Determine features and nn columns
    feats, n_cols = determine_feats(param_dict['with_displacement'],
                                    param_dict['with_deltar'],
                                    param_dict['with_pid'])
    ####################################################################################################################

    # Preprocessing of j2 for prediction
    if type(weak_model_j2).__name__ != 'jet_mult_classifier':
        j2_inp = preproc_for_lstm(j2_data.copy(deep=True), feats, mask, n_constits)
        if param_dict['with_pid'] == "True":
            enc = create_one_hot_encoder(class_dict)
            j2_inp = nominal2onehot(j2_inp, class_dict, enc)
    else:
        j2_inp = j2_data.copy(deep=True)

    # Infer weak predictions
    weak_preds = np.array(weak_model_j2.predict(j2_inp)).flatten()

    if infer_only:
        return weak_preds
    ####################################################################################################################

    # Create weak labels
    print(f'len(weak_labels) = {len(weak_preds)}')
    print(f'len(j1_inp) = {len(j1_data)}')
    weak_labels, j1_inp, thresh = filter_quantile(j1_data, weak_preds, bkg_quant, sig_quant)
    print(f'len(weak_labels) = {len(weak_labels)}')
    print(f'len(j1_inp) = {len(j1_inp)}')

    # Preprocessing of j1 for training
    j1_inp = preproc_for_lstm(j1_data.copy(deep=True), feats, mask, n_constits)
    if param_dict['with_pid'] == "True":
        enc = create_one_hot_encoder(class_dict)
        j1_inp = nominal2onehot(j1_inp, class_dict, enc)

    # Create model
    stronger_model_j1, log = create_lstm_classifier(n_constits, n_cols, param_dict['reg_dict'], mask, log=log)

    # Train model
    if param_dict.get('train_nn', "True")=="True":
        hist, log = train_classifier(j1_inp, weak_labels, model=stronger_model_j1, model_save_path=model_save_path,
                                     epochs=param_dict['epochs'], log=log)
    else:
        stronger_model_j1 = keras.models.load_model(model_save_path)
        hist = False

    # nn_input hisograms
    plot_nn_inp_histograms(j1_inp, plot_save_dir=model_save_path)

    return hist, log, thresh, stronger_model_j1

def main_semisup(B_path, S_path, Btest_path, Stest_path, exp_dir_path, Ntrain=int(1e5), Ntest=int(1e4), sig_frac=0.2,
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

    Path(exp_dir_path).mkdir(parents=True, exist_ok=True)
    log_path = exp_dir_path + 'log.txt'

    ## Data prep
    print('Loading train data...')
    j1_df, j2_df, event_label = combine_SB(B_path, S_path, Ntrain, sig_frac)
    print('Training data loaded')

    print('Loading test data')
    j1_test_df, j2_test_df, event_label_test = combine_SB(Btest_path, Stest_path, Ntest, 0.5)
    print('Test data loaded')

    ## Iteration split. Create n_iter+1 slices corresponding to n_iter iterations and a test set.
    train_size = len(event_label)
    if split_data == "True":
        split_size = int(train_size/n_iter)
        split_idxs = tuple(slice(iteration*split_size, (iteration+1)*split_size) for iteration in range(n_iter))
    else:
        split_idxs = tuple(slice(train_size) for _ in range(n_iter))

    ## First (unsupervised) weak classifier
    model_j1 = jet_mult_classifier()
    model_j2 = jet_mult_classifier()

    print(f'Starting {n_iter} iterations...')
    for iteration in range(n_iter):
        print(f'Starting iteration {iteration}')
        train_idx = split_idxs[iteration]

        weak_model_j1 = model_j1
        weak_model_j2 = model_j2
        print('Training model on jet1')
        hist1, log1, weak_labs1, thresh1, model_j1 = train_infer_semisup_new(j2_df.iloc[train_idx],
                                                                             weak_model_j2,
                                                                             j1_df.iloc[train_idx],
                                                                             exp_dir_path+f'j1_{iteration}/',
                                                                             semisup_dict)
        print('Finished training model on jet1')
        print('Training model on jet2...')
        hist2, log2, weak_labs2, thresh2, model_j2 = train_infer_semisup_new(j1_df.iloc[train_idx],
                                                                             weak_model_j1,
                                                                             j2_df.iloc[train_idx],
                                                                             exp_dir_path+f'j2_{iteration}/',
                                                                             semisup_dict)
        print('Finished training model on jet2')
    print('Finished iterations')

    print('Testing on test data...')
    ## Average of both jet classifiers serves as a final event prediction.
    print('Infering jet 1...')
    j1_semisup_probS = train_infer_semisup_new(j1_test_df, model_j1, infer_only=True)
    print('Finished infering jet 1')
    print('Infering jet 2...')
    j2_semisup_probS = train_infer_semisup_new(j2_test_df, model_j2, infer_only=True)
    print('Finished infering jet 2')
    event_semisup_probS = (j1_semisup_probS + j2_semisup_probS)/2

    # unsupervised prediction for benchmark
    j1_unsup_probS = jet_mult_classifier().predict(j1_test_df)
    j2_unsup_probS = jet_mult_classifier().predict(j2_test_df)
    event_unsup_probS = (j1_unsup_probS + j2_unsup_probS)/2

    ## Logs and plots
    print('Creating plots and logs...')
    # Logs
    log_args(log_path, B_path, S_path, exp_dir_path, unsup_dict, semisup_dict, n_iter)
    log_events_info(log_path, event_label)
    log_semisup_labels_info(log_path, weak_labs1, weak_labs2, thresh1, thresh2, event_label[split_idxs[n_iter-1]])
    log_nn_inp_info(log_path, log1, log2)
    with open(log_path, 'a') as f:
        f.write('Classifiers correlation\n')
        f.write(f'Unsup classifiers correlation: {np.corrcoef(j1_unsup_probS, j2_unsup_probS)[0, 1]:.3f}\n')
        f.write(f'Semisup classifiers correlation: {np.corrcoef(j1_semisup_probS, j2_semisup_probS)[0, 1]:.3f}\n')
        f.write('----------\n')
        f.write('\n')

    # Plots
    plot_event_histograms(j1_df, j2_df, event_label, save_dir=exp_dir_path+'event_hists/')
    if hist1 and hist2:
        plot_learn_curve(hist1, save_path=exp_dir_path+'nn1_learn_curve.png')
        plot_learn_curve(hist2, save_path=exp_dir_path+'nn2_learn_curve.png')

    # rocs and nn histograms
    classifier_dicts = {'semisup event classifier': {'probS': event_semisup_probS, 'plot_dict': {'linestyle': '-'}},
                        'semisup classifier on j1': {'probS': j1_semisup_probS, 'plot_dict': {'linestyle': '-'}},
                        'semisup classifier on j2': {'probS': j2_semisup_probS, 'plot_dict': {'linestyle': '-'}},
                        'unsup event classifier': {'probS': event_unsup_probS, 'plot_dict': {'linestyle': '--'}},
                        'unsup classifier on j1': {'probS': j1_unsup_probS, 'plot_dict': {'linestyle': '--'}},
                        'unsup classifier on j2': {'probS': j2_unsup_probS, 'plot_dict': {'linestyle': '--'}}}
    plot_nn_hists(classifier_dicts=classifier_dicts, true_lab=event_label[split_idxs[-1]],
                  semisup_labs=(weak_labs1, weak_labs2),
                  save_dir=exp_dir_path+'nn_out_hists/')
    plot_rocs(classifier_dicts=classifier_dicts, true_lab=event_label[split_idxs[-1]],
              save_path=exp_dir_path+'log_ROC.png')

    # save classifier outputs
    classifier_preds_save_dir = exp_dir_path + 'classifier_preds/'
    Path(classifier_preds_save_dir).mkdir(parents=True, exist_ok=True)
    for classifier_name, classifier_dict in zip(classifier_dicts.keys(), classifier_dicts.values()):
        probS = classifier_dict['probS']
        np.save(classifier_preds_save_dir+classifier_name+'.npy', probS)
    np.save(classifier_preds_save_dir+'event_labels.npy', event_label[split_idxs[-1]])
    print('Finished creating plots asn logs')

def parse_args(argv):
    ## Data prep params
    B_path, S_path, Btest_path, Stest_path, exp_dir_path =  argv[1], argv[2], argv[3], argv[4], argv[5]
    Ntrain, Ntest, sig_frac = int(float(argv[6])), int(float(argv[7])), float(argv[8])

    ## unsup classifier params
    unsup_type = 'constituent_mult'
    unsup_dict = {}

    ## semisup classifier params
    # General
    with_displacement, with_deltar, with_pid = argv[9], argv[10], argv[11]
    semisup_dict = {'epochs': int(argv[12]),
                    'reg_dict': {},
                    'with_displacement': with_displacement,
                    'with_deltar': with_deltar,
                    'with_pid': with_pid}

    # Regularization
    # Weight regularization
    weight_reg_params = ["kernel_regularizer", "recurrent_regularizer", "bias_regularizer"]
    weight_reg_dict = {param: None if arg=="None" else
                       keras.regularizers.l2(float(arg)) for param, arg in zip(weight_reg_params, argv[13:16])}
    # Dropout
    drop_params = ["dropout", "recurrent_dropout"]
    drop_dict = {param: float(arg) for param, arg in zip(drop_params, argv[16:18])}

    semisup_dict['reg_dict'] = {**weight_reg_dict, **drop_dict}

    n_iter = int(argv[18])
    split_data = argv[19]

    if len(argv)>20:
        semisup_dict['train_nn'] = argv[20]

    return (B_path, S_path, Btest_path, Stest_path, exp_dir_path, Ntrain, Ntest, sig_frac,
            unsup_type, unsup_dict, semisup_dict, n_iter, split_data)

if __name__ == '__main__':
    #set_tensorflow_threads(n_threads=30)
    start = timer()
    print('Starting excecution')
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