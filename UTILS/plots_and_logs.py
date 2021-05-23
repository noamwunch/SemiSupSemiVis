import numpy as np
from matplotlib import pyplot as plt
import sklearn.metrics
from pathlib import Path

def log_args(log_path, B_path, S_path, exp_dir_path, unsup_dict, semisup_dict, n_iter):
    with open(log_path, 'w') as f:
        f.write(f'B_path = {B_path}\n')
        f.write(f'S_path = {S_path}\n')
        f.write(f'exp_dir_path = {exp_dir_path}\n')
        f.write('\n')

        f.write('usupervised classifier info:\n')
        f.write(f'number of iterations: {n_iter}\n')
        for param_name, param_value in zip(unsup_dict.keys(), unsup_dict.values()):
            f.write(f'{param_name}: {param_value}\n')
        f.write('\n')

        f.write('semi-supervised classifier info:\n')
        weight_regs = ["kernel_regularizer", "recurrent_regularizer", "bias_regularizer"]
        for param_name, param_value in zip(semisup_dict.keys(), semisup_dict.values()):
            if param_name == 'reg_dict':
                for reg_name, reg_value in zip(param_value.keys(), param_value.values()):
                    if (reg_name in weight_regs) and (reg_value is not None):
                        f.write(f'{reg_name}: {reg_value.l2:.2e}\n')
                    else:
                        f.write(f'{reg_name}: {reg_value}\n')
            else:
                f.write(f'{param_name}: {param_value}\n')
        f.write('----------\n')
        f.write('\n')

def log_events_info(log_path, event_label):
    with open(log_path, 'a') as f:
        f.write(f'N = {len(event_label)}\n')
        f.write(f'sig_frac = {sum(event_label)/len(event_label):.2f}\n')
        f.write(f'#B = {len(event_label)-sum(event_label)}\n')
        f.write(f'#S = {sum(event_label)}\n')
        f.write('----------\n')
        f.write('\n')

def log_semisup_labels_info(log_path, j1_unsup_lab, j2_unsup_lab, j1_thresh, j2_thresh, event_label):
    with open(log_path, 'a') as f:
        n_S_tag = sum(j1_unsup_lab)
        n_B_tag = len(j1_unsup_lab) - n_S_tag
        S_in_Stag = sum(event_label * j1_unsup_lab)
        S_in_Btag = sum(event_label) - S_in_Stag
        f.write('j1 split info (split by unsup classifier on j2):\n')
        f.write(f'thresh = {j1_thresh}\n')
        f.write(f'#B\' = {n_B_tag}\n')
        f.write(f'#S\' = {n_S_tag}\n')
        f.write(f'B\' sig_frac = {S_in_Btag/n_B_tag:.3f}\n')
        f.write(f'S\' sig_frac = {S_in_Stag/n_S_tag:.3f}\n')
        f.write('\n')

        n_S_tag = sum(j2_unsup_lab)
        n_B_tag = len(j2_unsup_lab) - n_S_tag
        S_in_Stag = sum(event_label * j2_unsup_lab)
        S_in_Btag = sum(event_label) - S_in_Stag
        f.write('j2 split info (split by unsup classifier on j1:\n')
        f.write(f'thresh = {j2_thresh}\n')
        f.write(f'#B\' = {n_B_tag}\n')
        f.write(f'#S\' = {n_S_tag}\n')
        f.write(f'B\' sig_frac = {S_in_Btag / n_B_tag:.3f}\n')
        f.write(f'S\' sig_frac = {S_in_Stag / n_S_tag:.3f}\n')
        f.write('----------\n')
        f.write('\n')

def log_semisup_labels_info_new(log_path, j1_unsup_lab, j2_unsup_lab, j1_thresh, j2_thresh, true_lab1, true_lab2):
    with open(log_path, 'a') as f:
        unsup_sig_mask = j1_unsup_lab.astype(bool)
        true_sig_mask = true_lab1.astype(bool)
        n_S_tag = sum(unsup_sig_mask)
        n_B_tag = sum(~unsup_sig_mask)
        S_in_Stag = sum(true_sig_mask & unsup_sig_mask)
        S_in_Btag = sum(true_sig_mask & (~unsup_sig_mask))
        f.write('j1 split info (split by unsup classifier on j2):\n')
        f.write(f'thresh = {j1_thresh}\n')
        f.write(f'#B\' = {n_B_tag}\n')
        f.write(f'#S\' = {n_S_tag}\n')
        f.write(f'B\' sig_frac = {S_in_Btag/n_B_tag:.3f}\n')
        f.write(f'S\' sig_frac = {S_in_Stag/n_S_tag:.3f}\n')
        f.write('\n')

        unsup_sig_mask = j2_unsup_lab.astype(bool)
        true_sig_mask = true_lab2.astype(bool)
        n_S_tag = sum(unsup_sig_mask)
        n_B_tag = sum(~unsup_sig_mask)
        S_in_Stag = sum(true_sig_mask & unsup_sig_mask)
        S_in_Btag = sum(true_sig_mask & (~unsup_sig_mask))
        f.write('j2 split info (split by unsup classifier on j1:\n')
        f.write(f'thresh = {j2_thresh}\n')
        f.write(f'#B\' = {n_B_tag}\n')
        f.write(f'#S\' = {n_S_tag}\n')
        f.write(f'B\' sig_frac = {S_in_Btag / n_B_tag:.3f}\n')
        f.write(f'S\' sig_frac = {S_in_Stag / n_S_tag:.3f}\n')
        f.write('----------\n')
        f.write('\n')

def log_nn_inp_info(log_path, log1, log2):
    with open(log_path, 'a') as f:
        f.write('nn1 model and input info:\n')
        f.write(log1)
        f.write('nn2 model and input info:\n')
        f.write(log2)
        f.write('----------\n')
        f.write('\n')

def plot_learn_curve(hist, save_path):
    ## Minimum loss
    best_epoch = np.argmin(hist.history['val_loss'])
    min_val_loss = hist.history['val_loss'][best_epoch]
    ## Plot
    plt.figure()
    plt.title(f'learning curve: val_loss at best epoch({best_epoch + 1}) = {min_val_loss:.3f}')
    plt.plot(np.arange(len(hist.history['loss'])) + 1, hist.history['loss'], label='training loss')
    plt.plot(np.arange(len(hist.history['val_loss'])) + 1, hist.history['val_loss'], label='validation loss')
    plt.xlabel('epoch number')
    plt.ylabel('loss')
    plt.xlim([1, len(hist.history['loss'])])
    plt.legend()
    plt.gcf().set_size_inches(8.3, 5.85)
    plt.savefig(save_path)

    plt.close('all')

def plot_rocs(classifier_dicts, true_lab, save_path):
    plt.figure()
    for classifier_name, classifier_dict in zip(classifier_dicts.keys(), classifier_dicts.values()):
        probS = classifier_dict['probS']
        plot_dict = classifier_dict['plot_dict']

        bkg_eff, sig_eff, thresh = sklearn.metrics.roc_curve(true_lab, probS)
        AUC = sklearn.metrics.auc(bkg_eff, sig_eff)
        plt.semilogy(sig_eff, 1/bkg_eff, label=f'{classifier_name}: AUC = {AUC:.2f}', **plot_dict)

    plt.xlim([0, 1])
    plt.grid(which='both')
    plt.legend(loc='best')
    plt.xlabel('Signal efficiency')
    plt.ylabel('Background rejection (1/bkg_eff)')
    plt.gcf().set_size_inches(10, 10)
    plt.savefig(save_path)

    plt.close('all')

def plot_nn_hists(classifier_dicts, true_lab, save_dir):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    hist_params = {'histtype': 'step', 'density': True, 'bins': 40}

    true_sig_idx, true_bkg_idx = true_lab.astype(bool), ~true_lab.astype(bool)
    for classifier_name, classifier_dict in zip(classifier_dicts.keys(), classifier_dicts.values()):
        probS = classifier_dict['probS']

        plt.figure()
        plt.hist(probS[true_sig_idx], label='true signal', **hist_params)
        plt.hist(probS[true_bkg_idx], label='true background', **hist_params)
        plt.xlabel('Classifier output')
        plt.legend()
        plt.gcf().set_size_inches(4, 4)
        plt.savefig(save_dir+classifier_name+'_hist_truelab.pdf')

    plt.close('all')

def plot_mult(sig_mult_j1, sig_mult_j2, bkg_mult_j1, bkg_mult_j2, save_path):
    min_range = min(sig_mult_j1.min(), sig_mult_j2.min(), bkg_mult_j1.min(), bkg_mult_j2.min())
    max_range = max(sig_mult_j1.max(), sig_mult_j2.max(), bkg_mult_j1.max(), bkg_mult_j2.max())+2
    plt.figure()
    plt.hist([sig_mult_j1, bkg_mult_j1, sig_mult_j2, bkg_mult_j2],
             bins=np.arange(min_range, max_range),
             align='left', histtype='step',
             label=['sig jet1', 'bkg jet1', 'sig jet2', 'bkg jet2'])
    plt.xlabel('Constituent multiplicity')
    plt.ylabel('Events')
    plt.legend()
    plt.savefig(save_path)

'''
    pseudo_sig_idx1, pseudo_bkg_idx1 = semisup_labs[0].astype(bool), ~semisup_labs[0].astype(bool)
    name = 'semisup classifier on j1'
    probS = classifier_dicts[name]['probS']
    plt.figure()
    plt.hist(probS[pseudo_sig_idx1], label='pseudo signal', **hist_params)
    plt.hist(probS[pseudo_bkg_idx1], label='pseudo background', **hist_params)
    plt.xlabel('Classifier output')
    plt.legend()
    plt.gcf().set_size_inches(10, 10)
    plt.savefig(save_dir+name+'_hist_pseudo_lab.pdf', format='pdf')

    pseudo_sig_idx2, pseudo_bkg_idx2 = semisup_labs[1].astype(bool), ~semisup_labs[1].astype(bool)
    name = 'semisup classifier on j2'
    probS = classifier_dicts[name]['probS']
    plt.figure()
    plt.hist(probS[pseudo_sig_idx2], label='pseudo signal', **hist_params)
    plt.hist(probS[pseudo_bkg_idx2], label='pseudo background', **hist_params)
    plt.xlabel('Classifier output')
    plt.legend()
    plt.gcf().set_size_inches(10, 10)
    plt.savefig(save_dir+name+'_hist_pseudo_lab.pdf', format='pdf')
'''

