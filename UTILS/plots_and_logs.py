import numpy as np
from matplotlib import pyplot as plt
import sklearn
from pathlib import Path

def log_args(log_path, B_path, S_path, exp_dir_path, unsup_dict, semisup_dict):
    with open(log_path, 'w') as f:
        f.write(f'B_path = {B_path}\n')
        f.write(f'S_path = {S_path}\n')
        f.write(f'exp_dir_path = {exp_dir_path}\n')
        f.write('\n')

        f.write('usupervised classifier info:\n')
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

def log_unsup_labels_info(log_path, j1_unsup_lab, j2_unsup_lab, j1_thresh, j2_thresh, event_label):
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
        f.write(f'S\' sig_frac = {S_in_Stag/n_B_tag:.3f}\n')
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
        f.write(f'S\' sig_frac = {S_in_Stag / n_B_tag:.3f}\n')
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

def plot_nn_inp_histograms(j_semisup_inp, plot_save_dir):
    plt.figure()
    plt.hist(j_semisup_inp[:, 0, 0], label='track 1', bins=100, histtype='step', range=[0, 10])
    plt.hist(j_semisup_inp[:, 1, 0], label='track 2', bins=100, histtype='step', range=[0, 10])
    plt.hist(j_semisup_inp[:, 4, 0], label='track 5', bins=100, histtype='step', range=[0, 10])
    plt.hist(j_semisup_inp[:, 9, 0], label='track 10', bins=100, histtype='step', range=[0, 10])
    plt.legend(loc='best')
    plt.xlabel('relPT')
    plt.savefig(plot_save_dir + 'PT')

def plot_event_histograms(j1_df, j2_df, event_label, save_dir):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    track_nums = [1, 2, 5, 10]
    plot_dict = {'constit_D0': {'range': [-2, 2], 'xlabel': 'D0 [mm]'},
                 'constit_DZ': {'range': [-2, 2], 'xlabel': 'Dz [mm]'},
                 'constit_PT': {'range': [0, 100], 'xlabel': 'PT [GeV]'},
                 'constit_Eta': {'range': None, 'xlabel': 'Eta'},
                 'constit_Phi': {'range': None, 'xlabel': 'Phi [rad]'},
                 'constit_relDZ': {'range': [-2, 2], 'xlabel': 'relDz [mm]'},
                 'constit_relPT': {'range': [0, 1], 'xlabel': 'relPT [GeV]'},
                 'constit_relEta': {'range': [-1, 1], 'xlabel': 'relEta'},
                 'constit_relPhi': {'range': [-1, 1], 'xlabel': 'relPhi [rad]'},
                 'constit_deltaR': {'range': None, 'xlabel': 'deltaR'}
                 }
    for feat in plot_dict.keys():
        save_path = save_dir + feat
        fig, axes = plt.subplots(nrows=1, ncols=len(track_nums), figsize=(10, 10), sharex='row', sharey='row')
        for ax, track_num in zip(axes, track_nums):
            j1_df[~event_label.astype(bool)][j1_df.mult >= track_num][feat].map(lambda x: x[track_num - 1]).hist(
                ax=ax, label='j1 bkg', range=plot_dict[feat]['range'], density=True,
                histtype='step', bins=100, color='black')
            j1_df[event_label.astype(bool)][j1_df.mult >= track_num][feat].map(lambda x: x[track_num - 1]).hist(
                ax=ax, label='j1 sig', range=plot_dict[feat]['range'], density=True,
                histtype='step', bins=100, color='red')

            j2_df[~event_label.astype(bool)][j2_df.mult >= track_num][feat].map(lambda x: x[track_num - 1]).hist(
                ax=ax, label='j2 bkg', range=plot_dict[feat]['range'], density=True,
                histtype='step', bins=100, color='green')
            j2_df[event_label.astype(bool)][j2_df.mult >= track_num][feat].map(lambda x: x[track_num - 1]).hist(
                ax=ax, label='j2 sig', range=plot_dict[feat]['range'], density=True,
                histtype='step', bins=100, color='blue')

            ax.set_title(f'track #{track_num}')
            ax.legend(loc='best')
            ax.set_yticks([])
            ax.set_xlabel(plot_dict[feat]['xlabel'])
        fig.tight_layout()
        plt.gcf().set_size_inches(15, 10)
        plt.savefig(save_path)

def plot_learn_curve(hist, save_path):
    ## Minimum loss
    best_epoch = np.argmin(hist.history['val_loss'])
    min_val_loss = hist.history['val_loss'][best_epoch]
    ## Plot
    plt.figure()
    plt.title(f'learning curve: val_loss @ best epoch(#{best_epoch + 1}) = {min_val_loss:.3f}')
    plt.plot(np.arange(len(hist.history['loss'])) + 1, hist.history['loss'], label='training loss')
    plt.plot(np.arange(len(hist.history['val_loss'])) + 1, hist.history['val_loss'], label='validation loss')
    plt.xlabel('epoch number')
    plt.ylabel('loss')
    plt.xlim([1, len(hist.history['loss'])])
    plt.legend()
    plt.gcf().set_size_inches(8.3, 5.85)
    plt.savefig(save_path, format='pdf')

def plot_rocs(probS_dict, true_lab, save_path):
    linestyles = ['-', '-', '-', '-.', '-.', '-.']
    plt.figure()
    for classifier_name, probS, linestyle in zip(probS_dict.keys(), probS_dict.values(), linestyles):
        bkg_eff, sig_eff, thresh = sklearn.metrics.roc_curve(true_lab, probS)
        AUC = sklearn.metrics.auc(bkg_eff, sig_eff)
        plt.semilogy(sig_eff, 1/bkg_eff, linestyle, label=f'{classifier_name}: AUC = {AUC:.2f}')
    plt.xlim([0, 1])
    plt.grid(which='both')
    plt.legend(loc='best')
    plt.xlabel('Signal efficiency')
    plt.ylabel('Background rejection (1/bkg_eff)')
    plt.gcf().set_size_inches(10, 10)
    plt.savefig(save_path, format='pdf')

def plot_nn_hists(probS_dict, true_lab, unsup_labs, save_dir):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    print('made dir')
    hist_params = {'histtype': 'step', 'density': True, 'bins': 100}

    true_sig_idx, true_bkg_idx = true_lab.astype(bool), ~true_lab.astype(bool)
    for classifier_name, probS in zip(probS_dict.keys(), probS_dict.values()):
        plt.figure()
        plt.hist(probS[true_sig_idx], label='true signal', **hist_params)
        plt.hist(probS[true_bkg_idx], label='true background', **hist_params)
        plt.legend()

        plt.gcf().set_size_inches(10, 10)
        plt.savefig(save_dir+classifier_name+'_hist_truelab.pdf', format='pdf')

    pseudo_sig_idx1, pseudo_bkg_idx1 = unsup_labs[0].astype(bool), ~unsup_labs[0].astype(bool)
    name = 'semisup classifier on j1'
    probS = probS_dict[name]
    plt.figure()
    plt.hist(probS[pseudo_sig_idx1], label='pseudo signal', **hist_params)
    plt.hist(probS[pseudo_bkg_idx1], label='pseudo background', **hist_params)
    plt.gcf().set_size_inches(10, 10)
    plt.savefig(save_dir+name+'_hist_pseudo_lab.pdf', format='pdf')

    pseudo_sig_idx2, pseudo_bkg_idx2 = unsup_labs[1].astype(bool), ~unsup_labs[1].astype(bool)
    name = 'semisup classifier on j2'
    probS = probS_dict[name]
    plt.figure()
    plt.hist(probS[pseudo_sig_idx2], label='pseudo signal', **hist_params)
    plt.hist(probS[pseudo_bkg_idx2], label='pseudo background', **hist_params)
    plt.gcf().set_size_inches(10, 10)
    plt.savefig(save_dir+name+'_hist_pseudo_lab.pdf', format='pdf')
