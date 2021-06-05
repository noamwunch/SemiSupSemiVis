from UTILS.plots_and_logs import set_mpl_rc
import numpy as np
from matplotlib import pyplot as plt
import sklearn.metrics


def plot_rocs(classifier_dicts, save_path):
    set_mpl_rc()
    fig, ax = plt.subplots()
    for classifier_name, classifier_dict in zip(classifier_dicts.keys(), classifier_dicts.values()):
        probS = classifier_dict['probS']
        plot_dict = classifier_dict['plot_dict']
        true_lab = classifier_dict['lab']

        bkg_eff, sig_eff, thresh = sklearn.metrics.roc_curve(true_lab, probS)
        AUC = sklearn.metrics.auc(bkg_eff, sig_eff)
        plt.semilogy(sig_eff, 1 / bkg_eff, label=f'{classifier_name}', **plot_dict)
        # (AUC = {AUC:.2f})

    plt.xlim([0, 1])
    plt.ylim(top=ax.get_ylim()[1] * 1.2)
    plt.legend(loc='best')
    plt.xlabel('$\\epsilon_{S}$')
    plt.ylabel('Background rejection ($1/\\epsilon_{B}$)')
    fig.savefig(save_path)

    plt.close('all')


def plot_rocs_significance(classifier_dicts, true_lab, save_path):
    set_mpl_rc()
    fig, ax = plt.subplots()
    for classifier_name, classifier_dict in zip(classifier_dicts.keys(), classifier_dicts.values()):
        probS = classifier_dict['probS']
        plot_dict = classifier_dict['plot_dict']

        bkg_eff, sig_eff, thresh = sklearn.metrics.roc_curve(true_lab, probS)
        AUC = sklearn.metrics.auc(bkg_eff, sig_eff)
        plt.plot(sig_eff, sig_eff / np.sqrt(bkg_eff), label=f'{classifier_name}', **plot_dict)
        # (AUC = {AUC:.2f})

    plt.xlim([0.4, 1])
    plt.ylim(bottom=0)
    plt.ylim(top=10)
    plt.legend(loc='best')
    plt.xlabel('$\\epsilon_{S}$')
    plt.ylabel('$\\epsilon_{S}/\\sqrt{\\epsilon_{B}} (\\sigma)$')
    fig.savefig(save_path)

    plt.close('all')


## Plot significance gain
plot_significance = False
if plot_significance:
    exp_dir_path = "RESULTS/05062021_sf0.005_40eps_1e5evs_forseminar_new_withsig/"

    classifier_preds_save_dir = exp_dir_path + 'classifier_preds/'
    # for classifier_name, classifier_dict in zip(classifier_dicts.keys(), classifier_dicts.values()):
    #     probS = classifier_dict['probS']
    #     np.save(classifier_preds_save_dir+classifier_name+'.npy', probS)

    event_label_test = np.load(classifier_preds_save_dir + 'event_labels.npy')

    event_semisup_probS = np.load(classifier_preds_save_dir + 'event NN' + '.npy')
    event_unsup_probS = np.load(classifier_preds_save_dir + 'event multiplicity' + '.npy')

    classifier_dicts = {'event NN': {'probS': event_semisup_probS, 'plot_dict': {'linestyle': '-', 'color': 'black'}},
                        'event multiplicity': {'probS': event_unsup_probS,
                                               'plot_dict': {'linestyle': '--', 'color': 'black'}}}

    plot_rocs_significance(classifier_dicts=classifier_dicts, true_lab=event_label_test,
                           save_path=exp_dir_path + 'log_ROC_significance_new.pdf')

    # classifier_dicts = {'event NN': {'probS': event_semisup_probS, 'plot_dict': {'linestyle': '-', 'color': 'black'}},
    #                     'j1 NN': {'probS': j1_semisup_probS, 'plot_dict': {'linestyle': '-', 'color': 'blue'}},
    #                     'j2 NN': {'probS': j2_semisup_probS, 'plot_dict': {'linestyle': '-', 'color': 'green'}},
    #                     'event multiplicity': {'probS': event_unsup_probS, 'plot_dict': {'linestyle': '--', 'color': 'black'}},
    #                     'j1 multiplicity': {'probS': j1_unsup_probS, 'plot_dict': {'linestyle': '--', 'color': 'blue'}},
    #                     'j2 multiplicity': {'probS': j2_unsup_probS, 'plot_dict': {'linestyle': '--', 'color': 'green'}}}

## Compare dense/lstm with diff sigfracs
comp_dense_lstm_sigfracs = True
if comp_dense_lstm_sigfracs:
    densesf1_exp_dir_path = "RESULTS/05062021_sf0.005_40eps_1e5evs_forseminar_new_withsig/"
    densesf2_exp_dir_path = "RESULTS/05062021_sf0.05_40eps_1e5evs_forseminar_dense/"
    lstmsf1_exp_dir_path = "RESULTS/05062021_sf0.005_40eps_1e5evs_forseminar_lstm/"
    lstmsf2_exp_dir_path = "RESULTS/05062021_sf0.05_40eps_1e5evs_forseminar_lstm/"


    def load_preds_labs(exp_dir_path):
        preds_dir = exp_dir_path + 'classifier_preds/'
        evlab = np.load(preds_dir + 'event_labels.npy')
        preds = np.load(preds_dir + 'event NN' + '.npy')
        return evlab, preds


    densesf1_evlab, densesf1_preds = load_preds_labs(densesf1_exp_dir_path)
    densesf2_evlab, densesf2_preds = load_preds_labs(densesf2_exp_dir_path)
    lstmsf1_evlab, lstmsf1_preds = load_preds_labs(lstmsf1_exp_dir_path)
    lstmsf2_evlab, lstmsf2_preds = load_preds_labs(lstmsf2_exp_dir_path)

    densesf1_evlab_dict = {'probS': densesf1_preds, 'lab': densesf1_evlab,
                           'plot_dict': {'linestyle': '-', 'color': 'black'}}
    densesf2_evlab_dict = {'probS': densesf2_preds, 'lab': densesf2_evlab,
                           'plot_dict': {'linestyle': '-', 'color': 'red'}}
    lstmsf1_evlab_dict = {'probS': lstmsf1_preds, 'lab': lstmsf1_evlab,
                          'plot_dict': {'linestyle': '--', 'color': 'black'}}
    lstmsf2_evlab_dict = {'probS': lstmsf2_preds, 'lab': lstmsf2_evlab,
                          'plot_dict': {'linestyle': '--', 'color': 'red'}}

    classifier_dicts = {'Dense $sf = 5 \\times 10^{-3}$': densesf1_evlab_dict,
                        'Dense $sf = 5 \\times 10^{-2}$': densesf2_evlab_dict,
                        'LSTM $sf = 5 \\times 10^{-3}$': lstmsf1_evlab_dict,
                        'LSTM $sf = 5 \\times 10^{-2}$': lstmsf2_evlab_dict}

    plot_rocs(classifier_dicts=classifier_dicts, save_path='lstm_dense_compare.pdf')
