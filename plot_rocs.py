from UTILS.plots_and_logs import set_mpl_rc
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
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
    plt.legend(bbox_to_anchor=(1.2, 1.0))
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
    densesf2_exp_dir_path = "RESULTS/05062021_sf0.007_40eps_1e5evs_forseminar_dense/"
    densesf3_exp_dir_path = "RESULTS/05062021_sf0.01_40eps_1e5evs_forseminar_dense/"
    densesf4_exp_dir_path = "RESULTS/05062021_sf0.03_40eps_1e5evs_forseminar_dense/"
    densesf5_exp_dir_path = "RESULTS/05062021_sf0.05_40eps_1e5evs_forseminar_dense/"

    lstmsf1_exp_dir_path = "RESULTS/05062021_sf0.005_40eps_1e5evs_forseminar_lstm/"
    lstmsf2_exp_dir_path = "RESULTS/05062021_sf0.007_40eps_1e5evs_forseminar_lstm/"
    lstmsf3_exp_dir_path = "RESULTS/05062021_sf0.01_40eps_1e5evs_forseminar_lstm/"
    lstmsf4_exp_dir_path = "RESULTS/05062021_sf0.03_40eps_1e5evs_forseminar_lstm/"
    lstmsf5_exp_dir_path = "RESULTS/05062021_sf0.05_40eps_1e5evs_forseminar_lstm/"

    def load_preds_labs(exp_dir_path):
        preds_dir = exp_dir_path + 'classifier_preds/'
        evlab = np.load(preds_dir + 'event_labels.npy')
        preds = np.load(preds_dir + 'event NN' + '.npy')
        return evlab, preds

    lstmsf1_evlab, lstmsf1_preds = load_preds_labs(lstmsf1_exp_dir_path)
    lstmsf2_evlab, lstmsf2_preds = load_preds_labs(lstmsf2_exp_dir_path)
    lstmsf3_evlab, lstmsf3_preds = load_preds_labs(lstmsf3_exp_dir_path)
    lstmsf4_evlab, lstmsf4_preds = load_preds_labs(lstmsf4_exp_dir_path)
    lstmsf5_evlab, lstmsf5_preds = load_preds_labs(lstmsf5_exp_dir_path)

    densesf1_evlab, densesf1_preds = load_preds_labs(densesf1_exp_dir_path)
    densesf2_evlab, densesf2_preds = load_preds_labs(densesf2_exp_dir_path)
    densesf3_evlab, densesf3_preds = load_preds_labs(densesf3_exp_dir_path)
    densesf4_evlab, densesf4_preds = load_preds_labs(densesf4_exp_dir_path)
    densesf5_evlab, densesf5_preds = load_preds_labs(densesf5_exp_dir_path)

    x = np.linspace(0.2, 1, 5)
    red_colors = cm.Reds(x)
    blue_colors = cm.Blues(x)

    densesf1_evlab_dict = {'probS': densesf1_preds, 'lab': densesf1_evlab,
                           'plot_dict': {'linestyle': '-', 'color': red_colors[0]}}
    densesf2_evlab_dict = {'probS': densesf2_preds, 'lab': densesf2_evlab,
                           'plot_dict': {'linestyle': '-', 'color': red_colors[1]}}
    densesf3_evlab_dict = {'probS': densesf4_preds, 'lab': densesf3_evlab,
                           'plot_dict': {'linestyle': '-', 'color': red_colors[2]}}
    densesf4_evlab_dict = {'probS': densesf4_preds, 'lab': densesf4_evlab,
                           'plot_dict': {'linestyle': '-', 'color': red_colors[3]}}
    densesf5_evlab_dict = {'probS': densesf5_preds, 'lab': densesf5_evlab,
                           'plot_dict': {'linestyle': '-', 'color': red_colors[4]}}

    lstmsf1_evlab_dict = {'probS': lstmsf1_preds, 'lab': lstmsf1_evlab,
                          'plot_dict': {'linestyle': '--', 'color': blue_colors[0]}}
    lstmsf2_evlab_dict = {'probS': lstmsf2_preds, 'lab': lstmsf2_evlab,
                          'plot_dict': {'linestyle': '--', 'color': blue_colors[1]}}
    lstmsf3_evlab_dict = {'probS': lstmsf3_preds, 'lab': lstmsf3_evlab,
                          'plot_dict': {'linestyle': '--', 'color': blue_colors[2]}}
    lstmsf4_evlab_dict = {'probS': lstmsf4_preds, 'lab': lstmsf4_evlab,
                          'plot_dict': {'linestyle': '--', 'color': blue_colors[3]}}
    lstmsf5_evlab_dict = {'probS': lstmsf5_preds, 'lab': lstmsf5_evlab,
                          'plot_dict': {'linestyle': '--', 'color': blue_colors[4]}}

    classifier_dicts = {'Dense $sf = 5 \\times 10^{-3}$': densesf1_evlab_dict,
                        'Dense $sf = 7 \\times 10^{-3}$': densesf2_evlab_dict,
                        'Dense $sf = 1 \\times 10^{-2}$': densesf3_evlab_dict,
                        'Dense $sf = 3 \\times 10^{-2}$': densesf4_evlab_dict,
                        'Dense $sf = 5 \\times 10^{-2}$': densesf5_evlab_dict,
                        'LSTM $sf = 5 \\times 10^{-3}$': lstmsf1_evlab_dict,
                        'LSTM $sf = 7 \\times 10^{-3}$': lstmsf2_evlab_dict,
                        'LSTM $sf = 1 \\times 10^{-2}$': lstmsf3_evlab_dict,
                        'LSTM $sf = 3 \\times 10^{-2}$': lstmsf4_evlab_dict,
                        'LSTM $sf = 5 \\times 10^{-2}$': lstmsf5_evlab_dict}

    plot_rocs(classifier_dicts=classifier_dicts, save_path='lstm_dense_compare.pdf')
