from UTILS.plots_and_logs import plot_rocs_significance
import numpy as np

exp_dir_path = "RESULTS/05062021_sf0.005_40eps_1e5evs_forseminar_new_withsig/"

classifier_preds_save_dir = exp_dir_path + 'classifier_preds/'
# for classifier_name, classifier_dict in zip(classifier_dicts.keys(), classifier_dicts.values()):
#     probS = classifier_dict['probS']
#     np.save(classifier_preds_save_dir+classifier_name+'.npy', probS)

event_label_test = np.load(classifier_preds_save_dir+'event_labels.npy')

event_semisup_probS = np.load(classifier_preds_save_dir+'event NN'+'.npy')
event_unsup_probS = np.load(classifier_preds_save_dir+'event multiplicity'+'.npy')

classifier_dicts = {'event NN': {'probS': event_semisup_probS, 'plot_dict': {'linestyle': '-', 'color': 'black'}},
                    'event multiplicity': {'probS': event_unsup_probS, 'plot_dict': {'linestyle': '--', 'color': 'black'}}}

plot_rocs_significance(classifier_dicts=classifier_dicts, true_lab=event_label_test,
                       save_path=exp_dir_path+'log_ROC_significance_new.pdf')

# classifier_dicts = {'event NN': {'probS': event_semisup_probS, 'plot_dict': {'linestyle': '-', 'color': 'black'}},
#                     'j1 NN': {'probS': j1_semisup_probS, 'plot_dict': {'linestyle': '-', 'color': 'blue'}},
#                     'j2 NN': {'probS': j2_semisup_probS, 'plot_dict': {'linestyle': '-', 'color': 'green'}},
#                     'event multiplicity': {'probS': event_unsup_probS, 'plot_dict': {'linestyle': '--', 'color': 'black'}},
#                     'j1 multiplicity': {'probS': j1_unsup_probS, 'plot_dict': {'linestyle': '--', 'color': 'blue'}},
#                     'j2 multiplicity': {'probS': j2_unsup_probS, 'plot_dict': {'linestyle': '--', 'color': 'green'}}}