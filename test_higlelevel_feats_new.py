from semisup import combine_SB
from UTILS.dense_classifier import preproc_for_dense
from UTILS.dense_classifier import plot_event_histograms_dense
from UTILS.dense_classifier import plot_preproced_feats_dense
from UTILS.dense_classifier import plot_nn_inp_histograms_dense

exp_dir_path = "/gpfs0/kats/users/wunch/SemiSupSemiVis/test_fullsup_30constits/"
model1_save_path = "/gpfs0/kats/users/wunch/SemiSupSemiVis/test_fullsup_30constits/j1/"
model2_save_path = "/gpfs0/kats/users/wunch/SemiSupSemiVis/test_fullsup_30constits/j2/"

B_path = "/gpfs0/kats/users/wunch/semisup_dataset/bkg_bb_GenMjjGt800_GenPtGt40_GenEtaSt3_MjjGt1000_PtGt50_EtaSt2.5_y*lt1/train"
S_path = "/gpfs0/kats/users/wunch/semisup_dataset/sig_dl0.5_rinv0.00_mZp1500_lambda20_GenMjjGt800_GenPtGt40_GenEtaSt3_MjjGt1000_PtGt50_EtaSt2.5_y*lt1/train"
Btest_path = "/gpfs0/kats/users/wunch/semisup_dataset/bkg_bb_GenMjjGt800_GenPtGt40_GenEtaSt3_MjjGt1000_PtGt50_EtaSt2.5_y*lt1/test"
Stest_path = "/gpfs0/kats/users/wunch/semisup_dataset/sig_dl0.5_rinv0.00_mZp1500_lambda20_GenMjjGt800_GenPtGt40_GenEtaSt3_MjjGt1000_PtGt50_EtaSt2.5_y*lt1/test"

Ntest = 2e3

print('Loading train data...')
j1_df, j2_df, event_labels = combine_SB(Btest_path, Stest_path, Ntest, 0.5)
print('Training data loaded')

print('Plotting event histograms')
pdf_path = 'event_hists.pdf'
plot_event_histograms_dense(j1_df, j2_df, event_labels=event_labels, pdf_path=pdf_path)
print('Finished plotting event histograms')

print('Preprocessing events')
j1_preproc, j2_preproc = preproc_for_dense(j1_df), preproc_for_dense(j2_df)
print('Finished preprocessing events')

print('Plotting preproced event histograms')
pdf_path = 'event_hists_preproc.pdf'
plot_preproced_feats_dense(j1_preproc, j2_preproc, event_labels=event_labels, pdf_path=pdf_path)
print('Finished plotting preproced event histograms')

preproc_args = dict(feats=('constit_mult', 'ptwmean_dR', 'ptwmean_absD0', 'ptwmean_absDz', 'c1b'))
print('Plotting preproced event histograms jet 1')
pdf_path = 'event_hists_preproc_jet1.pdf'
plot_nn_inp_histograms_dense(j1_preproc, event_labels=event_labels, preproc_args=preproc_args, pdf_path=pdf_path)
print('Finished plotting preproced event histograms jet 1')

print('Plotting preproced event histograms jet 2')
pdf_path = 'event_hists_preproc_jet2.pdf'
plot_nn_inp_histograms_dense(j1_preproc, event_labels=event_labels, preproc_args=preproc_args, pdf_path=pdf_path)
print('Finished plotting preproced event histograms jet 2')

