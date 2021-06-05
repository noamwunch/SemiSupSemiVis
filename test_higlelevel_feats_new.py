import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from UTILS.utils import evs_txt2jets_df_with_verts_withparton as load_data

from UTILS.dense_classifier import preproc_for_dense
from UTILS.dense_classifier import plot_event_histograms_dense
from UTILS.dense_classifier import plot_preproced_feats_dense
from UTILS.dense_classifier import plot_nn_inp_histograms_dense
from UTILS.dense_classifier import set_mpl_rc

from UTILS.dense_classifier import all_feats
import time

def plot_mult_corrs(j1_df, j2_df, event_labs):
    set_mpl_rc()
    event_labs = event_labs.astype(bool)
    multb1 = j1_df[~event_labs].mult
    multb2 = j2_df[~event_labs].mult
    mults1 = j1_df[event_labs].mult
    mults2 = j2_df[event_labs].mult

    corr_coeff_b = np.corrcoef(multb1, multb2)
    corr_coeff_s = np.corrcoef(mults1, mults2)
    corr_coeff_b = corr_coeff_b[1, 0]
    corr_coeff_s = corr_coeff_s[1, 0]

    corr_txt_b = f'{corr_coeff_b:.2g}'
    corr_txt_s = f'{corr_coeff_s:.2g}'

    pre_txt = r'$\rho_{mult_1, mult_2} = '
    lab_b = 'S: ' + pre_txt + corr_txt_b + '$'
    lab_s = 'B: ' + pre_txt + corr_txt_s + '$'

    fig, ax = plt.subplots()
    plt.scatter(multb1, multb2, 2, label=lab_b, color='blue', alpha=0.6)
    plt.scatter(mults1, mults2, 2, label=lab_s, color='red', alpha=0.6)

    plt.xlim([1, 100])
    plt.ylim([1, 100])

    plt.xticks([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.yticks([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.legend(loc='best', frameon=True, framealpha=0.95)
    plt.xlabel('$jet_1$ Constituent Multiplicity')
    plt.ylabel('$jet_2$ Constituent Multiplicity')

    fig.savefig('mult_corr_alpha.pdf')
    plt.clf()

    ## 2D hist
    bins = np.arange(0.5, 150, 5)
    hb, _, _, _ = plt.hist2d(multb1, multb2, bins=bins)
    hs, _, _, _ = plt.hist2d(mults1, mults2, bins=bins)

    fig_b, ax_b = plt.subplots()
    fig_s, ax_s = plt.subplots()

    ax_b.imshow(hb, cmap='Blues', extent=[1, 150, 1, 150])
    ax_s.imshow(hs, cmap='Reds', extent=[1, 150, 1, 150])

    xticks = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
    ax_b.set_xticks(xticks)
    ax_b.set_yticks(xticks)
    ax_b.set_xlabel('$jet_1$ Constituent Multiplicity')
    ax_b.set_ylabel('$jet_2$ Constituent Multiplicity')

    ax_s.set_xticks(xticks)
    ax_s.set_yticks(xticks)
    ax_s.set_xlabel('$jet_1$ Constituent Multiplicity')
    ax_s.set_ylabel('$jet_2$ Constituent Multiplicity')

    fig_b.savefig('multb_hist2d.pdf')
    fig_s.savefig('mults_hist2d.pdf')

    fig_b.clf()
    fig_s.clf()

def combine_SB(B_path, S_path, N, sig_frac):
    mjj_range = (1200, 1500)
    n_B, n_S = int(N*(1 - sig_frac)), int(N * sig_frac)

    idxs = np.arange(n_B+n_S)
    np.random.shuffle(idxs)

    event_label = np.array([0]*n_B + [1]*n_S)[idxs]

    B_j1_df, B_j2_df = load_data(B_path, n_ev=n_B, mjj_range=mjj_range)
    S_j1_df, S_j2_df = load_data(S_path, n_ev=n_S, mjj_range=mjj_range)

    j1_df = pd.concat([B_j1_df, S_j1_df]).iloc[idxs].reset_index(drop=True)
    j2_df = pd.concat([B_j2_df, S_j2_df]).iloc[idxs].reset_index(drop=True)

    return j1_df, j2_df, event_label

exp_dir_path = "/gpfs0/kats/users/wunch/SemiSupSemiVis/test_fullsup_30constits/"
model1_save_path = "/gpfs0/kats/users/wunch/SemiSupSemiVis/test_fullsup_30constits/j1/"
model2_save_path = "/gpfs0/kats/users/wunch/SemiSupSemiVis/test_fullsup_30constits/j2/"

B_path = "/gpfs0/kats/users/wunch/semisup_dataset/bkg_bb_GenMjjGt800_GenPtGt40_GenEtaSt3_MjjGt1000_PtGt50_EtaSt2.5_y*lt1/train"
S_path = "/gpfs0/kats/users/wunch/semisup_dataset/sig_dl0.5_rinv0.00_mZp1500_lambda20_GenMjjGt800_GenPtGt40_GenEtaSt3_MjjGt1000_PtGt50_EtaSt2.5_y*lt1/train"
Btest_path = "/gpfs0/kats/users/wunch/semisup_dataset/bkg_bb_GenMjjGt800_GenPtGt40_GenEtaSt3_MjjGt1000_PtGt50_EtaSt2.5_y*lt1/test"
Stest_path = "/gpfs0/kats/users/wunch/semisup_dataset/sig_dl0.5_rinv0.00_mZp1500_lambda20_GenMjjGt800_GenPtGt40_GenEtaSt3_MjjGt1000_PtGt50_EtaSt2.5_y*lt1/test"

Ntest = 2e4
feats = ['constit_mult',
         'vert_count',
         'ptwmean_dR',
         'ptwmean_absD0',
         'ptwmean_absDZ',
         'photonE_over_jetpt',
         'photonE_over_chadE',
         'c1b']

print('Loading train data...')
j1_df, j2_df, event_labels = combine_SB(Btest_path, Stest_path, Ntest, 0.5)
print('Training data loaded')

plot_mult_corrs(j1_df, j2_df, event_labels)

# t0 = time.time()
# print('Plotting event histograms')
# pdf_path = 'event_hists_c1b.pdf'
# plot_event_histograms_dense(j1_df, j2_df, event_labels=event_labels, pdf_path=pdf_path)
# print('Finished plotting event histograms')
# t1 = time.time()
#
# print(f'time = {t1-t0}')

# print('Preprocessing events')
# j1_preproc, j2_preproc = preproc_for_dense(j1_df, feats=feats), preproc_for_dense(j2_df, feats=feats)
# print('Finished preprocessing events')
#
# print('Plotting preproced event histograms')
# pdf_path = 'event_hists_preproc.pdf'
# plot_preproced_feats_dense(j1_preproc, j2_preproc, event_labels=event_labels, feats=feats, pdf_path=pdf_path)
# print('Finished plotting preproced event histograms')
#
# print('Plotting preproced event histograms jet 1')
# pdf_path = 'event_hists_preproc_jet1.pdf'
# plot_nn_inp_histograms_dense(j1_preproc, event_labels=event_labels, feats=feats, pdf_path=pdf_path)
# print('Finished plotting preproced event histograms jet 1')
#
# print('Plotting preproced event histograms jet 2')
# pdf_path = 'event_hists_preproc_jet2.pdf'
# plot_nn_inp_histograms_dense(j1_preproc, event_labels=event_labels, feats=feats, pdf_path=pdf_path)
# print('Finished plotting preproced event histograms jet 2')
#
