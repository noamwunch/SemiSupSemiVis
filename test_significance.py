import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from UTILS.utils import evs_txt2jets_df_with_verts_withparton as load_data

from UTILS.dense_classifier import preproc_for_dense
from UTILS.dense_classifier import plot_event_histograms_dense
from UTILS.dense_classifier import plot_preproced_feats_dense
from UTILS.dense_classifier import plot_nn_inp_histograms_dense

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

def eval_significance(B_path, S_path, N, sig_frac, fig_path):
    j1_df, j2_df, event_labels = combine_SB(B_path, S_path, N, sig_frac)
    event_labels = event_labels.astype(bool)
    mult_preds = np.array(j1_df.mult + j2_df.mult)

    plot_significance(mult_preds, event_labels, fig_path)

def plot_significance(preds, event_labels, fig_path):
    data_eff = np.logspace(0.005, 0.995, 50, base=2)
    significance = calc_significance(preds, event_labels, data_eff)

    fig, ax = plt.subplots()
    plt.plot(data_eff, significance)
    plt.xscale('log')
    plt.xlabel('$\\epsilon$')
    plt.ylabel('$N_{S}/\\sqrt{N_{B}}$')
    fig.savefig(fig_path)

def calc_significance(preds, ev_lab, data_effs):
    sorted_idxs = np.argsort(-preds)  # indices sorted by decending preds
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

B_path = "/gpfs0/kats/users/wunch/semisup_dataset/bkg_bb_GenMjjGt800_GenPtGt40_GenEtaSt3_MjjGt1000_PtGt50_EtaSt2.5_y*lt1/train"
S_path = "/gpfs0/kats/users/wunch/semisup_dataset/sig_dl0.5_rinv0.00_mZp1500_lambda20_GenMjjGt800_GenPtGt40_GenEtaSt3_MjjGt1000_PtGt50_EtaSt2.5_y*lt1/train"
N = 8e4
sig_frac = 0.005
fig_path = 'test.pdf'

eval_significance(B_path, S_path, N, sig_frac, fig_path)




