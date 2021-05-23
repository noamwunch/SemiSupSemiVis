import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from cycler import cycler

def calc_dphi(phi1, phi2):
    if np.abs(phi1 - phi2) <= np.pi:
        d_phi = np.abs(phi1 - phi2)
    else:
        d_phi = 2 * np.pi - np.abs(phi1 - phi2)
    return d_phi

def calc_dR(eta1, eta2, phi1, phi2):
    d_eta = eta1-eta2
    d_phi = calc_dphi(phi1, phi2)
    dR = (d_eta**2 + d_phi**2)**0.5
    return dR

def calc_ptwmean_dR(jet_feats):
    deltaR = jet_feats['constit_deltaR']
    PT = jet_feats['constit_PT']
    jet_PT = jet_feats['jet_PT']
    return np.sum(deltaR*PT)/jet_PT

def calc_ptwmean_absD0(jet_feats):
    absD0 = np.abs(jet_feats['constit_D0'])
    PT = jet_feats['constit_PT']
    is_track = absD0>0
    if any(is_track):
        return np.sum(absD0[is_track])/np.sum(PT[is_track])
    else:
        return -1

def calc_ptwmean_absDZ(jet_feats):
    absDZ = np.abs(jet_feats['constit_DZ'])
    PT = jet_feats['constit_PT']
    is_track = absDZ>0
    if any(is_track):
        return np.sum(absDZ[is_track])/np.sum(PT[is_track])
    else:
        return -1

def calc_c1b(jet_feats, R0=0.7, beta=0.2):
    PT = jet_feats['constit_PT']
    Eta = jet_feats['constit_Eta']
    Phi = jet_feats['constit_Phi']
    jet_PT = jet_feats['jet_PT']

    TPEC = 0
    for i in range(len(PT)):
        for j in range(i + 1, len(PT)):
            zi = PT[i]/jet_PT
            zj = PT[j]/jet_PT
            thetaij = calc_dR(Eta[i], Eta[j], Phi[i], Phi[j])/R0
            TPEC += zi*zj*thetaij**beta
    return TPEC

def calc_Eratio(col_dict):
    PID = col_dict['constit_PID']
    PT = col_dict['constit_PT']
    jet_PT = col_dict['jet_PT']

    pid = [-2212, -321, -211, -13, -11, 0, 1, 11, 13, 211, 321, 2212]
    particle_name = ['h-', 'h-', 'h-', 'mu-', 'e-', 'photon', 'h0', 'e+', 'mu+', 'h+', 'h+', 'h+']
    pid_particle_map = dict(zip(pid, particle_name))
    particle = list(map(pid_particle_map.get, PID))

    Ecal_particles = ['photon', 'h0', 'e+', 'e-']
    Ecal = np.sum(np.isin(particle, Ecal_particles)*PT)
    return Ecal/jet_PT

def create_dense_classifier(nfeats, log=''):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(32, input_shape=(nfeats, )))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Dense(16))
    model.add(keras.layers.ELU())
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Dense(4))
    model.add(keras.layers.ELU())

    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='Adam', loss='binary_crossentropy')
    model.summary()
    print("")

    summary_str_list = []
    model.summary(print_fn=lambda x: summary_str_list.append(x))
    log = log +  '\n'.join(summary_str_list) + '\n\n'

    return model, log

def preproc_for_dense(j_df, feats='all'):
    if feats == 'all':
        feats = ['constit_mult', 'ptwmean_dR', 'ptwmean_absD0', 'ptwmean_absDZ', 'c1b']

    nn_inp = []
    if 'constit_mult' in feats:
        mult = j_df.mult
        mult = (mult-30) / 30
        nn_inp.append(mult)
    if 'ptwmean_dR' in feats:
        ptwmean_dR = j_df.apply(calc_ptwmean_dR, axis=1)
        nn_inp.append(ptwmean_dR)
    if 'ptwmean_absD0' in feats:
        ptwmedian_D0 = j_df.apply(calc_ptwmean_absD0, axis=1)
        ptwmedian_D0 = ptwmedian_D0 * 5
        nn_inp.append(ptwmedian_D0)
    if 'ptwmean_absDZ' in feats:
        ptwmedian_DZ = j_df.apply(calc_ptwmean_absDZ, axis=1)
        ptwmedian_DZ = ptwmedian_DZ * 5
        nn_inp.append(ptwmedian_DZ)
    if 'c1b' in feats:
        c1b = j_df.apply(calc_c1b, axis=1)
        nn_inp.append(c1b)

    nn_inp = np.stack(nn_inp, axis=1)

    return nn_inp

def set_mpl_rc():
    plt.rcdefaults()

    font_dict = {'family': 'sans-serif', 'size': 10}
    fig_dict = {'figsize': (4, 4), 'dpi': 150, 'autolayout': True}
    savefig_dict = {'dpi': 50}
    txt_dict = {'usetex': True}

    plt.rc('axes', prop_cycle=(cycler('linestyle', ['-', '-', '--', '--'])))
    plt.rc('font', **font_dict)
    plt.rc('text', **txt_dict)
    plt.rc('savefig', **savefig_dict)
    plt.rc('figure', **fig_dict)
    plt.rc('lines', linewidth=1.0)
    plt.rc('xtick', labelsize='medium', direction="in", top=True)
    plt.rc('ytick', labelsize='medium', direction="in", right=True)
    plt.rc('legend', fontsize='small', numpoints=1, frameon=False, handlelength=1)
    plt.rc('axes', linewidth=0.5, labelsize='x-large')

def plot_hist2jet(feat1, feat2, event_labels, hist_dict=None, xlabel='', ylabel='counts/bin - normalized',  fig_name='', pdf=None, log=False):
    feat_sig1, feat_bkg1 = feat1[event_labels.astype(bool)], feat1[~event_labels.astype(bool)]
    feat_sig2, feat_bkg2 = feat2[event_labels.astype(bool)], feat2[~event_labels.astype(bool)]

    if hist_dict is None:
        label = ['S-jets - $jet_1$', 'B-jets - $jet_1$', 'S-jets - $jet_2$', 'B-jets - $jet_2$']
        color = ['red', 'blue', 'red', 'blue']
        hist_dict = dict(label=label, histtype='step', align='mid', color=color, density=True)

    fig, ax = plt.subplots()
    plt.hist([feat_sig1, feat_bkg1, feat_sig2, feat_bkg2], **hist_dict)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yticks([])
    if log:
        plt.yscale('log')
    plt.legend()

    if pdf is None:
        fig.savefig(fig_name)
    else:
        pdf.savefig(fig)
    plt.clf()

def plot_hist1jet(feat, event_labels, hist_dict=None, xlabel='', ylabel='counts/bin - normalized',  fig_name='', pdf=None, log=False):
    feat_sig, feat_bkg = feat[event_labels.astype(bool)], feat[~event_labels.astype(bool)]

    if hist_dict is None:
        label = ['S-jets', 'B-jets']
        hist_dict = dict(label=label, histtype='step', align='mid', density=True)

    fig, ax = plt.subplots()
    plt.hist([feat_sig, feat_bkg], **hist_dict)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yticks([])
    if log:
        plt.yscale('log')
    plt.legend()
    plt.legend()

    if pdf is None:
        fig.savefig(fig_name)
    else:
        pdf.savefig(fig)
    plt.clf()

def plot_event_histograms_dense(j1_df, j2_df, event_labels, pdf_path):
    set_mpl_rc()
    ylabel = 'counts/bin - normalized'
    label = ['S-jets - $jet_1$', 'B-jets - $jet_1$', 'S-jets - $jet_2$', 'B-jets - $jet_2$']
    color = ['red', 'blue', 'red', 'blue']
    with PdfPages(pdf_path) as pdf:
        # multiplicity
        constit_mult1 = j1_df.mult
        constit_mult2 = j2_df.mult

        max_mult = np.max([np.max(constit_mult1), np.max(constit_mult2)])
        bins = np.arange(0.5, max_mult+0.5)
        xlabel = 'Constituent multiplicity'
        hist_dict = dict(label=label, histtype='step', align='mid', color=color, bins=bins, density=True)
        plot_hist2jet(constit_mult1, constit_mult2, event_labels, hist_dict=hist_dict, xlabel=xlabel, ylabel=ylabel, pdf=pdf)

        # mean delta R
        ptwmean_dR1 = j1_df.apply(calc_ptwmean_dR, axis=1)
        ptwmean_dR2 = j2_df.apply(calc_ptwmean_dR, axis=1)

        max_ptwmean_dR = np.max([np.max(ptwmean_dR1), np.max(ptwmean_dR1)])
        bins = np.linspace(0, max_ptwmean_dR, 40)
        xlabel = 'mean($\\Delta R$) - $P_T$ weighted'
        hist_dict = dict(label=label, histtype='step', align='mid', color=color, bins=bins, density=True)
        plot_hist2jet(ptwmean_dR1, ptwmean_dR2, event_labels, hist_dict=hist_dict, xlabel=xlabel, ylabel=ylabel, pdf=pdf)

        # mean abs(D0)
        ptwmean_absD01 = j1_df.apply(calc_ptwmean_absD0, axis=1)
        ptwmean_absD02 = j2_df.apply(calc_ptwmean_absD0, axis=1)

        bins = np.linspace(0, 2, 40)
        xlabel = 'mean(abs($D_0$)) - $P_T$ weighted [mm]'
        hist_dict = dict(label=label, histtype='step', align='mid', color=color, bins=bins, density=True)
        plot_hist2jet(ptwmean_absD01, ptwmean_absD02, event_labels, hist_dict=hist_dict, xlabel=xlabel, ylabel=ylabel, pdf=pdf, log=True)

        # mean abs(DZ)
        ptwmean_absDZ1 = j1_df.apply(calc_ptwmean_absDZ, axis=1)
        ptwmean_absDZ2 = j2_df.apply(calc_ptwmean_absDZ, axis=1)

        bins = np.linspace(0, 1, 40)
        xlabel = 'mean(abs($D_Z$)) - $P_T$ weighted  [mm]'
        hist_dict = dict(label=label, histtype='step', align='mid', color=color, bins=bins, density=True)
        plot_hist2jet(ptwmean_absDZ1, ptwmean_absDZ2, event_labels, hist_dict=hist_dict, xlabel=xlabel, ylabel=ylabel, pdf=pdf, log=True)

        # c1b
        c1b1 = j1_df.apply(calc_c1b, axis=1)
        c1b2 = j2_df.apply(calc_c1b, axis=1)

        bins = np.linspace(0, 0.6, 40)
        xlabel = '$C_1^{(0.2)}$'
        hist_dict = dict(label=label, histtype='step', align='mid', color=color, bins=bins, density=True)
        plot_hist2jet(c1b1, c1b2, event_labels, hist_dict=hist_dict, xlabel=xlabel, ylabel=ylabel, pdf=pdf)

def plot_nn_inp_histograms_dense(nn_inp, event_labels, feats, pdf_path):
    set_mpl_rc()
    ylabel = 'counts/bin - normalized'
    label = ['S-jets - $jet_1$', 'B-jets - $jet_1$', 'S-jets - $jet_2$', 'B-jets - $jet_2$']
    color = ['red', 'blue']
    col = 0
    with PdfPages(pdf_path) as pdf:
        if 'constit_mult' in feats:
            max_mult = np.max(nn_inp[:, col])
            min_mult = np.min(nn_inp[:, col])
            bins = np.linspace(min_mult, max_mult, 40)
            xlabel = 'Scaled - Constituent multiplicity'
            hist_dict = dict(label=label, histtype='step', align='mid', color=color, bins=bins, density=True)

            plot_hist1jet(nn_inp[:, col], event_labels, hist_dict=hist_dict, xlabel=xlabel, ylabel=ylabel, pdf=pdf)
            col = col+1

        if 'ptwmean_dR' in feats:
            max_ptwmean_dR = np.max(nn_inp[:, col])
            bins = np.linspace(0, max_ptwmean_dR, 40)
            xlabel = 'Scaled - mean($\\Delta R$) - $P_T$ weighted'
            hist_dict = dict(label=label, histtype='step', align='mid', color=color, bins=bins, density=True)
            plot_hist1jet(nn_inp[:, col], event_labels, hist_dict=hist_dict, xlabel=xlabel, ylabel=ylabel, pdf=pdf)
            col = col+1

        if 'ptwmean_absD0' in feats:
            bins = np.linspace(0, 10, 40)
            xlabel = 'Scaled - mean(abs($D_0$)) - $P_T$ weighted [mm]'
            hist_dict = dict(label=label, histtype='step', align='mid', color=color, bins=bins, density=True)
            plot_hist1jet(nn_inp[:, col], event_labels, hist_dict=hist_dict, xlabel=xlabel, ylabel=ylabel, pdf=pdf, log=True)
            col = col+1

        if 'ptwmean_absDZ' in feats:
            bins = np.linspace(0, 5, 40)
            xlabel = 'Scaled - mean(abs($D_Z$)) - $P_T$ weighted  [mm]'
            hist_dict = dict(label=label, histtype='step', align='mid', color=color, bins=bins, density=True)
            plot_hist1jet(nn_inp[:, col], event_labels, hist_dict=hist_dict, xlabel=xlabel, ylabel=ylabel, pdf=pdf, log=True)
            col = col+1

        if 'c1b' in feats:
            bins = np.linspace(0, 0.6, 40)
            xlabel = 'Scaled - $C_1^{(0.2)}$'
            hist_dict = dict(label=label, histtype='step', align='mid', color=color, bins=bins, density=True)
            plot_hist1jet(nn_inp[:, col], event_labels, hist_dict=hist_dict, xlabel=xlabel, ylabel=ylabel, pdf=pdf)

def plot_preproced_feats_dense(nn_inp1, nn_inp2, event_labels, feats, pdf_path):
    set_mpl_rc()
    ylabel = 'counts/bin - normalized'
    label = ['S-jets - $jet_1$', 'B-jets - $jet_1$', 'S-jets - $jet_2$', 'B-jets - $jet_2$']
    color = ['red', 'blue', 'red', 'blue']
    col = 0
    with PdfPages(pdf_path) as pdf:
        if 'constit_mult' in feats:
            max_mult = np.max([nn_inp1[:, col], nn_inp2[:, col]])
            min_mult = np.min([nn_inp1[:, col], nn_inp2[:, col]])
            bins = np.linspace(min_mult, max_mult, 40)
            xlabel = 'Scaled - Constituent multiplicity'
            hist_dict = dict(label=label, histtype='step', align='mid', color=color, bins=bins, density=True)
            plot_hist2jet(nn_inp1[:, col], nn_inp2[:, col], event_labels, hist_dict=hist_dict, xlabel=xlabel, ylabel=ylabel, pdf=pdf)
            col = col+1

        if 'ptwmean_dR' in feats:
            max_ptwmean_dR = np.max([nn_inp1[:, col], nn_inp2[:, col]])
            bins = np.linspace(0, max_ptwmean_dR, 40)
            xlabel = 'Scaled - mean($\\Delta R$) - $P_T$ weighted'
            hist_dict = dict(label=label, histtype='step', align='mid', color=color, bins=bins, density=True)
            plot_hist2jet(nn_inp1[:, col], nn_inp2[:, col], event_labels, hist_dict=hist_dict, xlabel=xlabel, ylabel=ylabel, pdf=pdf)
            col = col+1

        if 'ptwmean_absD0' in feats:
            bins = np.linspace(0, 10, 40)
            xlabel = 'Scaled - mean(abs($D_0$)) - $P_T$ weighted [mm]'
            hist_dict = dict(label=label, histtype='step', align='mid', color=color, bins=bins, density=True)
            plot_hist2jet(nn_inp1[:, col], nn_inp2[:, col], event_labels, hist_dict=hist_dict, xlabel=xlabel, ylabel=ylabel, pdf=pdf, log=True)
            col = col+1

        if 'ptwmean_absDZ' in feats:
            bins = np.linspace(0, 5, 40)
            xlabel = 'Scaled - mean(abs($D_Z$)) - $P_T$ weighted  [mm]'
            hist_dict = dict(label=label, histtype='step', align='mid', color=color, bins=bins, density=True)
            plot_hist2jet(nn_inp1[:, col], nn_inp2[:, col], event_labels, hist_dict=hist_dict, xlabel=xlabel, ylabel=ylabel, pdf=pdf, log=True)
            col = col+1

        if 'c1b' in feats:
            bins = np.linspace(0, 0.6, 40)
            xlabel = 'Scaled - $C_1^{(0.2)}$'
            hist_dict = dict(label=label, histtype='step', align='mid', color=color, bins=bins, density=True)
            plot_hist2jet(nn_inp1[:, col], nn_inp2[:, col], event_labels, hist_dict=hist_dict, xlabel=xlabel, ylabel=ylabel, pdf=pdf)