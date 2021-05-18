import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt

from semisup import combine_SB, determine_feats
from UTILS.lstm_classifier import preproc_for_lstm, create_lstm_classifier, train_classifier
from UTILS.utils import create_one_hot_encoder, nominal2onehot
from UTILS.plots_and_logs import plot_rocs


exp_dir_path = "/gpfs0/kats/users/wunch/SemiSupSemiVis/test_fullsup_30constits/"
model1_save_path = "/gpfs0/kats/users/wunch/SemiSupSemiVis/test_fullsup_30constits/j1/"
model2_save_path = "/gpfs0/kats/users/wunch/SemiSupSemiVis/test_fullsup_30constits/j2/"

B_path = "/gpfs0/kats/users/wunch/semisup_dataset/bkg_bb_GenMjjGt800_GenPtGt40_GenEtaSt3_MjjGt1000_PtGt50_EtaSt2.5_y*lt1/train"
S_path = "/gpfs0/kats/users/wunch/semisup_dataset/sig_dl0.5_rinv0.00_mZp1500_lambda20_GenMjjGt800_GenPtGt40_GenEtaSt3_MjjGt1000_PtGt50_EtaSt2.5_y*lt1/train"
Btest_path = "/gpfs0/kats/users/wunch/semisup_dataset/bkg_bb_GenMjjGt800_GenPtGt40_GenEtaSt3_MjjGt1000_PtGt50_EtaSt2.5_y*lt1/test"
Stest_path = "/gpfs0/kats/users/wunch/semisup_dataset/sig_dl0.5_rinv0.00_mZp1500_lambda20_GenMjjGt800_GenPtGt40_GenEtaSt3_MjjGt1000_PtGt50_EtaSt2.5_y*lt1/test"

Ntest = 2e4

print('Loading train data...')
j1_df, j2_df, event_labels = combine_SB(Btest_path, Stest_path, Ntest, 0.5)
print('Training data loaded')

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

def calc_fake_thrust(col_dict):
    deltaR = col_dict['constit_deltaR']
    PT = col_dict['constit_PT']
    jet_PT = col_dict['jet_PT']
    return np.sum(deltaR*PT)/jet_PT

def calc_c1b(col_dict, R0=0.7, beta=0.2):
    PT = col_dict['constit_PT']
    Eta = col_dict['constit_Eta']
    Phi = col_dict['constit_Phi']
    jet_PT = col_dict['jet_PT']

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
    particle = map(PID, pid_particle_map)

    Ecal_particles = ['photon', 'h0', 'e+', 'e-']
    Ecal = np.sum(np.isin(particle, Ecal_particles)*PT)
    Eratio =  Ecal/jet_PT
    return Eratio

def calc_disp_median(col_dict, col_name=None):
    col = col_dict[col_name]
    col = col[np.abs(col)>1e-6]
    if len(col)>0:
        return np.median(col)
    else:
        return -1

def calc_median(col_dict, col_name=None):
    col = col_dict[col_name]
    return np.median(col)

mult = j1_df.mult
n_verts = j1_df.n_verts
fake_thrust = j1_df.apply(calc_fake_thrust, axis=1)
med_d0 = j1_df.apply(calc_disp_median, col_name='constit_D0', axis=1)
med_dz = j1_df.apply(calc_disp_median, col_name='constit_DZ', axis=1)
med_dR = j1_df.apply(calc_median, col_name='constit_deltaR', axis=1)
c1b = j1_df.apply(calc_c1b, axis=1)
Eratio = j1_df.apply(calc_Eratio, axis=1)

mult_sig, mult_bkg = mult[event_labels.astype(bool)], mult[~event_labels.astype(bool)]
n_verts_sig, n_verts_bkg = n_verts[event_labels.astype(bool)], n_verts[~event_labels.astype(bool)]
fake_thrust_sig, fake_thrust_bkg = fake_thrust[event_labels.astype(bool)], fake_thrust[~event_labels.astype(bool)]
med_d0_sig, med_d0_bkg = med_d0[event_labels.astype(bool)&(med_d0!=-1)], med_d0[~event_labels.astype(bool)&(med_d0!=-1)]
med_dz_sig, med_dz_bkg = med_dz[event_labels.astype(bool)&(med_dz!=-1)], med_dz[~event_labels.astype(bool)&(med_dz!=-1)]
med_dR_sig, med_dR_bkg = med_dR[event_labels.astype(bool)], med_dR[~event_labels.astype(bool)]
c1b_sig, c1b_bkg = c1b[event_labels.astype(bool)], c1b[~event_labels.astype(bool)]
Eratio_sig, Eratio_bkg = Eratio[event_labels.astype(bool)], Eratio[~event_labels.astype(bool)]

plt.figure()
histdict = dict(label=['S', 'B'], histtype='step', align='mid')

plt.subplot(2, 3, 1)
plt.tight_layout()
bins = np.arange(-0.5, np.max([np.max(mult_bkg), np.max(mult_sig)])+0.5)
plt.hist([mult_sig, mult_bkg], bins=bins, **histdict)
plt.legend()
plt.yticks([])
plt.xlabel('Constituent multiplicity')

plt.subplot(2, 3, 2)
plt.tight_layout()
bins = np.arange(-0.5, np.max([np.max(n_verts_bkg), np.max(n_verts_sig)])+0.5)
plt.hist([n_verts_sig, n_verts_bkg], bins=bins, **histdict)
plt.yticks([])
plt.xlabel('Vertex count')


plt.subplot(2, 3, 3)
plt.tight_layout()
bins = np.arange(-0.2, 0.2, 0.01)
plt.hist([med_dz_sig, med_dz_bkg], bins=bins, **histdict)
plt.yticks([])
plt.xlabel('median DZ displacement')

plt.subplot(2, 3, 4)
plt.tight_layout()
bins = np.arange(-0.2, 0.2, 0.01)
plt.hist([med_d0_sig, med_d0_bkg], bins=bins, **histdict)
plt.yticks([])
plt.xlabel('median D0 displacement')

plt.subplot(2, 3, 5)
plt.tight_layout()
plt.hist([fake_thrust_sig, fake_thrust_bkg], **histdict)
plt.yticks([])
plt.xlabel('Fake thrust')

plt.subplot(2, 3, 6)
plt.tight_layout()
plt.hist([med_dR_sig, med_dR_bkg], **histdict)
plt.yticks([])
plt.xlabel('Median distance from jet axis')


plt.savefig('highlevelfeats1.png')
plt.clf()

plt.figure()
histdict = dict(label=['S', 'B'], histtype='step', align='mid')

plt.subplot(2, 1, 1)
plt.tight_layout()
plt.hist([c1b_sig, c1b_bkg], **histdict)
plt.yticks([])
plt.xlabel('c1b')

plt.subplot(2, 1, 2)
plt.tight_layout()
plt.hist([Eratio_sig, Eratio_bkg], **histdict)
plt.yticks([])
plt.xlabel('Eratio')


plt.savefig('highlevelfeats2.png')
plt.clf()
