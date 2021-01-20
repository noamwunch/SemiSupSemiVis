from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from UTILS.utils import evs_txt2jets_df_old as load_data_old
from UTILS.utils import evs_txt2jets_df as load_data_new

pt_cut = True
plot_path = "RESULTS/test"
Path(plot_path).mkdir(parents=True, exist_ok=True)

S_old_path = "/gpfs0/kats/users/wunch/semisup_data/sig"
S_new_path = "/gpfs0/kats/users/wunch/semisup_evs/sig_rinv_0.00_mjj_500"
j1_old_df, j2_old_df = load_data_old(S_old_path, n_ev=10000)
j1_new_df, j2_new_df = load_data_new(S_new_path, n_ev=10000)
mult1_sig_old = j1_old_df.mult
mult1_sig_new = j1_new_df.mult

B_old_path = "/gpfs0/kats/users/wunch/semisup_data/bkg"
B_new_path = "/gpfs0/kats/users/wunch/semisup_evs/bkg/train"
j1_old_df, j2_old_df = load_data_old(B_old_path, n_ev=10000)
j1_new_df, j2_new_df = load_data_new(B_new_path, n_ev=10000)

if pt_cut:
    PT_min = 100
    PT_max = 200
    valid_idx = (j1_new_df.jet_PT > PT_min) & (j2_new_df.jet_PT > PT_min) & (
                 j1_new_df.jet_PT < PT_max) & (j2_new_df.jet_PT < PT_max)
    j1_new_df = j1_new_df[valid_idx]
    j2_new_df = j2_new_df[valid_idx]

mult1_bkg_old = j1_old_df.mult
mult1_bkg_new = j1_new_df.mult


plt.figure()
plt.hist([mult1_sig_old, mult1_bkg_old],
         bins=np.arange(min(mult1_sig_old.min(), mult1_bkg_old.min()), max(mult1_sig_old.max(), mult1_bkg_old.max())+2),
         align='left', density=True, histtype='step')
plt.legend(['sig', 'bkg'])
plt.savefig('sig_vs_bkg_oldest.png')

plt.figure()
plt.hist([mult1_sig_new, mult1_bkg_new],
         bins=np.arange(min(mult1_sig_new.min(), mult1_bkg_new.min()), max(mult1_sig_new.max(), mult1_bkg_new.max())+2),
         align='left', density=True, histtype='step')
plt.legend(['sig', 'bkg'])
plt.savefig('sig_vs_bkg_newest.png')
