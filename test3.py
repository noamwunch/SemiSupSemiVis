from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from UTILS.utils import evs_txt2jets_df_old as load_data_old
from UTILS.utils import evs_txt2jets_df as load_data_new

plot_path = "RESULTS/test"
Path(plot_path).mkdir(parents=True, exist_ok=True)

S_old_path = "/gpfs0/kats/users/wunch/semisup_data/sig"
S_new_path = "/gpfs0/kats/users/wunch/semisup_evs/sig_rinv_0.50_mjj_500"
j1_old_df, j2_old_df = load_data_old(S_old_path, n_ev=10000)
j1_new_df, j2_new_df = load_data_new(S_new_path, n_ev=10000)
pt1_sig_old = j1_old_df.jet_PT
pt1_sig_new = j1_new_df.jet_PT

B_old_path = "/gpfs0/kats/users/wunch/semisup_data/bkg"
B_new_path = "/gpfs0/kats/users/wunch/semisup_evs/bkg"
j1_old_df, j2_old_df = load_data_old(B_old_path, n_ev=10000)
j1_new_df, j2_new_df = load_data_new(B_new_path, n_ev=10000)
pt1_bkg_old = j1_old_df.jet_PT
pt1_bkg_new = j1_new_df.jet_PT

plt.figure()
plt.hist([pt1_sig_old, pt1_bkg_old],
         bins=100,
         align='left', density=True, histtype='step')
plt.legend(['sig', 'bkg'])
plt.savefig('sig_vs_bkg_old_PT.png')

plt.figure()
plt.hist([pt1_sig_new, pt1_bkg_new],
         bins=100, range=[0, 400],
         align='left', density=True, histtype='step')
plt.legend(['sig', 'bkg'])
plt.savefig('sig_vs_bkg_new_prev_PT.png')
