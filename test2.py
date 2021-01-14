from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from UTILS.utils import evs_txt2jets_df_old as load_data_old
from UTILS.utils import evs_txt2jets_df as load_data_new

plot_path = "RESULTS/test"
Path(plot_path).mkdir(parents=True, exist_ok=True)

S_old_path = "/gpfs0/kats/users/wunch/semisup_data/sig"
S_new_path = "/gpfs0/kats/users/wunch/semisup_evs/sig_rinv_0.50_mjj_500_gen_ptcut_rem_mjjcut"

j1_old_df, j2_old_df = load_data_old(S_old_path, n_ev=10000)
j1_new_df, j2_new_df = load_data_new(S_new_path, n_ev=10000)

mult1_old = j1_old_df.mult
mult1_new = j1_new_df.mult

plt.figure()
plt.hist([mult1_old, mult1_new],
         bins=np.arange(min(mult1_old.min(), mult1_new.min()), max(mult1_old.max(), mult1_new.max())+2), align='left',
         density=True, histtype='step')
plt.legend(['old', 'new'])
plt.savefig('mult1_new_ptcut_no_mjjcut.png')
