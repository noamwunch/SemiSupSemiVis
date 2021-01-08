from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from UTILS.utils import evs_txt2jets_df as load_data

plot_path = "RESULTS/mjj"
Path(plot_path).mkdir(parents=True, exist_ok=True)

S_path = "/gpfs0/kats/users/wunch/semisup_evs/rinv_0.0_mjj_500"
_, S_j1_df = load_data(S_path)

S_ev_df = S_j1_df[["MET", "Mjj"]]

plt.figure()
plt.hist(np.sqrt(S_ev_df.Mjj), bins=100, density=True, range=[50, 2000])
plt.xlabel('$M_jj$ [GeV]')
plt.savefig(plot_path+"/mjj_signal_dist.png")
