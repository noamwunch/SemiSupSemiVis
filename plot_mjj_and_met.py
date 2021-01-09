from pathlib import Path
import sys

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from UTILS.utils import evs_txt2jets_df as load_data

plot_path = "RESULTS/mjj"
Path(plot_path).mkdir(parents=True, exist_ok=True)

B_path = "/gpfs0/kats/users/wunch/semisup_evs/bkg"
S_rinv0_path = "/gpfs0/kats/users/wunch/semisup_evs/"
S_rinv1_path = "/gpfs0/kats/users/wunch/semisup_evs/bkg"
S_rinv2_path = "/gpfs0/kats/users/wunch/semisup_evs/bkg"

######Background#######
if sys.argv[0]=="0":
    _, j1_df = load_data(B_path)
    B_ev_df = j1_df[["MET", "Mjj"]]
    print(f"Number of bkg events: {len(B_ev_df)}")

    plt.figure()
    plt.hist(np.sqrt(B_ev_df.Mjj), bins=100, density=True)  # , range=[50, 1500])
    plt.xlabel('$M_{jj}$ [GeV]')
    plt.savefig(plot_path+"/mjj_bkg_dist.png")
    plt.text(0.9, 0.9, f'N = {len(B_ev_df)}',
             horizontalalignment='center',
             verticalalignment='center',
             transform=plt.gca().transAxes)

    plt.figure()
    plt.hist(np.sqrt(B_ev_df.MET), bins=100, density=True)
    plt.xlabel('MET [GeV]')
    plt.savefig(plot_path+"/met_bkg_dist.png")

######Signal rinv=0#######
if sys.argv[0]=="1":
    _, j1_df = load_data(S_rinv0_path)
    S0_ev_df = j1_df[["MET", "Mjj"]]
    print(f"Number of sig rinv=0 events: {len(S0_ev_df)}")

    plt.figure()
    plt.hist(np.sqrt(S0_ev_df.Mjj), bins=100, density=True)  # , range=[50, 1500])
    plt.xlabel('$M_{jj}$ [GeV]')
    plt.savefig(plot_path+"/mjj_sig0_dist.png")

    plt.figure()
    plt.hist(np.sqrt(S0_ev_df.MET), bins=100, density=True)
    plt.xlabel('MET [GeV]')
    plt.savefig(plot_path+"/met_sig0_dist.png")

######Signal rinv=0.2#######
if sys.argv[0]=="2":
    _, j1_df = load_data(S_rinv1_path)
    S1_ev_df = j1_df[["MET", "Mjj"]]
    print(f"Number of sig rinv=0.2 events: {len(S1_ev_df)}")

    plt.figure()
    plt.hist(np.sqrt(S1_ev_df.Mjj), bins=100, density=True)  # , range=[50, 1500])
    plt.xlabel('$M_{jj}$ [GeV]')
    plt.savefig(plot_path+"/mjj_sig1_dist.png")

    plt.figure()
    plt.hist(np.sqrt(S1_ev_df.MET), bins=100, density=True)
    plt.xlabel('MET [GeV]')
    plt.savefig(plot_path+"/met_sig1_dist.png")

######Signal rinv=0.5#######
if sys.argv[0]=="3":
    _, j1_df = load_data(S_rinv2_path)
    S2_ev_df = j1_df[["MET", "Mjj"]]
    print(f"Number of sig rinv=0.5 events: {len(S2_ev_df)}")

    plt.figure()
    plt.hist(np.sqrt(S2_ev_df.Mjj), bins=100, density=True)  # , range=[50, 1500])
    plt.xlabel('$M_{jj}$ [GeV]')
    plt.savefig(plot_path+"/mjj_sig2_dist.png")

    plt.figure()
    plt.hist(np.sqrt(S2_ev_df.MET), bins=100, density=True)
    plt.xlabel('MET [GeV]')
    plt.savefig(plot_path+"/met_sig2_dist.png")