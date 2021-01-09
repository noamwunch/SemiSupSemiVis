from pathlib import Path
import sys

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from UTILS.utils import evs_txt2jets_df as load_data

plot_path = "RESULTS/mjj"
Path(plot_path).mkdir(parents=True, exist_ok=True)

B_path = "/gpfs0/kats/users/wunch/semisup_evs/bkg"
S_rinv0_path = "/gpfs0/kats/users/wunch/semisup_evs/sig_rinv_0.00_mjj_500"
S_rinv1_path = "/gpfs0/kats/users/wunch/semisup_evs/sig_rinv_0.25_mjj_500"
S_rinv2_path = "/gpfs0/kats/users/wunch/semisup_evs/sig_rinv_0.50_mjj_500"


if sys.argv[1]=="all":
    paths = [B_path, S_rinv0_path, S_rinv1_path, S_rinv2_path]
    names = ['Background (b b~)', 'Signal rinv=0.00', 'Signal rinv=0.25', 'Signal rinv=0.50']
    hist_args = {'bins': 100, 'density': True, 'histtype': 'step'}
    met_fig = plt.figure()
    mjj_fig = plt.figure()
    for path in paths:
        _, j1_df = load_data(B_path)
        ev_df = j1_df[["MET", "Mjj"]]

        plt.figure(mjj_fig.number)
        plt.hist(np.sqrt(ev_df.Mjj), **hist_args)

        plt.figure(met_fig.number)
        plt.hist(np.sqrt(ev_df.MET), **hist_args)

    plt.figure(mjj_fig.number)
    plt.xlabel('$M_{jj}$ [GeV]')
    plt.savefig(plot_path+"/mjj_dist.png")
    plt.legend(names)

    plt.figure(met_fig.number)
    plt.xlabel('MET [GeV]')
    plt.savefig(plot_path+"/met_dist.png")
    plt.legend(names)

######Background#######
if sys.argv[1]=="0":
    _, j1_df = load_data(B_path)
    B_ev_df = j1_df[["MET", "Mjj"]]
    print(f"Number of bkg events: {len(B_ev_df)}")

    plt.figure()
    plt.hist(np.sqrt(B_ev_df.Mjj), bins=100, density=True)
    plt.xlabel('$M_{jj}$ [GeV]')
    plt.savefig(plot_path+"/mjj_bkg_dist.png")

    plt.figure()
    plt.hist(np.sqrt(B_ev_df.MET), bins=100, density=True)
    plt.xlabel('MET [GeV]')
    plt.savefig(plot_path+"/met_bkg_dist.png")

######Signal rinv=0#######
if sys.argv[1]=="1":
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
if sys.argv[1]=="2":
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
if sys.argv[1]=="3":
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