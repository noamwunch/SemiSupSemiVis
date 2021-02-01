from pathlib import Path
import sys

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from tensorflow import keras

from UTILS.utils import evs_txt2jets_df as load_data
from semisup import combine_SB, determine_feats
from UTILS.lstm_classifier import preproc_for_lstm
from UTILS.plots_and_logs import plot_mult

plot_path = "RESULTS/mjj_30_01_21"
fig_format = 'png'
Path(plot_path).mkdir(parents=True, exist_ok=True)

B_path = "/gpfs0/kats/users/wunch/semisup_evs/bkg_bb_GenMjjGt400_GenPtGt40_GenEtaSt3_MjjGt500_PtGt50_EtaSt2.5/test"
S_path = "/gpfs0/kats/users/wunch/semisup_evs/sig_rinv0.25_mZp1000_GenMjjGt400_GenPtGt40_GenEtaSt3_MjjGt500_PtGt50_EtaSt2.5/test"
Ntest = 100000

distibutions = False

# Bumphunt settings
bumphunt = True
mask = -10.0
n_constits = 80
feats, n_cols = determine_feats(with_displacement='True',
                                with_deltar='True',
                                with_pid='False')
dat_eff_metcut = 1e-2
dat_eff_nncut = 1e-2

sig_frac = 0.05
model1_path = "RESULTS/final_grid1/rinv0.25sf0.05_newest/j1_0"
model2_path = "RESULTS/final_grid1/rinv0.25sf0.05_newest/j2_0"

#### S/B comparison plots ####
if distibutions:
    hist_dict = {'histtype': 'step', }

    print('loading data for S/B comparison')
    bkg1, bkg2, _ = combine_SB(B_path, S_path, Ntest, 0)
    print('loaded signal')
    sig1, sig2, _ = combine_SB(B_path, S_path, Ntest, 1)
    print('loaded background')
    print(f'loaded data for S/B comparison: (bkg_evs, sig_evs) = {(len(bkg1), len(sig1))}\n')

    met_bkg, met_sig = bkg1.MET, sig1.MET
    mjj_bkg, mjj_sig = bkg1.Mjj, sig1.Mjj
    mult_bkg1, mult_sig1 = bkg1.mult, sig1.mult
    mult_bkg2, mult_sig2 = bkg2.mult, sig2.mult
    drpart_sig1, drpart_sig2 = sig1.dR_closest_parton, sig2.dR_closest_parton

    print('Plotting MET and Mjj and Mult')
    # MET
    plt.figure()
    plt.hist([met_bkg, met_sig], label=['$b\\bar{b}$', "$x\\bar{x}$ ($r_{inv}=0.25$, $m_{Z'}=1$ TeV)"], bins=np.arange(0, 400, 5), **hist_dict)
    plt.xlim([0, 400-5])
    plt.ylim([0, None])
    plt.xlabel('MET/GeV')
    plt.ylabel('events/(5 GeV)')
    plt.legend()
    plt.savefig(plot_path + '/met', format=fig_format, dpi=1000)

    # MJJ
    plt.figure()
    plt.hist([mjj_bkg, mjj_sig], label=['$b\\bar{b}$', "$x\\bar{x}$ ($r_{inv}=0.25$, $m_{Z'}=1$ TeV)"], bins=np.arange(500, 1500, 25), **hist_dict)
    plt.yscale('log')
    plt.xlim([500, 1500-25])
    plt.xlabel('$M_{jj}/GeV$')
    plt.ylabel('events/(25 GeV)')
    plt.legend()
    plt.savefig(plot_path + '/mjj', format=fig_format, dpi=1000)

    # Mult
    plot_mult(mult_bkg1, mult_sig1, mult_bkg2, mult_sig2, save_path=plot_path+'/mult.'+fig_format)
    print('Finished plotting MET, MJJ, and Mult\n')

    # Distance to closest partons
    dark_frac1 = sum(drpart_sig1<0.3)/len(drpart_sig1)
    dark_frac2 = sum(drpart_sig2<0.3)/len(drpart_sig2)
    print(f'Percentage of leading jets in dark events originating from dark parton:'
          f'\n leading jet: {dark_frac1*100:.2f}%'
          f'\n next-to-leading jet: {dark_frac2*100:.2f}%')

#### Bump hunt ####
if bumphunt:
    hist_dict = {'histtype': 'step', 'bins': np.arange(500, 1500, 25)}

    print('Beginning bump hunt...')
    j1, j2, label = combine_SB(B_path, S_path, Ntest, sig_frac)
    bkg_mask, sig_mask = ~label.astype(bool), label.astype(bool)
    mjj = j1.Mjj
    met = j1.MET
    mult = (j1.mult + j2.mult)/2

    # No cut
    plt.figure()
    plt.hist(mjj, label=f'signal fraction: {sig_frac}', **hist_dict)
    plt.yscale('log')
    plt.xlim([500, 1500-25])
    plt.xlabel('$M_{jj}/GeV$')
    plt.ylabel('events/(25 GeV)')
    plt.legend()
    plt.savefig(plot_path + f'/mjj_sf{sig_frac}.png')

    # Multiplicity cut
    thresh = np.quantile(mult, 1-dat_eff_metcut)
    valid = mult>thresh

    sigeff = np.sum(valid & sig_mask)/np.sum(sig_mask)
    bkgeff = np.sum(valid & bkg_mask)/np.sum(bkg_mask)
    sig_frac_post = np.sum(valid & sig_mask)/np.sum(valid)
    Npost = np.sum(valid)
    txt = f'Signal fraction (before cut, after cut):\n({sig_frac}, {sig_frac_post:.3f})' \
          f'\n\nTotal events (before cut, after cut):\n({Ntest}, {Npost})' \
          f'\n\nSignal efficiency of cut:\n{sigeff:.2f}' \
          f'\n\nBackground efficiency of cut:\n{bkgeff:.2e}'

    plt.figure()
    plt.hist([mjj, mjj.loc[valid]], label=['before multiplicity cut', 'after multiplicity cut'], **hist_dict)
    plt.yscale('log')
    plt.xlim([500, 1500-25])
    plt.xlabel('$M_{jj}/GeV$')
    plt.ylabel('events/(25 GeV)')
    plt.legend(loc='lower left')

    props = dict(facecolor='wheat', alpha=0.5)
    plt.text(0.55, 0.95, txt, transform=plt.gca().transAxes, fontsize=8,
             verticalalignment='top', bbox=props
             )

    plt.savefig(plot_path + f'/mjj_sf{sig_frac}_multcut.png')

    # MET cut
    thresh = np.quantile(met, 1-dat_eff_metcut)
    valid = met>thresh

    sigeff = np.sum(valid & sig_mask)/np.sum(sig_mask)
    bkgeff = np.sum(valid & bkg_mask)/np.sum(bkg_mask)
    sig_frac_post = np.sum(valid & sig_mask)/np.sum(valid)
    Npost = np.sum(valid)
    txt = f'Signal fraction (before cut, after cut):\n({sig_frac}, {sig_frac_post:.3f})' \
          f'\n\nTotal events (before cut, after cut):\n({Ntest}, {Npost})' \
          f'\n\nSignal efficiency of cut:\n{sigeff:.2f}' \
          f'\n\nBackground efficiency of cut:\n{bkgeff:.2e}'

    plt.figure()
    plt.hist([mjj, mjj.loc[valid]], label=['before met cut', 'after met cut'], **hist_dict)
    plt.yscale('log')
    plt.xlim([500, 1500-25])
    plt.xlabel('$M_{jj}/GeV$')
    plt.ylabel('events/(25 GeV)')
    plt.legend(loc='lower left')

    props = dict(facecolor='wheat', alpha=0.5)
    plt.text(0.55, 0.95, txt, transform=plt.gca().transAxes, fontsize=8,
             verticalalignment='top', bbox=props
             )

    plt.savefig(plot_path + f'/mjj_sf{sig_frac}_metcut.png')

    # NN cut
    print('Inferring jets...')
    model1 = keras.models.load_model(model1_path)
    model2 = keras.models.load_model(model2_path)

    inp1 = preproc_for_lstm(j1.copy(deep=True), feats, mask, n_constits)
    inp2 = preproc_for_lstm(j2.copy(deep=True), feats, mask, n_constits)

    pred1 = model1.predict(inp1, batch_size=512, verbose=1).flatten()
    pred2 = model2.predict(inp2, batch_size=512, verbose=1).flatten()
    pred = (pred1 + pred2)/2
    print('Inferred jets\n')

    thresh = np.quantile(pred, 1-dat_eff_nncut)
    valid = pred>thresh

    sigeff = np.sum(valid & sig_mask)/np.sum(sig_mask)
    bkgeff = np.sum(valid & bkg_mask)/np.sum(bkg_mask)
    sig_frac_post = np.sum(valid & sig_mask)/np.sum(valid)
    Npost = np.sum(valid)
    txt = f'Signal fraction (before cut, after cut):\n({sig_frac}, {sig_frac_post:.3f})' \
            f'\n\nTotal events (before cut, after cut):\n({Ntest}, {Npost})'\
            f'\n\nSignal efficiency of cut:\n{sigeff:.2f}' \
            f'\n\nBackground efficiency of cut:\n{bkgeff:.2e}'

    plt.figure()
    plt.hist([mjj, mjj.loc[valid]], label=['before nn cut', 'after nn cut'], **hist_dict)
    plt.yscale('log')
    plt.xlim([500, 1500-25])
    plt.xlabel('$M_{jj}/GeV$')
    plt.ylabel('events/(25 GeV)')
    plt.legend(loc='lower left')

    props = dict(facecolor='wheat', alpha=0.5)
    plt.text(0.55, 0.95, txt, transform=plt.gca().transAxes, fontsize=8,
             verticalalignment='top', bbox=props
             )

    plt.savefig(plot_path + f'/mjj_sf{sig_frac}_nncut.png')

print('Done!')
