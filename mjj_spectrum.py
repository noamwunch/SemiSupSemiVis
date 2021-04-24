from semisup import combine_SB
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

B_path = "/gpfs0/kats/users/wunch/semisup_dataset/bkg_bb_GenMjjGt800_GenPtGt40_GenEtaSt3_MjjGt1000_PtGt50_EtaSt2.5_y*lt1"
S_path = "/gpfs0/kats/users/wunch/semisup_dataset/sig_dl0.5_rinv0.00_mZp1500_lambda20_GenMjjGt800_GenPtGt40_GenEtaSt3_MjjGt1000_PtGt50_EtaSt2.5_y*lt1"

N_bkg = 50000
N_sig = 2000
tot_region = (1000, 3000)
sig_region = (1200, 1500)

j1_bkg, j2_bkg, _ = combine_SB(B_path, S_path, N_bkg, 0)
j1_sig, j2_sig, _ = combine_SB(B_path, S_path, N_sig, 1)

y_bkg = j1_bkg.Mjj
y_sig = j1_sig.Mjj
y_both = pd.concat([j1_bkg.Mjj, j1_sig.Mjj])
bin_size = 40  # GeV

N_sig_tot = np.sum(y_sig.between(*tot_region))
N_bkg_tot = np.sum(y_bkg.between(*tot_region))
N_sig_reg = np.sum(y_sig.between(*sig_region))
N_bkg_reg = np.sum(y_bkg.between(*sig_region))
sig_frac = N_sig_reg/N_bkg_reg
significance = N_sig_reg/(np.sqrt(N_sig_reg + N_bkg_reg))

bins = np.arange(tot_region[0], tot_region[1], bin_size)
hist_dict = dict(histtype='step', align='mid')
legend_dict = dict(loc='lower right', framealpha=0.0)
labels = ["Dark events", "QCD events", "QCD + Dark events"]

# Annotation
txt_Nreg = f"(QCD events, Dark events): ({N_bkg_reg}, {N_sig_reg})"
txt_sigfrac = f"signal-fraction: {sig_frac:.2f}"
txt_significance = "significance: %.2f $\\sigma$" %(significance)
txt_reg = f"In signal region (1200-1500 GeV): \n{txt_Nreg} \n{txt_sigfrac} \n{txt_significance}"
annot_reg_dict = dict(xy=(0.4, 0.8), xycoords='axes fraction')

txt_Ntot = f"(QCD events, Dark events): ({N_bkg_tot}, {N_sig_tot})"
txt_tot = f"In entire region (1000-3000 GeV): \n{txt_Ntot}"
annot_tot_dict = dict(xy=(0.4, 0.7), xycoords='axes fraction')

# plot
plt.figure()
plt.hist([y_sig, y_bkg, y_both], bins=bins, label=labels, **hist_dict)
plt.annotate(txt_reg, **annot_reg_dict)
plt.annotate(txt_tot, **annot_tot_dict)
plt.legend(**legend_dict)
plt.ylabel('Events/(40 GeV)')
plt.xlabel('$M_{jj}/GeV$')
plt.savefig('mjj_hist')

# plot log
plt.figure()
plt.yscale('log')
plt.hist([y_sig, y_bkg, y_both], bins=bins, label=labels, **hist_dict)
plt.annotate(txt_reg, **annot_reg_dict)
plt.annotate(txt_tot, **annot_tot_dict)
plt.legend(**legend_dict)
plt.ylabel('Events/(40 GeV)')
plt.xlabel('$M_{jj}/GeV$')
plt.savefig('mjj_hist_log')

