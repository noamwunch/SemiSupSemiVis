from semisup import combine_SB
import numpy as np
from matplotlib import pyplot as plt

B_path = "/gpfs0/kats/users/wunch/semisup_dataset/bkg_bb_GenMjjGt400_GenPtGt40_GenEtaSt3_MjjGt500_PtGt50_EtaSt2.5/test"
S_path = "/gpfs0/kats/users/wunch/semisup_dataset/sig_dl0.5_rinv0.00_mZp1500_lambda20_GenMjjGt600_GenPtGt40_GenEtaSt3_MjjGt800_PtGt50_EtaSt2.5/test"

N_bkg = 20000
N_sig = 1000
sig_region = (1200, 1500)

j1_bkg, j2_bkg, _ = combine_SB(B_path, S_path, N_bkg, 0)
j1_sig, j2_sig, _ = combine_SB(B_path, S_path, N_sig, 1)

# y*
y_bkg = np.abs((j1_bkg.jet_Eta - j2_bkg.jet_Eta)/2)
y_sig = np.abs((j1_sig.jet_Eta - j2_sig.jet_Eta)/2)

plt.figure()
plt.hist([y_sig, y_bkg], histtype='step', density=True, label=["Dark events", "QCD events"])
plt.legend()
plt.xlabel('y')
plt.savefig('y_hist_new')

# Mjj
y_bkg = j1_bkg.Mjj
y_sig = j1_sig.Mjj

N_sig_reg = np.sum(y_sig.between(*sig_region))
N_bkg_reg = np.sum(y_bkg.between(*sig_region))
sig_frac = N_sig_reg/N_bkg_reg
significance = N_sig_reg/(np.sqrt(N_sig_reg + N_bkg_reg))

bins = np.arange(1000, 3000, 40)
hist_dict = dict(histtype='step', align='mid')
txt_dict = dict(xy=(0.6, 0.7), xycoords='axes fraction')
labels = [f"{N_sig} Dark events", f"{N_bkg} QCD events"]

txt_sigfrac = f"signal-fraction: {N_sig/N_bkg:.2f}"
txt_significance = "significance: %.2f $\\sigma$" %(significance)
txt = f"In signal region (1200-1500 GeV): \n{txt_sigfrac} \n{txt_significance}"

plt.figure()
plt.hist([y_sig, y_bkg], bins=bins, label=labels, **hist_dict)
plt.annotate(txt, **txt_dict)
plt.legend()
plt.ylabel('Events/(40 GeV)')
plt.xlabel('$M_{jj}$')
plt.savefig('mjj_hist')

