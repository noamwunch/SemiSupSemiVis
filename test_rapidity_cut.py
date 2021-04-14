from semisup import combine_SB
import numpy as np
from matplotlib import pyplot as plt

B_path = "/gpfs0/kats/users/wunch/semisup_dataset/bkg_bb_GenMjjGt400_GenPtGt40_GenEtaSt3_MjjGt500_PtGt50_EtaSt2.5/test"
S_path = "/gpfs0/kats/users/wunch/semisup_dataset/sig_dl0.5_rinv0.20_mZp1250_lambda20_GenMjjGt400_GenPtGt40_GenEtaSt3_MjjGt500_PtGt50_EtaSt2.5/test"
N = 50000

j1_bkg, j2_bkg, _ = combine_SB(B_path, S_path, N, 0)
j1_sig, j2_sig, _ = combine_SB(B_path, S_path, N, 1)

y_sig = np.abs((j1_bkg.jet_Eta - j2_bkg.jet_Eta)/2)
y_bkg = np.abs((j1_sig.jet_Eta - j2_sig.jet_Eta)/2)

plt.hist([y_sig, y_bkg], histtype='step', density=True, label=["Dark events", "QCD events"])
plt.legend()
plt.xlabel('y')
plt.savefig('y_hist')
