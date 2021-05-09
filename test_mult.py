import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve

from semisup import combine_SB

exp_dir_path = "/gpfs0/kats/users/wunch/SemiSupSemiVis/test_semisup_cwola/"

B_path = "/gpfs0/kats/users/wunch/semisup_dataset/bkg_bb_GenMjjGt800_GenPtGt40_GenEtaSt3_MjjGt1000_PtGt50_EtaSt2.5_y*lt1/train"
S_path = "/gpfs0/kats/users/wunch/semisup_dataset/sig_dl0.5_rinv0.00_mZp1500_lambda20_GenMjjGt800_GenPtGt40_GenEtaSt3_MjjGt1000_PtGt50_EtaSt2.5_y*lt1/train"

N = 2e4

print('Loading test data...')
j1test_df, j2test_df, event_labels_test = combine_SB(B_path, S_path, N, 0.5)
print('Test data loaded')

print('Infer jet 1 of test set')
j1_multpreds = j1test_df.mult
print('Finished inferring jet 1 of test set')

print('Infer jet 2 of test set')
j2_multpreds = j2test_df.mult
print('Finished inferring jet 1 of test set')

print("ploting ROCs")
multpreds = j1_multpreds + j2_multpreds
bkg_eff, sig_eff, thresh = roc_curve(event_labels_test, multpreds)

plt.figure()
plt.plot(thresh, sig_eff/bkg_eff)
plt.xlabel('Multiplicity threshold')
plt.ylabel(r'$\epsilon_{S} / \\epsilon_{B}$')
plt.savefig('test_mult2.png')
plt.close()
