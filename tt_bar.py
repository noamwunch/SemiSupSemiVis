import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

met_t = pd.read_pickle("met_t")
met_b = pd.read_pickle("met_b")
nb = len(met_b)
nt = len(met_t)
print(f'nb = {nb}')
print(f'nt = {nt}')

sig_b = 268
sig_t = 0.29

nt = int(nb*(sig_t/sig_b))
met_t = met_t[:nt]
print(f'nt_new = {nt}')
print(f'len(met_t) = {len(met_t)}')
print(f'b/t = {nb/nt:.3f}')

N = 1000
thresh_vec = np.linspace(0, 350, N)
ratio_vec = []
for thresh in thresh_vec:
    t_passed = np.sum(met_t>thresh)
    b_passed = np.sum(met_b>thresh)
    ratio_vec.append(b_passed/t_passed)

plt.figure()
plt.plot(thresh_vec, ratio_vec)
plt.xlabel('MET threshold [GeV]')
plt.ylabel('Ratio bb~/tt~')
plt.xlim(left=0)
plt.grid()
plt.show()
