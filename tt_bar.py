import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

met_t = pd.read_pickle("met_t_new")
met_b = pd.read_pickle("met_b")
nb = len(met_b)
nt = len(met_t)
print(f'nb = {nb}')
print(f'nt = {nt}')

thresh0 = 300
xs_b = 268
xs_t = 0.57

N = 10000
thresh_vec = np.linspace(0, 350, N)
bbmeteff = []
ttmeteff = []
for thresh in thresh_vec:
    beff = np.sum(met_b>thresh)/len(met_b)
    teff = np.sum(met_t>thresh)/len(met_t)
    bbmeteff.append(beff)
    ttmeteff.append(teff)

plt.figure()
plt.plot(thresh_vec, xs_b*np.array(bbmeteff), label='$b\\bar{b}$')
plt.plot(thresh_vec, xs_t*np.array(ttmeteff), label='$t\\bar{t}$')
plt.xlabel('MET threshold [GeV]')
plt.ylabel('cross section * met cut efficiency')
plt.xlim([0, 350])
plt.yscale('log')
plt.legend()
plt.grid()

ttmeteff280 = ttmeteff[np.argmin(np.abs(thresh_vec-thresh0))]
bbmeteff280 = bbmeteff[np.argmin(np.abs(thresh_vec-thresh0))]
bkgmeteff280 = (xs_b*bbmeteff280 + xs_t*ttmeteff280) / (xs_b + xs_t)
print(f'ttmeteff @280 GeV MET cut = {ttmeteff280}')
print(f'bbmeteff @280 GeV MET cut = {bbmeteff280}')
print(f'bkgmeteff @280 GeV MET cut = {bkgmeteff280}')

print(ttmeteff280/bbmeteff280)
print(xs_b/xs_t)

plt.show()
