sig_frac = 0.2
N = 80000
rinv = 0.0
sig_frac = 0.01



no cut, met cut, tnt cut
S_path =
B_path =
hist_args = {'bins': 100, 'density': True, 'histtype': 'step'}

j1_df, _, _ = combine_SB(B_path, S_path, N, sig_frac)
ev_df = j1_df[["MET", "Mjj"]]
preprocess
load model
infer data
cut on data
plot mjj + exponential fit