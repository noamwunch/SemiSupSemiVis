import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path
import tensorflow as tf
import os

def jet_list2jet_df(jets_list):
    jets_df = pd.DataFrame(jets_list,
                            columns=["Event", "MET", "Mjj",
                                     "Jet", "jet_PT", "jet_Eta", "jet_Phi", "dR_closest_parton",
                                     "constit_PT", "constit_Eta", "constit_Phi",
                                     "constit_type", "constit_PID", "constit_D0", "constit_DZ", "abs_D0"])

    dtypes = {"Event": np.int, "MET": np.float, "Mjj": np.float,
              "Jet": np.int, "jet_PT": np.float, "jet_Eta": np.float, "jet_Phi": np.float,
              "dR_closest_parton": np.float}

    jets_df = jets_df.astype(dtypes)

    constit_feats = ["constit_PT", "constit_Eta", "constit_Phi",
                     "constit_type", "constit_PID", "constit_D0", "constit_DZ", "abs_D0"]

    jets_df[constit_feats] = jets_df[constit_feats].applymap(lambda x: np.array(x))

    jets_df['mult'] = jets_df.constit_PT.map(lambda x: len(x))
    jets_df['constit_relPT'] = jets_df.constit_PT/jets_df.jet_PT
    jets_df['constit_relPhi'] = jets_df.constit_Phi - jets_df.jet_Phi
    jets_df.constit_relPhi = jets_df.constit_relPhi.map(lambda phi: phi - 2*np.pi * (phi>np.pi) + 2*np.pi * (phi<-np.pi))
    jets_df['constit_relEta'] = jets_df.constit_Eta - jets_df.jet_Eta
    jets_df['constit_relDZ'] = jets_df['constit_DZ'] / np.cosh(jets_df['jet_Eta'])
    jets_df['constit_deltaR'] = np.power(np.power(jets_df.constit_relPhi, 2.0)
                                         + np.power(jets_df.constit_relEta, 2.0), 0.5)

    return jets_df

def evs_txt2jets_df(events_dir_path, n_ev=int(1e6), sort="PT"):
    """Takes event list path (string) and returns a pandas Dataframe with jet info"""
    # Redefine sort variable to correspond to column index
    if sort == "PT":
        sort = 0
    elif sort == "D0":
        sort = 7
    # Initialize jet list
    jets1_list = []
    jets2_list = []
    # Initialize temp jet info
    jet1_info = []
    jet2_info = []
    jet1_constits = []
    jet2_constits = []

    # Loop over txt file paths in events_paths
    ev_num = 0
    pathlist = Path(events_dir_path).glob('**/*root.txt')
    print(list(pathlist))
    for events_path in pathlist:
        if ev_num > n_ev:
            break
        # Loop over lines in txt file
        with open(str(events_path)) as events:
            print("looping over lines")
            for line in events:
                row = line.split()

                # New event
                if row[0] == "--":
                    print(f"new event = {n_ev}")
                    # Log previuos event
                    if ev_num > 0:
                        event_info = [ev_num, met, mjj]

                        jet1_constits = np.array(jet1_constits, dtype="float")
                        if sort is not None:
                            jet1_constits = jet1_constits[jet1_constits[:, sort].argsort()[::-1]]
                        jets1_list.append(event_info + jet1_info + list(jet1_constits.T))

                        jet2_constits = np.array(jet2_constits, dtype="float")
                        if sort is not None:
                            jet2_constits = jet2_constits[jet2_constits[:, sort].argsort()[::-1]]
                        jets2_list.append(event_info + jet2_info + list(jet2_constits.T))

                    # Initialize for next event
                    jet1_constits = []
                    jet2_constits = []
                    ev_num += 1
                    if ev_num > n_ev:
                        break
                    continue

                # General event info
                if row[0] == "MET:":
                    met = row[1]
                if row[0] == "MJJ:":
                    mjj = row[1]

                # General jet info
                if (row[0] == "Jet") and (row[1] == "1"):
                    jet1_info = [row[1], row[3], row[5], row[7], row[9]]
                    continue
                if (row[0] == "Jet") and (row[1] == "2"):
                    jet2_info = [row[1], row[3], row[5], row[7], row[9]]
                    continue

                # Constituents info
                if row[0].isdigit():
                    if int(row[0]) == 1:
                        jet1_constits.append(row[1:] + [abs(float(row[7]))])
                        continue
                    if int(row[0]) == 2:
                        jet2_constits.append(row[1:] + [abs(float(row[7]))])
                        continue

    jets1_df, jets2_df = jet_list2jet_df(jets1_list), jet_list2jet_df(jets2_list)

    return jets1_df, jets2_df

def create_one_hot_encoder(class_dict):
    enc = OneHotEncoder()
    unique_class = np.unique(list(class_dict.keys()))
    enc.fit(np.array(unique_class).reshape(-1, 1))
    return enc

def nominal2onehot(j_feats, class_dict, enc):
    j_cat_feats = j_feats[:, :, -1]
    j_num_feats = np.delete(j_feats, -1, axis=2)

    j_cat_feats = np.vectorize(class_dict.get)(j_cat_feats)

    j_cat_feats = j_cat_feats.flatten().reshape(-1, 1)
    j_one_hot = enc.transform(j_cat_feats).toarray()

    masked_idx = np.argwhere(np.array(enc.categories_[0]) == 'masked')[0][0]
    j_one_hot[j_one_hot[:, masked_idx] == 1, :] = -10
    j_one_hot = np.delete(j_one_hot, masked_idx, axis=1)
    j_one_hot = j_one_hot.reshape((j_feats.shape[0], j_feats.shape[1], len(set(class_dict.values()))-1))

    return np.concatenate([j_one_hot, j_num_feats], axis=2)

def set_tensorflow_threads(n_threads=20):
    tf.config.threading.set_intra_op_parallelism_threads(n_threads)
    tf.config.threading.set_inter_op_parallelism_threads(n_threads)
    os.environ['TF_NUM_INTEROP_THREADS'] = str(n_threads)
    os.environ['TF_NUM_INTRAOP_THREADS'] = str(n_threads)
