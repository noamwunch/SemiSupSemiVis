from pathlib import Path

import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

def calc_fake_thrust(col_dict):
    deltaR = col_dict['constit_deltaR']
    PT = col_dict['constit_PT']
    jet_PT = col_dict['jet_PT']
    return np.sum(deltaR*PT)/jet_PT

def calc_disp_median(col_dict, col_name=None):
    col = col_dict[col_name]
    col = col[np.abs(col)>1e-6]
    if len(col)>0:
        return np.median(col)
    else:
        return -1

def calc_median(col_dict, col_name=None):
    col = col_dict[col_name]
    return np.median(col)


def create_dense_classifier():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(32, input_shape=(5, )))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Dense(16))
    model.add(keras.layers.ELU())
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Dense(4))
    model.add(keras.layers.ELU())

    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='Adam', loss='binary_crossentropy')
    model.summary()
    print("")
    return model

def preproc_for_dense(j_inp):
    mult = j_inp.mult
    n_verts = j_inp.n_verts
    fake_thrust = j_inp.apply(calc_fake_thrust, axis=1)
    med_d0 = j_inp.apply(calc_disp_median, col_name='constit_D0', axis=1)
    med_dz = j_inp.apply(calc_disp_median, col_name='constit_DZ', axis=1)

    mult = (mult-30) / 30
    n_verts = (n_verts-3) / 3
    med_dz = med_dz * 5
    med_d0 = med_d0 * 5

    dense_inp = np.stack((mult, n_verts, fake_thrust, med_dz, med_d0), axis=1)
    print(f'dense_inp.shape = {dense_inp.shape}')

    return dense_inp
