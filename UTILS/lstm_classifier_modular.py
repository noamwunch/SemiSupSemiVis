from pathlib import Path

import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from UTILS.utils import create_one_hot_encoder, nominal2onehot

def create_lstm_classifier(n_constits=80, n_cols=5, reg_dict=None, mask_val=-10.0, log=''):
    if reg_dict is None:
        reg_dict = {}
    model = keras.models.Sequential()
    model.add(keras.layers.Masking(mask_value=mask_val, input_shape=(n_constits, n_cols)))
    print(f"input_shape = {(n_constits, n_cols)}")
    model.add(keras.layers.LSTM(50, return_sequences=False, **reg_dict))
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='Adam', loss='binary_crossentropy')

    model.summary()
    print("")

    summary_str_list = []
    model.summary(print_fn=lambda x: summary_str_list.append(x))
    log = log +  '\n'.join(summary_str_list) + '\n\n'

    return model, log

def mask_list(x, mask, n_constits):
    return np.append(x[:n_constits], [mask] * (n_constits - len(x)))

def preproc_for_lstm(j_df, feats, n_constits, mask):
    # Scale
    j_df.constit_relPT *= 10
    j_df.constit_relEta *= 5
    j_df.constit_relPhi *= 5
    if ('constit_D0' in feats) and ('constit_relDZ' in feats):
        pass  # Room for scaling displacement features

    # Mask and transform to np array
    j_inp = np.array([np.vstack(j_df[feat].apply(mask_list, args=(mask, n_constits))) for feat in feats]
                     ).transpose((1, 2, 0))

    if 'constit_PID' in feats:
        pid = [mask, -2212, -321, -211, -13, -11, 0, 1, 11, 13, 211, 321, 2212]
        classification = ['masked', 'h-', 'h-', 'h-', 'mu-', 'e-', 'photon', 'h0', 'e+', 'mu+', 'h+', 'h+', 'h+']
        class_dict = dict(zip(pid, classification))
        enc = create_one_hot_encoder(class_dict)
        j_inp = nominal2onehot(j_inp, class_dict, enc)
    return j_inp

def train_classifier(X, y, model, model_save_path, epochs, log=''):
    Path(model_save_path).mkdir(parents=True, exist_ok=True)
    # Train test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    # plt.figure()
    # plt.hist(X_train[:, 0, 0], label='track 1', bins=100, histtype='step', range=[0, 10])
    # plt.hist(X_train[:, 1, 0], label='track 2', bins=100, histtype='step', range=[0, 10])
    # plt.hist(X_train[:, 4, 0], label='track 5', bins=100, histtype='step', range=[0, 10])
    # plt.hist(X_train[:, 9, 0], label='track 10', bins=100, histtype='step', range=[0, 10])
    # plt.legend(loc='best')
    # plt.savefig(model_save_path + 'PT2')

    # Callbacks
    checkpoint = keras.callbacks.ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True)
    es = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.001)
    rlop = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_lr=0.0001)
    # Train model
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        batch_size=8192, epochs=epochs, callbacks=[checkpoint])
    # Log
    log = log + f'X_train shape = {X_train.shape}\n'
    log = log + f'y_train shape = {y_train.shape}\n'
    log = log + f'X_val shape = {X_val.shape}\n'
    log = log + f'y_val shape = {y_val.shape}\n'
    log = log + '\n'

    return history, log

def plot_event_histograms_lstm(j1_df, j2_df, event_label, save_dir):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    track_nums = [1, 2, 5, 10]
    plot_dict = {'constit_D0': {'range': [-2, 2], 'xlabel': 'D0 [mm]'},
                 'constit_DZ': {'range': [-2, 2], 'xlabel': 'Dz [mm]'},
                 'constit_PT': {'range': [0, 100], 'xlabel': 'PT [GeV]'},
                 'constit_Eta': {'range': None, 'xlabel': 'Eta'},
                 'constit_Phi': {'range': None, 'xlabel': 'Phi [rad]'},
                 'constit_relDZ': {'range': [-2, 2], 'xlabel': 'relDz [mm]'},
                 'constit_relPT': {'range': [0, 1], 'xlabel': 'relPT [GeV]'},
                 'constit_relEta': {'range': [-1, 1], 'xlabel': 'relEta'},
                 'constit_relPhi': {'range': [-1, 1], 'xlabel': 'relPhi [rad]'},
                 'constit_deltaR': {'range': None, 'xlabel': 'deltaR'}
                 }
    for feat in plot_dict.keys():
        save_path = save_dir + feat
        fig, axes = plt.subplots(nrows=1, ncols=len(track_nums), figsize=(10, 10), sharex='row', sharey='row')
        for ax, track_num in zip(axes, track_nums):
            j1_df[(~event_label.astype(bool)) & (j1_df.mult >= track_num)][feat].map(lambda x: x[track_num - 1]).hist(
                ax=ax, label='j1 bkg', range=plot_dict[feat]['range'], density=True,
                histtype='step', bins=100, color='black')
            j1_df[(event_label.astype(bool)) & (j1_df.mult >= track_num)][feat].map(lambda x: x[track_num - 1]).hist(
                ax=ax, label='j1 sig', range=plot_dict[feat]['range'], density=True,
                histtype='step', bins=100, color='red')

            j2_df[(~event_label.astype(bool)) & (j2_df.mult >= track_num)][feat].map(lambda x: x[track_num - 1]).hist(
                ax=ax, label='j2 bkg', range=plot_dict[feat]['range'], density=True,
                histtype='step', bins=100, color='green')
            j2_df[(event_label.astype(bool)) & (j2_df.mult >= track_num)][feat].map(lambda x: x[track_num - 1]).hist(
                ax=ax, label='j2 sig', range=plot_dict[feat]['range'], density=True,
                histtype='step', bins=100, color='blue')

            ax.set_title(f'track #{track_num}')
            ax.legend(loc='best')
            ax.set_yticks([])
            ax.set_xlabel(plot_dict[feat]['xlabel'])
        fig.tight_layout()
        plt.gcf().set_size_inches(15, 10)
        plt.savefig(save_path)

    plt.close('all')

def plot_nn_inp_histograms_lstm(j_inp, plot_save_dir, preproc_args=None):
    plt.figure()
    plt.hist(j_inp[:, 0, 0], label='track 1', bins=100, histtype='step', range=[0, 10])
    plt.hist(j_inp[:, 1, 0], label='track 2', bins=100, histtype='step', range=[0, 10])
    plt.hist(j_inp[:, 4, 0], label='track 5', bins=100, histtype='step', range=[0, 10])
    plt.hist(j_inp[:, 9, 0], label='track 10', bins=100, histtype='step', range=[0, 10])
    plt.legend(loc='best')
    plt.xlabel('relPT')
    plt.savefig(plot_save_dir + 'PT')

    plt.close()
