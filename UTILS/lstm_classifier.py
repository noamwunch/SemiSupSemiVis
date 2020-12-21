import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

def create_lstm_classifier(n_constits, n_cols, reg_dict, mask_val, log):
    model = keras.models.Sequential()
    model.add(keras.layers.Masking(mask_value=mask_val, input_shape=(n_constits, n_cols)))
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

def preproc_for_lstm(j_df, feats, mask, n_constits):
    j_df.constit_relPT *= 10
    j_df.constit_relEta *= 5
    j_df.constit_relPhi *= 5
    if ('constit_D0' in feats) and ('constit_relDZ' in feats):
        pass  # Room for scaling displacement features
    j_semisup_inp = np.array([np.vstack(j_df[feat].apply(mask_list, args=(mask, n_constits))) for feat in feats]
                             ).transpose((1, 2, 0))
    return j_semisup_inp

def train_classifier(X, y, model, model_save_path, epochs, log):
    # Train test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    plt.figure()
    plt.hist(X_train[:, 0, 0], label='track 1', bins=100, histtype='step', range=[0, 10])
    plt.hist(X_train[:, 1, 0], label='track 2', bins=100, histtype='step', range=[0, 10])
    plt.hist(X_train[:, 4, 0], label='track 5', bins=100, histtype='step', range=[0, 10])
    plt.hist(X_train[:, 9, 0], label='track 10', bins=100, histtype='step', range=[0, 10])
    plt.legend(loc='best')
    plt.savefig(model_save_path + 'PT2')

    # Callbacks
    checkpoint = keras.callbacks.ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True)
    es = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.001)
    rlop = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_lr=0.0001)
    # Train model
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        batch_size=512, epochs=epochs, callbacks=[checkpoint])
    # Log
    log = log + f'X_train shape = {X_train.shape}\n'
    log = log + f'y_train shape = {y_train.shape}\n'
    log = log + f'X_val shape = {X_val.shape}\n'
    log = log + f'y_val shape = {y_val.shape}\n'
    log = log + '\n'

    return history, log
