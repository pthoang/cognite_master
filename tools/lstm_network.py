import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import tools.constants as constants
from sklearn import linear_model, preprocessing
from tensorflow import keras
import matplotlib.pyplot as plt
import tools.plotting as plotting
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import numpy as np

def predict(X, Y, test_size, look_back=1):
    min_max_scaler_X = preprocessing.MinMaxScaler()
    min_max_scaler_Y = preprocessing.MinMaxScaler()
    X_scaled = min_max_scaler_X.fit_transform(X)
    min_max_scaler_Y.fit(Y.values.reshape(-1,1))
    Y_scaled = min_max_scaler_Y.transform(Y.values.reshape(-1,1))


    X_train = X_scaled[:-test_size]
    X_test = X_scaled[-test_size:]
    Y_train = Y_scaled[:-test_size]
    Y_test = Y_scaled[-test_size:]

    model = keras.Sequential()

    # model.add(keras.layers.Dense(37, input_dim=len(X.columns)))
    model.add(keras.layers.LSTM(37, return_sequences=True, stateful=True, batch_input_shape=(1,None, len(X.columns))))
    model.add(keras.layers.LSTM(37, return_sequences=True, stateful=True))
    model.add(keras.layers.LSTM(37, return_sequences=True, stateful=True))
    model.add(keras.layers.Dense(1, activation='linear'))

    model.compile(
        optimizer=tf.train.RMSPropOptimizer(0.001),
        loss='mae',
        # metrics=['mae']
    )
    print(X_train)
    print(Y_train)

    print(model.summary())

    # history = model.fit(X_train.reshape(1,len(X_train), len(X.columns)), Y_train.reshape(1,len(Y_train),1),
    #           epochs=100, batch_size=1)

    for i in range(100):
        model.fit(X_train.reshape(1, len(X_train), len(X.columns)), Y_train.reshape(1, len(Y_train), 1),
                  epochs=1, batch_size=1, shuffle=False)

        model.reset_states()

    model.fit(X_train.reshape(1, len(X_train), len(X.columns)), Y_train.reshape(1, len(Y_train), 1),
              epochs=1, batch_size=1, shuffle=False)

    Y_pred = model.predict_on_batch(X_test.reshape(1, len(X_test), len(X.columns)))


    return min_max_scaler_Y.inverse_transform(Y_test).reshape(-1), \
           min_max_scaler_Y.inverse_transform(Y_pred[0]).reshape(-1)

def split_train_batches(X, y, num_batches, batch_len):

    samples = []
    labels = []

    for i in range(num_batches):
        samples.append(X[i*batch_len: (i+1)*batch_len])
        labels.append(y[i*batch_len: (i+1)*batch_len])


    samples[-1] = np.append(samples[-1],np.zeros((batch_len - len(samples[-1]), X.shape[1])),axis=0)
    labels[-1] = np.append(labels[-1],np.zeros((batch_len - len(labels[-1]), y.shape[1])),axis=0)

    return np.array(samples), np.array(labels)

def build_prediction_model(train_model, n_features, hidden_size):
    model = keras.Sequential()

    model.add(keras.layers.Masking(mask_value=.0, batch_input_shape=(1, None, n_features)))
    model.add(keras.layers.LSTM(hidden_size, return_sequences=True, stateful=True))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.LSTM(hidden_size, return_sequences=True, stateful=True))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(1, activation='linear'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0005),
        loss='mse',
        # metrics=['mae']
    )
    model.set_weights(train_model.get_weights())


    return model


def run_lstm(X, y, X_test, y_test, X_val, y_val, dates_test, x_values, y_value, scaler, old_model, save=False):

    min_max_scaler_Y = preprocessing.MinMaxScaler()

    min_max_scaler_Y.partial_fit(y.values.reshape(-1,1))
    min_max_scaler_Y.partial_fit(y_test.values.reshape(-1,1))
    min_max_scaler_Y.partial_fit(y_val.values.reshape(-1,1))

    y = min_max_scaler_Y.transform(y.values.reshape(-1,1))
    y_val = min_max_scaler_Y.transform(y_val.values.reshape(-1,1))

    num_batches = 24
    batch_len = len(X)//num_batches

    X_split_batch, y_split_batch = split_train_batches(X, y, num_batches, batch_len)
    hidden_size = 29
    if not old_model:
        model = keras.Sequential()

        model.add(keras.layers.Masking(mask_value=.0, batch_input_shape=(None, None, len(X_test.columns))))
        model.add(keras.layers.LSTM(hidden_size, return_sequences=True))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.LSTM(hidden_size, return_sequences=True))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(1, activation='linear'))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.0005),
            loss='mse',
            # metrics=['mae']
        )

        print(model.summary())

        print(X_split_batch.shape)

        history = model.fit(
            # X.reshape(1, X.shape[0], X.shape[1]), y.reshape(1, len(y), 1),
            X_split_batch, y_split_batch,
            epochs=80, batch_size=num_batches, shuffle=False,
            validation_data=(X_val.reshape(1, X_val.shape[0], X_val.shape[1]), y_val.reshape(1, len(y_val), 1))
        )

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])

        if save:
            path = 'models/'+'_'.join(y_value.split(' ')) + '.h5'
            model.save(path)

    else:
        model = keras.models.load_model(old_model)



    pred_model = build_prediction_model(model, len(X_test.columns), hidden_size)

    X_test = scaler.transform(X_test)
    pre_num = 3600
    X_test_pre = X_test[:pre_num]
    X_test = X_test[pre_num:]
    y_test_pre = min_max_scaler_Y.transform(y_test.iloc[:pre_num].values.reshape(-1,1))
    y_test = y_test.iloc[pre_num:]
    dates_test = dates_test[pre_num:]

    pred_model.fit(
        X_test_pre.reshape(1, X_test_pre.shape[0], X_test_pre.shape[1]), y_test_pre.reshape(1, len(y_test_pre), 1),
        epochs=1, batch_size=1, shuffle=False,
    )

    y_pred = pred_model.predict(X_test.reshape(1, X_test.shape[0], X_test.shape[1]))

    y_pred = min_max_scaler_Y.inverse_transform(y_pred[0]).reshape(-1)
    plotting.plt_act_pred(y_test.rename('Actual').to_frame(),
                          pd.DataFrame({'Predicted': y_pred}), dates_test, y_value)

    print('MSE: ' + str(round(math.sqrt(mean_squared_error(y_test, y_pred)), 4)))
    print('MAE: ' + str(round(mean_absolute_error(y_test, y_pred), 4)))



