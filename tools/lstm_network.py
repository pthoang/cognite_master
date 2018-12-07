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


def split_train_batches(X, y, num_batches, batch_len):

    samples = []
    labels = []

    for i in range(num_batches+1):
        samples.append(X[i*batch_len: (i+1)*batch_len])
        labels.append(y[i*batch_len: (i+1)*batch_len])


    samples[-1] = np.append(samples[-1],np.zeros((batch_len - len(samples[-1]), X.shape[1])),axis=0)
    labels[-1] = np.append(labels[-1],np.zeros((batch_len - len(labels[-1]), y.shape[1])),axis=0)

    return np.array(samples), np.array(labels)

def build_prediction_model(train_model, n_features, hidden_size, loss):
    model = keras.Sequential()

    model.add(keras.layers.Masking(mask_value=.0, batch_input_shape=(1, None, n_features)))
    model.add(keras.layers.LSTM(hidden_size, return_sequences=True, stateful=True))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.LSTM(hidden_size, return_sequences=True, stateful=True))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.LSTM(hidden_size, return_sequences=True, stateful=True))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(1, activation='linear'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0001),
        loss=loss,
        # metrics=['mae']+
    )
    model.set_weights(train_model.get_weights())


    return model


def run_lstm(X, y, X_test, y_test, X_val, y_val, dates_test, x_values, y_value, lag, scaler, old_model, save=False, loss='mse'):

    min_max_scaler_Y = preprocessing.MinMaxScaler()

    min_max_scaler_Y.partial_fit(y.values.reshape(-1,1))
    min_max_scaler_Y.partial_fit(y_test.values.reshape(-1,1))
    min_max_scaler_Y.partial_fit(y_val.values.reshape(-1,1))

    y = min_max_scaler_Y.transform(y.values.reshape(-1,1))
    y_val = min_max_scaler_Y.transform(y_val.values.reshape(-1,1))

    num_batches = 24*60
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
        model.add(keras.layers.LSTM(hidden_size, return_sequences=True))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(1, activation='linear'))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=loss,
            # metrics=['mae']
        )

        print(model.summary())

        print(X_split_batch.shape)

        history = model.fit(
            # X.reshape(1, X.shape[0], X.shape[1]), y.reshape(1, len(y), 1),
            X_split_batch, y_split_batch,
            epochs=10, batch_size=num_batches+1, shuffle=False,
            validation_data=(X_val.reshape(1, X_val.shape[0], X_val.shape[1]), y_val.reshape(1, len(y_val), 1))
        )

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])

        if save:
            path = 'models/'+'_'.join(y_value.split(' ')) + '_' + loss + ('' if lag == 0 else str(lag))+ '.h5'
            model.save(path)

    else:
        model = keras.models.load_model(old_model)
        # history = model.fit(
        #     # X.reshape(1, X.shape[0], X.shape[1]), y.reshape(1, len(y), 1),
        #     X_split_batch, y_split_batch,
        #     epochs=100, batch_size=num_batches+1, shuffle=False,
        #     validation_data=(X_val.reshape(1, X_val.shape[0], X_val.shape[1]), y_val.reshape(1, len(y_val), 1))
        # )

        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])

        if save:
            path = 'models/'+'_'.join(y_value.split(' ')) + '_' + ('' if lag == 0 else str(lag))+ '.h5'
            model.save(path)


    pred_model = build_prediction_model(model, len(X_test.columns), hidden_size, loss)

    pre_num = 3600
    X_test_pre = scaler.transform(X_test[:pre_num])
    X_test = X_test.iloc[pre_num:]
    y_test_pre = min_max_scaler_Y.transform(y_test.iloc[:pre_num].values.reshape(-1,1))
    y_test = y_test.iloc[pre_num:]
    dates_test = dates_test[pre_num:]

    for i in range(3):
        pred_model.reset_states()
        pred_model.fit(
            X_test_pre.reshape(1, X_test_pre.shape[0], X_test_pre.shape[1]), y_test_pre.reshape(1, len(y_test_pre), 1),
            epochs=1, batch_size=1, shuffle=False,
        )

    y_pred = []

    if lag > 0:
        row_lag = X_test.iloc[0, -lag:]
        X_nolag = X_test[x_values]
        for i in range(len(X_nolag)):
            row = X_nolag.iloc[i].append(row_lag)
            pred = pred_model.predict_on_batch(scaler.transform(row.values.reshape(1, -1)).reshape(1,1,len(x_values) + lag))
            y_pred.append(pred[0][0])
            row_lag = row_lag.iloc[1:].append(pd.Series(min_max_scaler_Y.inverse_transform(pred[0])[0][0]))
        y_pred = min_max_scaler_Y.inverse_transform(np.array(y_pred)).reshape(-1)
    else:
        X_test = scaler.transform(X_test)
        y_pred = pred_model.predict(X_test.reshape(1, X_test.shape[0], X_test.shape[1]))
        y_pred = min_max_scaler_Y.inverse_transform(y_pred[0]).reshape(-1)

    plotting.plt_act_pred(y_test.rename('Actual').to_frame(),
                          pd.DataFrame({'Predicted': y_pred}), dates_test, y_value)

    print('MSE: ' + str(round(math.sqrt(mean_squared_error(y_test, y_pred)), 4)))
    print('MAE: ' + str(round(mean_absolute_error(y_test, y_pred), 4)))



