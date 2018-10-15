import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import tools.constants as constants
from sklearn import linear_model, preprocessing
from tensorflow import keras

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
