import tools.constants as constants
from sklearn import linear_model, preprocessing
from statsmodels.tsa import ar_model
import tools.plotting as plotting
import pandas as pd
import numpy as np
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math


def predict(X, Y, test_size):
    min_max_scaler = preprocessing.MinMaxScaler()
    X_scaled =  min_max_scaler.fit_transform(X)

    X_train = X_scaled[:-test_size]
    X_test  = X_scaled[-test_size:]
    Y_train = Y[:-test_size]
    Y_test = Y[-test_size:]


    regr = linear_model.LinearRegression()

    regr.fit(X_train, Y_train)

    Y_pred = regr.predict(X_test)

    return Y_test, Y_pred


def run_linear_regression(X, y, X_test, y_test, dates, dates_test, x_values, y_value, lag, scaler, one_step):


    model = linear_model.LinearRegression()

    hour = 3600
    X = np.append(X, scaler.transform(X_test[:hour]), axis=0)
    y = y.append(y_test[:hour])

    X_test = X_test[hour:]
    y_test = y_test[hour:]
    dates_test = dates_test[hour:]

    model.fit(X, y)

    y_pred = []

    if one_step:
        y_pred = model.predict(scaler.transform(X_test))
    else:
        if lag > 0:
            row_lag = X_test.iloc[0, -lag:]
            X_nolag = X_test[x_values]
            for i in range(len(X_nolag)):
                row = X_nolag.iloc[i].append(row_lag)
                pred = model.predict(scaler.transform(row.values.reshape(1, -1)))
                y_pred.append(pred[0])
                row_lag = row_lag.iloc[1:].append(pd.Series([pred[0]]))
        else:
            y_pred = model.predict(scaler.transform(X_test))

    plotting.plt_act_pred(y_test.rename('Actual').to_frame(), pd.DataFrame({'Predicted': y_pred}), dates_test, y_value)

    print('MSE: ' + str(round(math.sqrt(mean_squared_error(y_test, y_pred)), 4)))
    print('MAE: ' + str(round(mean_absolute_error(y_test, y_pred), 4)))


    # lr_act_temp, lr_pred_temp = predict(data[
    #                                            [
    #                                                constants.COMPRESSOR_SUCTION_PRESSURE + '|average',
    #                                                constants.COMPRESSOR_SUCTION_TEMPERATURE + '|average',
    #                                                constants.COMPRESSOR_GAS_INFLOW + '|average'
    #                                            ]
    #                                        ].interpolate(),
    #                                        data[constants.COMPRESSOR_DISCHARGE_TEMPERATURE + '|average'].interpolate(),
    #                                        test_size=test_size)
    #
    # lr_act_press, lr_pred_press = predict(data[
    #                                              [
    #                                                  constants.COMPRESSOR_SUCTION_PRESSURE + '|average',
    #                                                  constants.COMPRESSOR_SUCTION_TEMPERATURE + '|average',
    #                                                  constants.COMPRESSOR_GAS_INFLOW + '|average'
    #                                              ]
    #                                          ].interpolate(),
    #                                          data[constants.COMPRESSOR_DISCHARGE_PRESSURE + '|average'].interpolate(),
    #                                          test_size=test_size)

    # lr_act_temp_df = lr_act_temp.to_frame()
    # lr_act_temp_df.set_index(dates, inplace=True)
    # lr_act_temp_df.plot(label="Actual", c='b')
    # predicted_temp = pd.DataFrame({'Predicted Temp': lr_pred_temp})
    # predicted_temp.set_index(dates, inplace=True)
    # predicted_temp['Predicted Temp'].plot(label='Predicted temperature', c='r', linestyle='--')
    # plotting.plt_act_pred(lr_act_temp.to_frame(), pd.DataFrame({'Predicted': lr_pred_temp}), dates, 'Temperature')

    # lr_act_press_df = lr_act_press.to_frame()
    # lr_act_press_df.set_index(dates, inplace=True)
    # lr_act_press_df.plot(label='Actual', c='b')
    # predicted_press = pd.DataFrame({'Predicted': lr_pred_press})
    # predicted_press.set_index(dates, inplace=True)
    # predicted_press['Predicted'].plot(label='Predicted pressure', c='r', linestyle='--')
    # plotting.plt_act_pred(lr_act_press.to_frame(), pd.DataFrame({'Predicted': lr_pred_press}), dates, 'Pressure')

