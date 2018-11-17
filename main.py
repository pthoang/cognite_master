import datetime
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import tools.linear_regression as lr
import tools.lstm_network as lstm
import tools.constants as constants
import tools.plotting as plotting
import seaborn as sns

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from cognite.config import configure_session
from cognite.v05.timeseries import get_datapoints_frame
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import scatter_matrix
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import autocorrelation_plot

configure_session(os.environ.get('PUBLIC_DATA_KEY'), 'publicdata')


input_tags = {'VAL_23-FT-92512:X.Value|average': 'Gas inflow from separators',
              'VAL_23-PT-92532:X.Value|average' : 'Suction pressure',
              'VAL_23-TT-92533:X.Value|average' : 'Suction temperature'}

output_tags = {'VAL_23-FT-92537-01:X.Value|average' : 'Discharge mass flow',
               'VAL_23-FT-92537-04:X.Value|average' : 'Discharge volume flow',
               'VAL_23-PT-92539:X.Value|average' : 'Discharge pressure',
               'VAL_23-TT-92539:X.Value|average' : 'Discharge temperature'}

control_tags = {'VAL_23_ZT_92543:Z.X.Value|average' : 'Anti-surge valve position',
                'VAL_23_ZT_92538:Z.X.Value|average' : 'Suction throttle valve position',
                'VAL_23-KA-9101_ASP:VALUE|average' : 'Shaft power'}


# input_tags = {'VAL_23-FT-92512:X.Value|average': 'Inflow',
#               'VAL_23-PT-92532:X.Value|average' : 'In. press.',
#               'VAL_23-TT-92533:X.Value|average' : 'In. temp.'}
#
# output_tags = {'VAL_23-FT-92537-01:X.Value|average' : 'Out. m. flow',
#                'VAL_23-FT-92537-04:X.Value|average' : 'Out. v. flow',
#                'VAL_23-PT-92539:X.Value|average' : 'Out. press.',
#                'VAL_23-TT-92539:X.Value|average' : 'Out. temp.'}
#
# control_tags = {'VAL_23_ZT_92543:Z.X.Value|average' : 'Anti-surge',
#                 'VAL_23_ZT_92538:Z.X.Value|average' : 'In. throttle',
#                 'VAL_23-KA-9101_ASP:VALUE|average' : 'Power'}

def preprocess_data(X, y, X_test, y_test):
    X = X.interpolate()
    y = y.interpolate()
    X_test = X_test.interpolate()
    y_test = y_test.interpolate()

    print('Remaining NaN count')
    print(X.isna().sum())
    print(y.isna().sum())

    min_max_scaler = preprocessing.MinMaxScaler()

    min_max_scaler.partial_fit(X)
    min_max_scaler.partial_fit(X_test)

    X_scaled = min_max_scaler.transform(X)

    remaining_nan = max(X.isna().sum().max(), y.isna().sum())
    remaining_nan_test = max(X_test.isna().sum().max(), y_test.isna().sum())

    return X_scaled[remaining_nan:], y[remaining_nan:], X_test[remaining_nan_test:], y_test[remaining_nan_test:], \
           remaining_nan, remaining_nan_test, min_max_scaler


def add_lagged_var(X, y, y_value, num_lag, dates):
    X = X.iloc[num_lag:]

    y_lagged = []
    for i, lag in enumerate(range(num_lag, 0, -1)):
        new_y_lag = y.iloc[i:-lag].rename(y_value + ' lag ' + str(lag))
        new_y_lag = new_y_lag.to_frame()
        new_y_lag.set_index(dates.iloc[num_lag:], inplace=True)
        y_lagged.append(new_y_lag)



    for lag in y_lagged:
        X = X.join(lag)

    return X, y[num_lag:]

def test_stationarity(timeseries, title):
    # Determing rolling statistics
    rolmean = timeseries.rolling(30).mean()
    rolstd = timeseries.rolling(30).std()

    # Plot rolling statistics:
    orig = plt.plot(timeseries, label='Values')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title(title)
    plt.show(block=False)

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)



def main():
    #Plot 3h plots
    # start = datetime.datetime(2018,9,1, second=7)
    # end = datetime.datetime(2018,9,1, hour=3, second=7)
    #Plot correlation
    # start = datetime.datetime(2018,1,2)
    # end = datetime.datetime(2018,1,3)
    #Plot test stationary
    # start = datetime.datetime(2017,11,1)
    # end = datetime.datetime(2018,11,1)
    # granularity = '24h'

    #Data samples
    start = datetime.datetime(2018,1,1, hour=23, minute=59, second=23)
    end = datetime.datetime(2018,1,3, second=0)

    granularity = '1s'
    aggregates = [
        'avg',
        # 'min',
        # 'max'
    ]
    tags = [
        constants.COMPRESSOR_SUCTION_PRESSURE,
        constants.COMPRESSOR_SUCTION_TEMPERATURE,
        constants.COMPRESSOR_GAS_INFLOW,
        constants.COMPRESSOR_DISCHARGE_TEMPERATURE,
        constants.COMPRESSOR_DISCHARGE_PRESSURE,
        constants.COMPRESSOR_DISCHARGE_MASS_FLOW,
        constants.COMPRESSOR_DISCHARGE_VOLUME_FLOW,
        constants.ANTI_SURGE_VALVE_POSITION,
        constants.SUCTION_THROTTLE_VALVE_POSITION,
        constants.COMPRESSOR_SHAFT_POWER,
    ]
    test_size = 30
    data  = get_datapoints_frame(time_series=tags, start=start, end=end, granularity=granularity, aggregates=aggregates)
    data.set_index(pd.to_datetime(data['timestamp'], unit='ms'), inplace=True)
    dates = pd.to_datetime(data['timestamp'], unit='ms')
    # plotting.plot_input_control(data)
    # plotting.plot_output(data)
    data = data.drop(data.columns[0], axis=1)
    data = data.rename(index=str, columns=control_tags)
    data = data.rename(index=str, columns=input_tags)
    data = data.rename(index=str, columns=output_tags)

    #Test data
    start_minute = 59
    extra_seconds = (60 - start_minute)*60
    start = datetime.datetime(2018,10, 2, hour=23, minute=start_minute, second=1)
    end = datetime.datetime(2018, 10, 3, hour=1, second=1)
    # end = datetime.datetime(2018, 10, 4, second=1)
    data_test = get_datapoints_frame(time_series=tags, start=start, end=end, granularity=granularity, aggregates=aggregates)
    data_test.set_index(pd.to_datetime(data_test['timestamp'], unit='ms'), inplace=True)
    dates_test = pd.to_datetime(data_test['timestamp'], unit='ms')
    # plotting.plot_input_control(data_test)
    data_test = data_test.drop(data_test.columns[0], axis=1)
    data_test = data_test.rename(index=str, columns=control_tags)
    data_test = data_test.rename(index=str, columns=input_tags)
    data_test = data_test.rename(index=str, columns=output_tags)

    # scatter_matrix(data)

    # col = 'Discharge volume flow'
    # plot_acf(data[col].interpolate(),lags=60, title=col+' ACF')
    # plot_pacf(data[col].interpolate(),lags=60, title=col + ' PACF')

    # test_stationarity(data[constants.COMPRESSOR_DISCHARGE_VOLUME_FLOW + '|average'].interpolate(), 'Output Volume Flow')

    x_values = [
        'Suction temperature',
        'Suction pressure',
        'Gas inflow from separators',
        'Anti-surge valve position',
        'Suction throttle valve position',
        'Shaft power'
    ]

    y_value = 'Discharge temperature'

    X = data[x_values]
    y = data[y_value]
    X_test = data_test[x_values]
    y_test = data_test[y_value]

    lag = 0
    X, y = add_lagged_var(X, y, y_value, lag, dates)
    dates = dates[lag:]
    X_test, y_test = add_lagged_var(X_test, y_test, y_value, lag, dates_test)
    dates_test = dates_test[lag:]


    X, y, X_test, y_test, remaining_nan, remaining_nan_test, scaler = preprocess_data(X, y, X_test, y_test)
    dates = dates[remaining_nan:]
    dates_test = dates_test[extra_seconds:]


    X_test = X_test.iloc[extra_seconds-remaining_nan_test:]
    y_test = y_test.iloc[extra_seconds-remaining_nan_test:]

    lr.run_linear_regression(X, y, X_test, y_test, dates, dates_test,x_values, y_value, lag, scaler, one_step=False)



    # lstm_act_temp, lstm_pred_temp = lstm.predict(data[
    #                                                  [
    #                                                     constants.COMPRESSOR_SUCTION_PRESSURE + '|average',
    #                                                     constants.COMPRESSOR_SUCTION_TEMPERATURE + '|average',
    #                                                     constants.COMPRESSOR_GAS_INFLOW + '|average'
    #                                                  ]
    #                                              ].interpolate(),
    #                                              data[constants.COMPRESSOR_DISCHARGE_TEMPERATURE + '|average'].interpolate(),
    #                                              test_size=test_size)
    #
    #
    # plotting.plt_act_pred(pd.DataFrame({'Actual': lstm_act_temp}), pd.DataFrame({'Predicted': lstm_pred_temp}), dates, 'Temperature')



    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()