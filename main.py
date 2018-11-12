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

from sklearn.metrics import mean_squared_error, r2_score
from cognite.config import configure_session
from cognite.v05.timeseries import get_datapoints_frame
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import scatter_matrix

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


input_tags = {'VAL_23-FT-92512:X.Value|average': 'Inflow',
              'VAL_23-PT-92532:X.Value|average' : 'In. press.',
              'VAL_23-TT-92533:X.Value|average' : 'In. temp.'}

output_tags = {'VAL_23-FT-92537-01:X.Value|average' : 'Out. m. flow',
               'VAL_23-FT-92537-04:X.Value|average' : 'Out. v. flow',
               'VAL_23-PT-92539:X.Value|average' : 'Out. press.',
               'VAL_23-TT-92539:X.Value|average' : 'Out. temp.'}

control_tags = {'VAL_23_ZT_92543:Z.X.Value|average' : 'Anti-surge',
                'VAL_23_ZT_92538:Z.X.Value|average' : 'In. throttle',
                'VAL_23-KA-9101_ASP:VALUE|average' : 'Power'}


def run_linear_regression(data, test_size, dates):
    lr_act_temp, lr_pred_temp = lr.predict(data[
                                               [
                                                   constants.COMPRESSOR_SUCTION_PRESSURE + '|average',
                                                   constants.COMPRESSOR_SUCTION_TEMPERATURE + '|average',
                                                   constants.COMPRESSOR_GAS_INFLOW + '|average'
                                               ]
                                           ].interpolate(),
                                           data[constants.COMPRESSOR_DISCHARGE_TEMPERATURE + '|average'].interpolate(),
                                           test_size=test_size)

    lr_act_press, lr_pred_press = lr.predict(data[
                                                 [
                                                     constants.COMPRESSOR_SUCTION_PRESSURE + '|average',
                                                     constants.COMPRESSOR_SUCTION_TEMPERATURE + '|average',
                                                     constants.COMPRESSOR_GAS_INFLOW + '|average'
                                                 ]
                                             ].interpolate(),
                                             data[constants.COMPRESSOR_DISCHARGE_PRESSURE + '|average'].interpolate(),
                                             test_size=test_size)

    print(mean_squared_error(lr_act_temp, lr_pred_temp))
    print(r2_score(lr_act_temp, lr_pred_temp))

    print(mean_squared_error(lr_act_press, lr_pred_press))
    print(r2_score(lr_act_press, lr_pred_press))



    # lr_act_temp_df = lr_act_temp.to_frame()
    # lr_act_temp_df.set_index(dates, inplace=True)
    # lr_act_temp_df.plot(label="Actual", c='b')
    # predicted_temp = pd.DataFrame({'Predicted Temp': lr_pred_temp})
    # predicted_temp.set_index(dates, inplace=True)
    # predicted_temp['Predicted Temp'].plot(label='Predicted temperature', c='r', linestyle='--')
    plotting.plt_act_pred(lr_act_temp.to_frame(), pd.DataFrame({'Predicted': lr_pred_temp}), dates, 'Temperature')

    # lr_act_press_df = lr_act_press.to_frame()
    # lr_act_press_df.set_index(dates, inplace=True)
    # lr_act_press_df.plot(label='Actual', c='b')
    # predicted_press = pd.DataFrame({'Predicted': lr_pred_press})
    # predicted_press.set_index(dates, inplace=True)
    # predicted_press['Predicted'].plot(label='Predicted pressure', c='r', linestyle='--')
    plotting.plt_act_pred(lr_act_press.to_frame(), pd.DataFrame({'Predicted': lr_pred_press}), dates ,'Pressure')



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
    start = datetime.datetime(2018,1,2)
    # start = datetime.datetime(2017,11,1)
    end = datetime.datetime(2018,1,3)
    # end = datetime.datetime(2018,11,1)
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
    dates = pd.to_datetime(data['timestamp'][-test_size:])
    data = data.drop(data.columns[0], axis=1)
    data = data.rename(index=str, columns=control_tags)
    data = data.rename(index=str, columns=input_tags)
    data = data.rename(index=str, columns=output_tags)


    scatter_matrix(data)


    # plotting.plot_input_control(data)
    # plotting.plot_output(data)
    # test_stationarity(data[constants.COMPRESSOR_DISCHARGE_VOLUME_FLOW + '|average'].interpolate(), 'Output Volume Flow')



    # run_linear_regression(data, test_size, dates)

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