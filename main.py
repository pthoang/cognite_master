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

from sklearn.metrics import mean_squared_error, r2_score
from cognite.config import configure_session
from cognite.v05.timeseries import get_datapoints_frame

configure_session(os.environ.get('PUBLIC_DATA_KEY'), 'publicdata')


input_tags = {'VAL_23-FT-92512:X.Value':'Gas inflow from separators',
              'VAL_23-PT-92532:X.Value' : 'Compressor suction pressure',
              'VAL_23-TT-92533:X.Value' : 'Compressor suction temperature'}

output_tags = {'VAL_23-FT-92537-01:X.Value' : 'Compressor discharge mass flow',
               'VAL_23-FT-92537-04:X.Value' : 'Compressor discharge volume flow',
               'VAL_23-PT-92539:X.Value' : 'Compressor discharge pressure',
               'VAL_23-TT-92539:X.Value' : 'Compressor discharge temperature'}

control_tags = {'VAL_23_ZT_92543:Z.X.Value' : 'Anti-surge valve position',
                'VAL_23_ZT_92538:Z.X.Value' : 'Suction throttle valve position',
                'VAL_23-KA-9101_ASP:VALUE' : 'Compressor shaft power'}


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


def plot_input_control(data):
    fig1, axs1 = plt.subplots(3, 1, figsize=(10, 10))
    fig1.subplots_adjust(hspace=0.3)
    fig2, axs2 = plt.subplots(3, 1, figsize=(10, 10))
    fig2.legend().set_visible(False)
    fig2.subplots_adjust(hspace=0.3)

    axs1[0].plot(data[constants.COMPRESSOR_SUCTION_TEMPERATURE + '|average'], label='_nolegend_')
    axs1[0].set_title('Input temperature degC')

    axs1[1].plot(data[constants.COMPRESSOR_SUCTION_PRESSURE + '|average'], label='_nolegend_')
    axs1[1].set_title('Input pressure barg')

    axs1[2].plot(data[constants.COMPRESSOR_GAS_INFLOW + '|average'], label='_nolegend_')
    axs1[2].set_title('Gas inflow')

    axs2[0].plot(data[constants.ANTI_SURGE_VALVE_POSITION + '|average'].interpolate(), label='_nolegend_')
    axs2[0].set_title('Anti-surge valve position %')

    axs2[1].plot(data[constants.SUCTION_THROTTLE_VALVE_POSITION + '|average'].interpolate(), label='_nolegend_')
    axs2[1].set_title('Suction throttle valve position %')

    axs2[2].plot(data[constants.COMPRESSOR_SHAFT_POWER + '|average'].interpolate(), label='_nolegend_')
    axs2[2].set_title('Shaft power kW')

def main():
    start = datetime.datetime(2013,9,1)
    end = datetime.datetime(2018,9,7)
    granularity = '1h'
    aggregates = [
        'avg',
        'min',
        'max'
    ]
    tags = [
        constants.COMPRESSOR_SUCTION_PRESSURE,
        constants.COMPRESSOR_SUCTION_TEMPERATURE,
        constants.COMPRESSOR_DISCHARGE_TEMPERATURE,
        constants.COMPRESSOR_GAS_INFLOW,
        constants.COMPRESSOR_DISCHARGE_PRESSURE,
        constants.ANTI_SURGE_VALVE_POSITION,
        constants.SUCTION_THROTTLE_VALVE_POSITION,
        constants.COMPRESSOR_SHAFT_POWER,
    ]
    test_size = 30
    data  = get_datapoints_frame(time_series=tags, start=start, end=end, granularity=granularity, aggregates=aggregates)
    data.set_index(pd.to_datetime(data['timestamp'], unit='ms'), inplace=True)


    plot_input_control(data)


    dates = pd.to_datetime(data['timestamp'][-test_size:])

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