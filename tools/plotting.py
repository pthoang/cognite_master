import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import tools.constants as constants
import pandas as pd

def plt_act_pred(act, pred, dates, title):
    act.set_index(dates, inplace=True)
    act.plot(label='Actual', c='b', title=title)
    pred.set_index(dates, inplace=True)
    pred['Predicted'].plot(label='Predicted', c='r', linestyle='--')


def plot_input_control(data):
    fig1, axs1 = plt.subplots(3, 1, figsize=(10, 10))
    fig1.subplots_adjust(hspace=0.3)
    fig2, axs2 = plt.subplots(3, 1, figsize=(10, 10))
    fig2.legend().set_visible(False)
    fig2.subplots_adjust(hspace=0.3)

    axs1[0].plot(data[constants.COMPRESSOR_SUCTION_TEMPERATURE + '|average'].interpolate(), label='_nolegend_')
    axs1[0].set_title('Suction temperature')

    axs1[1].plot(data[constants.COMPRESSOR_SUCTION_PRESSURE + '|average'].interpolate(), label='_nolegend_')
    axs1[1].set_title('Suction pressure')

    axs1[2].plot(data[constants.COMPRESSOR_GAS_INFLOW + '|average'].interpolate(), label='_nolegend_')
    axs1[2].set_title('Gas inflow from separators')

    axs2[0].plot(data[constants.ANTI_SURGE_VALVE_POSITION + '|average'].interpolate(), label='_nolegend_')
    axs2[0].set_title('Anti-surge valve position')

    axs2[1].plot(data[constants.SUCTION_THROTTLE_VALVE_POSITION + '|average'].interpolate(), label='_nolegend_')
    axs2[1].set_title('Suction throttle valve position')

    axs2[2].plot(data[constants.COMPRESSOR_SHAFT_POWER + '|average'].interpolate(), label='_nolegend_')
    axs2[2].set_title('Shaft power')

    # test = {}
    # count = 0
    # for line in data[constants.COMPRESSOR_SHAFT_POWER + '|average']:
    #     print(pd.isna(line))
    #     if pd.isna(line):
    #         count += 1
    #     else:
    #         try:
    #             test[str(count)] += 1
    #         except:
    #             test[str(count)] = 1
    #
    #         count = 0


def plot_output(data):
    fig, axs = plt.subplots(4, 1, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.3)

    axs[0].plot(data[constants.COMPRESSOR_DISCHARGE_TEMPERATURE + '|average'], label='_nolegend_')
    axs[0].set_title('Output temperature degC')