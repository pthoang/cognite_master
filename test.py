import datetime
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

os.environ['CUDA_VISIBLE_DEVICES']='-1'

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


def main():
    # Let us first figure out what data is available
    start = datetime.datetime(1970, 1, 1)
    end = '1w-ago'
    granularity = '1d'
    aggregates = ['avg']
    tags = list(input_tags.keys())
    data = get_datapoints_frame(time_series = tags, start=start, end=end, granularity=granularity, aggregates=aggregates)

    T = pd.to_datetime(data.timestamp, unit='ms')
    print(T.iloc[0], T.iloc[-1])

    # Next, let us see the input pressure vs output pressure over the last year
    # on hourly aggregates
    start = '365d-ago'
    end = '1w-ago'
    granularity = '1h'
    aggregates = ['avg', 'min', 'max']
    tags = ['VAL_23-PT-92532:X.Value', 'VAL_23-PT-92539:X.Value']
    data = get_datapoints_frame(time_series=tags, start=start, end=end, granularity=granularity, aggregates=aggregates)

    fig, ax = plt.subplots(figsize=(8, 5))
    T = pd.to_datetime(data.timestamp, unit='ms')
    for var in data.drop(['timestamp'], axis=1).columns:
        ax.plot(T, data[var], label=var)

    plt.legend()
    plt.show()

    # Let us look at the min out - max inn
    data['minmax'] = data['VAL_23-PT-92539:X.Value|min'] - data['VAL_23-PT-92532:X.Value|max']
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(T, data['minmax'])

    plt.legend()
    plt.show()

    # Let us look at the difference at 1s resolution
    start = '2w-ago'
    end = '1w-ago'
    granularity = '1s'
    aggregates = ['avg']
    tags = ['VAL_23-PT-92532:X.Value', 'VAL_23-PT-92539:X.Value']
    data = get_datapoints_frame(time_series=tags, start=start, end=end, granularity=granularity, aggregates=aggregates)
    T = pd.to_datetime(data.timestamp, unit='ms')
    data['minmax'] = data['VAL_23-PT-92539:X.Value|average'] - data['VAL_23-PT-92532:X.Value|average']
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(T, data['minmax'])

    plt.show()

    # Seems pretty steady, compressing the gas by increasing the pressure with approximately 10BARS.

if __name__ == '__main__':
    main()
