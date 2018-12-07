import tools.constants as constants
from sklearn import neighbors, preprocessing, model_selection
import tools.plotting as plotting
import pandas as pd
import numpy as np
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import matplotlib.pyplot as plt

def test_for_k(X, y, X_val, y_val, title, true=False):

    results = []
    k_range = (1, 60+1)
    if true:
        for i in range(k_range[0], k_range[1]):
            kNN_regression = neighbors.KNeighborsRegressor(i, weights='distance')
            total = 0.0
            for j in range(100):
                X_train, X_test, y_train, y_test = model_selection.train_test_split(X_val, y_val, test_size=0.1)
                kNN_regression.fit(X_train, y_train)
                total += kNN_regression.score(X_test, y_test)
            results.append(total / 100)
    else:
        for i in range(k_range[0], k_range[1]):
            kNN_regression = neighbors.KNeighborsRegressor(i, weights='uniform')
            total = 0.0
            for j in range(100):
                X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.9)
                X__, X_test, y__, y_test = model_selection.train_test_split(X_val, y_val, test_size=0.1)
                kNN_regression.fit(X_train, y_train)
                total += kNN_regression.score(X_test, y_test)
            results.append(total / 100)

    plt.plot(range(k_range[0], k_range[1]), results)
    plt.xlabel("Neighbors")
    plt.ylabel("Average score")
    plt.title(title)

def run_knn(X, y, X_test, y_test, X_val, y_val, dates_test, x_values, y_value, lag,  scaler, actual):

    # hour = 3600
    # X = np.append(X, scaler.transform(X_test[:hour]), axis=0)
    # y = y.append(y_test[:hour])
    #
    # X_test = X_test[hour:]
    # y_test = y_test[hour:]
    # dates_test = dates_test[hour:]

    # test_for_k(X, y, X_val, y_val, y_value, True)
    # return

    model = neighbors.KNeighborsRegressor(n_neighbors=100, weights='uniform')

    model.fit(X, y)

    y_pred = []
    if actual:
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