import tools.constants as constants
from sklearn import linear_model, preprocessing


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
