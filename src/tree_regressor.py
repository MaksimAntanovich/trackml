import pickle
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from data_path import DATA_PATH

X = pickle.load(open(DATA_PATH + '/train_X.pkl', 'rb'))
print(X.shape)
print(X[0,:])
y = pickle.load(open(DATA_PATH + '/train_Y.pkl', 'rb'))
print(y.shape)
regressors = []
for i in range(0, y.shape[1]):
    regressor = xgb.XGBRegressor()
    regressor = regressor.fit(X=X[:2800], y=y[:2800, i])
    regressors.append(regressor)

predicts = []
for regressor in regressors:
    predicts.append(regressor.predict(data=X[2800:]))

y_hat = np.transpose(np.vstack(tuple(predicts)))
y_true = y[2800:]
print(mean_squared_error(y_true=y_true, y_pred=y_hat))